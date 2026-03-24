"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

Setup (one-time):
    modal setup
    modal volume create flashinfer-trace
    modal volume put flashinfer-trace /path/to/flashinfer-trace/
"""

import json
import sys
import statistics
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
# The volume root contains a nested flashinfer-trace/ directory (from the HuggingFace clone)
TRACE_SET_PATH = "/data/flashinfer-trace"

# cuda:12.8 devel image provides nvcc so load_inline can JIT-compile our CUDA kernel.
# B200 (Blackwell, SM_100) requires CUDA 12.8+.
# pip torch (2.6+) compiled against CUDA 12.8 supports SM_100.
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={TRACE_SET_PATH: trace_volume})
def run_benchmark(solution_json: str, config: BenchmarkConfig = None) -> dict:
    """Run benchmark on Modal B200 and return results."""
    # Deserialize on the remote side — avoids Pydantic private-attr issues with Modal
    solution = Solution.model_validate_json(solution_json)

    if config is None:
        config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    import os
    print(f"Volume mounted at /data: {os.listdir('/data') if os.path.exists('/data') else 'MISSING'}")
    print(f"Trace path: {TRACE_SET_PATH}")
    trace_set = TraceSet.from_path(TRACE_SET_PATH)
    print(f"Definitions: {list(trace_set.definitions.keys())}")

    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])

    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])

    # Print first non-success trace in full for debugging
    for trace in traces:
        if trace.evaluation and trace.evaluation.status.value != "success":
            print(f"\n=== FIRST ERROR TRACE ===")
            print(trace.evaluation.model_dump_json(indent=2)[:3000])
            print(f"=== END ===\n")
            break

    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            if trace.evaluation.status.value != "success":
                entry["error"] = str(trace.evaluation.model_dump())[:500]
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            results[definition.name][trace.workload.uuid] = entry

    return results


def print_results(results: dict):
    """Print benchmark results in a formatted way."""
    for def_name, traces in results.items():
        print(f"\n{def_name}:")
        for workload_uuid, result in traces.items():
            status = result.get("status")
            print(f"  Workload {workload_uuid[:8]}...: {status}", end="")

            if result.get("error") and status != "success":
                print(f"\n    Error: {result['error'][:300]}", end="")

            if result.get("latency_ms") is not None:
                print(f" | {result['latency_ms']:.3f} ms", end="")

            if result.get("speedup_factor") is not None:
                print(f" | {result['speedup_factor']:.2f}x speedup", end="")

            if result.get("max_abs_error") is not None:
                abs_err = result["max_abs_error"]
                rel_err = result.get("max_rel_error", 0)
                print(f" | abs_err={abs_err:.2e}, rel_err={rel_err:.2e}", end="")

            print()


def save_results(results: dict, solution):
    """Save raw JSON results + a human-readable summary to results/."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sol_name = getattr(solution, "name", "unknown").replace(" ", "_").replace("/", "_")
    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    # Raw JSON
    raw_path = out_dir / f"{sol_name}_{ts}.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    speedups = []
    passed = failed = 0
    for def_name, traces in results.items():
        for wl_uuid, r in traces.items():
            if r.get("status") in ("PASSED", "success"):
                passed += 1
            else:
                failed += 1
            if r.get("speedup_factor") is not None:
                speedups.append(r["speedup_factor"])

    summary_path = out_dir / f"{sol_name}_{ts}_summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Benchmark results — {sol_name}\n\n")
        f.write(f"- **Run time (UTC)**: {ts}\n")
        f.write(f"- **Hardware**: NVIDIA B200 (Modal)\n")
        f.write(f"- **Workloads**: {passed} PASSED, {failed} FAILED\n\n")
        if speedups:
            speedups_sorted = sorted(speedups)
            top_half = speedups_sorted[len(speedups_sorted) // 2:]
            f.write("## Speedup statistics (all workloads)\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Mean   | {statistics.mean(speedups):.2f}x |\n")
            f.write(f"| Median | {statistics.median(speedups):.2f}x |\n")
            f.write(f"| Min    | {min(speedups):.2f}x |\n")
            f.write(f"| Max    | {max(speedups):.2f}x |\n\n")
            f.write("## Speedup statistics (larger half)\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Mean   | {statistics.mean(top_half):.2f}x |\n")
            f.write(f"| Median | {statistics.median(top_half):.2f}x |\n")

    print(f"\nResults saved:")
    print(f"  Raw JSON : {raw_path}")
    print(f"  Summary  : {summary_path}")


@app.local_entrypoint()
def main():
    """Pack solution and run benchmark on Modal."""
    from pack_solution_custom import pack_solution

    print("Packing solution from source files...")
    solution_path = pack_solution()

    print("\nLoading solution...")
    solution_json = solution_path.read_text()
    solution = Solution.model_validate_json(solution_json)
    print(f"Loaded: {solution.name} ({solution.definition})")

    print("\nRunning benchmark on Modal B200...")
    # Pass raw JSON string — avoids Pydantic private-attr serialization issues
    results = run_benchmark.remote(solution_json)

    if not results:
        print("No results returned!")
        return

    print_results(results)
    save_results(results, solution)
