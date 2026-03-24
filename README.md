# [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)

Create high-performance GPU kernels for state-of-the-art LLM architectures on NVIDIA Blackwell GPUs with humans and/or AI agents.

---

<p align="center">
  <a href="https://www.nvidia.com"><img src="images/nvidia-logo.svg" alt="NVIDIA" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://modal.com"><img src="images/modal-logo.png" alt="Modal" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://mlsys.org"><img src="images/mlsys-logo.svg" alt="MLSys" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer"><img src="images/flashinfer-logo.png" alt="FlashInfer" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer-bench"><img src="images/fib_logo.png" alt="FlashInfer-Bench" height="50"/></a>
</p>

---

[FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) is our official framework to evaluate your AI-generated kernels.

## 🏆 Our Solution — DSA TopK Indexer (dsa_topk_indexer_fp8_h64_d128_topk2048_ps64)

**Winner: solution11** — 8-stream multi-stream CUDA graph with fork/join capture.

### Final Results (NVIDIA B200, Modal)

| Metric | Value |
|--------|-------|
| **Workloads passed** | **128 / 128** (abs_err = 0.00, rel_err = 0.00) |
| **Mean speedup** | **14.14×** |
| **Median speedup** | **14.19×** |
| **Large-half mean** | **17.07×** |
| Min speedup | 6.94× |
| Max speedup | 22.64× |

### How we got here — full progression

| Solution | Key idea | B200 mean | H100 mean | Passes |
|---|---|---|---|---|
| Reference (solution3) | Unoptimized Python baseline | 1.0× | 1.0× | ✅ 128/128 |
| solution/ (v8) | Custom CUDA kernels, single-stream, no graph | — | 4.44× | ✅ 128/128 |
| solution7 | 4 parallel CUDA streams | 5.43× | 5.84× | ✅ 128/128 |
| solution8 | 4 streams + TMA prefetch | 5.26× | — | ✅ 128/128 |
| solution9 | 1-stream CUDA graph (eliminates dispatch overhead) | 5.39× | 6.07× | ✅ 128/128 |
| solution10 | **4-stream CUDA graph** (fork/join) | 10.81× | 12.49× | ✅ 128/128 |
| **solution11** ✅ | **8-stream CUDA graph** | **14.14×** | **15.76×** | ✅ **128/128** |
| solution12 ❌ | FP8 GEMM (`_scaled_mm`) — correctness fails | 12.04× | — | ❌ 117/128 |

### Key engineering decisions in solution11

**1. Static-input CUDA graph** (`torch.cuda.CUDAGraph`)
Every call: async-copy live inputs → static GPU buffers → `g.replay()` → async-copy output back.
The graph replays in ~1 µs vs ~50+ µs per call for raw ATen dispatch, eliminating all cuBLAS handle-lookup and kernel-selection overhead.

**2. Fork/join multi-stream capture** (`N_STREAMS = 8`)
The 8 side streams join the capture tree via a persistent `fork_ev` recorded on the default stream inside the graph context. Each of the B batch items dispatches to `streams[b % 8]`.
For a 30-item batch: 4 items run serially per stream, 8 streams in parallel = `~4×` serialisation vs `~30×` serial.

**3. Persistent CUDA events**
`cudaEventCreate`/`Destroy` are CPU-only ops that cannot be captured. `fork_ev` and all `done_evs` are allocated once at initialisation and reused across every replay, ensuring graph safety.

**4. Per-stream pre-allocated dequant buffers**
Each stream owns its own `k_bufs[s]` (`[max_tokens, 128]` float32). No dynamic allocation inside the hot path; the graph captures fixed-address ops only.

**5. Why FP8 GEMM didn't work (solution12)**
`torch._scaled_mm` on H100/B200 uses FP16-precision multiply internally (not float32). This causes up to ~0.013 absolute score error, enough to reorder topk indices for near-tie tokens → fails abs_err = 0.00 requirement.

### Running solution11

```bash
# Local (H100)
FIB_DATASET_PATH=/path/to/flashinfer-trace python scripts/run_local.py

# Cloud (B200 via Modal)
modal run scripts/run_modal.py
```

`config.toml` already points to `solution11/triton`.

---

## Updates

* 2026.02.05: Full dataset for definitions and workloads are released at [HuggingFace](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)

## Competition Tracks

The competition features three tracks, each targeting a critical LLM operation:

| Track | Description |
|-------|-------------|
| **fused_moe** | Fused Mixture-of-Experts kernel for efficient expert routing and computation |
| **sparse_attention** | Sparse attention mechanisms for long-context inference |
| **gated_delta_net** | Gated delta network operations for efficient state updates |

**Fork this template once per track** you want to compete in (separate repos for each track).

## Getting Started

### 1. Fork This Template

Click "Use this template" or fork this repository to create your solution repo.

### 2. Install Dependencies

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal
```

### 3. Download the TraceSet

We provide kernel definitions and workloads in [FlashInfer-Trace format](https://bench.flashinfer.ai/docs/flashinfer-trace). Clone the competition dataset from HuggingFace:

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
```

Set the environment variable:

```bash
export FIB_DATASET_PATH=/path/to/flashinfer-trace
```

### 4. Configure Your Solution

Edit `config.toml` to set your track and team info:

```toml
[solution]
name = "my-team-solution-v1"      # Solution name
definition = "fused_moe"          # Track: fused_moe | sparse_attention | gated_delta_net
author = "team-name"              # Team/author name

[build]
language = "triton"               # triton | cuda
entry_point = "kernel"            # Kernel function name
```

### 5. Implement Your Kernel

**For Triton:**
Edit `solution/triton/kernel.py` with your implementation.

**For CUDA:**
Edit `solution/cuda/kernel.cu` and `solution/cuda/binding.py` with your implementation.

## Development Workflow

### Pack Your Solution

Generate `solution.json` from your source files:

```bash
python scripts/pack_solution.py
```

### Run Local Benchmarks

Test your solution on your local GPU:

```bash
python scripts/run_local.py
```

Requires: Local CUDA-capable GPU and `FIB_DATASET_PATH` environment variable.

### Run Cloud Benchmarks (Modal)

Test your solution on NVIDIA B200 GPUs via Modal:

**One-time setup:**

```bash
modal setup
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/flashinfer-trace
```

**Run benchmark:**

```bash
modal run scripts/run_modal.py
```

## Submission

To submit your solution for evaluation:

1. Ensure your implementation is complete and tested
2. Run `python scripts/pack_solution.py` to generate `solution.json`
3. Commit and push your changes
4. Tag your commit for evaluation (e.g., `git tag submission-v1`)

## Project Structure

```
flashinfer-bench-starter-kit/
├── README.md                    # This file
├── config.toml                  # Track configuration (edit this)
├── solution/                    # Solution source files
│   ├── triton/                  # Triton implementation
│   │   └── kernel.py           # Your Triton kernel
│   └── cuda/                    # CUDA implementation
│       ├── kernel.cu           # Your CUDA kernel
│       └── binding.py          # TVM FFI bindings
├── scripts/                     # Utility scripts
│   ├── run_local.py            # Local benchmark runner
│   ├── run_modal.py            # Modal cloud benchmark runner
│   └── pack_solution.py        # Pack source files into solution.json
└── images/                      # Sponsor logos
```

## Additional Resources

### FlashInfer Trace Viewer

FlashInfer Trace consists of multiple JSON objects (definitions, workloads, solutions, and traces), which can contain large code blocks. To easily visualize and inspect these objects, you can use the [FlashInfer Trace Viewer](https://bench.flashinfer.ai/viewer). Simply paste any FlashInfer Trace JSON into the viewer to get a friendly, structured view of its contents.

### Solution Handling API

```python
from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files, extract_solution_to_files

# Pack source files into a Solution object
spec = BuildSpec(
    language="triton",  # or "cuda"
    target_hardware=["cuda"],
    entry_point="my_kernel",
)
solution = pack_solution_from_files(
    path="./my_solution_dir",
    spec=spec,
    name="my_solution_v1",
    definition="fused_moe",
    author="your_name",
)

# Extract a Solution to files in a working directory
extract_solution_to_files(solution, "./output_dir")
```

### Running Sanitizers

```python
from flashinfer_bench.agents import flashinfer_bench_run_sanitizer

output = flashinfer_bench_run_sanitizer(
    solution=solution,
    workload=workload,
    sanitizer_types=["memcheck", "racecheck", "synccheck", "initcheck"],
    timeout=300,
)
print(output)
```

### NCU Profiling

```python
from flashinfer_bench.agents import flashinfer_bench_run_ncu

output = flashinfer_bench_run_ncu(
    solution=solution,
    workload=workload,
    set="detailed",
    page="details",
    timeout=120,
)
print(output)
```

### List Available Tools

```python
from flashinfer_bench.agents import get_all_tool_schemas

schemas = get_all_tool_schemas()
# Returns list of OpenAI-compatible function schemas
```

## Notes

### Destination Passing Style (DPS)

FlashInfer-Bench uses destination passing style (DPS) by default, where both inputs and outputs are passed as function parameters. DPS avoids measuring tensor allocation overhead, resulting in more accurate performance numbers. We recommend using DPS when possible, as it yields better benchmark results.

**Important:** Avoid using variadic input arguments in your kernel signatures, as they will fail the builder validation check.

If your kernel uses value-returning style (i.e., returns output tensors instead of writing to pre-allocated ones), set `destination_passing_style` to `false` in your solution's `spec`:

```json
{
  "name": "my_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "triton",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.py::my_kernel",
    "dependencies": [],
    "destination_passing_style": false
  },
  "sources": [...]
}
```

**Common error when DPS is mismatched:**

```
Destination-passing style callable: expected xx parameters, but got xx
```

This can happen for two reasons: (1) your kernel function signature has the wrong number of parameters, or (2) your kernel uses value-returning style but the solution still has `destination_passing_style` set to `true` by default. For the latter case, fix by setting `destination_passing_style` to `false`.

### CUDA Kernel Bindings

For CUDA kernel implementations, we recommend using [TVM FFI](https://tvm.apache.org/ffi/) for Python bindings. The `flashinfer_bench.agents` module provides TVM FFI agent instruction prompts to assist with development.

You can set the `binding` field in your solution's `spec` to specify the C++ binding type. Defaults to `"tvm-ffi"` if not specified. Supported values: `"tvm-ffi"`, `"torch"`.

```json
{
  "name": "my_cuda_solution",
  "definition": "gdn_decode_qk4_v8_d128_k_last",
  "author": "my_name",
  "spec": {
    "language": "cuda",
    "target_hardware": ["cuda"],
    "entry_point": "kernel.cu::my_kernel",
    "dependencies": [],
    "binding": "torch"
  },
  "sources": [...]
}
```
