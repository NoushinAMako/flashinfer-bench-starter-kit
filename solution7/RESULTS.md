# DSA TopK Indexer — Solution7 Results

## What it does

Solution7 optimizes the DSA TopK Indexer benchmark by reducing CPU dispatch overhead
and GPU memory allocations compared to the original solution, while maintaining
**bitwise-identical** numerical results.

The benchmark task: given a batch of FP8-quantized query/key tensors, compute attention
scores, apply ReLU + per-head weights, sum over heads, and return the top-2048 token
indices per batch item.

---

## Key optimizations over the original solution

### 1. CUDA multi-streaming (4 streams)
Batch items are dispatched across 4 independent CUDA streams (`b % 4`), allowing the
GPU to overlap dequant, GEMM, and reduction work across batch items in parallel.
Streams are allocated once (static) and reused across calls.

### 2. Eliminated per-item tensor allocations
The original solution allocates two temporary tensors **per batch item** every call:
- `pages_i32 = block_table[b].slice(...).to(kInt32)` — passed to dequant kernel
- `pages_long = block_table[b].slice(...).to(kLong)` — passed to index convert kernel

Solution7 converts `block_table` to int32 **once** outside the loop and passes raw
`int*` pointers directly into the kernels. With B=25–30 items, this saves ~150–180 µs
per call (2 allocs × B items × ~3 µs each).

### 3. Pre-allocated K buffers per stream
One float32 K buffer per stream, allocated once and reused. Grows on demand if
`max_tokens` increases.

### Correctness constraints (what was NOT changed)

These ops must stay identical to the reference to guarantee bitwise-correct TopK indices:

| Op | Must use | Why |
|----|----------|-----|
| GEMM | `at::mm` (cuBLAS) | Different accumulation order → different FP32 values → different TopK |
| ReLU | `v <= 0 ? 0 : v` | NaN-preserving form; `v > 0 ? v : 0` destroys NaN bytes (0x7F, 0xFF in e4m3fn) |
| Sum | `scores.sum(0)` (fresh alloc) | Pre-alloc buffers retain stale NaN bit-patterns → different tie-breaking |
| TopK | `at::topk` (fresh alloc) | Same reason — leftover buffer values affect sort order for tied scores |

---

## Benchmark results — NVIDIA B200

Both solutions run on Modal `gpu="B200:1"`, torch 2.11.0+cu130, CUDA 13.0.
128 workloads, all PASSED (abs_err=0.00, rel_err=0.00 on every workload).

### All 128 workloads

|                | Original | Solution7 | Improvement |
|----------------|----------|-----------|-------------|
| **Mean**       | 3.94x    | **5.43x** | +38%        |
| **Median**     | 3.49x    | **5.17x** | +48%        |
| Stdev          | 1.07x    | 0.97x     | more consistent |
| Min            | 2.88x    | 4.40x     |             |
| Max            | 8.31x    | 9.23x     |             |

### Larger half (64 hardest workloads — most tokens)

|                | Original | Solution7 | Improvement |
|----------------|----------|-----------|-------------|
| **Mean**       | 3.23x    | **4.78x** | +48%        |
| **Median**     | 3.22x    | **4.75x** | +47%        |
| Stdev          | 0.16x    | 0.20x     |             |
| Min            | 2.88x    | 4.40x     |             |
| Max            | 3.48x    | 5.23x     |             |

### H100 results (local benchmark)

128/128 PASSED, average speedup **7.68x** vs the reference (solution3/triton).

---

## Solution7 kernel (`solution7/triton/kernel.py`)

```python
"""
DSA TopK Indexer - Reduced-Overhead Multi-Stream (v4)

Correctness lessons learned:
  - relu: must use `v <= 0 ? 0 : v` not `v > 0 ? v : 0`  (NaN propagation)
  - sum:  must use scores.sum(0) not at::sum_out into pre-alloc
  - topk: must use at::topk not at::topk_out into pre-alloc
  - GEMM: must use at::mm (cuBLAS) not a custom tiled kernel

Safe overhead reductions:
  1. dequant_v2: raw block_table pointer — no pages_i32 alloc/GPU-op per item
  2. convert_v2: raw block_table pointer — no pages_long alloc/GPU-op per item
  3. Pre-allocated K_bufs per stream
  4. block_table converted to int32 once outside the loop
  5. N_STREAMS=4 CUDA streams for overlapping batch item execution
"""
```

CUDA kernels:
- **`dequant_fp8_v2`**: FP8 e4m3fn → float32 with per-token scale, reads block_table
  directly via `block_table[b * max_pages + p]` (no per-item tensor allocation)
- **`relu_weight_mul`**: fused ReLU + weight multiply in one pass, NaN-preserving
- **`convert_indices_v2`**: flat token index → physical address, reads block_table
  directly (no per-item pages_long allocation)

ATen ops (unchanged from reference):
- `at::mm` — cuBLAS GEMM, bit-identical results
- `scores.sum(0)` — head reduction
- `at::topk` — top-K selection
