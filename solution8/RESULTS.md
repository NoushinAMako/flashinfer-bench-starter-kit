# DSA TopK Indexer — Solution8 Results

## What it does

Solution8 builds on solution7 and adds **TMA (Tensor Memory Accelerator)** to the
FP8 dequantization kernel. The TMA hardware unit performs asynchronous bulk DMA
directly from global memory into shared memory, freeing threads to overlap scale
loading while the transfer is in flight.

All other parts of the pipeline (GEMM, ReLU+weight, sum, TopK) are identical to
solution7, preserving bitwise-correct results.

---

## Key change over solution7: TMA dequantization

### `dequant_fp8_tma` (replaces `dequant_fp8_v2`)

| Step | Who does it | What |
|------|-------------|------|
| 1 | Thread 0 | `cuTensorMapEncodeTiled` (host, cached) builds a 3D TMA descriptor over `k_cache_fp8` |
| 2 | Thread 0 | `mbarrier.init` + `mbarrier.arrive.expect_tx` declares expected bytes |
| 3 | Thread 0 | `cp.async.bulk.tensor.3d` issues async DMA: one full page of FP8 bytes → SMEM |
| 4 | All threads | `__ldg` loads per-token scales from global memory (overlaps with DMA) |
| 5 | All threads | `mbarrier.try_wait.parity` spins until DMA completes |
| 6 | All threads | Convert FP8 → float32 out of SMEM (coalesced, L1-friendly) |

The descriptor is built once on the host with `cuTensorMapEncodeTiled` and cached
(only rebuilt if `k_cache_fp8` pointer changes across calls). Linked against
`libcuda.so` via `extra_ldflags=["-lcuda"]`.

---

## Benchmark results — NVIDIA B200

Modal `gpu="B200:1"`, torch 2.11.0+cu130, CUDA 13.0 (SM_100).
128 workloads, all PASSED (abs_err=0.00, rel_err=0.00 on every workload).

### All 128 workloads

|                | Solution7 | Solution8 (TMA) | Δ |
|----------------|-----------|-----------------|---|
| **Mean**       | 5.43x     | **5.26x**       | -3% |
| **Median**     | 5.17x     | **5.00x**       | -3% |
| Min            | 4.40x     | 3.52x           |   |
| Max            | 9.23x     | 11.37x          |   |

### Larger half (64 hardest workloads — most tokens)

|                | Solution7 | Solution8 (TMA) | Δ |
|----------------|-----------|-----------------|---|
| **Mean**       | 4.78x     | **5.02x**       | +5% |
| **Median**     | 4.75x     | **4.87x**       | +3% |

### Observations

- On the **small workloads** (tiny batch), TMA adds mbarrier synchronization overhead
  that outweighs the DMA benefit — small transfers don't amortize the setup cost.
- On the **large workloads** (most tokens), TMA's async overlap of scale loads with
  DMA gives a modest improvement (+3–5% mean/median) over solution7.
- The very high small-workload speedups (up to 11.37x) are consistent with
  solution7's pattern: the kernel is fast; the reference is extremely slow on tiny inputs.

---

## What changed from solution7

```
solution7/triton/kernel.py  →  solution8/triton/kernel.py

Kernel: dequant_fp8_v2  →  dequant_fp8_tma
  - Added: CUtensorMap cached_k_tmap (host-side, static)
  - Added: cuTensorMapEncodeTiled (3D: [HEAD_DIM, PAGE_SIZE, ∞ pages])
  - Added: mbarrier_init / mbarrier_wait / tma_3d_load helpers
  - Kernel now stages FP8 bytes in __shared__ before conversion
  - Scales loaded via __ldg overlap with TMA DMA

Includes: added <cuda.h>, <cudaTypedefs.h>
Link:     added extra_ldflags=["-lcuda"]
```

Everything else (multi-streaming, pre-allocated K buffers, raw block_table pointer,
GEMM, sum, topk) is unchanged from solution7.
