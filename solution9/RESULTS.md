# DSA TopK Indexer — Solution9 Results

## What it does

Solution9 implements **static-input CUDA graph replay** on top of solution7's
multi-stream kernel.  The goal: eliminate Python→C++ dispatch and ATen op launch
overhead for the 97+ timed iterations that follow the 3 warmup calls.

### Static-input pattern

Because the benchmark clones all input tensors on every iteration (so tensor
`data_ptr()` values change on every call), a naive graph keyed by pointer won't
replay.  Solution9 uses a *stable* key and a copy-before-replay pattern:

| Step | What |
|------|------|
| Key | `(q.shape, k_cache.shape, block_table.shape, seq_lens_values)` — stable |
| Static bufs | One set of GPU tensors per key, allocated once |
| Copy in | `s_q.copy_(live_q)` etc. on the default stream before each replay |
| Graph replay | Single CUDA call, no Python dispatch |
| Copy out | `topk_indices.copy_(s_out)` on the default stream after replay |

`seq_lens.cpu()` is called on every iteration (same as solution7) to provide the
GPU sync barrier AND the seq_lens values used to compute the key.

### Trade-off vs solution7

Gains:
- 97+ timed iterations skip all ATen dispatch (~5–50 µs saved per call)

Costs:
- 4 GPU–GPU `copy_()` ops before each replay (but all fits in <1 µs for typical workloads)
- Single-stream execution inside the graph (loses the N_STREAMS=4 parallelism of solution7)

The single-stream limitation is the dominant cost for large-batch workloads: the
multi-stream kernel in solution7 processes B batch items across 4 streams
concurrently, while the graph path serialises them.

---

## Benchmark results — NVIDIA B200

Modal `gpu="B200:1"`, torch 2.11.0+cu130, CUDA 13.0 (SM_100).
128 workloads, **all PASSED** (abs_err=0.00, rel_err=0.00 on every workload).

### Three-way comparison — all 128 workloads (B200)

|                     | Solution7 | Solution8 (TMA) | Solution9 (CUDA Graphs) | Best |
|---------------------|-----------|-----------------|-------------------------|------|
| **Mean**            | **5.43x** | 5.26x           | 5.39x                   | S7   |
| **Median**          | 5.17x     | 5.00x           | **5.18x**               | S9≈S7|
| Min                 | **4.40x** | 3.52x           | 3.86x                   | S7   |
| Max                 | 9.23x     | **11.37x**      | 8.32x                   | S8   |

### Larger half — 64 hardest workloads (most tokens)

|                     | Solution7 | Solution8 (TMA) | Solution9 (CUDA Graphs) | Best |
|---------------------|-----------|-----------------|-------------------------|------|
| **Mean**            | 4.78x     | 5.02x           | **6.10x**               | **S9** |
| **Median**          | 4.75x     | 4.87x           | **5.93x**               | **S9** |

### H100 comparison (local runs, both solutions at load simultaneously)

|                      | Solution7 | Solution9 (CUDA Graphs) | Δ      |
|----------------------|-----------|-------------------------|--------|
| Mean                 | 5.84x     | **6.07x**               | +4%    |
| Median               | 5.81x     | **6.24x**               | +7%    |
| Max                  | 9.17x     | **11.19x**              | +22%   |
| Workloads improved   | —         | 104 / 128 (81%)         |        |

### What each solution wins at

- **Solution7** — best overall mean and minimum; simplest code; safest choice for
  a mixed workload distribution.
- **Solution8 (TMA)** — highest single-workload peak (11.37x); better than S7 on
  large-token sequences (+5% top-half); worse on small batches where mbarrier setup
  overhead dominates.
- **Solution9 (CUDA Graphs)** — dominant on large/hard workloads (+28% top-half
  mean vs S7, +21% vs S8); trades away small-workload performance (single-stream
  serialisation) for big wins when sequences are long.  Best choice if the
  deployment mix skews toward long sequences.

### Reading the tradeoffs

| Scenario | Recommended |
|----------|-------------|
| Mixed workload (small + large batches) | Solution7 |
| Mostly long sequences, large batches | Solution9 |
| Optimising peak single-workload throughput | Solution8 |

---

## Correctness

All 128 workloads pass bitwise (`abs_err=0.00, rel_err=0.00`).  The same FP8
decode, at::mm, sum(0), and at::topk code paths are used as in solution7; only the
scheduling (static-input graph replay vs direct multi-stream dispatch) changes.

---

## Design notes & lessons

### Why naive CUDA graphs don't work here

The benchmark calls `_clone_args(args)` on every iteration, cloning all input
tensors.  This means `data_ptr()` changes every call, so a graph keyed on
`data_ptr()` never replays.  The static-input approach is the correct workaround.

### Why the bug in the initial solution9 was subtle

The initial implementation cached `seq_lens.cpu()` by `data_ptr()`.  When the
allocator reused a GPU address from a previous workload's seq_lens clone for a
new workload with different seq_lens values, the cache returned stale values.
This caused `sl` (sequence length) to be wrong, leading to out-of-bounds slices
of `k_bufs[s]` → illegal memory access → "GPU context corrupted".

The fix: call `seq_lens.cpu()` unconditionally on every call (same as solution7),
using the values for the stable key instead of the tensor pointer.

### Why graph capture doesn't execute ops

`torch.cuda.CUDAGraph` with `torch.cuda.graph()` captures GPU ops but does NOT
execute them.  After capture, `g.replay()` must be called once to produce valid
output on the capture call — otherwise the output tensor is uninitialized.
