# DSA TopK Indexer — Solution10 Results

## What it does

Solution10 combines **multi-stream GPU parallelism** (solution7's N_STREAMS=4)
with **static-input CUDA graph replay** (solution9's copy-before-replay pattern)
using the standard CUDA **fork/join capture** technique.

### Why solution9 left performance on the table

Solution9 (single-stream CUDA graph) eliminated CPU dispatch overhead but
serialised all B batch items onto a single GPU stream.  For large workloads
(30 batch items, up to 2048 tokens each) this was a significant bottleneck.

### How solution10 fixes that

The challenge: multi-stream graphs require all side streams to be in the
CUDA *capture tree*, and CUDA only allows cross-stream ops inside the capture
using **captured events** (events recorded during the active capture).  Using
an event recorded outside the capture raises `cudaErrorStreamCaptureUnsupported`.

Solution10 uses a **fork/join event pattern** — entirely standard CUDA graph
technique:

| Step | What / Why |
|------|-----------|
| `cudaEventRecord(fork_ev, default_stream)` | Recorded INSIDE the graph on the capture stream; becomes a captured event |
| `cudaStreamWaitEvent(streams[s], fork_ev)` × 4 | Side streams join the capture tree by waiting for a captured event |
| Batch loop | B items dispatch to 4 side streams (identical to solution7) |
| `cudaEventRecord(done_evs[s], streams[s])` × 4 | Persistent done events — graph-safe |
| `cudaStreamWaitEvent(default_stream, done_evs[s])` × 4 | Default stream re-joins all side streams |

### Why input-copy safety still holds

The live→static copies happen on the default stream **before** `g.replay()`.
The graph's first op on the default stream is `cudaEventRecord(fork_ev, default)`.
CUDA's in-stream ordering guarantee means `fork_ev` fires only **after** copies
complete.  Side streams wait for `fork_ev` before reading any static buffer,
so they always see fresh data.

### Persistent events requirement

`cudaEventCreate` / `cudaEventDestroy` are CPU-only operations that do **not**
appear in the captured graph.  If these were called inside the C++ function,
replay would reference freed event handles.  All events (`fork_ev`, `done_evs[]`)
are allocated once at initialisation time and kept alive.

### Execution timeline per timed iteration

```
Default stream:  [copies live→static] [g.replay submits...]
                                       |
                                  fork_ev fires
                                  /   /   \   \
Side streams:                   [s0][s1][s2][s3]  ← run in parallel
                                  \   \   /   /
                                   done_evs join
Default stream:  [...default waits for all done_evs] [copies static out→live]
```

---

## Benchmark results — NVIDIA B200

Modal `gpu="B200:1"`, torch 2.11.0+cu130, CUDA 13.0 (SM_100).
128 workloads, **all PASSED** (abs_err=0.00, rel_err=0.00 on every workload).

### Four-way comparison — all 128 workloads (B200)

|                     | Solution7 | Solution8 (TMA) | Solution9 (1-stream graph) | **Solution10 (ms graph)** | Best |
|---------------------|-----------|-----------------|---------------------------|--------------------------|------|
| **Mean**            | 5.43x     | 5.26x           | 5.39x                     | **10.81x**               | **S10** |
| **Median**          | 5.17x     | 5.00x           | 5.18x                     | **10.79x**               | **S10** |
| Min                 | 4.40x     | 3.52x           | 3.86x                     | **7.00x**                | **S10** |
| Max                 | 9.23x     | 11.37x          | 8.32x                     | **16.34x**               | **S10** |

### Larger half — 64 hardest workloads (most tokens)

|                     | Solution7 | Solution8 (TMA) | Solution9 (1-stream graph) | **Solution10 (ms graph)** | Best |
|---------------------|-----------|-----------------|---------------------------|--------------------------|------|
| **Mean**            | 4.78x     | 5.02x           | 6.10x                     | **11.98x**               | **S10** |
| **Median**          | 4.75x     | 4.87x           | 5.93x                     | **11.87x**               | **S10** |

**Solution10 achieves ~2× the mean speedup of every prior solution on B200.**

---

## Benchmark results — NVIDIA H100 (local run)

128 workloads, all PASSED (bitwise exact).

|                     | Solution7 (H100) | Solution9 (H100) | Solution10 (H100) |
|---------------------|------------------|------------------|-------------------|
| **Mean**            | 5.84x            | 6.07x            | **12.49x**        |
| **Median**          | 5.81x            | 6.24x            | **12.36x**        |
| Min                 | —                | —                | 5.90x             |
| Max                 | 9.17x            | 11.19x           | **22.03x**        |

> Note: H100 runs at GPU contention (multiple users); absolute times are less
> reliable than B200 numbers.  The relative gain over prior solutions is still
> instructive.

---

## Why the gain is so large

Solution7 / 8 / 9 all pay CPU dispatch overhead on every timed iteration:
- Solution7/8: full ATen dispatch + cuBLAS handle lookup per batch item per call
- Solution9: 1 µs graph replay + 4 GPU-GPU copies, but serialised batch items

Solution10 combines the savings:
- **1 µs graph replay** (no ATen/cuBLAS dispatch overhead)
- **4-stream parallel batch loop** (no serialisation of batch items)
- **4 async GPU-GPU copies** (< 5 µs total) as the only overhead vs ideal

The fork/join pattern adds negligible cost: 5 `cudaEventRecord` + 4
`cudaStreamWaitEvent` calls are hardware-scheduled in nanoseconds.

---

## Correctness

All 128 workloads pass bitwise (`abs_err=0.00, rel_err=0.00`).  The same FP8
decode, `at::mm` (cuBLAS), `sum(0)`, and `at::topk` code paths are used as in
solution7; only the scheduling (multi-stream graph replay vs direct dispatch)
changes.

---

## Implementation notes

### The `cudaErrorStreamCaptureUnsupported` pitfall

An earlier version of solution10 tried to insert `s_copies_ev` (an event
recorded outside the capture) into the side streams via
`cudaStreamWaitEvent(streams[s], s_copies_ev, 0)` inside the graph.  CUDA
raises `cudaErrorStreamCaptureUnsupported` for any `cudaStreamWaitEvent` where
either stream involved is currently in capture mode AND the event is not a
captured event.  The fix is the fork/join pattern: record `fork_ev` on the
default stream **inside** the capture context so it becomes a captured event,
then use it to fork the side streams.

### Static k_bufs for multi-stream

Each side stream has its own dequantization buffer `k_bufs[s]` (same as
solution7).  These are C++ statics.  For sequential workload evaluation (the
benchmark processes workload W1 fully before starting W2), buffer addresses
remain stable for all of W1's warmup + capture + replay calls, then may be
reallocated for W2.  This is safe because W1's graph is never replayed after
W2's first call.
