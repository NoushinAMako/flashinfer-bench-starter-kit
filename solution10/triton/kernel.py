"""
DSA TopK Indexer - Solution10: Multi-stream CUDA Graph

Combines solution7's N_STREAMS=4 parallelism with solution9's static-input
graph replay using the standard CUDA fork/join capture pattern.

How the synchronisation works
------------------------------
The benchmark clones all input tensors on every call, so we keep a set of
static GPU buffers per workload key.  Before each replay we copy the live
(cloned) inputs to the static buffers on the DEFAULT stream.

Inside the captured graph the following sequence is recorded once:
  1. cudaEventRecord(fork_ev, default_stream)   — captured, fires after copies
  2. cudaStreamWaitEvent(streams[s], fork_ev)   — all side streams join capture
  3. Batch loop: B items dispatched across N_STREAMS side streams (solution7)
  4. cudaEventRecord(done_evs[s], streams[s])   — captured, one per side stream
  5. cudaStreamWaitEvent(default_stream, done_evs[s]) — default waits for all

On every timed call (after warmup + capture):
  a. async-copy live inputs → static buffers    (default stream)
  b. g.replay()                                 (~1 µs CPU overhead)
       default stream: (1) fires, side streams unblock, run in parallel
  c. async-copy static output → live output     (default stream, after (5))

Why fork_ev must be recorded INSIDE the graph
----------------------------------------------
CUDA raises cudaErrorStreamCaptureUnsupported if cudaStreamWaitEvent is called
with an event that was NOT recorded during an active capture, when any stream
involved is currently capturing.  The side streams join the capture tree only
via a "captured event" (one recorded inside the capture context).  The fork_ev
is recorded on the capture stream (default) inside the C++ function body, which
runs inside `with torch.cuda.graph(g):`.

Input-copy ordering is still safe: the copies are submitted to the default
stream BEFORE g.replay().  The graph's first op on the default stream is
cudaEventRecord(fork_ev, default).  Because CUDA guarantees in-order execution
within a single stream, fork_ev fires only AFTER the copies have completed.
Side streams start only after they receive fork_ev — so they never read stale
static-buffer contents.

Persistent events
-----------------
fork_ev and done_evs are allocated once and reused across replays.  Transient
events (create inside C++ → record → destroy) are NOT graph-safe: the destroy
is a CPU-only op that does not appear in the capture, so replay references a
freed event handle.
"""

import torch
from torch.utils.cpp_extension import load_inline

PAGE_SIZE    = 64
NUM_HEADS    = 64
HEAD_DIM     = 128
TOPK         = 2048
N_STREAMS    = 4
WARMUP_CALLS = 3   # warmup before graph capture (≥1 to prime cuBLAS workspace)

_cpp_src = r"""
#include <torch/extension.h>
#include <vector>

// Multi-stream host function — graph-capturable via the fork/join event pattern.
// Internally records fork_ev on the default stream, then has each side stream
// wait for it (adding them to the capture tree), runs the batch loop, and
// re-joins all side streams to the default stream via done_evs.
void dsa_topk_run_ms(
    torch::Tensor q_fp8,
    torch::Tensor k_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens_cpu,
    torch::Tensor block_table,
    torch::Tensor topk_indices);
"""

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

using c10::cuda::CUDAStream;
using c10::cuda::getStreamFromPool;
using c10::cuda::getCurrentCUDAStream;

#define PAGE_SIZE_C  64
#define HEAD_DIM_C   128
#define NUM_HEADS_C  64
#define TOPK_C       2048
#define PAGE_BYTES   8448   // PAGE_SIZE_C * HEAD_DIM_C + PAGE_SIZE_C * sizeof(float)
#define N_STREAMS_C  4

// -----------------------------------------------------------------------
// FP8 E4M3FN decode — bit-identical to hardware conversion.
// -----------------------------------------------------------------------
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    if ((x & 0x7F) == 0x7F) {
        uint32_t sign = (uint32_t)(x >> 7) << 31;
        return __uint_as_float(sign | 0x7FC00000u);
    }
    uint32_t sign = (uint32_t)(x >> 7) << 31;
    uint32_t exp  = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;
    if ((x & 0x7F) == 0) return __uint_as_float(sign);
    uint32_t f;
    if (exp == 0) {
        uint32_t hb = 31 - __clz(mant);
        f = sign | ((118u + hb) << 23) | ((mant ^ (1u << hb)) << (23 - hb));
    } else {
        f = sign | ((exp + 120u) << 23) | ((uint32_t)mant << 20);
    }
    return __uint_as_float(f);
}

// -----------------------------------------------------------------------
// FP8 dequant  Grid: (np_seq,)  Block: (PAGE_SIZE_C,)
// -----------------------------------------------------------------------
__global__ void dequant_fp8_v2(
    const uint8_t* __restrict__ k_cache,
    const int*     __restrict__ block_table,
    int b, int max_pages,
    float*         __restrict__ K_out,
    int num_pages
) {
    const int p   = blockIdx.x;
    if (p >= num_pages) return;
    const int tok = threadIdx.x;
    if (tok >= PAGE_SIZE_C) return;

    const int phys_page    = block_table[b * max_pages + p];
    const uint8_t* pg_base = k_cache + (long long)phys_page * PAGE_BYTES;
    const uint8_t* fp8_row = pg_base + tok * HEAD_DIM_C;
    const float scale = __ldg(reinterpret_cast<const float*>(
        pg_base + PAGE_SIZE_C * HEAD_DIM_C + tok * 4));

    float* out = K_out + ((long long)p * PAGE_SIZE_C + tok) * HEAD_DIM_C;
    #pragma unroll 8
    for (int d = 0; d < HEAD_DIM_C; d += 4) {
        uint32_t pk = __ldg(reinterpret_cast<const uint32_t*>(fp8_row + d));
        out[d+0] = fp8e4m3_to_float((uint8_t)(pk      )) * scale;
        out[d+1] = fp8e4m3_to_float((uint8_t)(pk >>  8)) * scale;
        out[d+2] = fp8e4m3_to_float((uint8_t)(pk >> 16)) * scale;
        out[d+3] = fp8e4m3_to_float((uint8_t)(pk >> 24)) * scale;
    }
}

// -----------------------------------------------------------------------
// In-place ReLU + weight multiply.
// NaN-preserving: v <= 0 ? 0 : v  (NaN > 0 is false → NaN passes through)
// -----------------------------------------------------------------------
__global__ void relu_weight_mul(
    float*       __restrict__ scores,
    const float* __restrict__ w,
    int sl, int total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int h = idx / sl;
    float v = scores[idx];
    scores[idx] = (v <= 0.0f ? 0.0f : v) * w[h];
}

// -----------------------------------------------------------------------
// Index convert
// -----------------------------------------------------------------------
__global__ void convert_indices_v2(
    const int64_t* __restrict__ topk_idx,
    const int*     __restrict__ block_table,
    int b, int max_pages,
    int*           __restrict__ out,
    int actual_k
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= actual_k) return;
    const int tok  = (int)topk_idx[i];
    const int page = tok / PAGE_SIZE_C;
    const int phys = block_table[b * max_pages + page];
    out[i] = phys * PAGE_SIZE_C + tok % PAGE_SIZE_C;
}

// -----------------------------------------------------------------------
// Multi-stream host function — graph-capturable via fork/join pattern.
//
// Fork/join protocol (all events are persistent — never destroyed):
//
//   fork_ev:  Recorded on the default stream BEFORE the batch loop.
//             Side streams wait for it, adding themselves to the capture tree.
//             On replay this fires AFTER any copies queued on the default
//             stream before g.replay() (CUDA stream ordering guarantee).
//
//   done_evs: One per side stream, recorded at end of that stream's work.
//             Default stream waits for all of them to re-join.
//
// Why persistent events are required:
//   cudaEventCreate / cudaEventDestroy are CPU-only operations that do NOT
//   appear in the captured graph.  If events were created inside the function
//   and destroyed before/after capture, replay would reference freed handles
//   causing undefined behavior.
// -----------------------------------------------------------------------
void dsa_topk_run_ms(
    torch::Tensor q_fp8,
    torch::Tensor k_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens_cpu,
    torch::Tensor block_table,
    torch::Tensor topk_indices
) {
    const at::cuda::CUDAGuard device_guard(q_fp8.device());
    auto device     = q_fp8.device();
    int  device_idx = device.index();

    const int B             = q_fp8.size(0);
    const int max_num_pages = block_table.size(1);
    const int max_tokens    = max_num_pages * PAGE_SIZE_C;

    topk_indices.fill_(-1);

    auto q_f32      = q_fp8.to(torch::kFloat32);
    auto k_cache_u8 = k_cache_fp8.view(torch::kUInt8).contiguous();
    auto bt_i32     = block_table.to(torch::kInt32).contiguous();

    const int*     sl_data = seq_lens_cpu.data_ptr<int>();
    const uint8_t* k_ptr   = k_cache_u8.data_ptr<uint8_t>();
    const int*     bt_ptr  = bt_i32.data_ptr<int>();

    // -----------------------------------------------------------------------
    // Static resources: streams, per-stream k_bufs, fork_ev, done_evs.
    // Allocated once per device; k_bufs grow on demand.
    // -----------------------------------------------------------------------
    static std::vector<CUDAStream>    streams;
    static std::vector<torch::Tensor> k_bufs;
    static std::vector<cudaEvent_t>   done_evs;
    static cudaEvent_t                fork_ev       = nullptr;
    static int                        cached_device = -1;
    static int                        cached_max_tokens = 0;

    if (cached_device != device_idx || (int)streams.size() < N_STREAMS_C) {
        // Destroy old events before reinitialising
        if (fork_ev) { cudaEventDestroy(fork_ev); fork_ev = nullptr; }
        for (auto& ev : done_evs) cudaEventDestroy(ev);
        streams.clear(); k_bufs.clear(); done_evs.clear();

        for (int s = 0; s < N_STREAMS_C; s++) {
            streams.push_back(getStreamFromPool(false, device_idx));
            k_bufs.push_back(torch::empty({max_tokens, HEAD_DIM_C},
                torch::dtype(torch::kFloat32).device(device)));
            cudaEvent_t ev;
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            done_evs.push_back(ev);
        }
        cudaEventCreateWithFlags(&fork_ev, cudaEventDisableTiming);
        cached_device     = device_idx;
        cached_max_tokens = max_tokens;

    } else if (cached_max_tokens < max_tokens) {
        for (int s = 0; s < N_STREAMS_C; s++) {
            k_bufs[s] = torch::empty({max_tokens, HEAD_DIM_C},
                torch::dtype(torch::kFloat32).device(device));
        }
        cached_max_tokens = max_tokens;
    }

    auto default_stream = getCurrentCUDAStream(device_idx);

    // -----------------------------------------------------------------------
    // Fork: record fork_ev on the default stream, then have all side streams
    // wait for it.  This is a GPU op (captured when inside graph context).
    //
    // During graph capture fork_ev becomes a "captured event".  The subsequent
    // cudaStreamWaitEvent calls on the side streams reference this captured
    // event, causing those streams to join the capture tree.  CUDA requires
    // events used in cross-stream graph capture to be captured events — using
    // an event recorded outside the capture context would return
    // cudaErrorStreamCaptureUnsupported.
    //
    // During warmup (outside any capture) this is just a normal stream-event
    // fork: side streams are not constrained until fork_ev fires, which
    // happens immediately after the preceding default-stream work completes.
    // -----------------------------------------------------------------------
    cudaEventRecord(fork_ev, default_stream.stream());
    for (int s = 0; s < N_STREAMS_C; s++) {
        cudaStreamWaitEvent(streams[s].stream(), fork_ev, 0);
    }

    // -----------------------------------------------------------------------
    // Batch loop — identical ops to solution7.
    // -----------------------------------------------------------------------
    for (int b = 0; b < B; b++) {
        const int sl = sl_data[b];
        if (sl == 0) continue;
        const int s        = b % N_STREAMS_C;
        const int np_seq   = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
        const int total    = NUM_HEADS_C * sl;
        const int actual_k = (sl < TOPK_C) ? sl : TOPK_C;

        c10::cuda::CUDAStreamGuard guard(streams[s]);

        dequant_fp8_v2<<<np_seq, PAGE_SIZE_C, 0, streams[s].stream()>>>(
            k_ptr, bt_ptr, b, max_num_pages,
            k_bufs[s].data_ptr<float>(), np_seq);

        auto K      = k_bufs[s].slice(0, 0, sl);
        auto scores = at::mm(q_f32[b], K.t());         // [64, sl]

        relu_weight_mul<<<(total+255)/256, 256, 0, streams[s].stream()>>>(
            scores.data_ptr<float>(),
            weights[b].contiguous().data_ptr<float>(),
            sl, total);

        auto final_scores = scores.sum(0);
        auto topk_result  = at::topk(final_scores, actual_k);
        auto idx          = std::get<1>(topk_result);

        convert_indices_v2<<<(actual_k+255)/256, 256, 0, streams[s].stream()>>>(
            idx.data_ptr<int64_t>(),
            bt_ptr, b, max_num_pages,
            topk_indices[b].data_ptr<int>(),
            actual_k);
    }

    // -----------------------------------------------------------------------
    // Join: record done_evs on each side stream, then default stream waits.
    // All events are persistent — safe for graph capture and replay.
    // -----------------------------------------------------------------------
    for (int s = 0; s < N_STREAMS_C; s++) {
        cudaEventRecord(done_evs[s], streams[s].stream());
        cudaStreamWaitEvent(default_stream.stream(), done_evs[s], 0);
    }
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_s10",
            cpp_sources=[_cpp_src],
            cuda_sources=[_cuda_src],
            functions=["dsa_topk_run_ms"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _module


# -----------------------------------------------------------------------
# Static-input CUDA graph cache
#
# Key : (q.shape, k.shape, bt.shape, seq_lens_values)
#
# For each unique key:
#   _static_bufs[key]  — fixed-address GPU tensors for inputs + output
#   _graph_cache[key]  — captured CUDAGraph
#   _call_count[key]   — iteration counter for warmup gating
#
# Ordering invariant during replay:
#   1. async-copy live inputs → static buffers  (default stream)
#   2. g.replay()                               (default stream)
#      → captures: fork_ev fires (after copies), side streams unblock,
#        work runs in parallel, done_evs sync back to default
#   3. async-copy static output → live output   (default stream, after done_evs)
#
#   Steps 1-3 are all on the default stream and therefore strictly ordered.
# -----------------------------------------------------------------------

_call_count : dict = {}
_static_bufs: dict = {}
_graph_cache: dict = {}


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    q_c  = q_index_fp8.contiguous()
    k_c  = k_index_cache_fp8.contiguous()
    w_c  = weights.contiguous()
    bt_c = block_table.contiguous()

    # seq_lens.cpu() serves two roles:
    #   1. GPU sync: waits for all prior async ops on the default stream
    #      (including the previous iteration's output copy and the joined
    #      done_evs that funnel through the default stream).
    #   2. Stable key: actual token counts baked into the cache key.
    seq_lens_cpu = seq_lens.cpu()
    sl_tuple     = tuple(seq_lens_cpu.tolist())

    key   = (tuple(q_c.shape), tuple(k_c.shape), tuple(bt_c.shape), sl_tuple)
    count = _call_count.get(key, 0)
    _call_count[key] = count + 1

    mod = _get_module()

    if key not in _graph_cache:
        if key not in _static_bufs:
            _static_bufs[key] = {
                'q':            torch.empty_like(q_c),
                'k':            torch.empty_like(k_c),
                'w':            torch.empty_like(w_c),
                'bt':           torch.empty_like(bt_c),
                'out':          torch.empty_like(topk_indices),
                'seq_lens_cpu': seq_lens_cpu.clone(),
            }

        sb = _static_bufs[key]

        if count < WARMUP_CALLS:
            # ── Warmup: run on live tensors ────────────────────────────────
            # Primes cuBLAS workspace, the C++ static streams/k_bufs/events,
            # and warms GPU caches.  The fork/join events inside the C++
            # function work identically outside of graph capture mode.
            mod.dsa_topk_run_ms(q_c, k_c, w_c, seq_lens_cpu, bt_c, topk_indices)

        else:
            # ── Graph capture (count == WARMUP_CALLS) ──────────────────────
            sb['q'].copy_(q_c)
            sb['k'].copy_(k_c)
            sb['w'].copy_(w_c)
            sb['bt'].copy_(bt_c)
            torch.cuda.synchronize()    # ensure static bufs are fully written

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                mod.dsa_topk_run_ms(
                    sb['q'], sb['k'], sb['w'],
                    sb['seq_lens_cpu'], sb['bt'], sb['out'],
                )
            _graph_cache[key] = g

            # Graph capture records ops but does NOT execute them.
            # Replay once to produce valid output for this capture iteration.
            g.replay()
            topk_indices.copy_(sb['out'])

    else:
        # ── All subsequent calls: copy → replay → copy out ──────────────────
        # All three operations are on the default stream and are strictly ordered.
        # The graph's first op (cudaEventRecord fork_ev on default) fires AFTER
        # the copies above complete, so side streams see fresh static-buffer data.
        sb = _static_bufs[key]
        sb['q'].copy_(q_c,  non_blocking=True)
        sb['k'].copy_(k_c,  non_blocking=True)
        sb['w'].copy_(w_c,  non_blocking=True)
        sb['bt'].copy_(bt_c, non_blocking=True)
        _graph_cache[key].replay()
        topk_indices.copy_(sb['out'], non_blocking=True)
