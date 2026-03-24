"""
DSA TopK Indexer - Solution9: CUDA Graph Replay with Static Inputs

Why CUDA graphs help here
-------------------------
The benchmark calls `fn(cloned_args)` for warmup_runs + iterations per workload,
which means the CPU must dispatch the full kernel sequence on every call.  With
N_STREAMS multi-streaming the Python-to-C++ and ATen dispatch cost is already
driven down (solution7), but there is still measurable per-call overhead from
cuBLAS handle lookup, ATen op dispatch, and the Python→C++ call boundary.

Graph replay replaces all of that with a single ~5 µs CPU call.

Why naive graphs fail
---------------------
The benchmark's `_clone_args` clones EVERY input tensor on EVERY iteration, so
tensor data_ptr() values change on every call.  A naive graph that reads from the
original tensor addresses would produce stale results on replay.

The static-input pattern
------------------------
  1. Allocate a set of STATIC GPU tensors (s_q, s_k, s_w, s_bt, s_out) once per
     unique workload shape + seq_lens configuration.
  2. At graph-capture time, capture the single-stream C++ kernel using the static
     tensors as inputs/output.
  3. On every timed replay:
         a. async-copy live inputs  →  static tensors  (tiny memcpy, ~1-5 µs GPU)
         b. graph.replay()                             (~1 µs CPU overhead)
         c. async-copy static output →  live output
     All three steps are on the default CUDA stream and are therefore ordered.

seq_lens handling
-----------------
seq_lens contains CPU-side control values (token counts per batch item) that drive
the C++ loop.  They are NOT a GPU tensor inside the graph.  We call seq_lens.cpu()
on every invocation for:
  (a) a guaranteed GPU sync (same role as solution7's internal `.cpu()` call),
  (b) the VALUES used to build the stable per-workload cache key.

Correctness
-----------
Bit-identical to solution7: same FP8 decode, same at::mm, same sum(0), same at::topk.
Only the dispatch strategy changes; numerics are untouched.
"""

import torch
from torch.utils.cpp_extension import load_inline

PAGE_SIZE   = 64
NUM_HEADS   = 64
HEAD_DIM    = 128
TOPK        = 2048
WARMUP_CALLS = 3   # warmup before graph capture (must be ≥ 1 to prime cuBLAS workspace)

_cpp_src = r"""
#include <torch/extension.h>

// Single-stream graph-safe entry point.
// seq_lens_cpu: CPU int32 tensor (values already transferred to host by caller).
void dsa_topk_run_sg(
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

using c10::cuda::getCurrentCUDAStream;

#define PAGE_SIZE_C  64
#define HEAD_DIM_C   128
#define NUM_HEADS_C  64
#define TOPK_C       2048
#define PAGE_BYTES   8448   // PAGE_SIZE_C * HEAD_DIM_C + PAGE_SIZE_C * 4

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

    const int phys_page = block_table[b * max_pages + p];
    const uint8_t* pg_base = k_cache + (long long)phys_page * PAGE_BYTES;
    const float scale = __ldg(reinterpret_cast<const float*>(
        pg_base + PAGE_SIZE_C * HEAD_DIM_C + tok * 4));

    const uint8_t* fp8_row = pg_base + tok * HEAD_DIM_C;
    float* out = K_out + ((long long)p * PAGE_SIZE_C + tok) * HEAD_DIM_C;

    #pragma unroll 8
    for (int d = 0; d < HEAD_DIM_C; d += 4) {
        uint32_t pk = *reinterpret_cast<const uint32_t*>(fp8_row + d);
        out[d+0] = fp8e4m3_to_float((uint8_t)(pk      )) * scale;
        out[d+1] = fp8e4m3_to_float((uint8_t)(pk >>  8)) * scale;
        out[d+2] = fp8e4m3_to_float((uint8_t)(pk >> 16)) * scale;
        out[d+3] = fp8e4m3_to_float((uint8_t)(pk >> 24)) * scale;
    }
}

// -----------------------------------------------------------------------
// In-place ReLU + weight multiply
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
// Single-stream host function — all ops on getCurrentCUDAStream().
//
// When called inside torch.cuda.graph(), that stream IS the capture stream,
// so every GPU op (custom kernels, cuBLAS, sum, topk) is recorded and will
// be replayed verbatim.  seq_lens_cpu is a CPU tensor; the loop structure
// is determined at capture time and baked into the graph.
// -----------------------------------------------------------------------
void dsa_topk_run_sg(
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

    // Per-call K buffer — static, grows on demand.
    // Using a single buffer (one stream) is safe and graph-capturable.
    static torch::Tensor sg_k_buf;
    static int sg_cached_device     = -1;
    static int sg_cached_max_tokens = 0;

    if (sg_cached_device != device_idx || sg_cached_max_tokens < max_tokens) {
        sg_k_buf = torch::empty({max_tokens, HEAD_DIM_C},
            torch::dtype(torch::kFloat32).device(device));
        sg_cached_device     = device_idx;
        sg_cached_max_tokens = max_tokens;
    }

    auto stream = getCurrentCUDAStream(device_idx);

    for (int b = 0; b < B; b++) {
        const int sl = sl_data[b];
        if (sl == 0) continue;
        const int np_seq   = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
        const int total    = NUM_HEADS_C * sl;
        const int actual_k = (sl < TOPK_C) ? sl : TOPK_C;

        dequant_fp8_v2<<<np_seq, PAGE_SIZE_C, 0, stream.stream()>>>(
            k_ptr, bt_ptr, b, max_num_pages,
            sg_k_buf.data_ptr<float>(), np_seq);

        auto K      = sg_k_buf.slice(0, 0, sl);
        auto scores = at::mm(q_f32[b], K.t());

        relu_weight_mul<<<(total+255)/256, 256, 0, stream.stream()>>>(
            scores.data_ptr<float>(),
            weights[b].contiguous().data_ptr<float>(),
            sl, total);

        auto final_scores = scores.sum(0);

        auto topk_result = at::topk(final_scores, actual_k);
        auto idx         = std::get<1>(topk_result);

        convert_indices_v2<<<(actual_k+255)/256, 256, 0, stream.stream()>>>(
            idx.data_ptr<int64_t>(),
            bt_ptr, b, max_num_pages,
            topk_indices[b].data_ptr<int>(),
            actual_k);
    }
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_s9",
            cpp_sources=[_cpp_src],
            cuda_sources=[_cuda_src],
            functions=["dsa_topk_run_sg"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _module


# -----------------------------------------------------------------------
# Static-input CUDA graph cache
#
# Key   : (q.shape, k.shape, bt.shape, seq_lens_values)
#         — stable across benchmark iterations even though tensor data_ptrs
#           change on every clone.
#
# Buffers: one set of static GPU tensors per key.  On each replay we copy
#          the live (cloned) inputs into these static tensors; the graph
#          always reads from the fixed static addresses.
# -----------------------------------------------------------------------

_call_count : dict = {}   # key  → int
_static_bufs: dict = {}   # key  → dict of static tensors
_graph_cache: dict = {}   # key  → torch.cuda.CUDAGraph


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    q_c  = q_index_fp8.contiguous()
    k_c  = k_index_cache_fp8.contiguous()
    w_c  = weights.contiguous()
    bt_c = block_table.contiguous()

    # seq_lens.cpu() does two things:
    #   1. Synchronises the default CUDA stream (waits for all prior GPU work,
    #      including from the previous call's single-stream operations) — same
    #      synchronisation guarantee as solution7.
    #   2. Provides the VALUES that make the cache key stable across iterations.
    seq_lens_cpu = seq_lens.cpu()
    sl_tuple     = tuple(seq_lens_cpu.tolist())

    key = (tuple(q_c.shape), tuple(k_c.shape), tuple(bt_c.shape), sl_tuple)

    count = _call_count.get(key, 0)
    _call_count[key] = count + 1

    mod = _get_module()

    if key not in _graph_cache:
        # ── First WARMUP_CALLS iterations: run on live tensors ──────────────
        # This primes cuBLAS workspace, caches, and the static sg_k_buf.
        if key not in _static_bufs:
            _static_bufs[key] = {
                'q':          torch.empty_like(q_c),
                'k':          torch.empty_like(k_c),
                'w':          torch.empty_like(w_c),
                'bt':         torch.empty_like(bt_c),
                'out':        torch.empty_like(topk_indices),
                'seq_lens_cpu': seq_lens_cpu.clone(),  # kept alive for graph
            }

        sb = _static_bufs[key]

        if count < WARMUP_CALLS:
            mod.dsa_topk_run_sg(q_c, k_c, w_c, seq_lens_cpu, bt_c, topk_indices)

        else:
            # ── Graph capture (count == WARMUP_CALLS) ──────────────────────
            # Copy live inputs into static buffers, then capture.
            sb['q'].copy_(q_c)
            sb['k'].copy_(k_c)
            sb['w'].copy_(w_c)
            sb['bt'].copy_(bt_c)
            torch.cuda.synchronize()   # ensure copies land before capture

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                mod.dsa_topk_run_sg(
                    sb['q'], sb['k'], sb['w'],
                    sb['seq_lens_cpu'], sb['bt'], sb['out'],
                )
            _graph_cache[key] = g

            # CUDA graph capture does NOT execute ops — it only records them.
            # Replay once here so this iteration also produces a valid output.
            g.replay()
            topk_indices.copy_(sb['out'])

    else:
        # ── All subsequent calls: copy → replay → copy out ──────────────────
        # The three copy_() / replay() calls are all on the default CUDA stream,
        # so they execute strictly in order.  The replay reads from sb[*] which
        # is guaranteed to hold the freshly copied values when it runs.
        sb = _static_bufs[key]
        sb['q'].copy_(q_c,  non_blocking=True)
        sb['k'].copy_(k_c,  non_blocking=True)
        sb['w'].copy_(w_c,  non_blocking=True)
        sb['bt'].copy_(bt_c, non_blocking=True)
        _graph_cache[key].replay()
        topk_indices.copy_(sb['out'], non_blocking=True)
