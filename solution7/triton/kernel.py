"""
DSA TopK Indexer - Reduced-Overhead Multi-Stream (v4)

Correctness lessons learned (see comments):
  - relu: must use `v <= 0 ? 0 : v` not `v > 0 ? v : 0`  (NaN propagation)
  - sum:  must use `scores.sum(0)` not `at::sum_out` into pre-alloc
          (pre-alloc buffers retain stale values; for NaN-heavy random data the
           GPU reduction can produce NaN with a different bit-pattern than a fresh
           allocation, causing different topk tie-breaking order)
  - topk: must use `at::topk` not `at::topk_out` into pre-alloc
          (same tie-breaking reason: leftover buffer content affects sort order)
  - GEMM: must use `at::mm` (cuBLAS) not a custom tiled kernel
          (different FP accumulation order → different values → different top-k)

Safe overhead reductions kept from solution3:
  1. dequant_v2: block_table[b*max_pages+p] directly → no pages_i32 alloc/GPU-op per item
  2. convert_v2: block_table[b*max_pages+p] directly → no pages_long alloc/GPU-op per item
  3. Pre-allocated K_bufs (same as solution3 already does)
  4. block_table once converted to int32 outside the loop

Per-item savings vs solution3:
  - Eliminated: 1 GPU type-conversion op (block[b].to(int32)) + 1 tensor alloc for pages_i32
  - Eliminated: 1 GPU type-conversion op (block[b].to(long)) + 1 tensor alloc for pages_long
  B=25-30 × 2 allocs × ~3µs each ≈ 150-180µs saved per call.

All other ops (GEMM, relu_weight_mul, sum, topk, convert) are identical to solution3.
"""

import torch
from torch.utils.cpp_extension import load_inline

PAGE_SIZE  = 64
NUM_HEADS  = 64
HEAD_DIM   = 128
TOPK       = 2048
N_STREAMS  = 4

_cpp_src = r"""
#include <torch/extension.h>
#include <vector>

void dsa_topk_run(
    torch::Tensor q_fp8,
    torch::Tensor k_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
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
#define PAGE_BYTES   8448
#define N_STREAMS_C  4

// FP8 E4M3FN decode — bit-identical to PyTorch's hardware conversion.
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
// v2: dequant using raw block_table pointer.
// Eliminates: pages_i32 = block_table[b].slice().to(kInt32) per item.
// Numerically IDENTICAL to solution3's dequant_fp8_kernel.
// Grid: (np_seq,)  Block: (PAGE_SIZE_C,)
// -----------------------------------------------------------------------
__global__ void dequant_fp8_v2(
    const uint8_t* __restrict__ k_cache,
    const int*     __restrict__ block_table,  // [B, max_pages] int32
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
// In-place ReLU + weight multiply — EXACT COPY of solution3's kernel.
// NaN semantics: (v <= 0 ? 0 : v) preserves NaN (NaN<=0 is FALSE → v=NaN).
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
// v2: convert indices using raw block_table — no per-item pages_long alloc.
// Eliminates: pages_long = block_table[b].slice().to(kLong) per item.
// Numerically IDENTICAL to solution3's convert_indices_kernel.
// -----------------------------------------------------------------------
__global__ void convert_indices_v2(
    const int64_t* __restrict__ topk_idx,
    const int*     __restrict__ block_table,  // [B, max_pages] int32
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

void dsa_topk_run(
    torch::Tensor q_fp8,
    torch::Tensor k_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
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

    auto q_f32       = q_fp8.to(torch::kFloat32);
    auto k_cache_u8  = k_cache_fp8.view(torch::kUInt8).contiguous();
    // Convert block_table to int32 ONCE outside the loop (not per-item).
    // block_table is int32 in all benchmark workloads, so .to(kInt32) is a no-op
    // (returns same tensor). No GPU operation, no synchronization concern.
    auto bt_i32      = block_table.to(torch::kInt32).contiguous();

    auto seq_lens_cpu = seq_lens.cpu();
    const int* sl_data = seq_lens_cpu.data_ptr<int>();

    // -----------------------------------------------------------------------
    // Static per-stream K_buf — pre-allocated once, grows on demand.
    // (Same pattern as solution3, just with raw block_table access.)
    // -----------------------------------------------------------------------
    static std::vector<CUDAStream>    streams;
    static std::vector<torch::Tensor> k_bufs;
    static int cached_device     = -1;
    static int cached_max_tokens = 0;

    if (cached_device != device_idx || (int)streams.size() < N_STREAMS_C) {
        streams.clear();
        k_bufs.clear();
        for (int s = 0; s < N_STREAMS_C; s++) {
            streams.push_back(getStreamFromPool(false, device_idx));
            k_bufs.push_back(torch::empty({max_tokens, HEAD_DIM_C},
                torch::dtype(torch::kFloat32).device(device)));
        }
        cached_device     = device_idx;
        cached_max_tokens = max_tokens;
    } else if (cached_max_tokens < max_tokens) {
        for (int s = 0; s < N_STREAMS_C; s++) {
            k_bufs[s] = torch::empty({max_tokens, HEAD_DIM_C},
                torch::dtype(torch::kFloat32).device(device));
        }
        cached_max_tokens = max_tokens;
    }

    const uint8_t* k_ptr  = k_cache_u8.data_ptr<uint8_t>();
    const int*     bt_ptr = bt_i32.data_ptr<int>();

    for (int b = 0; b < B; b++) {
        const int sl       = sl_data[b];
        if (sl == 0) continue;
        const int s        = b % N_STREAMS_C;
        const int np_seq   = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
        const int total    = NUM_HEADS_C * sl;
        const int actual_k = (sl < TOPK_C) ? sl : TOPK_C;

        c10::cuda::CUDAStreamGuard guard(streams[s]);

        // 1. FP8 dequant — no pages_i32 tensor per item (saves 1 alloc + 1 GPU-op)
        dequant_fp8_v2<<<np_seq, PAGE_SIZE_C, 0, streams[s].stream()>>>(
            k_ptr, bt_ptr, b, max_num_pages,
            k_bufs[s].data_ptr<float>(), np_seq);

        // 2. GEMM via cuBLAS (identical to solution3)
        auto K      = k_bufs[s].slice(0, 0, sl);
        auto scores = at::mm(q_f32[b], K.t());     // [64, sl]

        // 3. In-place relu + weight multiply (identical to solution3)
        relu_weight_mul<<<(total+255)/256, 256, 0, streams[s].stream()>>>(
            scores.data_ptr<float>(),
            weights[b].contiguous().data_ptr<float>(),
            sl, total);

        // 4. Sum over heads (identical to solution3 — must use fresh scores.sum(0),
        //    NOT at::sum_out into pre-alloc: NaN bit-patterns differ → different topk order)
        auto final_scores = scores.sum(0);

        // 5. TopK (identical to solution3 — must use at::topk, NOT at::topk_out into
        //    pre-alloc: leftover buffer values affect tie-breaking order for NaN tokens)
        auto topk_result = at::topk(final_scores, actual_k);
        auto idx         = std::get<1>(topk_result);

        // 6. Convert flat indices → physical addresses (no pages_long tensor per item)
        convert_indices_v2<<<(actual_k+255)/256, 256, 0, streams[s].stream()>>>(
            idx.data_ptr<int64_t>(),
            bt_ptr, b, max_num_pages,
            topk_indices[b].data_ptr<int>(),
            actual_k);
    }

    // Sync all streams back to the default stream
    auto default_stream = getCurrentCUDAStream(device_idx);
    for (int s = 0; s < N_STREAMS_C; s++) {
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, streams[s].stream());
        cudaStreamWaitEvent(default_stream.stream(), ev, 0);
        cudaEventDestroy(ev);
    }
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_v4",
            cpp_sources=[_cpp_src],
            cuda_sources=[_cuda_src],
            functions=["dsa_topk_run"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _module


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    mod = _get_module()
    mod.dsa_topk_run(
        q_index_fp8.contiguous(),
        k_index_cache_fp8.contiguous(),
        weights.contiguous(),
        seq_lens.contiguous(),
        block_table.contiguous(),
        topk_indices,
    )
