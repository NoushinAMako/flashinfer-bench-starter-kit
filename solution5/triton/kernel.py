"""
DSA TopK Indexer - Per-item CUDA Streams (V2)

Key idea: give each batch item its OWN CUDA stream so all items' operations
(dequant → mm → relu+weight+sum → topk → convert) can overlap on the GPU.

Why this helps:
  - For small items (sl=7..64): the cuBLAS GEMM uses only ~2-4 SMs out of 132.
    Running B=31 items in parallel with 31 streams = near-full GPU utilization.
  - For large items: their mm operations are already SM-saturating, but their
    topk/dequant/relu ops can overlap with adjacent items' mm ops.
  - Result: total GPU time approaches max(item_time) rather than sum(item_time).

Correctness: identical to kernel9 — same at::mm, same relu_weight_mul, same
scores.sum(0), same at::topk. Guaranteed by not changing any numerical ops.

N_STREAMS = 32 (max practical pool size for typical B values):
  - For B <= 32: each item gets its own dedicated stream.
  - For B > 32: items cycle, but adjacent items on the same stream are
    serialized on that stream while other streams can still overlap.
"""

import torch
from torch.utils.cpp_extension import load_inline

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM  = 128
TOPK      = 2048
N_STREAMS = 32   # one stream per item for B <= 32; cycle otherwise

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
#define N_STREAMS    32

__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    if ((x & 0x7F) == 0x7F) {
        return __uint_as_float(((uint32_t)(x >> 7) << 31) | 0x7FC00000u);
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

__global__ void dequant_fp8_kernel(
    const uint8_t* __restrict__ k_cache,
    const int*     __restrict__ page_ids,
    float*         __restrict__ K_out,
    int num_pages_needed
) {
    const int page_local = blockIdx.x;
    if (page_local >= num_pages_needed) return;
    const int tok = threadIdx.x;
    if (tok >= PAGE_SIZE_C) return;

    const int phys_page = page_ids[page_local];
    const uint8_t* page_base = k_cache + (long long)phys_page * PAGE_BYTES;
    const uint8_t* fp8_row   = page_base + tok * HEAD_DIM_C;
    const float scale = __ldg(reinterpret_cast<const float*>(
        page_base + PAGE_SIZE_C * HEAD_DIM_C + tok * 4));

    float* out_row = K_out + ((long long)page_local * PAGE_SIZE_C + tok) * HEAD_DIM_C;

    #pragma unroll 8
    for (int d = 0; d < HEAD_DIM_C; d += 4) {
        uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(fp8_row + d));
        out_row[d + 0] = fp8e4m3_to_float((uint8_t)(packed      )) * scale;
        out_row[d + 1] = fp8e4m3_to_float((uint8_t)(packed >>  8)) * scale;
        out_row[d + 2] = fp8e4m3_to_float((uint8_t)(packed >> 16)) * scale;
        out_row[d + 3] = fp8e4m3_to_float((uint8_t)(packed >> 24)) * scale;
    }
}

__global__ void relu_weight_mul_kernel(
    float*       __restrict__ scores,
    const float* __restrict__ w,
    int sl,
    int total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int h = idx / sl;
    float v = scores[idx];
    scores[idx] = (v <= 0.0f ? 0.0f : v) * w[h];
}

__global__ void convert_indices_kernel(
    const int64_t* __restrict__ topk_idx,
    const int64_t* __restrict__ page_ids,
    int*           __restrict__ out,
    int actual_k
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= actual_k) return;
    const int64_t idx = topk_idx[i];
    out[i] = (int)(page_ids[idx / PAGE_SIZE_C] * PAGE_SIZE_C + idx % PAGE_SIZE_C);
}

// -----------------------------------------------------------------------
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

    auto q_f32      = q_fp8.to(torch::kFloat32);
    auto k_cache_u8 = k_cache_fp8.view(torch::kUInt8).contiguous();
    const uint8_t* k_ptr = k_cache_u8.data_ptr<uint8_t>();

    auto seq_lens_cpu = seq_lens.cpu();
    const int* sl_data = seq_lens_cpu.data_ptr<int>();

    // Create N_STREAMS streams once; grow K_bufs if max_tokens grows
    static std::vector<CUDAStream>    streams;
    static std::vector<torch::Tensor> k_bufs;
    static int                        cached_device     = -1;
    static long long                  cached_max_tokens = 0;

    if (cached_device != device_idx || (int)streams.size() < N_STREAMS) {
        streams.clear();
        k_bufs.clear();
        for (int s = 0; s < N_STREAMS; s++) {
            streams.push_back(getStreamFromPool(/*isHighPriority=*/false, device_idx));
            k_bufs.push_back(torch::empty({(long long)max_tokens * HEAD_DIM_C},
                torch::dtype(torch::kFloat32).device(device)));
        }
        cached_device     = device_idx;
        cached_max_tokens = (long long)max_tokens;
    } else if (cached_max_tokens < (long long)max_tokens) {
        for (int s = 0; s < N_STREAMS; s++) {
            k_bufs[s] = torch::empty({(long long)max_tokens * HEAD_DIM_C},
                torch::dtype(torch::kFloat32).device(device));
        }
        cached_max_tokens = (long long)max_tokens;
    }

    // Dispatch each batch item onto its own stream (round-robin if B > N_STREAMS)
    for (int b = 0; b < B; b++) {
        const int sl = sl_data[b];
        if (sl == 0) continue;

        const int s        = b % N_STREAMS;
        const int np_seq   = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
        const int total    = NUM_HEADS_C * sl;
        const int actual_k = (sl < TOPK_C) ? sl : TOPK_C;

        c10::cuda::CUDAStreamGuard guard(streams[s]);

        auto pages_long = block_table[b].slice(0, 0, np_seq).to(torch::kLong);
        auto pages_i32  = block_table[b].slice(0, 0, np_seq).to(torch::kInt32).contiguous();

        dequant_fp8_kernel<<<np_seq, PAGE_SIZE_C, 0, streams[s].stream()>>>(
            k_ptr,
            pages_i32.data_ptr<int>(),
            k_bufs[s].data_ptr<float>(),
            np_seq
        );

        auto K_view = k_bufs[s].slice(0, 0, (long long)max_tokens * HEAD_DIM_C)
                               .view({max_tokens, HEAD_DIM_C}).slice(0, 0, sl);
        auto scores = at::mm(q_f32[b], K_view.t());

        {
            int threads = 256;
            int blocks  = (total + threads - 1) / threads;
            relu_weight_mul_kernel<<<blocks, threads, 0, streams[s].stream()>>>(
                scores.data_ptr<float>(),
                weights[b].contiguous().data_ptr<float>(),
                sl, total
            );
        }

        auto final_scores = scores.sum(0);
        auto topk_result  = at::topk(final_scores, actual_k);
        auto idx          = std::get<1>(topk_result);

        {
            int threads = 256;
            int blocks  = (actual_k + 255) / 256;
            convert_indices_kernel<<<blocks, threads, 0, streams[s].stream()>>>(
                idx.data_ptr<int64_t>(),
                pages_long.data_ptr<int64_t>(),
                topk_indices[b].data_ptr<int>(),
                actual_k
            );
        }
    }

    // Synchronize all streams back to the caller's stream
    auto caller_stream = at::cuda::getCurrentCUDAStream(device_idx);
    for (int s = 0; s < N_STREAMS; s++) {
        cudaEvent_t event;
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        cudaEventRecord(event, streams[s].stream());
        cudaStreamWaitEvent(caller_stream.stream(), event, 0);
        cudaEventDestroy(event);
    }
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_multistream_v3",
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
