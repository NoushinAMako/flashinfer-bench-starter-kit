"""
DSA TopK Indexer - kernel5 (fixed)

Fix: replaced relu_weight_sum_kernel (sequential per-token head accumulation)
with relu_weight_mul_kernel (in-place) + scores.sum(0) (PyTorch tree reduction).
Root cause: sequential accumulation gives different floating-point rounding from
PyTorch's reduction, causing different topk orderings at score boundaries.
"""

import torch
from torch.utils.cpp_extension import load_inline

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM = 128
TOPK = 2048

_cpp_src = r"""
#include <torch/extension.h>

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
#include <c10/cuda/CUDAGuard.h>

#define PAGE_SIZE 64
#define HEAD_DIM 128
#define NUM_HEADS 64
#define TOPK 2048
#define BYTES_PER_PAGE 8448

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

__global__ void dequant_fp8_kernel(
    const uint8_t* __restrict__ k_cache,
    const int*     __restrict__ page_ids,
    float*         __restrict__ K_out,
    int num_pages_needed
) {
    const int page_local = blockIdx.x;
    if (page_local >= num_pages_needed) return;

    const int tok = threadIdx.x;
    if (tok >= PAGE_SIZE) return;

    const int phys_page = page_ids[page_local];
    const uint8_t* page_base = k_cache + (long long)phys_page * BYTES_PER_PAGE;
    const uint8_t* fp8_row = page_base + tok * HEAD_DIM;
    const float scale = __ldg(reinterpret_cast<const float*>(
        page_base + PAGE_SIZE * HEAD_DIM + tok * 4));

    float* out_row = K_out + ((long long)page_local * PAGE_SIZE + tok) * HEAD_DIM;

    #pragma unroll 8
    for (int d = 0; d < HEAD_DIM; d += 4) {
        uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(fp8_row + d));
        out_row[d + 0] = fp8e4m3_to_float((uint8_t)(packed      )) * scale;
        out_row[d + 1] = fp8e4m3_to_float((uint8_t)(packed >>  8)) * scale;
        out_row[d + 2] = fp8e4m3_to_float((uint8_t)(packed >> 16)) * scale;
        out_row[d + 3] = fp8e4m3_to_float((uint8_t)(packed >> 24)) * scale;
    }
}

// In-place ReLU + weight multiply: scores[h, t] = relu(scores[h, t]) * w[h]
// Must be followed by scores.sum(0) to match reference numerically.
__global__ void relu_weight_mul_kernel(
    float*       __restrict__ scores,  // [NUM_HEADS, sl] row-major, modified in-place
    const float* __restrict__ w,       // [NUM_HEADS]
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
    out[i] = (int)(page_ids[idx / PAGE_SIZE] * PAGE_SIZE + idx % PAGE_SIZE);
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
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = q_fp8.device();

    const int B = q_fp8.size(0);
    const int max_num_pages = block_table.size(1);
    const int max_tokens = max_num_pages * PAGE_SIZE;

    topk_indices.fill_(-1);

    auto q_f32 = q_fp8.to(torch::kFloat32);

    auto k_cache_u8 = k_cache_fp8.view(torch::kUInt8).contiguous();
    const uint8_t* k_ptr = k_cache_u8.data_ptr<uint8_t>();

    auto K_buf = torch::empty({max_tokens, HEAD_DIM},
        torch::dtype(torch::kFloat32).device(device));

    auto seq_lens_cpu = seq_lens.cpu();
    const int* sl_data = seq_lens_cpu.data_ptr<int>();

    for (int b = 0; b < B; b++) {
        const int sl = sl_data[b];
        if (sl == 0) continue;

        const int np_seq = (sl + PAGE_SIZE - 1) / PAGE_SIZE;

        auto pages_long = block_table[b].slice(0, 0, np_seq).to(torch::kLong);
        auto pages_i32  = block_table[b].slice(0, 0, np_seq).to(torch::kInt32).contiguous();

        dequant_fp8_kernel<<<np_seq, PAGE_SIZE, 0, stream>>>(
            k_ptr,
            pages_i32.data_ptr<int>(),
            K_buf.data_ptr<float>(),
            np_seq);

        auto K = K_buf.slice(0, 0, sl);

        auto scores = at::mm(q_f32[b], K.t());  // [NUM_HEADS, sl]

        // In-place ReLU + weight multiply, then PyTorch sum(0) to match reference numerically
        const int total = NUM_HEADS * sl;
        {
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            relu_weight_mul_kernel<<<blocks, threads, 0, stream>>>(
                scores.data_ptr<float>(),
                weights[b].contiguous().data_ptr<float>(),
                sl, total);
        }
        auto final_scores = scores.sum(0);

        const int actual_k = (sl < TOPK) ? sl : TOPK;
        auto topk_result = at::topk(final_scores, actual_k);
        auto idx = std::get<1>(topk_result);

        {
            int threads = 256;
            int blocks = (actual_k + 255) / 256;
            convert_indices_kernel<<<blocks, threads, 0, stream>>>(
                idx.data_ptr<int64_t>(),
                pages_long.data_ptr<int64_t>(),
                topk_indices[b].data_ptr<int>(),
                actual_k);
        }
    }
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_kernel5_fixed",
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

