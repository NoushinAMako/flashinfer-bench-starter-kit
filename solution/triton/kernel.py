"""
DSA TopK Indexer v8 - Full CUDA kernels + cuBLAS matmul

CUDA Kernels:
  1. dequant_fp8_kernel: FP8 page gather + dequant + scale (NaN-correct)
  2. relu_weight_kernel: Fused ReLU + weight multiply (NaN-preserving)
  3. convert_indices_kernel: Local topk → global token indices

cuBLAS (via at::mm): matmul (bit-identical to reference)
at:: ops: sum(0) for head reduction, topk for selection
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

// ======================================================================
// FP8 e4m3fn -> float32 conversion (NaN-correct)
//
// Key: byte 0x7F and 0xFF are NaN in e4m3fn. Must produce float32 NaN.
// PyTorch's .view(float8_e4m3fn).float() does this correctly.
// ======================================================================
__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    // NaN check: e4m3fn NaN = exp=15, mant=7 (0x7F positive, 0xFF negative)
    if ((x & 0x7F) == 0x7F) {
        // Produce quiet NaN with correct sign
        uint32_t sign = (uint32_t)(x >> 7) << 31;
        return __uint_as_float(sign | 0x7FC00000u);
    }

    uint32_t sign = (uint32_t)(x >> 7) << 31;
    uint32_t exp  = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;

    // Zero (positive or negative)
    if ((x & 0x7F) == 0) return __uint_as_float(sign);

    uint32_t f;
    if (exp == 0) {
        // Subnormal: val = (-1)^s * 2^(-6) * (mant/8)
        uint32_t hb = 31 - __clz(mant);
        f = sign | ((118u + hb) << 23) | ((mant ^ (1u << hb)) << (23 - hb));
    } else {
        // Normal: val = (-1)^s * 2^(exp-7) * (1 + mant/8)
        f = sign | ((exp + 120u) << 23) | ((uint32_t)mant << 20);
    }
    return __uint_as_float(f);
}

// ======================================================================
// Kernel 1: FP8 Dequantization
// One block per page, 64 threads (one per token within page).
// Each thread dequantizes 128 FP8 values for its token.
//
// Memory layout (packed per page, 8448 bytes):
//   [0, 8192):    FP8 data - 64 tokens x 128 dims
//   [8192, 8448): scales  - 64 x float32
// ======================================================================
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

    // Vectorized 4-byte loads
    #pragma unroll 8
    for (int d = 0; d < HEAD_DIM; d += 4) {
        uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(fp8_row + d));
        out_row[d + 0] = fp8e4m3_to_float((uint8_t)(packed      )) * scale;
        out_row[d + 1] = fp8e4m3_to_float((uint8_t)(packed >>  8)) * scale;
        out_row[d + 2] = fp8e4m3_to_float((uint8_t)(packed >> 16)) * scale;
        out_row[d + 3] = fp8e4m3_to_float((uint8_t)(packed >> 24)) * scale;
    }
}

// ======================================================================
// Kernel 2: Fused ReLU + Weight Multiply (in-place)
//
// CRITICAL NaN handling: must use (v <= 0 ? 0 : v) NOT (v > 0 ? v : 0)
// The difference: NaN <= 0 is FALSE -> returns v (NaN preserved)
//                 NaN > 0  is FALSE -> returns 0 (NaN destroyed)
// PyTorch's relu_ uses the <= form, so we must match.
// ======================================================================
__global__ void relu_weight_kernel(
    float*       __restrict__ scores,   // [NUM_HEADS, sl] row-major
    const float* __restrict__ w,        // [NUM_HEADS]
    int total                           // NUM_HEADS * sl
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Compute which head this element belongs to
    // scores layout: [h=0: t0,t1,...,t_{sl-1}, h=1: t0,..., ...]
    // We need total / NUM_HEADS = sl, but we pass total = NUM_HEADS * sl
    // Use float division to find head: h = idx * NUM_HEADS / total
    // Actually simpler: just pass sl separately
    // For now, we do the relu in-place, weight mul separately
    float v = scores[idx];
    scores[idx] = v <= 0.0f ? 0.0f : v;
}

// Weight multiply: scores[h, t] *= w[h]
__global__ void weight_mul_kernel(
    float*       __restrict__ scores,  // [NUM_HEADS, sl] row-major
    const float* __restrict__ w,       // [NUM_HEADS]
    int sl,
    int total
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int h = idx / sl;
    scores[idx] *= w[h];
}

// ======================================================================
// Kernel 3: Index Conversion
// ======================================================================
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

// ======================================================================
// Host entry point
// ======================================================================
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

    // Pre-convert Q to float32
    auto q_f32 = q_fp8.to(torch::kFloat32);  // [B, 64, 128]

    // Raw byte pointer for KV cache
    auto k_cache_u8 = k_cache_fp8.view(torch::kUInt8).contiguous();
    const uint8_t* k_ptr = k_cache_u8.data_ptr<uint8_t>();

    // Pre-allocate reusable K buffer
    auto K_buf = torch::empty({max_tokens, HEAD_DIM},
        torch::dtype(torch::kFloat32).device(device));

    auto seq_lens_cpu = seq_lens.cpu();
    const int* sl_data = seq_lens_cpu.data_ptr<int>();

    for (int b = 0; b < B; b++) {
        const int sl = sl_data[b];
        if (sl == 0) continue;

        const int np_seq = (sl + PAGE_SIZE - 1) / PAGE_SIZE;
        const int total = NUM_HEADS * sl;

        // Page indices
        auto pages_long = block_table[b].slice(0, 0, np_seq).to(torch::kLong);
        auto pages_i32 = block_table[b].slice(0, 0, np_seq).to(torch::kInt32).contiguous();

        // ---- CUDA Kernel 1: FP8 dequant ----
        dequant_fp8_kernel<<<np_seq, PAGE_SIZE, 0, stream>>>(
            k_ptr,
            pages_i32.data_ptr<int>(),
            K_buf.data_ptr<float>(),
            np_seq);

        auto K = K_buf.slice(0, 0, sl);  // [sl, 128]

        // ---- cuBLAS matmul (at::mm = torch.mm, bit-identical) ----
        auto scores = at::mm(q_f32[b], K.t());  // [64, sl]

        // ---- CUDA Kernel 2a: ReLU (NaN-preserving, in-place) ----
        {
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            relu_weight_kernel<<<blocks, threads, 0, stream>>>(
                scores.data_ptr<float>(),
                weights[b].contiguous().data_ptr<float>(),
                total);
        }

        // ---- CUDA Kernel 2b: Weight multiply (in-place) ----
        {
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            weight_mul_kernel<<<blocks, threads, 0, stream>>>(
                scores.data_ptr<float>(),
                weights[b].contiguous().data_ptr<float>(),
                sl,
                total);
        }

        // ---- Head reduction (at::sum matches reference exactly) ----
        auto final_scores = scores.sum(0);  // [sl]

        // ---- TopK (at::topk matches reference exactly) ----
        const int actual_k = (sl < TOPK) ? sl : TOPK;
        auto topk_result = at::topk(final_scores, actual_k);
        auto idx = std::get<1>(topk_result);

        // ---- CUDA Kernel 3: index conversion ----
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
            name="dsa_topk_cuda_v8",
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

