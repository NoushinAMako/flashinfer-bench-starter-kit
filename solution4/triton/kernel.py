"""
DSA TopK Indexer - Mixed Batched/Sequential GEMM (V2)

Key insight from extensive empirical testing:
  - torch.bmm produces NUMERICALLY IDENTICAL results to sequential at::mm when:
    (1) The batch item's individual sl >= 192 (= 3 pages of 64 tokens)
    (2) N_padded for bmm >= sl (guaranteed: we pad to np_seq * PAGE_SIZE >= sl)
  - For sl < 192: cuBLAS picks a different algorithm for small N vs large N_padded,
    causing numerical divergence (even with no NaN). Must use sequential at::mm.

Strategy V2:
  - Split batch items into two groups:
    - "large" group (sl >= 192): batch together → ONE dequant launch + ONE bmm
    - "small" group (sl < 192):  sequential at::mm (identical to kernel9)
  - For large group: N_padded = max_np_large * PAGE_SIZE >= max(sl_large) >= 192
    → cuBLAS picks the same algorithm as sequential mm → identical results.
  - Both groups run on the same CUDA stream for correctness.
"""

import torch
from torch.utils.cpp_extension import load_inline

PAGE_SIZE  = 64
NUM_HEADS  = 64
HEAD_DIM   = 128
TOPK       = 2048
MIN_SL_BATCHED = 192   # items with sl >= this go to batched bmm path

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
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <vector>

#define PAGE_SIZE_C  64
#define HEAD_DIM_C   128
#define NUM_HEADS_C  64
#define TOPK_C       2048
#define PAGE_BYTES   8448   // PAGE_SIZE*HEAD_DIM + PAGE_SIZE*4 = 8192+256

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

// ------------------------------------------------------------------
// Dequant for a SUBSET of batch items (the "large" group).
// active_ids[0..B_large) maps local index → original batch index.
// grid = (B_large, max_pages_large), block = (PAGE_SIZE_C)
// Output: K_large[b_local, p*PAGE_SIZE+tok, d] for each valid page.
// ------------------------------------------------------------------
__global__ void dequant_fp8_large_kernel(
    const uint8_t* __restrict__ k_cache,
    const int*     __restrict__ block_table,   // [B_orig, max_num_pages_orig]
    const int*     __restrict__ active_ids,    // [B_large]
    const int*     __restrict__ seq_lens,      // [B_orig]
    float*         __restrict__ K_large,       // [B_large, max_pages_large*PAGE_SIZE, HEAD_DIM]
    int max_num_pages_orig,
    int max_pages_large
) {
    const int b_local = blockIdx.x;
    const int p       = blockIdx.y;
    const int tok     = threadIdx.x;

    const int b_orig = active_ids[b_local];
    const int sl     = seq_lens[b_orig];
    const int np_seq = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
    if (p >= np_seq || tok >= PAGE_SIZE_C) return;

    const int phys_page = block_table[b_orig * max_num_pages_orig + p];
    const uint8_t* page_base = k_cache + (long long)phys_page * PAGE_BYTES;
    const uint8_t* fp8_row   = page_base + tok * HEAD_DIM_C;
    const float scale = __ldg(reinterpret_cast<const float*>(
        page_base + PAGE_SIZE_C * HEAD_DIM_C + tok * 4));

    float* out = K_large +
        ((long long)b_local * max_pages_large * PAGE_SIZE_C + p * PAGE_SIZE_C + tok) * HEAD_DIM_C;

    #pragma unroll 8
    for (int d = 0; d < HEAD_DIM_C; d += 4) {
        uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(fp8_row + d));
        out[d + 0] = fp8e4m3_to_float((uint8_t)(packed      )) * scale;
        out[d + 1] = fp8e4m3_to_float((uint8_t)(packed >>  8)) * scale;
        out[d + 2] = fp8e4m3_to_float((uint8_t)(packed >> 16)) * scale;
        out[d + 3] = fp8e4m3_to_float((uint8_t)(packed >> 24)) * scale;
    }
}

// ------------------------------------------------------------------
// Sequential dequant for one batch item (same as kernel9).
// grid = (np_seq,), block = (PAGE_SIZE_C)
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// In-place ReLU + weight multiply (same as kernel9).
// ------------------------------------------------------------------
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

// ------------------------------------------------------------------
// Convert flat token indices → physical KV-cache addresses.
// ------------------------------------------------------------------
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
// Host launcher
// -----------------------------------------------------------------------
void dsa_topk_run(
    torch::Tensor q_fp8,
    torch::Tensor k_cache_fp8,
    torch::Tensor weights,
    torch::Tensor seq_lens,
    torch::Tensor block_table,
    torch::Tensor topk_indices
) {
    const c10::cuda::CUDAGuard device_guard(q_fp8.device());
    auto device     = q_fp8.device();
    int  device_idx = device.index();

    const int B             = q_fp8.size(0);
    const int max_num_pages = block_table.size(1);

    topk_indices.fill_(-1);

    auto q_f32      = q_fp8.to(torch::kFloat32);
    auto k_cache_u8 = k_cache_fp8.view(torch::kUInt8).contiguous();
    auto stream     = at::cuda::getCurrentCUDAStream(device_idx);

    auto seq_lens_gpu = seq_lens.contiguous();
    auto seq_lens_cpu = seq_lens.cpu();
    const int* sl_data = seq_lens_cpu.data_ptr<int>();

    // ---- Split items into large (sl>=192) and small (sl<192) groups ----
    std::vector<int> large_ids, small_ids;
    int max_pages_large = 0;

    for (int b = 0; b < B; b++) {
        int sl = sl_data[b];
        if (sl >= 192) {
            large_ids.push_back(b);
            int np = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
            if (np > max_pages_large) max_pages_large = np;
        } else {
            small_ids.push_back(b);
        }
    }

    int B_large    = (int)large_ids.size();
    int max_sl_large = max_pages_large * PAGE_SIZE_C;

    // ================================================================
    // LARGE GROUP: one dequant launch + one bmm (numerically correct)
    // ================================================================
    if (B_large > 0) {
        // Upload active indices to GPU
        auto active_cpu = torch::tensor(large_ids,
            torch::dtype(torch::kInt32).device(torch::kCPU));
        auto active_gpu = active_cpu.to(device);

        // K_large buffer [B_large, max_sl_large, HEAD_DIM]
        static torch::Tensor K_large_buf;
        static long long     K_large_cap = 0;
        long long K_needed = (long long)B_large * max_sl_large * HEAD_DIM_C;
        if (K_needed > K_large_cap) {
            K_large_buf = torch::empty({K_needed},
                torch::dtype(torch::kFloat32).device(device));
            K_large_cap = K_needed;
        }
        auto K_large = K_large_buf.slice(0, 0, K_needed)
                           .view({B_large, max_sl_large, HEAD_DIM_C});

        // 1. Dequant all large items in one kernel launch
        {
            dim3 grid(B_large, max_pages_large);
            dequant_fp8_large_kernel<<<grid, PAGE_SIZE_C, 0, stream.stream()>>>(
                k_cache_u8.data_ptr<uint8_t>(),
                block_table.contiguous().data_ptr<int>(),
                active_gpu.data_ptr<int>(),
                seq_lens_gpu.data_ptr<int>(),
                K_large.data_ptr<float>(),
                max_num_pages,
                max_pages_large
            );
        }

        // 2. Gather q rows for large items: [B_large, 64, 128]
        auto active_long = active_cpu.to(torch::kLong).to(device);
        auto q_large     = q_f32.index_select(0, active_long);
        auto w_large     = weights.index_select(0, active_long);

        // 3. ONE batched GEMM: [B_large, 64, max_sl_large]
        auto scores_large = at::bmm(q_large, K_large.transpose(1, 2));

        // 4. relu + weight (element-wise, identical to sequential kernel)
        scores_large.clamp_(0.0f);
        scores_large.mul_(w_large.unsqueeze(2));  // [B_large, 64, 1]

        // 5. Sum over heads → [B_large, max_sl_large]
        auto fs_large = scores_large.sum(1);

        // 6. Per-item: topk on [:sl] slice, then convert to physical indices
        for (int bl = 0; bl < B_large; bl++) {
            int b        = large_ids[bl];
            int sl       = sl_data[b];
            int actual_k = (sl < TOPK_C) ? sl : TOPK_C;
            int np_seq   = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;

            // Slice to valid token range (positions >= sl are never written by dequant)
            auto fs_item = fs_large[bl].slice(0, 0, sl);
            auto topk_result = at::topk(fs_item, actual_k);
            auto idx = std::get<1>(topk_result);

            auto pages_long = block_table[b].slice(0, 0, np_seq).to(torch::kLong);

            int thr = 256, blk = (actual_k + 255) / 256;
            convert_indices_kernel<<<blk, thr, 0, stream.stream()>>>(
                idx.data_ptr<int64_t>(),
                pages_long.data_ptr<int64_t>(),
                topk_indices[b].data_ptr<int>(),
                actual_k
            );
        }
    }

    // ================================================================
    // SMALL GROUP: sequential at::mm (identical to kernel9)
    // ================================================================
    if (!small_ids.empty()) {
        // Find max_sl for the small group (for K buffer sizing)
        int max_pages_small = 0;
        for (int b : small_ids) {
            int np = (sl_data[b] + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
            if (np > max_pages_small) max_pages_small = np;
        }
        int max_sl_small = max_pages_small * PAGE_SIZE_C;

        static torch::Tensor K_buf_seq;
        static long long     K_buf_cap = 0;
        long long K_needed = (long long)max_sl_small * HEAD_DIM_C;
        if (K_needed > K_buf_cap) {
            K_buf_seq = torch::empty({max_sl_small, HEAD_DIM_C},
                torch::dtype(torch::kFloat32).device(device));
            K_buf_cap = K_needed;
        }

        for (int b : small_ids) {
            const int sl      = sl_data[b];
            const int np_seq  = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
            const int total   = NUM_HEADS_C * sl;
            const int actual_k = (sl < TOPK_C) ? sl : TOPK_C;

            auto pages_i32  = block_table[b].slice(0, 0, np_seq)
                                .to(torch::kInt32).contiguous();
            auto pages_long = block_table[b].slice(0, 0, np_seq).to(torch::kLong);

            dequant_fp8_kernel<<<np_seq, PAGE_SIZE_C, 0, stream.stream()>>>(
                k_cache_u8.data_ptr<uint8_t>(),
                pages_i32.data_ptr<int>(),
                K_buf_seq.data_ptr<float>(),
                np_seq
            );

            auto K      = K_buf_seq.slice(0, 0, sl);
            auto scores = at::mm(q_f32[b], K.t());

            {
                int threads = 256;
                int blocks  = (total + threads - 1) / threads;
                relu_weight_mul_kernel<<<blocks, threads, 0, stream.stream()>>>(
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
                convert_indices_kernel<<<blocks, threads, 0, stream.stream()>>>(
                    idx.data_ptr<int64_t>(),
                    pages_long.data_ptr<int64_t>(),
                    topk_indices[b].data_ptr<int>(),
                    actual_k
                );
            }
        }
    }
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_batched_v2",
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
