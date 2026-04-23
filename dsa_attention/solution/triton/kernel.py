"""
DSA Sparse Attention v8 - Fused concat gather + fused softmax + C++ wrapper
CUDA: bf16->fp32 fused gather, fused scale+mask+softmax+lse kernel
cuBLAS: logits bmm + output bmm
"""

import math
import torch
from torch.utils.cpp_extension import load_inline

NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
TOPK = 2048
DIM_CONCAT = HEAD_DIM_CKV + HEAD_DIM_KPE  # 576

_cpp_src = r"""
#include <torch/extension.h>

void dsa_sparse_attention(
    torch::Tensor ckv_flat, torch::Tensor kpe_flat,
    torch::Tensor q_nope, torch::Tensor q_pe,
    torch::Tensor indices, double sm_scale,
    torch::Tensor output, torch::Tensor lse,
    int64_t num_tokens, int64_t topk);
"""

_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <limits>

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

#define DIM_CKV 512
#define DIM_KPE 64
#define DIM_TOTAL 576
#define NUM_HEADS 16

// ============================================================
// Fused concat gather: bf16 ckv + kpe -> fp32 [T, K, 576]
// ============================================================
__global__ __launch_bounds__(256)
void gather_kv_concat_kernel(
    const __nv_bfloat16* __restrict__ ckv_flat,
    const __nv_bfloat16* __restrict__ kpe_flat,
    const int* __restrict__ indices,
    float* __restrict__ out,
    int num_tokens, int topk)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int dim8 = DIM_TOTAL >> 3;
    int total = num_tokens * topk * dim8;
    if (tid >= total) return;

    int tmp = tid;
    int d8 = tmp % dim8; tmp /= dim8;
    int k = tmp % topk;
    int t = tmp / topk;

    int idx = __ldg(&indices[t * topk + k]);
    int d = d8 << 3;

    float4* dst = reinterpret_cast<float4*>(out + ((int64_t)(t * topk + k) * DIM_TOTAL + d));

    if (idx < 0) {
        dst[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        dst[1] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    const __nv_bfloat16* src;
    if (d < DIM_CKV) {
        src = ckv_flat + (int64_t)idx * DIM_CKV + d;
    } else {
        src = kpe_flat + (int64_t)idx * DIM_KPE + (d - DIM_CKV);
    }

    uint4 raw = __ldg(reinterpret_cast<const uint4*>(src));
    const __nv_bfloat16* vals = reinterpret_cast<const __nv_bfloat16*>(&raw);

    dst[0] = make_float4(
        __bfloat162float(vals[0]), __bfloat162float(vals[1]),
        __bfloat162float(vals[2]), __bfloat162float(vals[3]));
    dst[1] = make_float4(
        __bfloat162float(vals[4]), __bfloat162float(vals[5]),
        __bfloat162float(vals[6]), __bfloat162float(vals[7]));
}

// ============================================================
// Fused softmax + LSE kernel
// Replaces: scale, mask, logsumexp, div, softmax (5 ops -> 1)
// Each warp handles one (t, h) row of K=2048 elements
// Overwrites logits in-place with attention weights
// ============================================================
__global__ __launch_bounds__(256)
void fused_softmax_lse_kernel(
    float* __restrict__ logits,        // [T*H, K] -> overwritten with attn weights
    const int* __restrict__ indices,   // [T, K]
    float* __restrict__ lse_out,       // [T, H]
    float sm_scale,
    int num_tokens, int topk)
{
    // 8 warps per block, each warp handles one row
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 8 + warp_id;

    int total_rows = num_tokens * NUM_HEADS;
    if (row >= total_rows) return;

    int t = row / NUM_HEADS;
    float* logit_row = logits + (int64_t)row * topk;
    const int* idx_row = indices + t * topk;

    // 2048 / 32 = 64 elements per thread
    constexpr int ELEMS = 2048 / 32;
    float vals[ELEMS];

    // Phase 1: Scale + mask + local max
    float local_max = -INFINITY;
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) {
        int k = lane + 32 * i;
        float v = logit_row[k] * sm_scale;
        if (idx_row[k] < 0) v = -INFINITY;
        vals[i] = v;
        local_max = fmaxf(local_max, v);
    }

    // Warp reduce max
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    float row_max = local_max;

    // Phase 2: exp + local sum
    float local_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) {
        vals[i] = expf(vals[i] - row_max);
        local_sum += vals[i];
    }

    // Warp reduce sum
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    float row_sum = local_sum;

    // Phase 3: Normalize and write attn weights
    float inv_sum = 1.f / row_sum;
    #pragma unroll
    for (int i = 0; i < ELEMS; i++) {
        int k = lane + 32 * i;
        logit_row[k] = vals[i] * inv_sum;
    }

    // Write LSE (base-2)
    if (lane == 0) {
        lse_out[row] = (row_max + logf(row_sum)) * 1.4426950408889634f;
    }
}

// ============================================================
// Main C++ entry point
// ============================================================
void dsa_sparse_attention(
    torch::Tensor ckv_flat, torch::Tensor kpe_flat,
    torch::Tensor q_nope, torch::Tensor q_pe,
    torch::Tensor indices, double sm_scale,
    torch::Tensor output, torch::Tensor lse,
    int64_t num_tokens, int64_t topk)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = ckv_flat.device();

    // 1. Fused concat gather: [T, K, 576] fp32
    auto K_concat = torch::empty({num_tokens, topk, DIM_TOTAL},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    {
        int total = (int)(num_tokens * topk * (DIM_TOTAL >> 3));
        gather_kv_concat_kernel<<<div_up(total, 256), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(ckv_flat.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(kpe_flat.data_ptr()),
            indices.data_ptr<int>(),
            K_concat.data_ptr<float>(),
            (int)num_tokens, (int)topk);
    }

    // 2. Query concat + fp32 conversion: [T, H, 576] fp32
    auto q_concat = at::cat({q_nope.to(torch::kFloat32), q_pe.to(torch::kFloat32)}, -1);

    // 3. Single bmm for logits: [T, H, K] fp32
    auto logits = at::bmm(q_concat, K_concat.transpose(1, 2));

    // 4. Fused softmax+LSE: overwrites logits with attn weights, writes lse
    {
        int total_rows = (int)(num_tokens * NUM_HEADS);
        int nblocks = div_up(total_rows, 8);
        fused_softmax_lse_kernel<<<nblocks, 256, 0, stream>>>(
            logits.data_ptr<float>(),
            indices.data_ptr<int>(),
            lse.data_ptr<float>(),
            (float)sm_scale,
            (int)num_tokens, (int)topk);
    }

    // 5. Output bmm: attn (in logits) @ Kc -> bf16 output
    auto Kc = K_concat.slice(2, 0, DIM_CKV);
    output.copy_(at::bmm(logits, Kc).to(torch::kBFloat16));
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_sparse_v8",
            cpp_sources=_cpp_src,
            cuda_sources=_cuda_src,
            functions=[
                "dsa_sparse_attention",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _module


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale, output, lse):
    mod = _get_module()
    num_tokens = q_nope.shape[0]

    ckv_flat = ckv_cache.reshape(-1, HEAD_DIM_CKV)
    kpe_flat = kpe_cache.reshape(-1, HEAD_DIM_KPE)

    idx = sparse_indices.int() if sparse_indices.dtype != torch.int32 else sparse_indices

    mod.dsa_sparse_attention(
        ckv_flat, kpe_flat, q_nope, q_pe, idx, sm_scale,
        output, lse, num_tokens, TOPK)