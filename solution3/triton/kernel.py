"""
DSA TopK Indexer - Multi-stream parallel v1

Profiling showed the bottleneck is B sequential CUDA ops (dequant+mm+topk) per call.
Fix: spread batch items across N_STREAMS independent CUDA streams so that
dequant/mm/topk for item i+1 can run in parallel with dequant/mm/topk for item i.

Correctness guarantee: same numerics as solution (kernel9) —
  at::mm (cuBLAS) + relu_weight_mul in-place + scores.sum(0) (PyTorch reduction).
All these ops are enqueued on the per-batch CUDA stream via CUDAStreamGuard.

Design notes:
- Each stream gets its own K_buf to avoid data races during concurrent dequant writes.
- Streams are pre-created and reused across calls (stored in a C++ static).
- All streams are synchronized at the end of the host function.
"""

import torch
from torch.utils.cpp_extension import load_inline

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM = 128
TOPK = 2048
N_STREAMS = 4

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
#define N_STREAMS    4

// FP8 E4M3FN decode (identical to reference/solution)
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

// FP8 dequant: grid=(np_seq,), block=(PAGE_SIZE_C,)
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

// In-place ReLU + weight multiply: scores[h*sl+t] = relu(scores[h*sl+t]) * w[h]
// Followed by scores.sum(0) to produce final_scores — this pair matches reference numerics.
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

// Convert flat token indices -> physical KV cache addresses
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
    const at::cuda::CUDAGuard device_guard(q_fp8.device());
    auto device = q_fp8.device();
    int device_idx = device.index();

    const int B             = q_fp8.size(0);
    const int max_num_pages = block_table.size(1);
    const int max_tokens    = max_num_pages * PAGE_SIZE_C;

    topk_indices.fill_(-1);

    // Convert q once (matches reference: q_index_fp8.to(float32))
    auto q_f32      = q_fp8.to(torch::kFloat32);
    auto k_cache_u8 = k_cache_fp8.view(torch::kUInt8).contiguous();
    const uint8_t* k_ptr = k_cache_u8.data_ptr<uint8_t>();

    // Read seq_lens on CPU
    auto seq_lens_cpu = seq_lens.cpu();
    const int* sl_data = seq_lens_cpu.data_ptr<int>();

    // Create N_STREAMS streams on first call, reuse across calls
    static std::vector<CUDAStream>    streams;
    static std::vector<torch::Tensor> k_bufs;
    static int                        cached_device     = -1;
    static int                        cached_max_tokens = 0;

    if (cached_device != device_idx || (int)streams.size() < N_STREAMS) {
        streams.clear();
        k_bufs.clear();
        for (int s = 0; s < N_STREAMS; s++) {
            streams.push_back(getStreamFromPool(/*isHighPriority=*/false, device_idx));
            k_bufs.push_back(torch::empty({max_tokens, HEAD_DIM_C},
                torch::dtype(torch::kFloat32).device(device)));
        }
        cached_device     = device_idx;
        cached_max_tokens = max_tokens;
    } else if (cached_max_tokens < max_tokens) {
        // Grow K_buf if this workload needs more pages
        for (int s = 0; s < N_STREAMS; s++) {
            k_bufs[s] = torch::empty({max_tokens, HEAD_DIM_C},
                torch::dtype(torch::kFloat32).device(device));
        }
        cached_max_tokens = max_tokens;
    }

    // Process batch items across N_STREAMS streams
    for (int b = 0; b < B; b++) {
        const int sl = sl_data[b];
        if (sl == 0) continue;

        const int s        = b % N_STREAMS;
        const int np_seq   = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
        const int total    = NUM_HEADS_C * sl;
        const int actual_k = (sl < TOPK_C) ? sl : TOPK_C;

        // Switch to this batch item's stream for all ops
        c10::cuda::CUDAStreamGuard guard(streams[s]);

        auto pages_long = block_table[b].slice(0, 0, np_seq).to(torch::kLong);
        auto pages_i32  = block_table[b].slice(0, 0, np_seq).to(torch::kInt32).contiguous();

        // Step 1: FP8 dequant into this stream's K_buf
        dequant_fp8_kernel<<<np_seq, PAGE_SIZE_C, 0, streams[s].stream()>>>(
            k_ptr,
            pages_i32.data_ptr<int>(),
            k_bufs[s].data_ptr<float>(),
            np_seq
        );

        // Step 2: GEMM via cuBLAS on this stream (numerically identical to reference)
        auto K      = k_bufs[s].slice(0, 0, sl);
        auto scores = at::mm(q_f32[b], K.t());

        // Step 3: In-place relu + weight multiply
        {
            int threads = 256;
            int blocks  = (total + threads - 1) / threads;
            relu_weight_mul_kernel<<<blocks, threads, 0, streams[s].stream()>>>(
                scores.data_ptr<float>(),
                weights[b].contiguous().data_ptr<float>(),
                sl, total
            );
        }

        // Step 4: Sum over heads (PyTorch reduction on this stream — matches reference)
        auto final_scores = scores.sum(0);

        // Step 5: TopK (on this stream)
        auto topk_result = at::topk(final_scores, actual_k);
        auto idx         = std::get<1>(topk_result);

        // Step 6: Convert flat indices -> physical addresses (on this stream)
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

    // Synchronize all streams back to the default stream so the caller sees
    // completed results when the default stream is synced.
    auto default_stream = getCurrentCUDAStream(device_idx);
    for (int s = 0; s < N_STREAMS; s++) {
        // Record event on worker stream, wait on default stream
        cudaEvent_t event;
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        cudaEventRecord(event, streams[s].stream());
        cudaStreamWaitEvent(default_stream.stream(), event, 0);
        cudaEventDestroy(event);
    }
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_multistream_v1",
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
