"""
DSA TopK Indexer - Solution8: TMA dequant + Multi-Stream (B200-optimized)

Changes from solution7:
  - dequant_fp8_v2 (64 threads × 8 __ldg loads each = 512 instructions) replaced by
    dequant_fp8_tma (1 TMA instruction loads all 8192 FP8 bytes for a page into SMEM).

TMA design:
  k_cache modeled as 3D tensor [HEAD_DIM=128, PAGE_SIZE=64, total_pages].
  globalStrides = {128, PAGE_BYTES=8448} — naturally skips the 256 scale bytes
  that follow each page's FP8 data. Coordinate (x=0, y=0, z=phys_page) loads
  exactly one page's FP8 data. Scales (256 bytes) loaded via __ldg in parallel
  while TMA runs.

Everything else (GEMM, relu_weight_mul, sum, topk, convert) identical to solution7.
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
#include <cuda.h>
#include <cudaTypedefs.h>
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

// -----------------------------------------------------------------------
// FP8 E4M3FN decode — bit-identical to PyTorch hardware conversion.
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
// TMA helpers (from tcgen05.yaml / proven near-CUTLASS reference)
// -----------------------------------------------------------------------

// Convert generic SMEM pointer to uint32_t for inline PTX "r" constraint.
__device__ __forceinline__ uint32_t smem_to_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Initialize an mbarrier in shared memory.
__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count)
    );
}

// Spin-wait until mbarrier completes (phase flip).
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

// Issue a TMA 3D bulk load: global → shared memory.
// Coordinates (x, y, z) map to (dim0, dim1, dim2) of the CUtensorMap.
// The mbarrier at mbar_addr is signalled when bytes arrive.
__device__ __forceinline__ void tma_3d_load(
    int            smem_dst,
    const void*    tmap,
    int x, int y, int z,
    int            mbar_addr
) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(smem_dst), "l"(tmap),
           "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr)
        : "memory"
    );
}

// -----------------------------------------------------------------------
// TMA dequant kernel.
//
// Grid: (np_seq,)   Block: (PAGE_SIZE_C,) = 64 threads
//
// What changed vs solution7's dequant_fp8_v2:
//   BEFORE: 64 threads × 8 __ldg (4-byte) loads each = 512 loads from L1/L2.
//   AFTER:  1 TMA instruction loads all 8192 FP8 bytes into SMEM asynchronously.
//           Scales (256 B) load via __ldg in parallel while TMA runs.
//           Conversion reads from SMEM (sub-ns latency, zero cache pressure).
//
// Shared memory layout:
//   smem_fp8   [PAGE_SIZE_C × HEAD_DIM_C]  = 8192 bytes  (128-byte aligned)
//   smem_scale [PAGE_SIZE_C]               = 256  bytes
//   mbar       [1 × uint64_t]              = 8    bytes   (8-byte aligned)
// -----------------------------------------------------------------------
__global__ void dequant_fp8_tma(
    const __grid_constant__ CUtensorMap k_tmap,   // TMA descriptor for FP8 data
    const uint8_t* __restrict__ k_cache,           // raw cache ptr (for scales)
    const int*     __restrict__ block_table,
    int b, int max_pages,
    float*         __restrict__ K_out,
    int num_pages
) {
    const int p   = blockIdx.x;
    if (p >= num_pages) return;
    const int tok = threadIdx.x;
    if (tok >= PAGE_SIZE_C) return;

    // Shared memory: FP8 data + scales + mbarrier
    __shared__ __align__(128) uint8_t smem_fp8[PAGE_SIZE_C * HEAD_DIM_C]; // 8192 B
    __shared__ float smem_scale[PAGE_SIZE_C];                              // 256 B
    __shared__ __align__(8) uint64_t mbar;

    const int phys_page = block_table[b * max_pages + p];

    // ---- Step 1 (thread 0): init mbarrier, issue TMA ----
    if (tok == 0) {
        const int mbar_addr = smem_to_u32(&mbar);
        mbarrier_init(mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

        // Declare expected incoming bytes (8192) and count this as one arrive.
        asm volatile(
            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(PAGE_SIZE_C * HEAD_DIM_C) : "memory"
        );

        // Single TMA instruction → hardware DMA loads entire page FP8 data.
        // Coordinate (0, 0, phys_page): x=0 (HEAD_DIM offset), y=0 (tok offset),
        // z=phys_page (which physical page to load).
        tma_3d_load(smem_to_u32(smem_fp8), &k_tmap, 0, 0, phys_page, mbar_addr);
    }

    // Ensure mbar is initialized before other threads call mbarrier_wait.
    __syncthreads();

    // ---- Step 2 (all threads): load scales via __ldg, overlaps with TMA ----
    const uint8_t* pg_base = k_cache + (long long)phys_page * PAGE_BYTES;
    smem_scale[tok] = __ldg(reinterpret_cast<const float*>(
        pg_base + PAGE_SIZE_C * HEAD_DIM_C + tok * 4));

    // ---- Step 3 (all threads): wait for TMA to deliver smem_fp8 bytes ----
    mbarrier_wait(smem_to_u32(&mbar), 0);

    // Ensure smem_scale writes from all threads are visible before conversion.
    __syncthreads();

    // ---- Step 4: convert FP8 → float32 from SMEM ----
    const uint8_t* fp8_row = smem_fp8 + tok * HEAD_DIM_C;
    const float scale = smem_scale[tok];
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
// In-place ReLU + weight multiply — EXACT COPY of solution7.
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
// Index convert — EXACT COPY of solution7.
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
// Host entry point
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
    auto bt_i32     = block_table.to(torch::kInt32).contiguous();

    auto seq_lens_cpu = seq_lens.cpu();
    const int* sl_data = seq_lens_cpu.data_ptr<int>();

    // -----------------------------------------------------------------------
    // Static caches: streams, K buffers, TMA descriptor
    // -----------------------------------------------------------------------
    static std::vector<CUDAStream>    streams;
    static std::vector<torch::Tensor> k_bufs;
    static int            cached_device     = -1;
    static int            cached_max_tokens = 0;
    static CUtensorMap    cached_k_tmap;
    static const uint8_t* cached_k_ptr      = nullptr;

    const uint8_t* k_ptr  = k_cache_u8.data_ptr<uint8_t>();
    const int*     bt_ptr = bt_i32.data_ptr<int>();

    // Re-create streams / K buffers on device change or first call.
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

    // Build TMA descriptor once; rebuild only if k_cache pointer changes.
    if (k_ptr != cached_k_ptr) {
        // Model k_cache as 3D uint8 tensor: [HEAD_DIM, PAGE_SIZE, total_pages].
        // globalStrides = {HEAD_DIM, PAGE_BYTES}:
        //   - stride between tokens (dim1 step) = HEAD_DIM = 128 bytes (contiguous)
        //   - stride between pages  (dim2 step) = PAGE_BYTES = 8448 bytes
        //     (skips the 256 scale bytes appended after each page's FP8 data)
        // boxDim = {HEAD_DIM, PAGE_SIZE, 1}: one page per TMA call.
        // Coordinate (x=0, y=0, z=phys_page) → loads page phys_page's FP8 data.
        constexpr uint32_t rank = 3;
        uint64_t globalDim[3]    = {HEAD_DIM_C, PAGE_SIZE_C, (uint64_t)(1 << 24)};
        uint64_t globalStride[2] = {HEAD_DIM_C, PAGE_BYTES};
        uint32_t boxDim[3]       = {HEAD_DIM_C, PAGE_SIZE_C, 1};
        uint32_t elemStride[3]   = {1, 1, 1};

        CUresult err = cuTensorMapEncodeTiled(
            &cached_k_tmap,
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            rank,
            (void*)k_ptr,
            globalDim,
            globalStride,
            boxDim,
            elemStride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(err == CUDA_SUCCESS,
            "cuTensorMapEncodeTiled failed: ", (int)err);
        cached_k_ptr = k_ptr;
    }

    // SMEM per block: smem_fp8 (8192) + smem_scale (256) + mbar (8) + padding
    constexpr size_t SMEM_SIZE = PAGE_SIZE_C * HEAD_DIM_C       // smem_fp8
                               + PAGE_SIZE_C * sizeof(float)     // smem_scale
                               + sizeof(uint64_t);               // mbar

    for (int b = 0; b < B; b++) {
        const int sl       = sl_data[b];
        if (sl == 0) continue;
        const int s        = b % N_STREAMS_C;
        const int np_seq   = (sl + PAGE_SIZE_C - 1) / PAGE_SIZE_C;
        const int total    = NUM_HEADS_C * sl;
        const int actual_k = (sl < TOPK_C) ? sl : TOPK_C;

        c10::cuda::CUDAStreamGuard guard(streams[s]);

        // 1. FP8 dequant via TMA
        dequant_fp8_tma<<<np_seq, PAGE_SIZE_C, SMEM_SIZE, streams[s].stream()>>>(
            cached_k_tmap, k_ptr, bt_ptr, b, max_num_pages,
            k_bufs[s].data_ptr<float>(), np_seq);

        // 2. GEMM via cuBLAS (identical to solution7)
        auto K      = k_bufs[s].slice(0, 0, sl);
        auto scores = at::mm(q_f32[b], K.t());

        // 3. In-place relu + weight multiply (identical to solution7)
        relu_weight_mul<<<(total+255)/256, 256, 0, streams[s].stream()>>>(
            scores.data_ptr<float>(),
            weights[b].contiguous().data_ptr<float>(),
            sl, total);

        // 4. Sum over heads (identical to solution7)
        auto final_scores = scores.sum(0);

        // 5. TopK (identical to solution7)
        auto topk_result = at::topk(final_scores, actual_k);
        auto idx         = std::get<1>(topk_result);

        // 6. Convert flat indices → physical addresses (identical to solution7)
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
            name="dsa_topk_s8",
            cpp_sources=[_cpp_src],
            cuda_sources=[_cuda_src],
            functions=["dsa_topk_run"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=["-lcuda"],
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
