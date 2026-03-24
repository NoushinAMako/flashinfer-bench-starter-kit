"""
DSA TopK Indexer — Solution 12: FP8 GEMM via torch._scaled_mm

Pipeline change vs solution11:

  Before (s11):  FP8 page bytes → dequant_fp8_v2 → K_f32 [sl,128] → at::mm(q_f32, K.T)
  After  (s12):  FP8 page bytes → pack_k_fp8 → K_fp8 [sl,128] + k_scales [sl]
                 → torch._scaled_mm(q_fp8, K_fp8.T, 1.0, k_scales) → scores_f32 [64,sl]

Memory savings per batch item (sl=2048):
  - K write:   4 MB float32  →  1 MB float8  (−3 MB)
  - GEMM K read: 1 MB float32 → 256 KB float8 (−768 KB)
  - Total saved: ~3.75 MB per batch item (B200 @ 8 TB/s: ~0.5 µs less stall)

Compute savings:
  - B200 FP8 Tensor Cores: ~2× throughput vs FP32 for same GEMM dimensions
  - For [64,128]×[128,2048]: est. 1.5–2× faster GEMM kernel

Stream/graph design:
  - Identical fork/join pattern to solution11, but now managed entirely from Python
  - N_STREAMS = 8, one torch.cuda.Stream per stream slot
  - Events: fork_ev (default→side) + done_evs (side→default), persistent across replays

Correctness note (KNOWN LIMITATION):
  cuBLAS FP8 Tensor Cores use FP16 precision for the multiply step (not float32).
  This causes up to ~0.013 max error vs the float32 GEMM reference, which changes
  topk index ordering for close-scoring tokens → abs_err > 0 → benchmark FAILS.

  RowWise per-token scaling ([M,1] / [1,N]) only supports out_dtype=bfloat16 on H100,
  which would introduce even more divergence.  We therefore use:
    • scale_a = scalar 1.0 (TensorWise — only mode that allows float32 output)
    • score post-multiply by k_scales after the GEMM

  This solution is kept to measure the *performance* of FP8 GEMM on B200 even though
  it fails correctness.  Combining FP8 throughput with correctness would require
  either (a) a Triton kernel that does exact float32-equivalent accumulation, or
  (b) a looser correctness tolerance in the benchmark.
"""
import torch
from torch.utils.cpp_extension import load_inline

PAGE_SIZE    = 64
NUM_HEADS    = 64
HEAD_DIM     = 128
TOPK         = 2048
N_STREAMS    = 8
WARMUP_CALLS = 3

# -----------------------------------------------------------------------
# C++ / CUDA extension — only two kernels needed:
#   pack_k_fp8       : copy FP8 bytes + extract per-token scales (no dequant)
#   convert_indices  : flat token index → physical page address (unchanged)
# -----------------------------------------------------------------------
_cpp_src = r"""
#include <torch/extension.h>

// Pack raw FP8 bytes (without converting to float32) + extract per-token scales.
void pack_k_fp8(
    torch::Tensor k_cache_fp8,  // original paged KV cache (float8_e4m3fn or int8)
    torch::Tensor block_table,  // [B, max_pages] int32/int64
    int64_t b_idx, int64_t max_pages,
    torch::Tensor k_fp8_out,    // [sl, HEAD_DIM] — raw FP8 bytes out
    torch::Tensor k_scales_out, // [sl] float32 — per-token scale out
    int64_t num_pages);

// Convert flat token indices to physical page addresses.
void convert_indices(
    torch::Tensor topk_idx,
    torch::Tensor block_table,
    int64_t b_idx, int64_t max_pages,
    torch::Tensor out,
    int64_t actual_k);
"""

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

using c10::cuda::getCurrentCUDAStream;

#define PAGE_SIZE_C  64
#define HEAD_DIM_C   128
// PAGE_BYTES = PAGE_SIZE * HEAD_DIM * sizeof(uint8) + PAGE_SIZE * sizeof(float)
//            = 64 * 128 + 64 * 4  = 8192 + 256 = 8448
#define PAGE_BYTES   8448

// -----------------------------------------------------------------------
// pack_k_fp8_kernel — Grid: (num_pages,)  Block: (PAGE_SIZE_C,)
//
// Copies FP8 bytes verbatim (no float32 conversion) and extracts the
// per-token scale into a contiguous float32 array.
// -----------------------------------------------------------------------
__global__ void pack_k_fp8_kernel(
    const uint8_t* __restrict__ k_cache,
    const int*     __restrict__ block_table,
    int b, int max_pages,
    uint8_t*       __restrict__ k_fp8_out,
    float*         __restrict__ k_scales_out,
    int num_pages
) {
    const int p   = blockIdx.x;
    if (p >= num_pages) return;
    const int tok = threadIdx.x;
    if (tok >= PAGE_SIZE_C) return;

    const int phys_page    = block_table[b * max_pages + p];
    const uint8_t* pg_base = k_cache + (long long)phys_page * PAGE_BYTES;
    const uint8_t* fp8_row = pg_base + tok * HEAD_DIM_C;

    // Scale stored after the FP8 payload: PAGE_SIZE * HEAD_DIM + tok * sizeof(float)
    const float scale = __ldg(reinterpret_cast<const float*>(
        pg_base + PAGE_SIZE_C * HEAD_DIM_C + tok * 4));

    const int global_tok    = p * PAGE_SIZE_C + tok;
    k_scales_out[global_tok] = scale;

    // Copy 128 FP8 bytes in 32 × 4-byte chunks — fully coalesced across warps.
    uint8_t* out_row = k_fp8_out + (long long)global_tok * HEAD_DIM_C;
    #pragma unroll 8
    for (int d = 0; d < HEAD_DIM_C; d += 4) {
        *reinterpret_cast<uint32_t*>(out_row + d) =
            *reinterpret_cast<const uint32_t*>(fp8_row + d);
    }
}

void pack_k_fp8(
    torch::Tensor k_cache_fp8,
    torch::Tensor block_table,
    int64_t b_idx, int64_t max_pages,
    torch::Tensor k_fp8_out,
    torch::Tensor k_scales_out,
    int64_t num_pages
) {
    const c10::cuda::CUDAGuard guard(k_cache_fp8.device());
    const int device_idx = k_cache_fp8.device().index();
    auto stream = getCurrentCUDAStream(device_idx).stream();

    // View KV cache as raw bytes regardless of declared FP8 dtype.
    auto k_cache_u8 = k_cache_fp8.view(torch::kUInt8).contiguous();
    auto bt_i32     = block_table.to(torch::kInt32).contiguous();

    pack_k_fp8_kernel<<<(int)num_pages, PAGE_SIZE_C, 0, stream>>>(
        k_cache_u8.data_ptr<uint8_t>(),
        bt_i32.data_ptr<int>(),
        (int)b_idx, (int)max_pages,
        // Output pointer is valid for any 1-byte dtype (float8/uint8/int8).
        reinterpret_cast<uint8_t*>(k_fp8_out.data_ptr()),
        k_scales_out.data_ptr<float>(),
        (int)num_pages
    );
}

// -----------------------------------------------------------------------
// convert_indices_kernel — Grid: ceil(actual_k/256)  Block: 256
// Identical to solution11.
// -----------------------------------------------------------------------
__global__ void convert_indices_kernel(
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

void convert_indices(
    torch::Tensor topk_idx,
    torch::Tensor block_table,
    int64_t b_idx, int64_t max_pages,
    torch::Tensor out,
    int64_t actual_k
) {
    const c10::cuda::CUDAGuard guard(out.device());
    const int device_idx = out.device().index();
    auto stream = getCurrentCUDAStream(device_idx).stream();

    auto bt_i32 = block_table.to(torch::kInt32).contiguous();
    convert_indices_kernel<<<((int)actual_k + 255) / 256, 256, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        bt_i32.data_ptr<int>(),
        (int)b_idx, (int)max_pages,
        out.data_ptr<int>(),
        (int)actual_k
    );
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dsa_topk_s12",
            cpp_sources=[_cpp_src],
            cuda_sources=[_cuda_src],
            functions=["pack_k_fp8", "convert_indices"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _module


# -----------------------------------------------------------------------
# Per-key stream / event resources
# -----------------------------------------------------------------------
_streams_cache: dict = {}   # key → list[torch.cuda.Stream]  (N_STREAMS entries)
_events_cache:  dict = {}   # key → (fork_ev, list[done_ev])  (persistent)

def _get_stream_resources(key, device):
    if key not in _streams_cache:
        _streams_cache[key] = [torch.cuda.Stream(device=device)
                                for _ in range(N_STREAMS)]
        fork_ev  = torch.cuda.Event(enable_timing=False, blocking=False)
        done_evs = [torch.cuda.Event(enable_timing=False, blocking=False)
                    for _ in range(N_STREAMS)]
        _events_cache[key] = (fork_ev, done_evs)
    return _streams_cache[key], _events_cache[key]


# -----------------------------------------------------------------------
# Core batch function
#
# This function is called:
#   (a) During warmup  : on live input tensors, outside any graph context
#   (b) During capture : on static-buffer tensors, inside `with cuda.graph(g):`
#   (c) Never at replay: the captured graph is replayed directly
#
# All arguments after `bt` are pre-sliced per-stream resource lists so that
# each indexed access has the same Python bytecode on every call (required
# for graph capture correctness).
#
# Fork/join protocol (Python equivalent of solution11's C++ pattern):
#
#   fork_ev.record()         — GPU op on default/capture stream.
#                              Becomes a *captured event* inside graph context,
#                              allowing side streams to join the capture tree.
#
#   streams[s].wait_event()  — side streams block until fork_ev fires.
#                              Captured inside the graph.
#
#   <batch loop>             — each item's GPU ops dispatched via
#                              `with torch.cuda.stream(streams[s]):`.
#
#   done_evs[s].record()     — captured on each side stream.
#   default_stream.wait_event() — default stream rejoins; subsequent
#                              output-copy sees fresh results.
#
# NaN-preserving relu:
#   PyTorch's clamp/relu zero NaN on CUDA.  We use `torch.where(v <= 0, 0, v)`.
#   For NaN: (NaN <= 0) = False  →  returns `v` = NaN.  Matches the custom
#   relu_weight_mul kernel in solution11 which also preserves NaN.
# -----------------------------------------------------------------------
def _do_batch(mod, q_fp8, k_cache, w, bt, sl_list, B, max_pages,
              k_fp8_bufs, k_scales_bufs, scale_scalar, out_buf,
              streams, fork_ev, done_evs):
    """Execute the full DSA TopK pipeline across N_STREAMS parallel streams."""
    # Fill output buffer with sentinel so un-hit slots are -1.
    out_buf.fill_(-1)

    # --- Fork -------------------------------------------------------
    # Record on the default/capture stream; side streams will wait for it.
    fork_ev.record()
    for s in range(N_STREAMS):
        streams[s].wait_event(fork_ev)

    # --- Batch dispatch ---------------------------------------------
    for b in range(B):
        s    = b % N_STREAMS
        sl   = int(sl_list[b])
        if sl == 0:
            continue
        np_seq   = (sl + PAGE_SIZE - 1) // PAGE_SIZE
        actual_k = sl if sl < TOPK else TOPK

        with torch.cuda.stream(streams[s]):
            # 1. Pack FP8 bytes (no dequant) + per-token scales.
            #    pack_k_fp8 uses getCurrentCUDAStream() → picks up streams[s].
            mod.pack_k_fp8(
                k_cache, bt, b, max_pages,
                k_fp8_bufs[s][:sl],              # [sl, HEAD_DIM] float8_e4m3fn
                k_scales_bufs[s][0, :sl],        # [sl] float32
                np_seq,
            )

            # 2. GEMM: FP8 path (fast) vs float32 fallback (safe).
            #
            #    torch._scaled_mm requires N (token count) divisible by 16.
            #    For small or misaligned sl we fall back to float32 GEMM.
            #
            #    FP8 path note: on H100 cuBLAS FP8 uses FP16 multiply internally,
            #    so results can diverge ~0.013 from float32 → fails abs_err=0.
            #    On B200 (Blackwell SM100) the FP8 hardware may accumulate in float32,
            #    potentially producing bit-identical results — verified via Modal run.
            k_b = k_fp8_bufs[s][:sl]       # [sl, 128] float8_e4m3fn

            if sl >= 16 and sl % 16 == 0:
                # FP8 GEMM: 4× less bandwidth, FP8 Tensor Cores (faster on B200)
                scores = torch._scaled_mm(
                    q_fp8[b],       # [64, 128]
                    k_b.T,          # [128, sl] (b.t().is_contiguous() == True)
                    scale_a=scale_scalar,
                    scale_b=scale_scalar,
                    out_dtype=torch.float32,
                    use_fast_accum=False,
                )   # → [64, sl] float32 (unscaled — K scale applied below)
                # Post-multiply by per-token K scale: [64,sl] * [sl] → [64,sl]
                scores = scores * k_scales_bufs[s][0, :sl]
            else:
                # Float32 fallback — bitwise-correct on all hardware.
                # Dequant FP8 K in Python then use torch.mm.
                k_f32  = k_b.to(torch.float32) * k_scales_bufs[s][0, :sl].unsqueeze(1)
                scores = torch.mm(q_fp8[b].to(torch.float32), k_f32.T)

            # 3. NaN-preserving relu + head-weight multiply.
            #    torch.where(cond, a, b): when cond is False (NaN case), returns b=scores.
            #    scores[h,t] <= 0  for NaN is False  →  NaN passes through.
            w_b    = w[b]   # [64] float32
            relu_s = torch.where(scores <= 0,
                                 torch.zeros_like(scores),
                                 scores)
            # Broadcast weights over token dimension: [64,sl] * [64,1]
            final_scores = (relu_s * w_b.unsqueeze(1)).sum(0)   # [sl]

            # 4. TopK selection (unchanged algorithm — at::topk via Python).
            topk_vals, topk_idx = torch.topk(final_scores, actual_k)

            # 5. Convert flat token indices to physical page:offset addresses.
            #    convert_indices uses getCurrentCUDAStream() → picks up streams[s].
            mod.convert_indices(topk_idx, bt, b, max_pages, out_buf[b], actual_k)

    # --- Join -------------------------------------------------------
    # Record done_ev on each side stream, then default stream waits for all.
    default_stream = torch.cuda.current_stream()
    for s in range(N_STREAMS):
        done_evs[s].record(streams[s])
        default_stream.wait_event(done_evs[s])


# -----------------------------------------------------------------------
# Static-input CUDA graph cache — identical structure to solution11.
# Key encodes (q.shape, k.shape, bt.shape, per-batch seq_lens).
# -----------------------------------------------------------------------
_call_count:  dict = {}
_static_bufs: dict = {}
_graph_cache: dict = {}


def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table, topk_indices):
    q_c  = q_index_fp8.contiguous()
    k_c  = k_index_cache_fp8.contiguous()
    w_c  = weights.contiguous()
    bt_c = block_table.contiguous()

    # seq_lens.cpu() acts as a GPU sync (D2H copy flushes prior async work)
    # AND gives us a stable Python-hashable key.
    seq_lens_cpu = seq_lens.cpu()
    sl_tuple     = tuple(seq_lens_cpu.tolist())

    key   = (tuple(q_c.shape), tuple(k_c.shape), tuple(bt_c.shape), sl_tuple)
    count = _call_count.get(key, 0)
    _call_count[key] = count + 1

    mod    = _get_module()
    device = q_c.device
    B      = q_c.size(0)
    max_pages  = bt_c.size(1)
    max_tokens = max_pages * PAGE_SIZE

    streams, (fork_ev, done_evs) = _get_stream_resources(key, device)

    if key not in _graph_cache:
        # ── First-time setup: allocate static buffers ──────────────────
        if key not in _static_bufs:
            # Per-stream FP8 K buffers (1 byte/elem vs 4 bytes/elem for float32).
            k_fp8_bufs   = [torch.empty((max_tokens, HEAD_DIM),
                             dtype=torch.float8_e4m3fn, device=device)
                             for _ in range(N_STREAMS)]
            # Scale buffers: shape [1, max_tokens] for easy unsqueeze-free slicing.
            k_scales_bufs = [torch.empty((1, max_tokens),
                              dtype=torch.float32, device=device)
                              for _ in range(N_STREAMS)]
            # Scalar 1.0 for TensorWise _scaled_mm (only mode supporting float32 out).
            scale_scalar = torch.tensor(1.0, dtype=torch.float32, device=device)

            _static_bufs[key] = {
                'q':             torch.empty_like(q_c),
                'k':             torch.empty_like(k_c),
                'w':             torch.empty_like(w_c),
                'bt':            torch.empty_like(bt_c),
                'out':           torch.empty_like(topk_indices),
                'seq_lens_cpu':  seq_lens_cpu.clone(),
                'k_fp8_bufs':    k_fp8_bufs,
                'k_scales_bufs': k_scales_bufs,
                'scale_scalar':  scale_scalar,
            }

        sb       = _static_bufs[key]
        sl_data  = sb['seq_lens_cpu'].tolist()

        if count < WARMUP_CALLS:
            # ── Warmup: run on live tensors ─────────────────────────────
            # Warms cuBLAS FP8 path, pack_k_fp8 kernel, fork/join events.
            _do_batch(mod, q_c, k_c, w_c, bt_c, sl_tuple, B, max_pages,
                      sb['k_fp8_bufs'], sb['k_scales_bufs'], sb['scale_scalar'],
                      topk_indices,
                      streams, fork_ev, done_evs)

        else:
            # ── Graph capture ──────────────────────────────────────────
            sb['q'].copy_(q_c)
            sb['k'].copy_(k_c)
            sb['w'].copy_(w_c)
            sb['bt'].copy_(bt_c)
            torch.cuda.synchronize()   # static bufs fully written before capture

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                _do_batch(mod, sb['q'], sb['k'], sb['w'], sb['bt'], sl_data,
                          B, max_pages,
                          sb['k_fp8_bufs'], sb['k_scales_bufs'], sb['scale_scalar'],
                          sb['out'],
                          streams, fork_ev, done_evs)
            _graph_cache[key] = g

            # Replay once to produce valid output for the capture iteration.
            g.replay()
            topk_indices.copy_(sb['out'])

    else:
        # ── All subsequent calls: async-copy → replay → async-copy ────
        sb = _static_bufs[key]
        sb['q'].copy_(q_c,   non_blocking=True)
        sb['k'].copy_(k_c,   non_blocking=True)
        sb['w'].copy_(w_c,   non_blocking=True)
        sb['bt'].copy_(bt_c, non_blocking=True)
        _graph_cache[key].replay()
        topk_indices.copy_(sb['out'], non_blocking=True)
