# Technical Report — DSA Track (Sparse Attention)

**Team:** Makora  
**Track:** Dense Sparse Attention (DSA)  
**Kernels:** DSA TopK Indexer + DSA Sparse Attention  
**Approach:** Agent-Assisted

---

## Overview

The DSA track requires two kernels that work together:

1. **DSA TopK Indexer** — given FP8-quantized query and key caches, produce a list of the top-K most relevant page indices for each token in the batch.
2. **DSA Sparse Attention** — given the sparse page indices from the indexer, compute multi-latent attention (MLA) over only the selected top-K pages.

Both kernels use **destination-passing style (DPS)**: they write results into pre-allocated output buffers rather than allocating new tensors on each call. This is a critical optimization since the FlashInfer baselines allocate tensors on every call, adding significant overhead that our kernels avoid.

---

## Kernel 1: DSA TopK Indexer

**Definition:** `dsa_topk_indexer_fp8_h64_d128_topk2048_ps64`  
**Baseline:** `flashinfer_deepgemm_wrapper_2ba145` (DeepGEMM FP8 + FlashInfer top_k_page_table_transform)

### Approach

The indexer computes attention scores between each query token and all cached key pages, then selects the top-K (2048) highest-scoring pages. The core challenge is efficiently processing a batch of up to 30 tokens, each with different context lengths, while maximizing GPU utilization.

### Key Optimizations

#### 1. Multi-Stream Parallelism (N_STREAMS = 8)

Each token in the batch is an independent scoring problem. Assigning tokens to a pool of 8 CUDA streams allows the GPU to execute up to 8 scoring pipelines concurrently:

```
Stream 0: token 0 → token 8 → token 16 → token 24
Stream 1: token 1 → token 9 → token 17 → token 25
...
Stream 7: token 7 → token 15 → token 23 → token 29
```

With 8 streams on the B200's 144 SMs, each GEMM (`[64, 128] × [128, seq_len]`) is small enough that multiple can run simultaneously without contention. The fork/join event pattern ensures all streams synchronize correctly within the graph capture framework.

For large batches (B = 25–30), this reduces the worst-case latency from ~7-8 serial items per stream to ~3-4, roughly halving the total runtime for those workloads.

#### 2. CUDA Graph Capture

After 3 warmup calls (to prime cuBLAS workspace and C++ static state), the full 8-stream pipeline is captured as a CUDA graph. Subsequent calls become a sequence of:
1. Copy live inputs → static buffers (default stream, non-blocking)
2. `graph.replay()` — replays the entire pipeline with zero kernel-launch overhead
3. Copy static output → live output buffer (default stream, non-blocking)

This eliminates hundreds of CPU-side kernel dispatch calls per benchmark iteration.

#### 3. Static Buffer Pre-allocation

Per-shape static GPU buffers are allocated once and reused. The cache key includes the full input shape and sequence length values, ensuring correctness when context lengths change between benchmark calls.

#### 4. FP8 Pipeline

Each stream runs: FP8 dequantization → FP32 GEMM (via cuBLAS) → ReLU-sum scoring → `at::topk` → int32 index conversion. The FP8 format is the `deep_gemm`-compatible layout (FP8 data + per-page FP32 scales).

### Performance

| Metric | Value |
|--------|-------|
| Workloads | 128 (all PASSED) |
| Correctness | Perfect (abs_err = 0.00 on all workloads) |
| Speedup range | 7.4x – 21.2x |
| Arithmetic mean | 13.65x |
| **Geometric mean** | **13.26x** |

---

## Kernel 2: DSA Sparse Attention

**Definition:** `dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64`  
**Baseline:** `flashinfer_wrapper_5af199` (FlashInfer TRT-LLM MLA decode, DPS=False)

### Approach

Given the top-K indices from the indexer, this kernel computes multi-latent attention (MLA) with:
- 16 query/output heads
- 512-dim compressed KV (`ckv`) + 64-dim rotary PE key (`kpe`) = 576-dim concatenated key
- 2048 selected top-K pages per token

### Key Optimizations

#### 1. Fused BF16 → FP32 Concat-Gather Kernel

A single custom CUDA kernel reads the `ckv` and `kpe` caches at the sparse page indices and writes a contiguous `[T, K, 576]` FP32 tensor in one pass. This avoids:
- A separate gather step
- Intermediate BF16 tensors
- A separate concatenation along the head dimension

The kernel vectorizes reads in groups of 8 elements (`dim8 = 576/8 = 72` threads per row-group) to maximize memory bandwidth.

#### 2. Fused Scale + Mask + Softmax + LSE

Instead of computing attention weights and log-sum-exp in separate passes, a single kernel handles the entire softmax numerically in one thread block per attention row:
- Loads all K logits into shared memory
- Applies softmax scale and `-inf` masking (for -1 index padding)
- Computes the stable-softmax in one pass with online normalization
- Writes the attention weights (normalized) and LSE simultaneously

#### 3. cuBLAS BMMs for GEMM Steps

The two large matrix multiplications (query-key logits and attention-value output) use cuBLAS, which is always available and optimally tuned for the B200 tensor cores.

#### 4. Destination-Passing Style

The `run()` function writes directly into the caller-provided `output` and `lse` tensors. The baseline allocates fresh output tensors on every call — DPS eliminates this allocation cost, which is especially significant in benchmarking tight loops.

### Performance

| Metric | Value |
|--------|-------|
| Workloads | 23 (all PASSED) |
| Speedup range | 12.0x – 17.4x |
| Arithmetic mean | 14.85x |
| **Geometric mean** | **14.68x** |

---

## Track Score

Per the scoring rules, the track speedup is the arithmetic mean of the two kernel speedups:

```
DSA track score ≈ (13.65 + 14.85) / 2 = 14.25x
```
