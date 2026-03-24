#include <cuda_runtime.h>
#include <stdint.h>
#include <tvm/ffi/function.h>

// -----------------------------------------------------------------------
// Helper: FP8 E4M3 -> float conversion
// TODO: implement fp8 to float lookup table or inline conversion
// -----------------------------------------------------------------------


// -----------------------------------------------------------------------
// Kernel 1: compute per-token scores
//
// Inputs:
//   q_fp8          - [batch_size, 64, 128] FP8 query
//   k_cache_raw    - [num_pages, 64, 1, 132] FP8 key cache (128 data + 4 scale bytes per token)
//   weights        - [batch_size, 64] float32 per-head weights
//   seq_lens       - [batch_size] int32 sequence lengths
//   block_table    - [batch_size, max_num_pages] int32 page mapping
//   max_num_pages  - int
//
// Output:
//   final_scores   - [batch_size, max_num_pages * 64] float32
//
// For each token t in sequence b:
//   score[b][t] = sum over heads h: relu(q[b][h] dot k[t][h]) * weights[b][h]
//
// TODO: implement
// -----------------------------------------------------------------------
__global__ void compute_scores_kernel(
    const uint8_t* q_fp8,
    const uint8_t* k_cache_raw,
    const float*   weights,
    const int*     seq_lens,
    const int*     block_table,
    float*         final_scores,
    int            max_num_pages
) {
    // TODO
}


// -----------------------------------------------------------------------
// Kernel 2: top-K selection
//
// Input:
//   final_scores   - [batch_size, max_num_pages * 64] float32
//   seq_lens       - [batch_size] int32
//   block_table    - [batch_size, max_num_pages] int32
//   max_num_pages  - int
//
// Output:
//   topk_indices   - [batch_size, 2048] int32
//                    global token indices (page * 64 + offset), -1 for padding
//
// TODO: implement (e.g. bitonic sort or radix select)
// -----------------------------------------------------------------------
__global__ void topk_kernel(
    const float* final_scores,
    const int*   seq_lens,
    const int*   block_table,
    int*         topk_indices,
    int          max_num_pages
) {
    // TODO
}


// -----------------------------------------------------------------------
// TVM FFI entry point
//
// Uses TVM_FFI_DLL_EXPORT_TYPED_FUNC to export "kernel" so the framework
// can find it via tvm_ffi.load_module().
//
// Signature matches definition inputs + output (DPS style):
//   q_index_fp8        [batch_size, 64, 128]       float8_e4m3fn
//   k_index_cache_fp8  [num_pages, 64, 1, 132]     int8
//   weights            [batch_size, 64]             float32
//   seq_lens           [batch_size]                 int32
//   block_table        [batch_size, max_num_pages]  int32
//   topk_indices       [batch_size, 2048]           int32  (output, pre-allocated)
// -----------------------------------------------------------------------
void kernel_(DLTensor* q_index_fp8,
             DLTensor* k_index_cache_fp8,
             DLTensor* weights,
             DLTensor* seq_lens,
             DLTensor* block_table,
             DLTensor* topk_indices) {
    int batch_size    = q_index_fp8->shape[0];
    int max_num_pages = block_table->shape[1];

    // TODO: allocate final_scores [batch_size, max_num_pages * 64] on device
    // TODO: launch compute_scores_kernel
    // TODO: launch topk_kernel
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(kernel, kernel_);

