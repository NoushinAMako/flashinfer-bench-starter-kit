"""
TVM FFI Bindings for DSA TopK CUDA Kernel.

Compiles kernel.cu at import time via nvcc, loads the resulting shared
library via ctypes, and registers the run() entry point with TVM FFI.
"""

import ctypes
import os
import subprocess

import torch
from tvm.ffi import register_func

_dir = os.path.dirname(os.path.abspath(__file__))
_so  = os.path.join(_dir, "libkernel.so")

# -----------------------------------------------------------------------
# Compile kernel.cu -> libkernel.so at import time
# TODO: adjust --arch flag to match target GPU (e.g. sm_90a for B200)
# -----------------------------------------------------------------------
subprocess.run(
    [
        "nvcc", "-O3", "-shared", "-fPIC",
        "--arch=sm_90a",
        "-o", _so,
        os.path.join(_dir, "kernel.cu"),
    ],
    check=True,
)

# -----------------------------------------------------------------------
# Load compiled shared library and declare C function signature
# -----------------------------------------------------------------------
_lib = ctypes.CDLL(_so)
_lib.kernel.restype = None
_lib.kernel.argtypes = [
    ctypes.c_void_p,  # q_index_fp8
    ctypes.c_void_p,  # k_index_cache_fp8
    ctypes.c_void_p,  # weights
    ctypes.c_void_p,  # seq_lens
    ctypes.c_void_p,  # block_table
    ctypes.c_void_p,  # topk_indices (output)
    ctypes.c_void_p,  # final_scores (intermediate)
    ctypes.c_int,     # batch_size
    ctypes.c_int,     # max_num_pages
]


# -----------------------------------------------------------------------
# TVM FFI entry point — name must match entry_point in config.toml
#
# Inputs (from benchmark framework):
#   q_index_fp8        - [batch_size, 64, 128] torch.float8_e4m3fn
#   k_index_cache_fp8  - [num_pages, 64, 1, 132] torch.int8
#   weights            - [batch_size, 64] torch.float32
#   seq_lens           - [batch_size] torch.int32
#   block_table        - [batch_size, max_num_pages] torch.int32
#
# Returns:
#   (topk_indices,)    - tuple with [batch_size, 2048] torch.int32
#                        global token indices, -1 for padding
# -----------------------------------------------------------------------
@register_func("kernel")
def kernel(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size    = q_index_fp8.shape[0]
    max_num_pages = block_table.shape[1]
    device        = q_index_fp8.device

    # Intermediate buffer for per-token scores
    final_scores = torch.full(
        (batch_size, max_num_pages * 64), float("-inf"),
        dtype=torch.float32, device=device
    )

    # Output buffer
    topk_indices = torch.full(
        (batch_size, 2048), -1,
        dtype=torch.int32, device=device
    )

    _lib.kernel(
        ctypes.c_void_p(q_index_fp8.data_ptr()),
        ctypes.c_void_p(k_index_cache_fp8.data_ptr()),
        ctypes.c_void_p(weights.data_ptr()),
        ctypes.c_void_p(seq_lens.data_ptr()),
        ctypes.c_void_p(block_table.data_ptr()),
        ctypes.c_void_p(topk_indices.data_ptr()),
        ctypes.c_void_p(final_scores.data_ptr()),
        ctypes.c_int(batch_size),
        ctypes.c_int(max_num_pages),
    )

    return (topk_indices,)

