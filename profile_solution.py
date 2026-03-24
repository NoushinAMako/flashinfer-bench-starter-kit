"""
Profile solution (kernel9) to understand time breakdown per step.
Runs a large workload and times each phase with CUDA events.
"""
import torch
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, "/home/noushin/flashinfer/index/flashinfer-bench-starter-kit")

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM = 128
TOPK = 2048
NUM_PAGES = 11923
TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")

# Load a large workload (high batch size, many pages)
WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
import safetensors.torch as st

workloads = [json.loads(l) for l in open(WL_FILE)]
# Pick a large workload
large_wls = [w for w in workloads if w['workload']['axes']['batch_size'] >= 20 and
             w['workload']['axes']['max_num_pages'] >= 8]
wl = large_wls[0]
print(f"Workload: {wl['workload']['axes']}")

axes = wl['workload']['axes']
B = axes['batch_size']
max_pages = axes['max_num_pages']
inputs_spec = wl['workload']['inputs']

torch.manual_seed(42)
q = torch.randn(B, NUM_HEADS, HEAD_DIM).clamp_(-2, 2).to(torch.float8_e4m3fn).cuda()
k_cache = torch.randint(-128, 128, (NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM + 4), dtype=torch.int8).cuda()
weights = torch.randn(B, NUM_HEADS, dtype=torch.float32).cuda()

sl_tensors = st.load_file(str(TRACE_ROOT / inputs_spec['seq_lens']['path']))
seq_lens = sl_tensors[inputs_spec['seq_lens']['tensor_key']].cuda()
bt_tensors = st.load_file(str(TRACE_ROOT / inputs_spec['block_table']['path']))
block_table = bt_tensors[inputs_spec['block_table']['tensor_key']].cuda()

seq_lens_cpu = seq_lens.cpu().tolist()
print(f"seq_lens: min={min(seq_lens_cpu)} max={max(seq_lens_cpu)} sum={sum(seq_lens_cpu)}")

# Manually profile each step
from solution.triton.kernel import _get_module
mod = _get_module()

# Warmup
topk_out = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
for _ in range(5):
    mod.dsa_topk_run(q.contiguous(), k_cache.contiguous(), weights.contiguous(),
                     seq_lens.contiguous(), block_table.contiguous(), topk_out)
torch.cuda.synchronize()

# Profile the full solution
N = 50
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    mod.dsa_topk_run(q.contiguous(), k_cache.contiguous(), weights.contiguous(),
                     seq_lens.contiguous(), block_table.contiguous(), topk_out)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"\nFull solution: {(t1-t0)/N*1000:.3f} ms/iter")

# Profile individual steps manually
q_f32 = q.to(torch.float32)
k_u8 = k_cache.view(torch.uint8).contiguous()
sl_data = seq_lens.cpu()

# Phase timings using CUDA events
def time_phase(fn, warmup=3, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

print("\n--- Phase breakdown (averaged over all batch items) ---")

# 1. seq_lens D2H
def f1(): return seq_lens.cpu()
t_cpu = time_phase(f1)
print(f"seq_lens.cpu():        {t_cpu:.3f} ms")

# 2. q FP8→F32
def f2(): return q.to(torch.float32)
t_q = time_phase(f2)
print(f"q FP8→F32:             {t_q:.3f} ms")

# 3. fill_(-1)
def f3(): topk_out.fill_(-1)
t_fill = time_phase(f3)
print(f"topk_out.fill_(-1):    {t_fill:.3f} ms")

# 4. Per-batch dequant (total)
from torch.utils.cpp_extension import load_inline
dequant_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define PAGE_SIZE 64
#define HEAD_DIM 128
#define BYTES_PER_PAGE 8448

__device__ __forceinline__ float fp8e4m3_to_float(uint8_t x) {
    if ((x & 0x7F) == 0x7F) return __uint_as_float(((uint32_t)(x>>7)<<31)|0x7FC00000u);
    uint32_t sign=(uint32_t)(x>>7)<<31,exp=(x>>3)&0xF,mant=x&0x7;
    if ((x&0x7F)==0) return __uint_as_float(sign);
    uint32_t f;
    if (exp==0){uint32_t hb=31-__clz(mant);f=sign|((118u+hb)<<23)|((mant^(1u<<hb))<<(23-hb));}
    else f=sign|((exp+120u)<<23)|((uint32_t)mant<<20);
    return __uint_as_float(f);
}
__global__ void dequant_fp8_kernel(const uint8_t* __restrict__ kc, const int* __restrict__ pids,
    float* __restrict__ K, int np) {
    int pl=blockIdx.x; if(pl>=np) return;
    int tok=threadIdx.x; if(tok>=PAGE_SIZE) return;
    int pp=pids[pl];
    const uint8_t* pb=kc+(long long)pp*BYTES_PER_PAGE;
    const uint8_t* row=pb+tok*HEAD_DIM;
    float sc=__ldg(reinterpret_cast<const float*>(pb+PAGE_SIZE*HEAD_DIM+tok*4));
    float* out=K+((long long)pl*PAGE_SIZE+tok)*HEAD_DIM;
    #pragma unroll 8
    for(int d=0;d<HEAD_DIM;d+=4){
        uint32_t p=__ldg(reinterpret_cast<const uint32_t*>(row+d));
        out[d]=fp8e4m3_to_float((uint8_t)(p))     *sc;
        out[d+1]=fp8e4m3_to_float((uint8_t)(p>>8)) *sc;
        out[d+2]=fp8e4m3_to_float((uint8_t)(p>>16))*sc;
        out[d+3]=fp8e4m3_to_float((uint8_t)(p>>24))*sc;
    }
}
void run_dequant(torch::Tensor kc, torch::Tensor pids, torch::Tensor K, int np) {
    auto stream=at::cuda::getCurrentCUDAStream();
    dequant_fp8_kernel<<<np, PAGE_SIZE, 0, stream>>>(
        kc.data_ptr<uint8_t>(), pids.data_ptr<int>(), K.data_ptr<float>(), np);
}
"""
dmod = load_inline("profile_dequant", cpp_sources="void run_dequant(torch::Tensor,torch::Tensor,torch::Tensor,int);",
                   cuda_sources=dequant_src, functions=["run_dequant"],
                   extra_cuda_cflags=["-O3"], extra_cflags=["-O3"], verbose=False)

K_buf = torch.empty(max_pages * PAGE_SIZE, HEAD_DIM, device='cuda')
sl_list = seq_lens.cpu().tolist()

def f4_dequant():
    for b in range(B):
        sl = sl_list[b]
        np_seq = (sl + PAGE_SIZE - 1) // PAGE_SIZE
        pids = block_table[b, :np_seq].to(torch.int32).contiguous()
        dmod.run_dequant(k_u8, pids, K_buf, np_seq)
t_dequant = time_phase(f4_dequant)
print(f"all dequants (total):  {t_dequant:.3f} ms  ({t_dequant/B:.4f} ms/item)")

# 5. Per-batch mm (total)
K_bufs = []
for b in range(B):
    sl = sl_list[b]
    np_seq = (sl + PAGE_SIZE - 1) // PAGE_SIZE
    pids = block_table[b, :np_seq].to(torch.int32).contiguous()
    dmod.run_dequant(k_u8, pids, K_buf, np_seq)
    K_bufs.append(K_buf[:sl].clone())
torch.cuda.synchronize()

def f5_mm():
    for b in range(B):
        _ = torch.mm(q_f32[b], K_bufs[b].t())
t_mm = time_phase(f5_mm)
print(f"all at::mm (total):    {t_mm:.3f} ms  ({t_mm/B:.4f} ms/item)")

# 6. relu_weight_mul + sum (total)
scores_list = [torch.mm(q_f32[b], K_bufs[b].t()) for b in range(B)]
torch.cuda.synchronize()

def f6_postproc():
    for b in range(B):
        sl = sl_list[b]
        s = scores_list[b].clone()
        # relu + weight mul
        mask = s > 0
        s = s * mask * weights[b:b+1, :, None].squeeze(0).unsqueeze(1)
        _ = s.sum(0)
t_post = time_phase(f6_postproc)
print(f"relu+weight+sum total: {t_post:.3f} ms  ({t_post/B:.4f} ms/item)")

# 7. topk (total)
finals = [torch.mm(q_f32[b], K_bufs[b].t()).sum(0) for b in range(B)]
torch.cuda.synchronize()

def f7_topk():
    for b in range(B):
        sl = sl_list[b]
        actual_k = min(TOPK, sl)
        _ = torch.topk(finals[b], actual_k)
t_topk = time_phase(f7_topk)
print(f"all topk (total):      {t_topk:.3f} ms  ({t_topk/B:.4f} ms/item)")

print(f"\nSum of phases: {t_cpu+t_q+t_fill+t_dequant+t_mm+t_post+t_topk:.3f} ms")
print(f"(vs actual {(t1-t0)/N*1000:.3f} ms -- difference = CUDA scheduling overhead etc)")
