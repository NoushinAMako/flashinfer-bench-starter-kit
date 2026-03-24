"""
Test whether torch.bmm with zero-padded K gives the same results as
per-batch torch.mm — critical for correctness if we switch to batched GEMM.
"""
import torch
import sys, json
from pathlib import Path
sys.path.insert(0, "/home/noushin/flashinfer/index/flashinfer-bench-starter-kit")

PAGE_SIZE = 64; NUM_HEADS = 64; HEAD_DIM = 128; NUM_PAGES = 11923
TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")
import safetensors.torch as st

WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = [json.loads(l) for l in open(WL_FILE)]

# Test on several workloads with varying seq_lens
from solution.triton.kernel import _get_module
mod = _get_module()

import torch.utils.cpp_extension as ext
dequant_mod = ext.load_inline(
    name="dequant_test",
    cpp_sources="void run(torch::Tensor,torch::Tensor,torch::Tensor,int);",
    cuda_sources=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#define PAGE_SIZE 64
#define HEAD_DIM 128
#define BYTES 8448
__device__ float f8(uint8_t x){
    if((x&127)==127)return __uint_as_float(((uint32_t)(x>>7)<<31)|0x7FC00000u);
    uint32_t s=(uint32_t)(x>>7)<<31,e=(x>>3)&15,m=x&7;
    if((x&127)==0)return __uint_as_float(s);
    uint32_t f;if(e==0){uint32_t hb=31-__clz(m);f=s|((118+hb)<<23)|((m^(1u<<hb))<<(23-hb));}
    else f=s|((e+120u)<<23)|((uint32_t)m<<20);return __uint_as_float(f);}
__global__ void dq(const uint8_t*kc,const int*pids,float*K,int np){
    int pl=blockIdx.x;if(pl>=np)return;int tok=threadIdx.x;if(tok>=PAGE_SIZE)return;
    int pp=pids[pl];const uint8_t*pb=kc+(long long)pp*BYTES;
    float sc=__ldg((const float*)(pb+PAGE_SIZE*HEAD_DIM+tok*4));
    float*out=K+((long long)pl*PAGE_SIZE+tok)*HEAD_DIM;
    for(int d=0;d<HEAD_DIM;d+=4){uint32_t p=__ldg((const uint32_t*)(pb+tok*HEAD_DIM+d));
        out[d]=f8(p)*sc;out[d+1]=f8(p>>8)*sc;out[d+2]=f8(p>>16)*sc;out[d+3]=f8(p>>24)*sc;}}
void run(torch::Tensor kc,torch::Tensor pids,torch::Tensor K,int np){
    auto s=at::cuda::getCurrentCUDAStream();dq<<<np,PAGE_SIZE,0,s>>>(
        kc.data_ptr<uint8_t>(),pids.data_ptr<int>(),K.data_ptr<float>(),np);}
""", functions=["run"],
    extra_cuda_cflags=["-O3"], extra_cflags=["-O3"], verbose=False)

n_match = 0; n_total = 0
for trial, wl in enumerate(workloads[:20]):
    axes = wl['workload']['axes']
    B = axes['batch_size']; max_pages = axes['max_num_pages']; np_total = axes['num_pages']
    inp = wl['workload']['inputs']

    torch.manual_seed(trial)
    q = torch.randn(B, NUM_HEADS, HEAD_DIM).clamp_(-2,2).to(torch.float8_e4m3fn).cuda()
    k = torch.randint(-128,128,(np_total,PAGE_SIZE,1,HEAD_DIM+4),dtype=torch.int8).cuda()
    k_u8 = k.view(torch.uint8).contiguous()
    sl_t = st.load_file(str(TRACE_ROOT/inp['seq_lens']['path']))[inp['seq_lens']['tensor_key']].cuda()
    bt   = st.load_file(str(TRACE_ROOT/inp['block_table']['path']))[inp['block_table']['tensor_key']].cuda()

    sl_list = sl_t.cpu().tolist()
    q_f32 = q.to(torch.float32)
    max_sl = max_pages * PAGE_SIZE

    # Build padded K batch: K_batch[b, :sl_b, :] = decoded, K_batch[b, sl_b:, :] = 0
    K_batch = torch.zeros(B, max_sl, HEAD_DIM, device='cuda')
    K_individual = []
    for b in range(B):
        sl = sl_list[b]; np_seq = (sl + PAGE_SIZE - 1) // PAGE_SIZE
        pids = bt[b, :np_seq].to(torch.int32).contiguous()
        K_buf = torch.empty(np_seq * PAGE_SIZE, HEAD_DIM, device='cuda')
        dequant_mod.run(k_u8, pids, K_buf, np_seq)
        K_batch[b, :sl] = K_buf[:sl]
        K_individual.append(K_buf[:sl].clone())
    torch.cuda.synchronize()

    # Method 1: individual mm per batch item (current approach)
    scores_seq = [torch.mm(q_f32[b], K_individual[b].t()) for b in range(B)]
    torch.cuda.synchronize()

    # Method 2: batched mm with zero-padded K
    scores_batch = torch.bmm(q_f32, K_batch.transpose(1, 2))  # [B, 64, max_sl]
    torch.cuda.synchronize()

    # Compare for valid positions only
    all_match = True
    for b in range(B):
        sl = sl_list[b]
        s1 = scores_seq[b]             # [64, sl]
        s2 = scores_batch[b, :, :sl]   # [64, sl]
        if not torch.equal(s1, s2):
            diff = (s1 - s2).abs()
            all_match = False
            print(f"  MISMATCH trial={trial} b={b} sl={sl}: max_diff={diff.max().item():.6e} "
                  f"n_diff={(diff>0).sum().item()}/{diff.numel()}")
            break

    if all_match:
        n_match += 1
    n_total += 1

print(f"\nbmm == sequential mm: {n_match}/{n_total} workloads match exactly")
