"""Find the exact source of divergence in solution4 for a failing workload."""
import torch, sys, json
from pathlib import Path
sys.path.insert(0, "/home/noushin/flashinfer/index/flashinfer-bench-starter-kit")

PAGE_SIZE = 64; NUM_HEADS = 64; HEAD_DIM = 128; TOPK = 2048
TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")
import safetensors.torch as st

WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = [json.loads(l) for l in open(WL_FILE)]

# Use failing workload (wl=92, max_num_pages=18)
wl = workloads[92]
axes = wl['workload']['axes']; inp = wl['workload']['inputs']
B = axes['batch_size']; max_pages = axes['max_num_pages']
print(f"Workload axes: {axes}")

torch.manual_seed(92)
q = torch.randn(B, NUM_HEADS, HEAD_DIM).clamp_(-2,2).to(torch.float8_e4m3fn).cuda()
k = torch.randint(-128,128,(axes['num_pages'],PAGE_SIZE,1,HEAD_DIM+4),dtype=torch.int8).cuda()
w = torch.randn(B, NUM_HEADS, dtype=torch.float32).cuda()
sl_t = st.load_file(str(TRACE_ROOT/inp['seq_lens']['path']))[inp['seq_lens']['tensor_key']].cuda()
bt   = st.load_file(str(TRACE_ROOT/inp['block_table']['path']))[inp['block_table']['tensor_key']].cuda()

sl_list = sl_t.cpu().tolist()
print(f"seq_lens: {sl_list}")

from torch.utils.cpp_extension import load_inline
# Load the same dequant kernel as solution (kernel9)
dq_mod = load_inline(
    name="dq_compare",
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
__global__ void dq_kernel(const uint8_t*kc,const int*pids,float*K,int np){
    int pl=blockIdx.x;if(pl>=np)return;int tok=threadIdx.x;if(tok>=PAGE_SIZE)return;
    int pp=pids[pl];const uint8_t*pb=kc+(long long)pp*BYTES;
    float sc=__ldg((const float*)(pb+PAGE_SIZE*HEAD_DIM+tok*4));
    float*out=K+((long long)pl*PAGE_SIZE+tok)*HEAD_DIM;
    for(int d=0;d<HEAD_DIM;d+=4){uint32_t p=__ldg((const uint32_t*)(pb+tok*HEAD_DIM+d));
        out[d]=f8(p)*sc;out[d+1]=f8(p>>8)*sc;out[d+2]=f8(p>>16)*sc;out[d+3]=f8(p>>24)*sc;}}
void run(torch::Tensor kc,torch::Tensor pids,torch::Tensor K,int np){
    auto s=at::cuda::getCurrentCUDAStream();dq_kernel<<<np,PAGE_SIZE,0,s>>>(
        kc.data_ptr<uint8_t>(),pids.data_ptr<int>(),K.data_ptr<float>(),np);}
""", functions=["run"],
    extra_cuda_cflags=["-O3"], extra_cflags=["-O3"], verbose=False)

k_u8 = k.view(torch.uint8).contiguous()
q_f32 = q.to(torch.float32)
max_sl = max_pages * PAGE_SIZE

# Step 1: Compare decoded K values: per-batch vs all-batch
print("\n=== Comparing decoded K values ===")
K_seq = []
for b in range(B):
    sl = sl_list[b]
    np_seq = (sl + PAGE_SIZE - 1) // PAGE_SIZE
    pids = bt[b, :np_seq].to(torch.int32).contiguous()
    K_buf = torch.zeros(max_sl, HEAD_DIM, device='cuda')  # zero-padded!
    dq_mod.run(k_u8, pids, K_buf, np_seq)
    K_seq.append(K_buf[:sl].clone())
torch.cuda.synchronize()

# Now the "all-batch" version with actual solution4 kernel
from solution4.triton.kernel import _get_module
mod4 = _get_module()
# We'll simulate the all-batch decode using a simpler per-batch approach with zero padding
K_batch = torch.zeros(B, max_sl, HEAD_DIM, device='cuda')
for b in range(B):
    sl = sl_list[b]
    np_seq = (sl + PAGE_SIZE - 1) // PAGE_SIZE
    pids = bt[b, :np_seq].to(torch.int32).contiguous()
    K_buf = torch.empty(np_seq * PAGE_SIZE, HEAD_DIM, device='cuda')
    dq_mod.run(k_u8, pids, K_buf, np_seq)
    K_batch[b, :sl] = K_buf[:sl]  # zero-padded beyond sl
torch.cuda.synchronize()

# Step 2: Compare scores from sequential mm vs batched bmm
print("=== Comparing GEMM scores ===")
scores_seq = [torch.mm(q_f32[b], K_seq[b].t()) for b in range(B)]  # [64, sl_b]
scores_batch_all = torch.bmm(q_f32, K_batch.transpose(1, 2))  # [B, 64, max_sl]
torch.cuda.synchronize()

n_mismatch = 0
for b in range(B):
    sl = sl_list[b]
    s1 = scores_seq[b]          # [64, sl]
    s2 = scores_batch_all[b, :, :sl]  # [64, sl]
    # Compare ignoring NaN positions (where both are NaN)
    both_nan = torch.isnan(s1) & torch.isnan(s2)
    not_nan = ~(torch.isnan(s1) | torch.isnan(s2))
    finite_match = torch.equal(s1[not_nan], s2[not_nan]) if not_nan.any() else True
    nan_match = (torch.isnan(s1) == torch.isnan(s2)).all()
    if not (finite_match and nan_match):
        diff = (s1 - s2).abs()
        print(f"  b={b} sl={sl}: MISMATCH max_diff={diff.nanmean():.2e} nan_positions={torch.isnan(s1).sum().item()}")
        n_mismatch += 1
    else:
        nan_count = torch.isnan(s1).sum().item()

if n_mismatch == 0:
    print("  All GEMM scores match! (ignoring NaN-at-same-positions)")
    print("  Checking for NaN patterns...")
    for b in range(B):
        sl = sl_list[b]
        s1 = scores_seq[b]
        n_nan = torch.isnan(s1).sum().item()
        if n_nan > 0:
            print(f"  b={b} sl={sl}: {n_nan}/{s1.numel()} NaN scores")

# Step 3: Compare final_scores after relu+weight+sum
print("\n=== Comparing final_scores ===")
final_seq = []
for b in range(B):
    s = scores_seq[b].clone()
    s.clamp_(min=0)  # relu
    s.mul_(w[b].unsqueeze(1))        # weight mul
    final_seq.append(s.sum(0))

scores_batch_all.clamp_(0)
scores_batch_all.mul_(w.unsqueeze(2))  # [B, 64, 1]
final_batch = scores_batch_all.sum(1)  # [B, max_sl]
torch.cuda.synchronize()

for b in range(B):
    sl = sl_list[b]
    f1 = final_seq[b]         # [sl]
    f2 = final_batch[b, :sl]  # [sl]
    both_nan = torch.isnan(f1) & torch.isnan(f2)
    not_nan = ~(torch.isnan(f1) | torch.isnan(f2))
    finite_match = torch.equal(f1[not_nan], f2[not_nan]) if not_nan.any() else True
    nan_match = (torch.isnan(f1) == torch.isnan(f2)).all()
    if not (finite_match and nan_match):
        diff = (f1 - f2).abs()
        print(f"  b={b} sl={sl}: final_scores MISMATCH! diff={diff[~torch.isnan(diff)].max():.2e}")
    else:
        print(f"  b={b} sl={sl}: final_scores match")
