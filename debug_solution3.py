"""Quick correctness + performance comparison: solution3 (multi-stream) vs solution (baseline)."""
import torch, sys, json, time
from pathlib import Path
sys.path.insert(0, "/home/noushin/flashinfer/index/flashinfer-bench-starter-kit")

PAGE_SIZE = 64; NUM_HEADS = 64; HEAD_DIM = 128; TOPK = 2048; NUM_PAGES = 11923
TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")
import safetensors.torch as st

WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = [json.loads(l) for l in open(WL_FILE)]

from solution.triton.kernel import run as run_baseline
from solution3.triton.kernel import run as run_s3

def make(wl, seed=42):
    axes = wl['workload']['axes']; inp = wl['workload']['inputs']
    B = axes['batch_size']; np = axes['num_pages']
    torch.manual_seed(seed)
    q = torch.randn(B, NUM_HEADS, HEAD_DIM).clamp_(-2,2).to(torch.float8_e4m3fn).cuda()
    k = torch.randint(-128,128,(np,PAGE_SIZE,1,HEAD_DIM+4),dtype=torch.int8).cuda()
    w = torch.randn(B, NUM_HEADS, dtype=torch.float32).cuda()
    sl = st.load_file(str(TRACE_ROOT/inp['seq_lens']['path']))[inp['seq_lens']['tensor_key']].cuda()
    bt = st.load_file(str(TRACE_ROOT/inp['block_table']['path']))[inp['block_table']['tensor_key']].cuda()
    return q, k, w, sl, bt, B

print("=== Correctness check ===")
n_pass = 0
for i, wl in enumerate(workloads):
    q, k, w, sl, bt, B = make(wl, seed=i)
    ref = torch.full((B,TOPK),-1,dtype=torch.int32,device='cuda')
    got = torch.full((B,TOPK),-1,dtype=torch.int32,device='cuda')
    run_baseline(q,k,w,sl,bt,ref)
    run_s3(q,k,w,sl,bt,got)
    torch.cuda.synchronize()
    err = (ref.float()-got.float()).abs().max().item()
    if err == 0.0:
        n_pass += 1
    else:
        axes = wl['workload']['axes']
        print(f"  FAIL wl={i} axes={axes} abs_err={err:.1f}")
print(f"Passed: {n_pass}/128")

print("\n=== Performance comparison (large workloads) ===")
large = [w for w in workloads if w['workload']['axes']['batch_size'] >= 20][:5]
for wl in large:
    axes = wl['workload']['axes']
    q, k, w, sl, bt, B = make(wl)
    ref_out = torch.full((B,TOPK),-1,dtype=torch.int32,device='cuda')
    s3_out  = torch.full((B,TOPK),-1,dtype=torch.int32,device='cuda')
    # warmup
    for _ in range(5):
        run_baseline(q,k,w,sl,bt,ref_out)
        run_s3(q,k,w,sl,bt,s3_out)
    torch.cuda.synchronize()
    N=50
    t0=time.perf_counter()
    for _ in range(N): run_baseline(q,k,w,sl,bt,ref_out)
    torch.cuda.synchronize(); tb=(time.perf_counter()-t0)/N*1000
    t0=time.perf_counter()
    for _ in range(N): run_s3(q,k,w,sl,bt,s3_out)
    torch.cuda.synchronize(); ts=(time.perf_counter()-t0)/N*1000
    print(f"  B={axes['batch_size']} max_pages={axes['max_num_pages']}: baseline={tb:.3f}ms  multistream={ts:.3f}ms  speedup={tb/ts:.2f}x")
