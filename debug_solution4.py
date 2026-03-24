"""Correctness + performance comparison: solution4 (batched) vs solution (baseline)."""
import torch, sys, json, time
from pathlib import Path
sys.path.insert(0, "/home/noushin/flashinfer/index/flashinfer-bench-starter-kit")

PAGE_SIZE = 64; NUM_HEADS = 64; HEAD_DIM = 128; TOPK = 2048; NUM_PAGES = 11923
TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")
import safetensors.torch as st

WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = [json.loads(l) for l in open(WL_FILE)]

from solution.triton.kernel import run as run_baseline
from solution4.triton.kernel import run as run_s4

def make(wl, seed=42):
    axes = wl['workload']['axes']; inp = wl['workload']['inputs']
    B = axes['batch_size']; np = axes['num_pages']
    torch.manual_seed(seed)
    q = torch.randn(B, NUM_HEADS, HEAD_DIM).clamp_(-2,2).to(torch.float8_e4m3fn).cuda()
    k = torch.randint(-128,128,(np,PAGE_SIZE,1,HEAD_DIM+4),dtype=torch.int8).cuda()
    w = torch.randn(B, NUM_HEADS, dtype=torch.float32).cuda()
    sl = st.load_file(str(TRACE_ROOT/inp['seq_lens']['path']))[inp['seq_lens']['tensor_key']].cuda()
    bt = st.load_file(str(TRACE_ROOT/inp['block_table']['path']))[inp['block_table']['tensor_key']].cuda()
    return q, k, w, sl, bt, B, axes

print("=== Correctness check (all 128 workloads) ===")
n_pass = 0; n_batched = 0; n_seq = 0
for i, wl in enumerate(workloads):
    q, k, w, sl, bt, B, axes = make(wl, seed=i)
    ref = torch.full((B,TOPK),-1,dtype=torch.int32,device='cuda')
    got = torch.full((B,TOPK),-1,dtype=torch.int32,device='cuda')
    run_baseline(q,k,w,sl,bt,ref)
    run_s4(q,k,w,sl,bt,got)
    torch.cuda.synchronize()
    err = (ref.float()-got.float()).abs().max().item()
    if err == 0.0:
        n_pass += 1
        if axes['max_num_pages'] >= 3: n_batched += 1
        else: n_seq += 1
    else:
        print(f"  FAIL wl={i} axes={axes} abs_err={err:.1f}")
print(f"Passed: {n_pass}/128  (batched path: {n_batched}, seq fallback: {n_seq})")

print("\n=== Performance by workload size ===")
by_size = sorted(workloads, key=lambda w: w['workload']['axes']['batch_size'])
groups = [(b, [w for w in workloads if w['workload']['axes']['batch_size']==b])
          for b in sorted(set(w['workload']['axes']['batch_size'] for w in workloads))]

for b_size, wls in groups:
    wl = wls[0]
    axes = wl['workload']['axes']
    if axes['max_num_pages'] < 1: continue
    q, k, w, sl, bt, B, axes = make(wl)
    ref_out = torch.full((B,TOPK),-1,dtype=torch.int32,device='cuda')
    s4_out  = torch.full((B,TOPK),-1,dtype=torch.int32,device='cuda')
    # warmup
    for _ in range(5):
        run_baseline(q,k,w,sl,bt,ref_out)
        run_s4(q,k,w,sl,bt,s4_out)
    torch.cuda.synchronize()
    N=100
    t0=time.perf_counter()
    for _ in range(N): run_baseline(q,k,w,sl,bt,ref_out)
    torch.cuda.synchronize(); tb=(time.perf_counter()-t0)/N*1000
    t0=time.perf_counter()
    for _ in range(N): run_s4(q,k,w,sl,bt,s4_out)
    torch.cuda.synchronize(); ts=(time.perf_counter()-t0)/N*1000
    path = "BATCHED" if axes['max_num_pages'] >= 3 else "SEQ"
    print(f"  B={B:2d} max_pages={axes['max_num_pages']:3d} [{path}]: "
          f"baseline={tb:.3f}ms  s4={ts:.3f}ms  speedup={tb/ts:.2f}x")
