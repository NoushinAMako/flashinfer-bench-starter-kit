"""
Correctness + performance comparison: solution5 (multi-stream 32) vs solution (baseline kernel9).
"""
import torch, sys, json, time
from pathlib import Path

PAGE_SIZE = 64; NUM_HEADS = 64; HEAD_DIM = 128; TOPK = 2048
TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")
WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = [json.loads(l) for l in open(WL_FILE)]

import safetensors.torch as st
from solution.triton.kernel  import run as run_baseline
from solution5.triton.kernel import run as run_s5

def make(wl, seed=42):
    axes = wl['workload']['axes']; inp = wl['workload']['inputs']
    B = axes['batch_size']; np_ = axes['num_pages']
    torch.manual_seed(seed)
    q  = torch.randn(B, NUM_HEADS, HEAD_DIM).clamp_(-2, 2).to(torch.float8_e4m3fn).cuda()
    k  = torch.randint(-128, 128, (np_, PAGE_SIZE, 1, HEAD_DIM + 4), dtype=torch.int8).cuda()
    w  = torch.randn(B, NUM_HEADS, dtype=torch.float32).cuda()
    sl = st.load_file(str(TRACE_ROOT/inp['seq_lens']['path']))[inp['seq_lens']['tensor_key']].cuda()
    bt = st.load_file(str(TRACE_ROOT/inp['block_table']['path']))[inp['block_table']['tensor_key']].cuda()
    return q, k, w, sl, bt, B, axes

print("Warming up (JIT compilation)...")
q, k, w, sl, bt, B, axes = make(workloads[0])
ref_out = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
s5_out  = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
run_baseline(q, k, w, sl, bt, ref_out)
run_s5(q, k, w, sl, bt, s5_out)
torch.cuda.synchronize()
print("Done.\n")

print("=== Correctness check (all 128 workloads) ===")
n_pass = 0; n_fail = 0
for i, wl in enumerate(workloads):
    q, k, w, sl, bt, B, axes = make(wl, seed=i)
    ref_out = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
    s5_out  = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
    run_baseline(q, k, w, sl, bt, ref_out)
    run_s5(q, k, w, sl, bt, s5_out)
    torch.cuda.synchronize()
    if torch.equal(ref_out, s5_out):
        n_pass += 1
    else:
        n_fail += 1
        print(f"  FAIL wl={i:3d} B={B} max_pages={axes['max_num_pages']:3d}")

print(f"\nCorrectness: {n_pass}/128 PASS, {n_fail}/128 FAIL")

print("\n=== Performance (representative workloads) ===")
N = 200
for i in [0, 64, 92, 110, 127]:
    wl = workloads[i]
    q, k, w, sl, bt, B, axes = make(wl)
    ref_out = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
    s5_out  = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
    for _ in range(10):
        run_baseline(q, k, w, sl, bt, ref_out)
        run_s5(q, k, w, sl, bt, s5_out)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N): run_baseline(q, k, w, sl, bt, ref_out)
    torch.cuda.synchronize()
    tb = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N): run_s5(q, k, w, sl, bt, s5_out)
    torch.cuda.synchronize()
    ts = (time.perf_counter() - t0) / N * 1000

    sls = sl.cpu().tolist()
    print(f"  WL{i:3d} B={B:2d} max_pages={axes['max_num_pages']:3d}: "
          f"baseline={tb:.3f}ms  s5={ts:.3f}ms  speedup={tb/ts:.2f}x "
          f"sl_min={min(sls)} sl_max={max(sls)}")
