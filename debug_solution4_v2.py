"""
Test solution4 V2 against kernel9 reference on all 128 workloads.
Reports pass/fail per workload, including which group each batch item falls into.
"""
import torch, sys, json, time
from pathlib import Path

PAGE_SIZE = 64; NUM_HEADS = 64; HEAD_DIM = 128; TOPK = 2048
TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")
WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = [json.loads(l) for l in open(WL_FILE)]

import safetensors.torch as st
from solution.triton.kernel  import run as run_baseline
from solution4.triton.kernel import run as run_s4

def make(wl, seed=42):
    axes = wl['workload']['axes']
    inp  = wl['workload']['inputs']
    B    = axes['batch_size']
    np_  = axes['num_pages']
    torch.manual_seed(seed)
    q  = torch.randn(B, NUM_HEADS, HEAD_DIM).clamp_(-2, 2).to(torch.float8_e4m3fn).cuda()
    k  = torch.randint(-128, 128, (np_, PAGE_SIZE, 1, HEAD_DIM + 4), dtype=torch.int8).cuda()
    w  = torch.randn(B, NUM_HEADS, dtype=torch.float32).cuda()
    sl = st.load_file(str(TRACE_ROOT / inp['seq_lens']['path']))[inp['seq_lens']['tensor_key']].cuda()
    bt = st.load_file(str(TRACE_ROOT / inp['block_table']['path']))[inp['block_table']['tensor_key']].cuda()
    return q, k, w, sl, bt, B, axes

print("Warming up (JIT compilation)...")
q, k, w, sl, bt, B, axes = make(workloads[0])
ref_out = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
s4_out  = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
run_baseline(q, k, w, sl, bt, ref_out)
run_s4(q, k, w, sl, bt, s4_out)
torch.cuda.synchronize()
print("Done.\n")

print("=== Correctness check (all 128 workloads) ===")
n_pass = 0; n_fail = 0; n_large_items = 0; n_small_items = 0

for i, wl in enumerate(workloads):
    q, k, w, sl, bt, B, axes = make(wl, seed=i)
    ref_out = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
    s4_out  = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
    run_baseline(q, k, w, sl, bt, ref_out)
    run_s4(q, k, w, sl, bt, s4_out)
    torch.cuda.synchronize()

    sls = sl.cpu().tolist()
    large = [s for s in sls if s >= 192]
    small = [s for s in sls if s <  192]

    err = (ref_out.float() - s4_out.float()).abs().max().item()
    if err == 0.0:
        n_pass += 1
        n_large_items += len(large)
        n_small_items += len(small)
    else:
        n_fail += 1
        # Find first mismatch
        for b in range(B):
            ak = min(sls[b], TOPK)
            r = ref_out[b, :ak].sort()[0]
            s = s4_out[b,  :ak].sort()[0]
            if not torch.equal(r, s):
                print(f"  FAIL wl={i:3d} B={B} max_pages={axes['max_num_pages']:3d} "
                      f"large={len(large)} small={len(small)} "
                      f"first_bad_b={b} sl={sls[b]}")
                break

print(f"\nResults: {n_pass}/128 PASS, {n_fail}/128 FAIL")
print(f"Passed workloads had: {n_large_items} large-path items, {n_small_items} small-path items")
