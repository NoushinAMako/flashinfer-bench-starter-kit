"""
Test solution1 against reference using the actual safetensors workload data.
"""
import torch
import sys
import json
from pathlib import Path
sys.path.insert(0, "/home/noushin/flashinfer/index/flashinfer-bench-starter-kit")

TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")
WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM = 128
TOPK = 2048

@torch.no_grad()
def run_reference(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    batch_size, num_index_heads, index_head_dim = q_index_fp8.shape
    topk = TOPK
    device = q_index_fp8.device
    q = q_index_fp8.to(torch.float32)
    raw = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = raw.shape
    head_dim = head_dim_sf - 4
    flat = raw.view(num_pages, page_size * head_dim_sf)
    fp8_vals = flat[:, :page_size * head_dim].contiguous().view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn)
    fp8_f32 = fp8_vals.to(torch.float32)
    scales = flat[:, page_size * head_dim:].contiguous().view(num_pages, page_size, 4).view(torch.float32)
    K_all = fp8_f32 * scales
    topk_indices = torch.full((batch_size, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            continue
        num_pages_for_seq = (seq_len + page_size - 1) // page_size
        page_indices = block_table[b, :num_pages_for_seq].to(torch.long)
        K = K_all[page_indices].reshape(-1, index_head_dim)[:seq_len]
        scores = q[b] @ K.T
        final_scores = (torch.relu(scores) * weights[b, :, None]).sum(dim=0)
        actual_topk = min(topk, seq_len)
        _, topk_idx = torch.topk(final_scores, actual_topk)
        phys_page = page_indices[topk_idx // page_size]
        topk_indices[b, :actual_topk] = (phys_page * page_size + topk_idx % page_size).to(torch.int32)
    return topk_indices

def run_solution1(q, k_cache, weights, seq_lens, block_table, topk_indices):
    from solution1.triton.kernel import run as sol1_run
    sol1_run(q, k_cache, weights, seq_lens, block_table, topk_indices)

def load_workload(wl_info):
    """Load inputs for a workload, using fixed safetensors for seq_lens/block_table."""
    import safetensors.torch as st
    axes = wl_info['workload']['axes']
    B = axes['batch_size']
    max_num_pages = axes['max_num_pages']
    num_pages = axes['num_pages']
    inputs = wl_info['workload']['inputs']

    torch.manual_seed(0)
    q = torch.randint(-127, 127, (B, NUM_HEADS, HEAD_DIM), dtype=torch.int8).cuda().view(torch.float8_e4m3fn)
    k_cache = torch.randint(-127, 127, (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4), dtype=torch.int8).cuda()
    weights = torch.rand(B, NUM_HEADS, dtype=torch.float32).cuda()

    # Load fixed tensors from safetensors
    sl_path = TRACE_ROOT / inputs['seq_lens']['path']
    bt_path = TRACE_ROOT / inputs['block_table']['path']
    sl_key = inputs['seq_lens']['tensor_key']
    bt_key = inputs['block_table']['tensor_key']
    tensors = st.load_file(str(sl_path))
    seq_lens = tensors[sl_key].cuda()
    bt_tensors = st.load_file(str(bt_path))
    block_table = bt_tensors[bt_key].cuda()

    return q, k_cache, weights, seq_lens, block_table

workloads = [json.loads(l) for l in open(WL_FILE)]

# Test first 5 workloads
for i, wl in enumerate(workloads[:10]):
    axes = wl['workload']['axes']
    uuid = wl['workload']['uuid'][:8]
    B = axes['batch_size']
    max_pages = axes['max_num_pages']

    q, k, w, sl, bt = load_workload(wl)
    ref = run_reference(q, k, w, sl, bt)
    got = torch.full((B, TOPK), -1, dtype=torch.int32, device='cuda')
    run_solution1(q, k, w, sl, bt, got)
    torch.cuda.synchronize()

    abs_err = (ref.float() - got.float()).abs().max().item()
    seq_lens_cpu = sl.cpu().tolist()
    status = "PASS" if abs_err == 0.0 else "FAIL"
    print(f"[{status}] uuid={uuid} B={B} max_pages={max_pages} seq_lens={seq_lens_cpu} abs_err={abs_err:.1f}")

print("\nDone.")
