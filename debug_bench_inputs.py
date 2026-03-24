"""
Test with inputs generated exactly like the benchmark framework does.
"""
import torch
import sys
import json
from pathlib import Path
sys.path.insert(0, "/home/noushin/flashinfer/index/flashinfer-bench-starter-kit")

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM = 128
TOPK = 2048
NUM_PAGES = 11923
TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")

def gen_like_benchmark(B, max_num_pages, seq_lens_tensor, block_table_tensor):
    """Generate inputs exactly like the benchmark framework does."""
    # q: float8_e4m3fn - generated as randn clamped then converted
    q_f32 = torch.randn(B, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device='cuda').clamp_(-2.0, 2.0)
    q = q_f32.to(torch.float8_e4m3fn)

    # k_cache: int8 - generated as randint(-128, 128)  
    k_cache = torch.randint(-128, 128, (NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM + 4), dtype=torch.int8, device='cuda')

    # weights: float32 - generated as randn
    weights = torch.randn(B, NUM_HEADS, dtype=torch.float32, device='cuda')

    return q, k_cache, weights, seq_lens_tensor, block_table_tensor

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

import safetensors.torch as st

WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = [json.loads(l) for l in open(WL_FILE)]

print("Testing with benchmark-style inputs (randn q, randn weights):\n")

for i, wl in enumerate(workloads[:15]):
    axes = wl['workload']['axes']
    uuid = wl['workload']['uuid'][:8]
    B = axes['batch_size']
    max_pages = axes['max_num_pages']
    inputs_spec = wl['workload']['inputs']

    # Load fixed tensors
    sl_path = TRACE_ROOT / inputs_spec['seq_lens']['path']
    tensors = st.load_file(str(sl_path))
    seq_lens = tensors[inputs_spec['seq_lens']['tensor_key']].cuda()
    block_table = tensors[inputs_spec['block_table']['tensor_key']].cuda()

    for trial in range(3):  # 3 trials like benchmark
        torch.manual_seed(trial * 1000 + i)  # different seed per trial
        q, k, w, sl, bt = gen_like_benchmark(B, max_pages, seq_lens, block_table)

        ref = run_reference(q, k, w, sl, bt)
        got = torch.empty((B, TOPK), dtype=torch.int32, device='cuda')  # empty like benchmark
        run_solution1(q, k, w, sl, bt, got)
        torch.cuda.synchronize()

        abs_err = (ref.float() - got.float()).abs().max().item()
        if abs_err > 0:
            seq_lens_cpu = seq_lens.cpu().tolist()
            print(f"[FAIL] uuid={uuid} B={B} max_pages={max_pages} trial={trial} abs_err={abs_err:.1f}")
            print(f"       seq_lens={seq_lens_cpu}")
            break
    else:
        print(f"[PASS] uuid={uuid} B={B} max_pages={max_pages}")

print("\nDone.")
