"""
Quick debug script to find where solution1 (hybrid_v3) diverges from the reference.
"""
import torch
import sys
sys.path.insert(0, "/home/noushin/flashinfer/index/flashinfer-bench-starter-kit")

PAGE_SIZE = 64
NUM_HEADS = 64
HEAD_DIM = 128
TOPK = 2048
torch.manual_seed(42)

def make_inputs(batch_size, seq_lens_list, num_pages=200):
    max_num_pages = max((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens_list)
    q = torch.randint(-127, 127, (batch_size, NUM_HEADS, HEAD_DIM), dtype=torch.int8).cuda().view(torch.float8_e4m3fn)
    k_cache = torch.randint(-127, 127, (num_pages, PAGE_SIZE, 1, HEAD_DIM + 4), dtype=torch.int8).cuda()
    weights = torch.rand(batch_size, NUM_HEADS, dtype=torch.float32).cuda()
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32).cuda()
    block_table = torch.randint(0, num_pages, (batch_size, max_num_pages), dtype=torch.int32).cuda()
    return q, k_cache, weights, seq_lens, block_table

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

def compare(ref, got, label):
    ref_valid = ref[ref != -1]
    got_valid = got[got != -1]
    ref_s = torch.sort(ref_valid)[0]
    got_s = torch.sort(got_valid)[0]
    match = torch.equal(ref_s, got_s)
    abs_err = (ref.float() - got.float()).abs().max().item()
    print(f"  {label}: match={match}, max_abs_err={abs_err:.1f}, ref_count={ref_s.numel()}, got_count={got_s.numel()}")
    if not match and ref_s.numel() > 0 and got_s.numel() > 0:
        ref_set = set(ref_s.cpu().tolist())
        got_set = set(got_s.cpu().tolist())
        common = len(ref_set & got_set)
        print(f"    Common indices: {common}/{len(ref_set)} ({100*common/max(1,len(ref_set)):.1f}%)")

def run_test(label, batch_size, seq_lens_list, num_pages=200):
    print(f"\n{label}")
    q, k, w, sl, bt = make_inputs(batch_size, seq_lens_list, num_pages)
    ref = run_reference(q, k, w, sl, bt)
    got = torch.full((batch_size, TOPK), -1, dtype=torch.int32, device='cuda')
    run_solution1(q, k, w, sl, bt, got)
    torch.cuda.synchronize()
    for b, s in enumerate(seq_lens_list):
        compare(ref[b], got[b], f"b={b} sl={s}")

run_test("Test 1: batch=1, sl=50 (single partial page)", 1, [50])
run_test("Test 2: batch=1, sl=64 (exactly 1 full page)", 1, [64])
run_test("Test 3: batch=1, sl=65 (2 pages, last partial)", 1, [65])
run_test("Test 4: batch=1, sl=128 (exactly 2 full pages)", 1, [128])
run_test("Test 5: batch=1, sl=130 (3 pages, last partial)", 1, [130])
run_test("Test 6: batch=2, sl=[50, 64]", 2, [50, 64])
run_test("Test 7: batch=2, sl=[64, 65]", 2, [64, 65])
run_test("Test 8: batch=2, sl=[128, 128]", 2, [128, 128])
run_test("Test 9: batch=3, sl=[100, 200, 300]", 3, [100, 200, 300])
run_test("Test 10: batch=1, sl=500 (many pages)", 1, [500])

print("\nDone.")
