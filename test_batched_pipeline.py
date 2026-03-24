"""
Full pipeline test: does batched (bmm + relu_weight_mul + sum(1))
match sequential (mm + relu_weight_mul + sum(0)) numerically?

Strategy:
  - Group batch items by np_seq (same page count → same N = np_seq*64)
  - For N >= 192 (np_seq >= 3): use bmm (proven to match for GEMM)
  - For N < 192: use sequential mm (safe fallback)
"""
import torch
import time
torch.manual_seed(0)

PAGE_SIZE = 64; NUM_HEADS = 64; HEAD_DIM = 128

def test_pipeline(B, N, label=""):
    """Test: batched bmm+sum(1) vs sequential mm+sum(0)."""
    # Inputs
    q = torch.randn(B, NUM_HEADS, HEAD_DIM, device='cuda')   # already float32
    K = torch.randn(B, N, HEAD_DIM, device='cuda')            # decoded K for all items
    w = torch.randn(B, NUM_HEADS, device='cuda')              # weights

    # Sequential reference (same as solution kernel9)
    final_seq = []
    for b in range(B):
        s = torch.mm(q[b], K[b].t())   # [64, N]
        # relu_weight_mul: scores[h,t] = relu(s[h,t]) * w[b,h]
        mask = s > 0
        s = s * mask * w[b].unsqueeze(1)  # [64, N] * [64, 1] element-wise
        final_seq.append(s.sum(0))        # [N]
    torch.cuda.synchronize()

    # Batched approach
    scores_batch = torch.bmm(q, K.transpose(1, 2))  # [B, 64, N]
    # relu_weight_mul in-place on batched
    mask_b = scores_batch > 0
    scores_batch = scores_batch * mask_b * w.unsqueeze(2)   # [B, 64, N] * [B, 64, 1]
    final_batch = scores_batch.sum(1)  # [B, N]
    torch.cuda.synchronize()

    # Compare
    all_ok = True
    for b in range(B):
        if not torch.equal(final_seq[b], final_batch[b]):
            diff = (final_seq[b] - final_batch[b]).abs()
            n_mismatch = (diff > 0).sum().item()
            print(f"  {label} B={B} N={N} b={b}: MISMATCH max_diff={diff.max():.2e} n={n_mismatch}/{N}")
            all_ok = False
            break
    if all_ok:
        print(f"  {label} B={B} N={N}: ALL MATCH")

print("=== Pipeline correctness: batched vs sequential ===")
for N in [64, 128, 192, 256, 512]:
    for B in [1, 4, 16, 32]:
        test_pipeline(B, N, f"N={N:4d}")

# ---- Use the SAME operations as solution (relu_weight_mul kernel + sum(0)) ----
print("\n=== With exact solution kernel ops (relu in-place mul + .sum(0)/.sum(1)) ===")
def test_exact_ops(B, N):
    q = torch.randn(B, NUM_HEADS, HEAD_DIM, device='cuda')
    K = torch.randn(B, N, HEAD_DIM, device='cuda')
    w = torch.randn(B, NUM_HEADS, device='cuda')

    # Sequential: exact same ops as solution kernel9
    final_seq = []
    for b in range(B):
        s = torch.mm(q[b], K[b].t())    # [64, N]
        # relu_weight_mul: in-place, exact same as C++ kernel
        s.clamp_(min=0)
        s.mul_(w[b].unsqueeze(1))
        final_seq.append(s.sum(0))
    torch.cuda.synchronize()

    # Batched: bmm + in-place relu*weight + sum(1)
    scores = torch.bmm(q, K.transpose(1,2))  # [B, 64, N]
    scores.clamp_(min=0)
    scores.mul_(w.unsqueeze(2))
    final_batch = scores.sum(1)  # [B, N]
    torch.cuda.synchronize()

    match = all(torch.equal(final_seq[b], final_batch[b]) for b in range(B))
    if not match:
        for b in range(B):
            if not torch.equal(final_seq[b], final_batch[b]):
                diff = (final_seq[b] - final_batch[b]).abs()
                print(f"  B={B} N={N:4d} b={b}: MISMATCH max_diff={diff.max():.2e}")
                break
    else:
        print(f"  B={B:2d} N={N:4d}: MATCH")

for N in [64, 128, 192, 256, 512, 1024, 2048]:
    for B in [1, 8, 31]:
        test_exact_ops(B, N)

# ---- Performance ----
print("\n=== Performance: batched vs sequential ===")
B = 31
for N in [192, 256, 512, 1024]:
    q = torch.randn(B, NUM_HEADS, HEAD_DIM, device='cuda')
    K = torch.randn(B, N, HEAD_DIM, device='cuda')
    w = torch.randn(B, NUM_HEADS, device='cuda')

    for _ in range(5):
        torch.bmm(q, K.transpose(1,2)).clamp_(min=0).mul_(w.unsqueeze(2)).sum(1)
        for b in range(B): torch.mm(q[b], K[b].t()).clamp_(min=0).mul_(w[b].unsqueeze(1)).sum(0)
    torch.cuda.synchronize()

    N_ITER = 100
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        scores = torch.bmm(q, K.transpose(1,2))
        scores.clamp_(min=0).mul_(w.unsqueeze(2))
        final = scores.sum(1)
    torch.cuda.synchronize()
    t_batched = (time.perf_counter()-t0)/N_ITER*1000

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        for b in range(B):
            s = torch.mm(q[b], K[b].t())
            s.clamp_(min=0).mul_(w[b].unsqueeze(1))
            s.sum(0)
    torch.cuda.synchronize()
    t_seq = (time.perf_counter()-t0)/N_ITER*1000

    print(f"  B={B} N={N:4d}: batched={t_batched:.3f}ms seq={t_seq:.3f}ms speedup={t_seq/t_batched:.1f}x")
