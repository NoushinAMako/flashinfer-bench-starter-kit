"""
Test: does cublasSgemmStridedBatched (via torch.bmm on same-shape matrices)
give identical results to individual torch.mm calls?

Key hypothesis: strided batched GEMM is defined as B INDEPENDENT sub-problems
with no cross-batch interaction, so it should match individual GEMMs exactly
when M, K, N are the same for all batch elements.
"""
import torch
import time
torch.manual_seed(0)

def test_strided_vs_sequential(M, K, N, batch_sizes):
    print(f"\nM={M}, K={N_K}, N={N}:")
    for B in batch_sizes:
        A = torch.randn(B, M, K, device='cuda', dtype=torch.float32)
        X = torch.randn(B, N, K, device='cuda', dtype=torch.float32)  # each row is K-vector

        # Batched: A [B, M, K] × X.transpose(1,2) [B, K, N] = [B, M, N]
        out_batch = torch.bmm(A, X.transpose(1, 2))

        # Sequential: A[b] [M, K] × X[b].t() [K, N] = [M, N]
        out_seq = torch.stack([torch.mm(A[b], X[b].t()) for b in range(B)])
        torch.cuda.synchronize()

        match = torch.equal(out_batch, out_seq)
        if not match:
            diff = (out_batch - out_seq).abs()
            n_diff = (diff > 0).sum().item()
            print(f"  B={B:3d}: MISMATCH  max_diff={diff.max().item():.2e}  n_diff={n_diff}/{diff.numel()}")
        else:
            print(f"  B={B:3d}: MATCH")

# Test matrix dimensions that match our problem
# M=NUM_HEADS=64, K=HEAD_DIM=128, N=PAGE_SIZE*np_seq (multiples of 64)
N_K = 128
for M in [64]:
    for N in [64, 128, 192, 256, 512, 1024, 2048]:
        test_strided_vs_sequential(M, N_K, N, [1, 4, 8, 16, 32])

# Also test specifically with values that look like real K data
print("\n\n=== Test with decoded FP8-like K data (typical value range) ===")
M, K_size, N = 64, 128, 128
B = 16
# Simulate decoded FP8 K: small values (FP8 E4M3FN range ~[-448, 448] but usually small)
A = torch.randn(B, M, K_size, device='cuda').clamp_(-2, 2)
X = torch.randn(B, N, K_size, device='cuda').clamp_(-10, 10)

out_batch = torch.bmm(A, X.transpose(1, 2))
out_seq = torch.stack([torch.mm(A[b], X[b].t()) for b in range(B)])
torch.cuda.synchronize()

match = torch.equal(out_batch, out_seq)
print(f"FP8-like values, B={B}: {'MATCH' if match else 'MISMATCH'}")
if not match:
    diff = (out_batch - out_seq).abs()
    print(f"  max_diff={diff.max().item():.2e}  n_diff={(diff>0).sum().item()}")

# Performance comparison
print("\n=== Performance: bmm vs sequential mm ===")
for B in [8, 16, 31]:
    A = torch.randn(B, 64, 128, device='cuda')
    X = torch.randn(B, 128, 128, device='cuda')
    for _ in range(10):
        torch.bmm(A, X.transpose(1,2))
        for b in range(B): torch.mm(A[b], X[b].t())
    torch.cuda.synchronize()

    N_ITER = 100
    t0 = time.perf_counter()
    for _ in range(N_ITER): torch.bmm(A, X.transpose(1,2))
    torch.cuda.synchronize()
    t_bmm = (time.perf_counter()-t0)/N_ITER*1000

    t0 = time.perf_counter()
    for _ in range(N_ITER):
        for b in range(B): torch.mm(A[b], X[b].t())
    torch.cuda.synchronize()
    t_seq = (time.perf_counter()-t0)/N_ITER*1000

    print(f"  B={B:2d}: bmm={t_bmm:.3f}ms  seq_mm={t_seq:.3f}ms  speedup={t_seq/t_bmm:.2f}x")
