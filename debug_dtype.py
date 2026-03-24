"""Check dtypes of all input tensors."""
import torch, sys, json
from pathlib import Path
sys.path.insert(0, "/home/noushin/flashinfer/index/flashinfer-bench-starter-kit")
import safetensors.torch as st

TRACE_ROOT = Path("/home/noushin/flashinfer/flashinfer-trace")
WL_FILE = TRACE_ROOT / "workloads/dsa_paged/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64.jsonl"
workloads = [json.loads(l) for l in open(WL_FILE)]

wl = workloads[0]
inp = wl['workload']['inputs']
sl_t = st.load_file(str(TRACE_ROOT/inp['seq_lens']['path']))[inp['seq_lens']['tensor_key']]
bt_t = st.load_file(str(TRACE_ROOT/inp['block_table']['path']))[inp['block_table']['tensor_key']]
print(f"seq_lens dtype: {sl_t.dtype}, shape: {sl_t.shape}")
print(f"block_table dtype: {bt_t.dtype}, shape: {bt_t.shape}")
print(f"block_table[0,:5] = {bt_t[0,:5]}")
print(f"seq_lens[:5] = {sl_t[:5]}")
