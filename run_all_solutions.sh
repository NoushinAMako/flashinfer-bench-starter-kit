#!/bin/bash
export FIB_DATASET_PATH=/home/noushin/flashinfer/flashinfer-trace

SOLUTIONS=("solution/triton" "solution1/triton" "solution2/triton" "solution3/triton" "solution4/triton" "solution5/triton" "solution6/triton" "solution6/cuda")

for sol in "${SOLUTIONS[@]}"; do
    echo ""
    echo "======================================================"
    echo ">>> RUNNING: $sol"
    echo "======================================================"
    
    # Update config.toml
    cat > config.toml << TOML
[solution]
name = "dsa-topk-v1"
definition = "dsa_topk_indexer_fp8_h64_d128_topk2048_ps64"
author = "makora/team"

[build]
language = "triton"
source_dir = "$sol"
entry_point = "kernel.py::run"
TOML

    python scripts/run_local.py 2>&1 | tail -20
    echo ">>> DONE: $sol"
done

echo ""
echo "======================================================"
echo "ALL DONE"
echo "======================================================"
