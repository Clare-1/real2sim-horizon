#!/bin/bash
# Experiment-1 Table 1 at 3 banks (72 runs), queued behind the Table-9 bias sweep so only one
# batch touches the GPU at a time.
set -euo pipefail
cd /home/aurora/real2sim/horizon
DONE=results/experiment5_bias_levels_20261123/completion.json
while [ ! -f "$DONE" ]; do sleep 60; done
echo "table-9 done at $(date -Is); starting experiment-1 at 3 banks"
PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
  --config configs/experiment1_table1_3banks.yaml \
  --out results/experiment1_table1_3banks_20261123 --workers 4
echo "experiment-1 3 banks done at $(date -Is)"
