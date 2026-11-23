#!/bin/bash
# Table 9 (bias-response curve) queued behind the Fig. 7 budget sweep so the GPU is not shared by
# three batches. Waits for fb48/completion.json, then runs 3 bias levels x 2 banks x 3 selectors.
set -euo pipefail
cd /home/aurora/real2sim/horizon
DONE=results/budget_sweep_20261123/fb48/completion.json
while [ ! -f "$DONE" ]; do sleep 60; done
echo "sweep done at $(date -Is); starting table-9 bias levels"
PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
  --config configs/experiment5_bias_levels.yaml \
  --out results/experiment5_bias_levels_20261123 --workers 4
echo "table-9 done at $(date -Is)"
