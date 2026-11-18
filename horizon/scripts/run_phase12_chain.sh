#!/usr/bin/env bash
# Phase 1 mixed-cause grid, then Phase 2 counterfactual sweep + value model.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[$(date +%H:%M:%S)] mixed grid start"
PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
    --config configs/minimal_validation_mixed_grid.yaml \
    --out results/mixed_grid_20261118 --workers 12

echo "[$(date +%H:%M:%S)] operation sweep start"
PYTHONPATH=src .venv-linux/bin/python scripts/run_operation_sweep.py \
    --config configs/minimal_validation_4causes.yaml \
    --results results/four_causes_20261118_rule \
    --out results/operation_sweep_20261118 --banks 2 --workers 12

echo "[$(date +%H:%M:%S)] value model"
.venv-linux/bin/python scripts/train_value_model.py \
    --sweep results/operation_sweep_20261118/sweep.jsonl \
    --out results/operation_sweep_20261118/value_model.json

echo "[$(date +%H:%M:%S)] DONE"
