#!/usr/bin/env bash
# Phase 2 online evaluation: LinUCB + learned operation-value selector, after the sweep produced
# the value model. Reuses the four-cause initial data via hard links.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=results/operation_sweep_20261118/value_model.json
OUT=results/four_causes_learned_20261118

echo "[$(date +%H:%M:%S)] waiting for $MODEL (max 480m)"
for _ in $(seq 1 480); do
    [ -f "$MODEL" ] && break
    sleep 60
done
[ -f "$MODEL" ] || { echo "timeout waiting for the value model"; exit 1; }

mkdir -p "$OUT/shared" "$OUT/tasks_train" "$OUT/tasks_deploy"
for d in shared tasks_train tasks_deploy; do
    cp -al "results/four_causes_20261118_rule/$d/." "$OUT/$d/" 2>/dev/null || true
done

echo "[$(date +%H:%M:%S)] learned selectors start"
PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
    --config configs/minimal_validation_learned.yaml --out "$OUT" --workers 12

echo "[$(date +%H:%M:%S)] report"
.venv-linux/bin/python scripts/report_four_causes.py "$OUT"
echo "[$(date +%H:%M:%S)] DONE"
