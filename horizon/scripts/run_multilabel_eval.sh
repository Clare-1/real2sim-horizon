#!/usr/bin/env bash
# Phase 2 learned-selector evaluation: multi-label diagnosis -> operation sequence.
# Waits for the LinUCB/value-model evaluation to finish so the two do not compete for workers.
set -euo pipefail
cd "$(dirname "$0")/.."

DEP=results/four_causes_learned_20261118/completion.json
OUT=results/four_causes_multilabel_20261118

echo "[$(date +%H:%M:%S)] waiting for $DEP (max 600m)"
for _ in $(seq 1 600); do
    [ -f "$DEP" ] && break
    sleep 60
done
[ -f "$DEP" ] || { echo "timeout waiting for the learned evaluation"; exit 1; }

mkdir -p "$OUT/shared" "$OUT/tasks_train" "$OUT/tasks_deploy"
for d in shared tasks_train tasks_deploy; do
    cp -al "results/four_causes_20261118_rule/$d/." "$OUT/$d/" 2>/dev/null || true
done

echo "[$(date +%H:%M:%S)] multilabel selector start"
PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
    --config configs/minimal_validation_multilabel.yaml --out "$OUT" --workers 12

echo "[$(date +%H:%M:%S)] report"
.venv-linux/bin/python scripts/report_four_causes.py "$OUT"
echo "[$(date +%H:%M:%S)] DONE"
