#!/usr/bin/env bash
# Minimal validation per the advisor's spec, stage 1: 2 causes x 10 banks x 1 seed x 4 selectors
# = 80 runs. fixed / rule / bandit / oracle; no offline-trained selector, so no model leakage.
# Reuses the seed-0 geometry/coverage init data from the four-cause run via hard links.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/minimal_validation_advisor_20261121
mkdir -p "$OUT/shared" "$OUT/tasks_train" "$OUT/tasks_deploy"
for d in shared tasks_train tasks_deploy; do
    cp -al "results/four_causes_20261118_rule/$d/." "$OUT/$d/" 2>/dev/null || true
done

echo "[$(date +%H:%M:%S)] minimal validation: 2 x 10 banks x 3 seeds x 4 selectors (240 cells; seed 0 records are reused)"
PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
    --config configs/minimal_validation_advisor.yaml --out "$OUT" --workers 12

echo "[$(date +%H:%M:%S)] report"
.venv-linux/bin/python scripts/report_paper8.py "$OUT" || true
echo "[$(date +%H:%M:%S)] DONE"
