#!/usr/bin/env bash
# Wait for the 48-run four-cause batch, then re-run the stale rule column with the CURRENT code,
# merge it back, and regenerate every table/figure. See 重跑风险审计_2026-11-18.md section 10.
#
#   nohup bash scripts/finish_four_causes.sh > /tmp/finish_four.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

MAIN=results/four_causes_20261118
FIX=results/four_causes_20261118_rule
PY=.venv-linux/bin/python
WAIT_MIN=${WAIT_MIN:-240}

echo "[$(date +%H:%M:%S)] waiting for $MAIN/completion.json (max ${WAIT_MIN}m)"
for _ in $(seq 1 "$WAIT_MIN"); do
    [ -f "$MAIN/completion.json" ] && break
    sleep 60
done
[ -f "$MAIN/completion.json" ] || { echo "timeout waiting for completion.json"; exit 1; }

mkdir -p "$FIX/shared" "$FIX/tasks_train" "$FIX/tasks_deploy"
for d in shared tasks_train tasks_deploy; do
    cp -al "$MAIN/$d/." "$FIX/$d/" 2>/dev/null || true
done

echo "[$(date +%H:%M:%S)] rulefix start (8 runs; initial data are re-collected because the batch
never saved them)"
PYTHONPATH=src "$PY" scripts/run_minimal_validation.py \
    --config configs/minimal_validation_4causes_rulefix.yaml --out "$FIX" --workers 12

echo "[$(date +%H:%M:%S)] merge rulefix into $MAIN"
"$PY" scripts/merge_rule_records.py --main "$MAIN" --fix "$FIX" --selectors rule

echo "[$(date +%H:%M:%S)] regenerate reports"
"$PY" scripts/report_four_causes.py
"$PY" scripts/make_fig3_action.py --per-selector
"$PY" scripts/make_fig4_trajectory.py
"$PY" scripts/make_all_tables.py
echo "[$(date +%H:%M:%S)] DONE"
