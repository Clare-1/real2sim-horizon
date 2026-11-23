#!/bin/bash
# LLM queue: only the LLM experiments that can appear in the paper.
#   1. mixed causes, LLM vs the published rule        (24 cells, ~24 LLM calls)
#   2. mixed causes, rule with tau_labels_low fixed   ( 6 cells, control for #1)
#   3. paper-grade LLM rows: 4 causes x 3 banks x 3 seeds x {llm, llm_memory}  (72 cells)
# Waits for scripts/run_paper_queue.sh so only one batch owns the GPU.
set -euo pipefail
cd /home/aurora/real2sim/horizon

while pgrep -f 'scripts/run_paper_queue.sh' > /dev/null; do sleep 60; done
echo "[$(date -Is)] paper queue finished, starting LLM queue"

set -a
# shellcheck disable=SC1090
. "$HOME/.config/horizon/llm.env"
set +a

run () {  # $1 = config, $2 = out dir, $3 = label
  echo "[$(date -Is)] start $3"
  PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
    --config "$1" --out "$2" --workers 4
  echo "[$(date -Is)] done $3"
}

run configs/experiment_llm_mixed.yaml results/experiment_llm_mixed_20261123 "llm-mixed"
run configs/experiment_llm_mixed_rulefix.yaml results/experiment_llm_mixed_rulefix_20261123 "rule-fixed-labels"
run configs/paper_llm_rows.yaml results/paper_llm_rows_20261123 "paper-llm-rows"

echo "[$(date -Is)] reports"
for d in results/experiment_llm_mixed_20261123 results/experiment_llm_mixed_rulefix_20261123 \
         results/paper_llm_rows_20261123; do
  PYTHONPATH=src .venv-linux/bin/python scripts/report_paper8.py "$d" > "$d/paper8_report.md" || true
done
PYTHONPATH=src .venv-linux/bin/python scripts/report_experiment5.py > /tmp/exp5_report.log 2>&1 || true
echo "[$(date -Is)] llm queue done"
