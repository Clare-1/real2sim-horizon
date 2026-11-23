#!/bin/bash
# Conflict/order stage, run after the LLM queue so only one batch owns the GPU.
#
# The paper queue used to run this stage directly, but it invoked a copy of the patcher from /tmp
# whose `__file__`-based path resolution pointed at `/src/...`; with `set -euo pipefail` that
# killed the whole queue.  The patcher now resolves from the cwd and is idempotent, and this stage
# is independent of the table-1 batch.
set -euo pipefail
cd /home/aurora/real2sim/horizon

while pgrep -f 'scripts/run_llm_queue.sh' > /dev/null; do sleep 60; done
echo "[$(date -Is)] llm queue finished, starting conflict/order"

python3 scripts/patch_seq_selector.py
PYTHONPATH=src .venv-linux/bin/python - <<'PY'
from horizon.harness.selector_experiment import make_selector
s = make_selector("seq:calibrate_geometry+regenerate_targeted", "geometry_bias")
print("seq smoke ok:", s.name, s.actions)
PY

PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
  --config configs/experiment_conflict_order.yaml \
  --out results/experiment_conflict_order_20261123 --workers 4

PYTHONPATH=src .venv-linux/bin/python scripts/report_conflict_order.py \
  results/experiment_conflict_order_20261123
echo "[$(date -Is)] conflict/order done"
