#!/bin/bash
# Paper queue (only experiments that appear in the paper).
#   1. experiment-1 Table 1 at 3 banks  (advisor Table 1)
#   2. conflict/order 2-op sequences    (new core: operations are not commutative)
# Each batch waits for the previous completion.json so only one batch owns the GPU.
set -euo pipefail
cd /home/aurora/real2sim/horizon

run () {  # $1 = config, $2 = out dir, $3 = label
  echo "[$(date -Is)] start $3"
  PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
    --config "$1" --out "$2" --workers 4
  echo "[$(date -Is)] done $3"
}

# 1) Table 1 at 3 banks
run configs/experiment1_table1_3banks.yaml results/experiment1_table1_3banks_20261123 "table1-3banks"

# 2) conflict / order (needs the seq: selector, registered only now that no worker is importing)
python3 scripts/patch_seq_selector.py
PYTHONPATH=src .venv-linux/bin/python - <<'PY'
from horizon.harness.selector_experiment import make_selector
s = make_selector("seq:calibrate_geometry+regenerate_targeted", "geometry_bias")
print("seq smoke ok:", s.name, s.actions)
PY
run configs/experiment_conflict_order.yaml results/experiment_conflict_order_20261123 "conflict-order"

echo "[$(date -Is)] paper queue finished"
