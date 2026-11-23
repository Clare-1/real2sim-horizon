#!/bin/bash
# Conflict/order experiment (32 runs) queued behind the experiment-1 3-bank batch.
# Registers the `seq:` selector only after the GPU queue is free, so no running worker can import
# a half-written module.
set -euo pipefail
cd /home/aurora/real2sim/horizon
DONE=results/experiment1_table1_3banks_20261123/completion.json
while [ ! -f "$DONE" ]; do sleep 60; done
echo "experiment-1 3 banks done at $(date -Is); registering seq: selector"
python3 /tmp/patch_seq_register.py
PYTHONPATH=src .venv-linux/bin/python - <<'PY'
from horizon.harness.selector_experiment import make_selector
s = make_selector("seq:calibrate_geometry+regenerate_targeted", "geometry_bias")
print("smoke ok:", s.name, s.actions)
PY
echo "starting conflict/order batch"
PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
  --config configs/experiment_conflict_order.yaml \
  --out results/experiment_conflict_order_20261123 --workers 4
echo "conflict/order done at $(date -Is)"
