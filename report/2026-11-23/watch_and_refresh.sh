#!/bin/bash
# Wait for the three batches to write completion.json, then refresh the folder.
# Detached: nohup bash watch_and_refresh.sh > /tmp/watch_refresh.log 2>&1 &
set -uo pipefail
OUT="$(cd "$(dirname "$0")" && pwd)"
H="$OUT/../horizon"

pending () {  # prints how many of the three batches are still incomplete
  local n=0
  for d in experiment_conflict_order_20261123 experiment_a1_perturbed_20261123 paper_llm_rows_20261123; do
    if ! python3 -c "import json,sys;d=json.load(open('$H/results/$d/completion.json'));sys.exit(0 if d.get('complete') else 1)" 2>/dev/null; then
      n=$((n+1))
    fi
  done
  echo "$n"
}

echo "[$(date -Is)] watcher start"
for i in $(seq 1 2880); do   # up to 48 h
  left=$(pending)
  if [ "$left" -eq 0 ]; then
    echo "[$(date -Is)] all batches complete, refreshing"
    bash "$OUT/refresh.sh"
    echo "[$(date -Is)] watcher done"
    exit 0
  fi
  sleep 60
done
echo "[$(date -Is)] watcher timeout with $left batch(es) pending; refreshing anyway"
bash "$OUT/refresh.sh"