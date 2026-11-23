#!/bin/bash
# Keep the folder current while the batches run: refresh whenever the number of finished cells
# changes, so a group meeting does not have to wait for the slowest batch. Stops once all three
# batches report complete and one final refresh has run.
# Detached: nohup bash watch_and_refresh.sh > /tmp/watch_refresh.log 2>&1 &
set -uo pipefail
OUT="$(cd "$(dirname "$0")" && pwd)"
H="$OUT/../horizon"

counts () {
  for d in experiment_conflict_order_20261123 experiment_a1_perturbed_20261123 paper_llm_rows_20261123; do
    printf '%s:%s ' "${d%%_2026*}" "$(find "$H/results/$d" -name record.json 2>/dev/null | wc -l)"
  done
}

all_done () {
  for d in experiment_conflict_order_20261123 experiment_a1_perturbed_20261123 paper_llm_rows_20261123; do
    python3 -c "import json,sys;d=json.load(open('$H/results/$d/completion.json'));sys.exit(0 if d.get('complete') else 1)" 2>/dev/null || return 1
  done
  return 0
}

echo "[$(date -Is)] watcher start"
last=""
for i in $(seq 1 2880); do   # up to 48 h at one minute per tick
  now=$(counts)
  if [ "$now" != "$last" ]; then
    echo "[$(date -Is)] progress $now -> refresh"
    bash "$OUT/refresh.sh" > /tmp/refresh_run.log 2>&1
    last="$now"
    if all_done; then
      echo "[$(date -Is)] all batches complete; watcher done"
      exit 0
    fi
  fi
  sleep 60
done
echo "[$(date -Is)] watcher timeout"