"""Bounded parallel scheduling of the preregistered development repair screen."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--conditions", default="triple_unit,triple_cost,healthy")
    ap.add_argument("--out", type=Path, default=Path("results/protocol_screen_20261025"))
    args = ap.parse_args()
    if not 1 <= args.jobs <= 6 or not 1 <= args.workers <= 8:
        raise ValueError("Use bounded concurrency: jobs <= 6, workers <= 8")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    tasks = []
    for condition in args.conditions.split(","):
        config_path = Path(f"configs/protocol_screen_{condition}.yaml")
        raw = config_path.read_bytes()
        conf = yaml.safe_load(raw)
        if conf["common"]["evaluation_split"] != "development":
            raise ValueError("This scheduler only runs development experiments")
        for method in conf["methods"]:
            for seed in map(int, args.seeds.split(",")):
                tasks.append({"condition": condition, "method": method, "seed": seed,
                              "config": str(config_path), "config_sha256": hashlib.sha256(raw).hexdigest()})
    schedule_path = args.out / "schedule.json"
    schedule = {"purpose": "development; no confirmation evaluation", "jobs": args.jobs,
                "workers_per_job": args.workers, "tasks": tasks}
    if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule:
        raise ValueError("Existing schedule differs; use a new output directory")
    schedule_path.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def run(task):
        started = time.time()
        condition, method, seed = task["condition"], task["method"], task["seed"]
        logfile = args.out / "logs" / f"{condition}_{method}_s{seed}.log"
        cmd = [sys.executable, "scripts/run_repair_probe.py", "--config", task["config"],
               "--methods", method, "--seeds", str(seed), "--workers", str(args.workers),
               "--out", str(args.out / condition), "--skip-existing"]
        with logfile.open("w", encoding="utf-8") as log:
            process = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return {**task, "returncode": process.returncode, "seconds": time.time() - started, "log": str(logfile)}

    completed = []
    print(f"Starting {len(tasks)} development runs, {args.jobs} concurrent x {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(pending):
            result = future.result()
            completed.append(result)
            (args.out / "status.json").write_text(json.dumps({"total": len(tasks), "completed": completed}, indent=2), encoding="utf-8")
            label = f"{result['condition']}/{result['method']}/s{result['seed']}"
            print(f"[{len(completed)}/{len(tasks)}] {label}: exit={result['returncode']}, {result['seconds']:.1f}s", flush=True)
    failed = [r for r in completed if r["returncode"]]
    if failed:
        raise SystemExit(f"{len(failed)} failed runs; inspect logs before reporting")
    print("All scheduled development runs completed. Confirmation bank was not evaluated.", flush=True)


if __name__ == "__main__":
    main()
