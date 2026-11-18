"""Dump individual failing rollouts instead of only aggregate success rates.

Advisor 回复学生 §二.2: "每个实验都要看原始数据。每个条件至少看 20 条失败的 rollout".
This prints one row per failure with the attributed subgoal, failure type, confidence, steps and
perturbation events, and writes the same table to markdown so it can be read before any summary.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from horizon.datagen.parallel import Runner
from horizon.harness.artifacts import file_hash, load_policy
from horizon.harness.experiment import TaskStore, check_results, eval_jobs
from horizon.monitor.attribution import attribute_auto
from horizon.real2sim.recon_error import ReconModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--bank", type=int, required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bias", type=float, default=None, help="override support-height factor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pert", type=float, default=0.15)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    params = load_policy({"path": str(Path(args.policy).resolve()), "sha256": file_hash(args.policy)})
    recon = (ReconModel() if args.bias is None
             else ReconModel(bias={"plate": args.bias, "tray": args.bias}))
    store = TaskStore(out / "tasks", {**cfg, "randomization": {"layout_prior": "full"}}, Path(".").resolve())

    rows = []
    with Runner(args.workers, policy={"params": params}) as runner:
        results = runner.map(eval_jobs(store, cfg, recon, args.bank, args.n, args.pert))
    check_results(results)
    for r in results:
        log = r.log
        att = attribute_auto(log)
        events = [k for _, k, _ in log.events]
        rows.append({
            "task_seed": r.job.task_seed, "H": r.job.H, "rollout_seed": r.job.seed,
            "success": bool(log.success),
            "failed_subgoal": (att.subgoal if att else None),
            "failure_type": (att.ftype if att else None),
            "confidence": (round(att.confidence, 2) if att else None),
            "steps": int(log.steps),
            "satisfied_at_end": int(sum(log.meta["final_sat"])),
            "of": int(len(log.meta["final_sat"])),
            "perturbations": len(events),
        })

    fails = [r for r in rows if not r["success"]]
    lines = ["# Failure inspection", "",
             f"policy `{Path(args.policy).name}`  bank={args.bank}  bias={args.bias}  episodes={len(rows)}",
             f"**failures={len(fails)}/{len(rows)} = {len(fails)/max(len(rows),1):.1%}**", "",
             "| # | task | H | rollout | failed subgoal | type | conf | steps | satisfied | perturb |",
             "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for i, r in enumerate(fails, 1):
        lines.append(f"| {i} | {r['task_seed']} | {r['H']} | {r['rollout_seed']} | {r['failed_subgoal']} | "
                     f"{r['failure_type']} | {r['confidence']} | {r['steps']} | "
                     f"{r['satisfied_at_end']}/{r['of']} | {r['perturbations']} |")
    by_type = Counter(r["failure_type"] for r in fails)
    by_goal = Counter(r["failed_subgoal"] for r in fails)
    lines += ["", f"failure types: {dict(by_type)}", f"failed subgoals: {dict(by_goal)}", ""]
    (out / "failures.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "failures.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"... {len(fails)} failures written to {out}/failures.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
