"""Train a reference pi0 at the protocol budget (80k labels, 8000 steps) for calibration.

The reduced-budget screening policy (20k/4000) scores 0 even with a *correct* twin, so it cannot
be used to calibrate anything. This rebuilds the protocol policy on the full-support prior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from horizon.datagen.parallel import Runner
from horizon.harness.artifacts import save_blocks, write_json
from horizon.harness.experiment import TaskStore, collect_recorded, nominal_factory, train_version
from horizon.real2sim.recon_error import ReconModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init-budget", type=int, default=80000)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--prior", default="full")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = json.loads(json.dumps(cfg))  # deep copy without dataclasses
    cfg["init_budget"] = args.init_budget
    cfg.setdefault("train", {})["steps"] = args.steps
    cfg["randomization"] = {"layout_prior": args.prior}
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    project_root = Path(".").resolve()
    store = TaskStore(out / "tasks", cfg, project_root)
    recon = ReconModel()
    with Runner(args.workers) as runner:
        blocks, ledger, _ = collect_recorded(runner, nominal_factory(store, cfg, recon, 0),
                                             cfg["init_budget"], recon, require_success=True)
    save_blocks(out / "data.npz", blocks)
    params, record, stats = train_version(blocks, cfg, 0, out / "policy_0.npz")
    write_json(out / "train.json", {"ledger": ledger, "record": record, "stats": stats,
                                    "labels": sum(b.n for b in blocks)})
    print(f"trained {sum(b.n for b in blocks)} labels, {args.steps} steps, "
          f"wall={stats['wall_seconds']:.1f}s -> {record['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
