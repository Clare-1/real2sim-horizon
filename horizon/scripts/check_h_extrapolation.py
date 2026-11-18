"""Advisor 共用规矩 4 / 表 11: train on H=2,4 only, then test H=6,8,10.

Historical records report 0% at every unseen H under the old observation encoding. This retrains
with the current code and reports the same grid, so the claim can be checked rather than cited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from horizon.datagen.parallel import Runner
from horizon.harness.experiment import (TaskStore, collect_recorded, evaluate, nominal_factory,
                                        train_version)
from horizon.real2sim.recon_error import ReconModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/minimal_validation.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-h", default="2,4")
    ap.add_argument("--test-h", default="2,4,6,8,10")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    train_h = [int(x) for x in args.train_h.split(",")]
    test_h = [int(x) for x in args.test_h.split(",")]

    cfg = dict(base); cfg["H_values"] = train_h
    cfg["randomization"] = {"layout_prior": "full"}
    store = TaskStore(out / "tasks", {**cfg, "H_values": test_h}, Path(".").resolve())
    recon = ReconModel()
    print(f"training on H={train_h}", flush=True)
    with Runner(args.workers) as runner:
        blocks, ledger, _ = collect_recorded(
            runner, nominal_factory(store, cfg, recon, 0), cfg["init_budget"], recon, require_success=True)
    params, _, stats = train_version(list(blocks), cfg, 0, out / "policy_0.npz")
    print(f"trained {sum(b.n for b in blocks)} labels, wall={stats['wall_seconds']:.1f}s", flush=True)

    rows = []
    print(f"{'H':>4} {'in train?':>10} {'SR nominal':>11} {'SR perturbed':>13}")
    for H in test_h:
        cfg_h = {**cfg, "H_values": [H], "curve_n_per": base.get("curve_n_per", 20)}
        ev, _ = evaluate(params, cfg_h, store, recon, base["banks"]["curve"], base.get("curve_n_per", 20),
                         args.workers)
        row = {"H": H, "in_training": H in train_h,
               "sr_nominal": ev["nominal"]["overall"], "sr_perturbed": ev["perturbed"]["overall"]}
        rows.append(row)
        print(f"{H:>4} {str(H in train_h):>10} {row['sr_nominal']:>11.3f} {row['sr_perturbed']:>13.3f}", flush=True)
    (out / "h_extrapolation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
