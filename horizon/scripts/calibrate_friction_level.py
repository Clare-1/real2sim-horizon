"""Calibrate the friction gap by pi0 SR, the same rule used for the height bias.

The deployment world is given a friction multiplier; the twin keeps 1.0. We sweep the multiplier
and report SR so the "dynamics" cause can be set to a level where the policy really struggles
(~40%), instead of a level where nothing happens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from horizon.harness.artifacts import file_hash, load_policy
from horizon.harness.experiment import TaskStore, evaluate
from horizon.real2sim.recon_error import ReconModel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/minimal_validation.yaml")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-per", type=int, default=20)
    ap.add_argument("--bank", type=int, default=91000000)
    ap.add_argument("--levels", default="1.0,0.7,0.55,0.4,0.3,0.2")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    params = load_policy({"path": str(Path(args.policy).resolve()), "sha256": file_hash(args.policy)})
    store = TaskStore(out / "tasks", {**cfg, "randomization": {"layout_prior": "full"}}, Path(".").resolve())
    recon = ReconModel()
    rows = []
    print("%10s %12s %14s" % ("friction", "SR nominal", "SR perturbed"))
    for f in [float(x) for x in args.levels.split(",")]:
        local = {**cfg, "world": {**cfg.get("world", {}), "friction_scale": f}}
        ev, _ = evaluate(params, local, store, recon, args.bank, args.n_per, args.workers)
        row = {"friction_scale": f, "sr_nominal": ev["nominal"]["overall"],
               "sr_perturbed": ev["perturbed"]["overall"]}
        rows.append(row)
        print("%10.2f %12.3f %14.3f" % (f, row["sr_nominal"], row["sr_perturbed"]))
    best = min(rows, key=lambda r: abs(r["sr_nominal"] - 0.40))
    print(f"\nclosest to 40%: friction_scale={best['friction_scale']} (SR={best['sr_nominal']:.3f})")
    (out / "friction_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
