"""Calibrate the support-height bias level by pi0's success rate, not by millimetres.

The advisor's rule (回复学生 §五 实验一): pick the bias level that puts the initial policy at
roughly 70/40/10% SR, because 6 mm-vs-20 mm saturates the metric at zero and carries no
information. This sweeps the multiplicative bias factor on a trained pi0 and reports SR.
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
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bank", type=int, default=85000000)
    ap.add_argument("--n-per", type=int, default=8)
    ap.add_argument("--factors", default="1.0,0.9,0.8,0.65,0.5,0.3",
                    help="comma-separated multiplicative support-height factors to probe")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    params = load_policy({"path": str(Path(args.policy).resolve()), "sha256": file_hash(args.policy)})
    store = TaskStore(out / "tasks", {**cfg, "randomization": {"layout_prior": "full"}}, Path(".").resolve())

    rows = []
    print(f"{'tray factor':>12} {'believed':>10} {'SR nominal':>11} {'SR perturbed':>13}")
    for factor in [float(f) for f in args.factors.split(",") if f.strip()]:
        recon = ReconModel() if factor == 1.0 else ReconModel(bias={"plate": factor, "tray": factor})
        ev, ledger = evaluate(params, cfg, store, recon, args.bank, args.n_per, args.workers)
        row = {"factor": factor, "believed_mm": round(20.0 * factor, 2),
               "sr_nominal": ev["nominal"]["overall"], "sr_perturbed": ev["perturbed"]["overall"],
               "episodes": ledger["episodes"]}
        rows.append(row)
        print(f"{factor:>12.2f} {20*factor:>9.1f}mm {ev['nominal']['overall']:>11.3f} "
              f"{ev['perturbed']['overall']:>13.3f}")

    target = 0.40
    best = min(rows, key=lambda r: abs(r["sr_nominal"] - target))
    print(f"\nclosest to {target:.0%} nominal: factor={best['factor']} "
          f"({best['believed_mm']} mm, SR={best['sr_nominal']:.3f})")
    if all(r["sr_nominal"] == 0.0 for r in rows):
        print("WARNING: every level is 0 — this bias type only has two outcomes on this task; "
              "switch to a position or friction bias instead of tuning the factor.")
    (out / "bias_sweep.json").write_text(json.dumps({"rows": rows, "target": target, "chosen": best},
                                                    indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
