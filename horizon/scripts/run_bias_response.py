"""Advisor 表 9: how the calibration benefit scales with bias size (small / medium / large).

One fixed policy is evaluated at several support-height biases, before and after a single robust
calibration, so the only thing changing is the twin. Answers "小偏差时校准还有没有必要".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from horizon.harness.artifacts import file_hash, load_policy
from horizon.harness.experiment import TaskStore, evaluate, observed_feedback
from horizon.harness.selector_experiment import _collect, _store
from horizon.real2sim.recon_error import ReconModel
from horizon.rsi.simreal import geometry_repair


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/minimal_validation.yaml")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--biases", default="1.0,0.65,0.46,0.30")
    ap.add_argument("--bank", type=int, default=91000000)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    params = load_policy({"path": str(Path(args.policy).resolve()), "sha256": file_hash(args.policy)})
    deploy = _store(out / "tasks", cfg, Path(".").resolve(), "full")
    rows = []
    print("%8s %10s %10s %10s %10s" % ("bias", "believed", "SR before", "SR after", "delta"))
    for b in [float(x) for x in args.biases.split(",")]:
        recon = ReconModel() if b == 1.0 else ReconModel(bias={"plate": b, "tray": b})
        before, _ = evaluate(params, cfg, deploy, recon, args.bank, cfg["curve_n_per"], args.workers)
        results, _ = _collect(params, cfg, deploy, recon, 0, 1, args.workers, out / f"fb_{b}")
        calibrated, report = geometry_repair(
            recon, [observed_feedback(r.log) for r in results], robust=True)
        after, _ = evaluate(params, cfg, deploy, calibrated, args.bank, cfg["curve_n_per"], args.workers)
        sr0, sr1 = before["nominal"]["overall"], after["nominal"]["overall"]
        est = None
        for v in report.values():
            est = v.get("estimated")
        row = {"bias": b, "believed_mm": round(20.0 * b, 1),
               "estimated_mm": None if est is None else round(est * 1000, 2),
               "sr_before": sr0, "sr_after": sr1, "delta": sr1 - sr0,
               "n_obs": {k: v.get("n") for k, v in report.items()}}
        rows.append(row)
        print("%8.2f %9.1fmm %10.3f %10.3f %+10.3f" % (b, 20 * b, sr0, sr1, sr1 - sr0))
    (out / "bias_response.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
