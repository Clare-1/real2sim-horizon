"""Counterfactual operation sweep: what would each operation have bought from this state?

For every (cause, bank) the same pi0, feedback episodes and recon are used, then each candidate
operation is applied once and the curve-bank SR is re-measured. The output is the training set for
a learned operation-value model:

    (profile_vector, action, label_cost) -> ΔSR

    python scripts/run_operation_sweep.py --banks 2 --workers 12
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon.harness.artifacts import load_blocks, load_params  # noqa: E402
from horizon.harness.experiment import (evaluate, observed_feedback, train_version)  # noqa: E402
from horizon.harness.selector_experiment import _act, _collect, _profile, _store  # noqa: E402
from horizon.real2sim.recon_error import ReconModel  # noqa: E402
from horizon.rsi.selector import profile_vector  # noqa: E402
from horizon.rsi.simreal import geometry_repair  # noqa: E402

ACTIONS = ("calibrate_geometry", "regenerate_uniform", "regenerate_targeted",
           "add_recovery_data", "dagger_round")


def scenario_cfg(base: dict, scenario: dict) -> dict:
    cfg = copy.deepcopy(base)
    if scenario.get("open_loop"):
        cfg["open_loop"] = {**base.get("open_loop", {}), **scenario["open_loop"]}
    for key in ("feedback_pert", "eval_pert", "eval_mode", "friction_range"):
        if scenario.get(key) is not None:
            cfg[key] = scenario[key]
    if scenario.get("world"):
        cfg["world"] = {**base.get("world", {}), **scenario["world"]}
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/minimal_validation_4causes.yaml")
    ap.add_argument("--results", default="results/four_causes_20261118_rule")
    ap.add_argument("--out", default="results/operation_sweep_20261118")
    ap.add_argument("--banks", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()
    base = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    root = (ROOT / args.results).resolve()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in base["scenarios"]:
        name = scenario["name"]
        shared = root / "shared" / name / "s0"
        manifest = shared / "data.manifest.json"
        if not manifest.exists():
            print(f"skip {name}: no data.manifest.json", flush=True)
            continue
        cfg = scenario_cfg(base, scenario)
        store_train = _store(root / "tasks_train" / name, cfg,
                             Path(args.project_root).resolve(), "full")
        store_deploy = _store(root / "tasks_deploy" / name, cfg,
                              Path(args.project_root).resolve(), "full")
        recon0 = ReconModel.from_dict(scenario.get("conditions", {}).get("recon", {}))
        base_blocks = load_blocks(json.loads(manifest.read_text(encoding="utf-8")))
        pi0 = load_params(shared / "policy_0.npz")
        for bank in range(args.banks):
            feedback_bank = cfg["banks"]["feedback"] + bank * 1_000_000
            curve_bank = cfg["banks"]["curve"] + bank * 1_000_000
            fb, _ = _collect(pi0, cfg, store_deploy, recon0, 0, 1, args.workers,
                             out / "_tmp" / name / f"b{bank}", bank_base=feedback_bank)
            prof, _ = _profile(fb, recon0)
            ev, _ = evaluate(pi0, cfg, store_deploy, recon0, curve_bank,
                             cfg["curve_n_per"], args.workers)
            sr0 = ev[cfg.get("eval_mode", "nominal")]["overall"]
            for action in ACTIONS:
                if action == "calibrate_geometry":
                    recon1, _ = geometry_repair(
                        recon0, [observed_feedback(r.log) for r in fb], robust=True)
                    added, label_cost = [], 0
                else:
                    recon1, added, _, _, act_cost = _act(
                        action, None, cfg, store_train, store_deploy, recon0, 0, 1,
                        args.workers, pi0)
                    label_cost = int(act_cost.get("accepted_labels", 0))
                params = pi0
                if added:
                    params, _, _ = train_version(list(base_blocks) + added, cfg, 0,
                                                 out / "_tmp" / name / f"b{bank}_{action}.npz")
                ev1, _ = evaluate(params, cfg, store_deploy, recon1, curve_bank,
                                  cfg["curve_n_per"], args.workers)
                sr1 = ev1[cfg.get("eval_mode", "nominal")]["overall"]
                rows.append({
                    "cause": scenario["cause"], "scenario": name, "bank_index": bank,
                    "action": action, "profile": list(profile_vector(prof)),
                    "sr_before": sr0, "sr_after": sr1, "delta": sr1 - sr0,
                    "label_cost": label_cost,
                    "failure_types": dict(prof.failure_types),
                })
                print(f"{name} b{bank} {action:20s} {sr0:.3f} -> {sr1:.3f} "
                      f"({sr1-sr0:+.3f}) labels={label_cost}", flush=True)
    path = out / "sweep.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    print(f"[written] {path} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
