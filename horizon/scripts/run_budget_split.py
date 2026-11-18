"""Advisor §四.4 step 2, quantitative half: does "do every operation" win at an EQUAL total budget?

single : calibrate once, then one regeneration with the full label budget B
do_all : calibrate once, then uniform + targeted regeneration sharing the same B (B/2 each)
Same D0 and pi0 per (cause, seed); only the split changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from horizon.datagen.parallel import Runner
from horizon.harness.artifacts import write_json
from horizon.harness.experiment import (collect_recorded, evaluate, nominal_factory, observed_feedback,
                                        train_version)
from horizon.harness.selector_experiment import _act, _collect, _store
from horizon.policy.dataset import Dataset
from horizon.real2sim.recon_error import ReconModel
from horizon.rsi.simreal import geometry_repair


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1_matrix.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--budget", type=int, default=12000)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    recs = []
    for scen in cfg["scenarios"]:
        name, cause = scen["name"], scen["cause"]
        prior = scen.get("train_randomization", {}).get("layout_prior", "full")
        store_train = _store(out / "tasks_train" / name, cfg, Path(".").resolve(), prior)
        store_deploy = _store(out / "tasks_deploy" / name, cfg, Path(".").resolve(), "full")
        base = ReconModel.from_dict(scen.get("conditions", {}).get("recon", {}))
        for seed in [int(x) for x in args.seeds.split(",")]:
            shared = out / "shared" / name / f"s{seed}"
            shared.mkdir(parents=True, exist_ok=True)
            with Runner(args.workers) as runner:
                d0, _, _ = collect_recorded(runner, nominal_factory(store_train, cfg, base, seed),
                                            cfg["init_budget"], base, require_success=True)
            pi0, _, _ = train_version(list(d0), cfg, seed, shared / "policy_0.npz")
            print(f"INIT {name}/s{seed}: {sum(b.n for b in d0)} labels", flush=True)
            for arm in ("single", "do_all"):
                recon = ReconModel.from_dict(base.to_dict())
                obs, _ = _collect(pi0, cfg, store_deploy, recon, seed, 1, args.workers,
                                  out / "fb" / f"{name}_s{seed}")
                recon, _ = geometry_repair(recon, [observed_feedback(o.log) for o in obs], robust=True)
                halves = [args.budget] if arm == "single" else [args.budget // 2, args.budget // 2]
                added = []
                for i, budget in enumerate(halves):
                    op = "regenerate_uniform" if i == 0 else "regenerate_targeted"
                    local = dict(cfg); local["round_budget"] = budget
                    _, b, _, _, _ = _act(op, None, local, store_train, store_deploy, recon,
                                         seed, 300 + i, args.workers, pi0)
                    added += b
                data = Dataset(list(d0)); data.add(added)
                params, _, _ = train_version(list(data.blocks), cfg, seed, shared / f"{arm}.npz")
                ev, _ = evaluate(params, cfg, store_deploy, recon, cfg["banks"]["curve"],
                                 cfg["curve_n_per"], args.workers)
                rec = {"scenario": name, "cause": cause, "seed": seed, "arm": arm,
                       "labels_total": args.budget, "labels_added": sum(b.n for b in added),
                       "sr_nominal": ev["nominal"]["overall"], "sr_perturbed": ev["perturbed"]["overall"]}
                recs.append(rec)
                print(f"  {arm:7s} {name}/s{seed}: SR {rec['sr_nominal']:.3f}/{rec['sr_perturbed']:.3f}"
                      f"  (+{rec['labels_added']} labels)", flush=True)
    write_json(out / "summary.json", {"records": recs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
