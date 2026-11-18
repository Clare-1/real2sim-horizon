"""Challenge 3, second attempt: a POLICY-FREE verification of a candidate twin update.

The first attempt (run_verify_ablation.py) used twin success rate and was refuted: a policy trained
under the wrong twin scores *better* in the wrong twin, so the check is anti-correlated with the
true objective. This version never looks at the policy. It splits the feedback episodes into
fit/check and accepts a candidate only if it reduces held-out prediction error of the observable
support heights. That is exactly the "was the twin pushed off" test the advisor asked for.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from horizon.datagen.parallel import Runner
from horizon.harness.artifacts import write_json
from horizon.harness.experiment import (collect_recorded, evaluate, nominal_factory, observed_feedback,
                                        train_version)
from horizon.harness.selector_experiment import _collect, _store
from horizon.real2sim.recon_error import ReconModel
from horizon.rsi.simreal import geometry_repair, support_observations


def per_episode_heights(obs_list, idx):
    out = defaultdict(list)
    for i in idx:
        for name, vals in support_observations([obs_list[i]]).items():
            if vals:
                out[name].append(float(np.median(vals)))
    return out


def believed_top(recon, base_top, name):
    return base_top.get(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1_matrix.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tol", type=float, default=0.002, help="allowed held-out error increase, m")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["rounds"] = int(cfg.get("rounds", 3))
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
            print(f"INIT {name}/s{seed}", flush=True)
            for arm in ("no_verify", "verify_pred"):
                recon = ReconModel.from_dict(base.to_dict())
                rejected, log = 0, []
                for r in range(1, cfg["rounds"] + 1):
                    obs, _ = _collect(pi0, cfg, store_deploy, recon, seed, r, args.workers,
                                      out / "fb" / f"{name}_s{seed}_r{r}")
                    obs = [observed_feedback(o.log) for o in obs]
                    n = len(obs)
                    fit = [i for i in range(n) if i % 2 == 0]
                    chk = [i for i in range(n) if i % 2 == 1]
                    cand, report = geometry_repair(recon, [obs[i] for i in fit], robust=True)
                    if arm == "verify_pred" and report:
                        chk_h = per_episode_heights(obs, chk)
                        ok = True
                        for fx, entry in report.items():
                            est = entry["estimated"]
                            hs = chk_h.get(fx) or []
                            if len(hs) < 2:
                                continue
                            cur = entry["believed"]
                            err_before = float(np.mean([abs(h - cur) for h in hs]))
                            err_after = float(np.mean([abs(h - est) for h in hs]))
                            log.append({"fixture": fx, "believed_mm": round(cur * 1000, 2),
                                        "cand_mm": round(est * 1000, 2),
                                        "err_before_mm": round(err_before * 1000, 2),
                                        "err_after_mm": round(err_after * 1000, 2)})
                            if err_after > err_before + args.tol:
                                ok = False
                        if not ok:
                            rejected += 1
                            cand = recon
                    recon = cand
                ev, _ = evaluate(pi0, cfg, store_deploy, recon, cfg["banks"]["curve"],
                                 cfg["curve_n_per"], args.workers)
                rec = {"scenario": name, "cause": cause, "seed": seed, "arm": arm,
                       "rejected": rejected, "checks": log,
                       "sr_nominal": ev["nominal"]["overall"],
                       "sr_perturbed": ev["perturbed"]["overall"]}
                recs.append(rec)
                print(f"  {arm:11s} {name}/s{seed}: SR {rec['sr_nominal']:.3f}/{rec['sr_perturbed']:.3f}"
                      f"  rejected={rejected}", flush=True)
    write_json(out / "summary.json", {"records": recs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
