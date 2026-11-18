"""Advisor Challenge 3: verify a candidate twin update before committing it.

no_verify : commit every calibration the estimator proposes (current behaviour)
verify    : evaluate the policy in the twin under the old and the candidate twin on a held-out
            twin bank; reject the candidate if the twin success rate drops by more than tol.
This is the module the method chapter lists as Challenge 3 and had no experiment behind it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from horizon.datagen.parallel import Runner
from horizon.datagen.rollout import PerturbationConfig
from horizon.harness.artifacts import write_json
from horizon.harness.experiment import (check_results, collect_recorded, eval_jobs, evaluate,
                                        job_for, nominal_factory, observed_feedback, train_version)
from horizon.harness.selector_experiment import _collect, _store
from horizon.policy.dataset import Dataset
from horizon.real2sim.recon_error import ReconModel
from horizon.rsi.simreal import geometry_repair


def twin_sr(params, cfg, store, recon, bank, n, workers):
    """Success rate measured *inside the twin* (real=False), i.e. the sandbox the advisor describes."""
    jobs = []
    for family in cfg["families"]:
        for H in cfg["H_values"]:
            for i in range(n):
                ts = bank + i
                jobs.append(job_for(store, family, H, ts, ts + 7, cfg, recon, real=False,
                                    controller="hier", pert=PerturbationConfig.suite(0.0),
                                    want_samples=False, keep_log=True))
    with Runner(workers, policy={"params": params}) as runner:
        results = runner.map(jobs)
    check_results(results)
    return sum(1 for r in results if r.log.success) / len(results)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1_matrix.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--twin-n", type=int, default=10)
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
            for arm in ("no_verify", "verify"):
                recon = ReconModel.from_dict(base.to_dict())
                rejected = 0
                for r in range(1, cfg["rounds"] + 1):
                    obs, _ = _collect(pi0, cfg, store_deploy, recon, seed, r, args.workers,
                                      out / "fb" / f"{name}_s{seed}_r{r}")
                    cand, _ = geometry_repair(recon, [observed_feedback(o.log) for o in obs], robust=True)
                    if arm == "verify":
                        bank = cfg["banks"]["curve"] + r * 1000
                        old = twin_sr(pi0, cfg, store_train, recon, bank, args.twin_n, args.workers)
                        new = twin_sr(pi0, cfg, store_train, cand, bank, args.twin_n, args.workers)
                        if new < old - args.tol:
                            rejected += 1
                            cand = recon
                    recon = cand
                ev, _ = evaluate(pi0, cfg, store_deploy, recon, cfg["banks"]["curve"],
                                 cfg["curve_n_per"], args.workers)
                rec = {"scenario": name, "cause": cause, "seed": seed, "arm": arm,
                       "rejected": rejected, "sr_nominal": ev["nominal"]["overall"],
                       "sr_perturbed": ev["perturbed"]["overall"]}
                recs.append(rec)
                print(f"  {arm:9s} {name}/s{seed}: SR {rec['sr_nominal']:.3f}/{rec['sr_perturbed']:.3f}"
                      f"  rejected={rejected}", flush=True)
    write_json(out / "summary.json", {"records": recs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
