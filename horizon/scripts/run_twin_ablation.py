"""实验一 (advisor §五) missing rows: domain randomization, deploy-env DAgger, mixed training.

These three are what answers the reviewer questions "why not just use deployment data" and
"why not domain randomization". Each arm starts from the same D0 and pi0 so the comparison is paired.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import yaml

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, open_loop_job
from horizon.datagen.parallel import Runner
from horizon.datagen.rollout import PerturbationConfig
from horizon.harness.artifacts import write_json
from horizon.harness.experiment import (TaskStore, collect_recorded, evaluate, job_for, nominal_factory,
                                        train_version)
from horizon.harness.selector_experiment import _store
from horizon.monitor.verifier import VerifierConfig
from horizon.policy.dataset import Dataset
from horizon.real2sim.recon_error import ReconModel
from horizon.harness.selector_experiment import _collect
from horizon.rsi.simreal import geometry_repair


def randomized_factory(store, cfg, seed, lo, hi):
    """Domain randomisation: every demo is generated under a differently-believed twin."""
    dist = TaskDistribution(tuple(cfg["families"]), tuple(cfg["H_values"]), seed_base=cfg["banks"]["training"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 77]))
    def make():
        j = open_loop_job(dist, OpenLoopConfig(**cfg.get("open_loop", {})), rng)
        factor = float(rng.uniform(lo, hi))
        j.recon = ReconModel(bias={"plate": factor, "tray": factor}).to_dict()
        j.world = cfg["world"]
        j.observation_mode = cfg.get("training_observation_mode", "legacy")
        j.verifier_cfg = VerifierConfig(tol_scale=1.0)
        j.task_payload, ref = store.get(j.family, j.H, j.task_seed)
        j.tag = {"task_artifact": ref, "twin_sha256": "randomised"}
        return j
    return make


def deploy_dagger_factory(store, cfg, recon, seed, round_id):
    """Expert labels on the deployment environment only (no twin in the loop)."""
    cells = [(f, H) for f in cfg["families"] for H in cfg["H_values"]]
    def make(i=[0]):
        fam, H = cells[i[0] % len(cells)]
        ts = cfg["banks"]["feedback"] + seed * 100_000 + round_id * 10_000 + H * 100 + i[0]
        i[0] += 1
        return job_for(store, fam, H, ts, ts + 41, cfg, recon, real=True, controller="mixed",
                       learner="hier", beta=0.5 ** round_id, label_expert=True, max_samples=300,
                       pert=PerturbationConfig.suite(cfg.get("collection_pert", 0.1)),
                       provenance=f"deploy_dagger_r{round_id}")
    return make


ARMS = ("fixed_twin", "calibrate_frozen", "robust_repeat", "domain_rand", "deploy_dagger",
        "mixed", "true_params")


def run(cfg, out: Path, workers: int, bias: float, seeds, lo: float, hi: float, arms=ARMS):
    recs = []
    store_train = _store(out / "tasks_train", cfg, Path(".").resolve(), "full")
    store_deploy = _store(out / "tasks_deploy", cfg, Path(".").resolve(), "full")
    base_recon = ReconModel(bias={"plate": bias, "tray": bias})
    for seed in seeds:
        shared = out / "shared" / f"s{seed}"
        shared.mkdir(parents=True, exist_ok=True)
        with Runner(workers) as runner:
            blocks, _, _ = collect_recorded(runner, nominal_factory(store_train, cfg, ReconModel(), seed),
                                            cfg["init_budget"], ReconModel(), require_success=True)
        pi0, _, _ = train_version(list(blocks), cfg, seed, shared / "policy_0.npz")
        print(f"INIT s{seed}: {sum(b.n for b in blocks)} labels", flush=True)
        for arm in arms:
            data = Dataset(list(blocks))
            params = dict(pi0)
            recon = ReconModel.from_dict(base_recon.to_dict())
            if arm == "true_params":
                recon = ReconModel()  # no gap: the upper bound the advisor names method 8
            added_total = 0
            if arm in ("calibrate_frozen", "robust_repeat"):
                obs, _ = _collect(params, cfg, store_deploy, recon, seed, 1, workers, out / "fb" / f"{arm}_s{seed}")
                from horizon.harness.experiment import observed_feedback
                recon, _ = geometry_repair(recon, [observed_feedback(o.log) for o in obs], robust=True)
            for r in range(1, int(cfg["rounds"]) + 1):
                if arm == "calibrate_frozen" and r > 1:
                    break
                if arm in ("fixed_twin", "calibrate_frozen"):
                    factory = nominal_factory(store_train, cfg, recon, seed)
                elif arm == "domain_rand":
                    factory = randomized_factory(store_train, cfg, seed, lo, hi)
                elif arm == "deploy_dagger":
                    factory = deploy_dagger_factory(store_deploy, cfg, recon, seed, r)
                else:  # mixed: alternate twin-generated demos and deployment labels
                    factory = (nominal_factory(store_train, cfg, recon, seed) if r % 2 == 1
                               else deploy_dagger_factory(store_deploy, cfg, recon, seed, r))
                if arm == "robust_repeat" and r > 1:
                    from horizon.harness.experiment import observed_feedback as _of
                    _obs, _ = _collect(params, cfg, store_deploy, recon, seed, r, workers,
                                       out / "fb" / (arm + "_s%d_r%d" % (seed, r)))
                    recon, _ = geometry_repair(recon, [_of(o.log) for o in _obs], robust=True)
                with Runner(workers, policy={"params": params}) as runner:
                    new, _, _ = collect_recorded(runner, factory, cfg["round_budget"], recon,
                                                 require_success=False)
                data.add(new)
                added_total += sum(b.n for b in new)
                params, _, _ = train_version(list(data.blocks), cfg, seed, shared / f"{arm}_{r}.npz")
            ev, _ = evaluate(params, cfg, store_deploy,
                             base_recon if arm == "calibrate_frozen" else recon,
                             cfg["banks"]["curve"], cfg["curve_n_per"], workers)
            rec = {"arm": arm, "seed": seed, "bias": bias, "sr_nominal": ev["nominal"]["overall"],
                   "sr_perturbed": ev["perturbed"]["overall"], "new_labels": added_total}
            recs.append(rec)
            print(f"  {arm:18s} s{seed}: SR {rec['sr_nominal']:.3f}/{rec['sr_perturbed']:.3f}  "
                  f"(+{added_total} labels)", flush=True)
    write_json(out / "summary.json", {"records": recs})
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1_matrix.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bias", type=float, default=0.46)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dr-lo", type=float, default=0.30)
    ap.add_argument("--dr-hi", type=float, default=1.00)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--feedback-episodes", type=int, default=0)
    ap.add_argument("--H-values", default="")
    ap.add_argument("--rounds", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg["rounds"] = args.rounds or int(cfg.get("rounds", 3))
    if args.feedback_episodes:
        cfg["feedback_episodes"] = args.feedback_episodes
    if args.H_values:
        cfg["H_values"] = [int(x) for x in args.H_values.split(",")]
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    run(cfg, out, args.workers, args.bias, [int(x) for x in args.seeds.split(",")], args.dr_lo,
        args.dr_hi, tuple(args.arms.split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
