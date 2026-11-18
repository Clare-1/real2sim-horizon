"""R1 matrix: measure, rather than assume, which operation helps which failure cause.

This is the advisor's §四.4 step 1 at minimal scale: 5 single operations x 2 causes x N seeds.
It produces the cause -> operation map that the selectors use as "correct", replacing the
hard-coded table that the 18-run screening refuted.

Rows (each cell starts from the same D0 and pi0 so the comparison is paired):
  nothing              no update at all
  calibrate_geometry   commit one robust calibration from deployment feedback
  regenerate_uniform   resample from the training prior
  regenerate_targeted  resample where deployment actually puts objects
  all                  calibrate, then resample from the training prior

Every cell pays for its feedback episodes and for any labels it adds.
"""
from __future__ import annotations

import copy
from pathlib import Path

from ..datagen.parallel import Runner
from ..policy.dataset import Dataset
from ..real2sim.recon_error import ReconModel
from ..rsi.simreal import geometry_repair
from .artifacts import canonical_hash, write_json
from .experiment import (check_results, collect_recorded, evaluate, nominal_factory,
                         observed_feedback, train_version)
from .selector_experiment import _act, _collect, _store

R1_OPS = ("nothing", "calibrate_geometry", "regenerate_uniform", "regenerate_targeted", "all")


def _calibrate(recon, results):
    return geometry_repair(recon, [observed_feedback(r.log) for r in results], robust=True)[0]


def run_r1_matrix(cfg, out, project_root, workers=8):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = canonical_hash({"config": cfg, "matrix": "r1"})
    write_json(out / "experiment.json", {"config": cfg, "fingerprint": fingerprint})
    records = []
    for scenario in cfg["scenarios"]:
        name, cause = scenario["name"], scenario["cause"]
        prior = scenario.get("train_randomization", {}).get("layout_prior", "full")
        store_train = _store(out / "tasks_train" / name, cfg, project_root, prior)
        store_deploy = _store(out / "tasks_deploy" / name, cfg, project_root, "full")
        recon0 = ReconModel.from_dict(scenario.get("conditions", {}).get("recon", {}))
        for seed in cfg["seeds"]:
            shared = out / "shared" / name / f"s{seed}"
            shared.mkdir(parents=True, exist_ok=True)
            with Runner(workers) as runner:
                blocks, ledger, _ = collect_recorded(
                    runner, nominal_factory(store_train, cfg, recon0, seed),
                    cfg["init_budget"], recon0, require_success=True)
            write_json(shared / "collection.json", {"ledger": ledger})
            base = list(blocks)
            pi0, _, _ = train_version(base, cfg, seed, shared / "policy_0.npz")
            ev0, _ = evaluate(pi0, cfg, store_deploy, recon0, cfg["banks"]["curve"],
                              cfg["curve_n_per"], workers)
            sr0 = ev0["nominal"]["overall"]
            print(f"INIT {name}/s{seed}: {sum(b.n for b in base)} labels, SR={sr0:.3f}", flush=True)
            for op in R1_OPS:
                records.append(_cell(out, cfg, name, cause, seed, op, store_train, store_deploy,
                                     recon0, base, pi0, sr0, workers))
    write_json(out / "summary.json", {"fingerprint": fingerprint, "records": records})
    write_json(out / "completion.json", {"complete": True, "fingerprint": fingerprint,
                                         "runs": len(records)})
    return records


def _cell(out, cfg, name, cause, seed, op, store_train, store_deploy, recon0, base, pi0, sr0, workers):
    """One (cause, operation, seed) cell, paired on the same D0 and pi0."""
    d = out / "cells" / name / op / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    recon = ReconModel.from_dict(recon0.to_dict())
    data = Dataset(list(base))
    params = dict(pi0)
    deploy, labels, calib = 0, 0, 0
    cost = {}
    if op != "nothing":
        results, fb_cost = _collect(params, cfg, store_deploy, recon, seed, 901, workers, d / "feedback")
        cost.update({k: cost.get(k, 0) + fb_cost.get(k, 0) for k in fb_cost})
        deploy += int(cfg["feedback_episodes"])
    if op in ("calibrate_geometry", "all"):
        recon = _calibrate(recon, results)
        calib += 1
    if op in ("regenerate_uniform", "all"):
        recon, added, _, _, act_cost = _act("regenerate_uniform", None, cfg, store_train, store_deploy,
                                            recon, seed, 902, workers, params)
        cost.update({k: cost.get(k, 0) + act_cost.get(k, 0) for k in act_cost})
        data.add(added)
        labels += sum(b.n for b in added)
    elif op == "regenerate_targeted":
        recon, added, _, _, act_cost = _act("regenerate_targeted", None, cfg, store_train, store_deploy,
                                            recon, seed, 903, workers, params)
        cost.update({k: cost.get(k, 0) + act_cost.get(k, 0) for k in act_cost})
        data.add(added)
        labels += sum(b.n for b in added)
    if labels:
        params, _, _ = train_version(list(data.blocks), cfg, seed, d / "policy.npz")
    ev, ev_cost = evaluate(params, cfg, store_deploy, recon, cfg["banks"]["curve"], cfg["curve_n_per"], workers)
    cost.update({k: cost.get(k, 0) + ev_cost.get(k, 0) for k in ev_cost})
    sr1 = ev["nominal"]["overall"]
    rec = {"scenario": name, "cause": cause, "seed": seed, "op": op,
           "sr_before": sr0, "sr_after": sr1, "delta": sr1 - sr0,
           "feedback_episodes": deploy, "new_labels": labels, "calibrations": calib,
           "cost": cost,
           "recon_bias": dict(recon.bias), "recon_corrections": dict(recon.corrections)}
    write_json(d / "cell.json", rec)
    print(f"  {name}/{op}/s{seed}: {sr0:.3f} -> {sr1:.3f}  (+{labels} labels)", flush=True)
    return rec
