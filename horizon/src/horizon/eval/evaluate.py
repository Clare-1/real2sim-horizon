"""Held-out evaluation on a fixed test set shared by all methods (paper Sec. 6.5, 6.8).

Test tasks use a seed range disjoint from training (``TEST_SEED_BASE``), so layouts, object
sizes and colours are unseen. The same (task, noise) seeds are used for every method,
making per-episode comparisons paired.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..datagen.parallel import Job, Runner
from ..datagen.rollout import PerturbationConfig
from ..monitor.attribution import attribute_auto, attribute_oracle
from ..monitor.verifier import VerifierConfig
from .metrics import fit_horizon_decay, wilson

TEST_SEED_BASE = 10_000_000


def test_jobs(policy: str, families, H_values, n_per: int, pert_p: float,
              verifier_cfg: VerifierConfig | None = None) -> list[Job]:
    jobs = []
    for fi, fam in enumerate(families):
        for H in H_values:
            for i in range(n_per):
                ts = TEST_SEED_BASE + fi * 100_000 + H * 1000 + i
                jobs.append(Job(family=fam, H=H, task_seed=ts, seed=ts + 17, controller=policy,
                                pert=PerturbationConfig.suite(pert_p), verifier_cfg=verifier_cfg,
                                want_samples=False, keep_log=True, provenance="test",
                                tag={"cond": "perturbed" if pert_p > 0 else "nominal"}))
    return jobs


def evaluate(params: dict, policy: str, families=("pack", "sort", "stack"), H_values=(2, 4, 6, 8, 10),
             n_per: int = 30, pert_p: float = 0.0, workers: int | None = None,
             verifier_cfg: VerifierConfig | None = None, runner: Runner | None = None) -> dict:
    jobs = test_jobs(policy, families, H_values, n_per, pert_p, verifier_cfg)
    if runner is None:
        with Runner(workers, policy={"params": params}) as r:
            res = r.map(jobs)
    else:
        res = runner.map(jobs)
    return summarize(res, H_values)


def summarize(res, H_values) -> dict:
    by_h: dict[int, list[int]] = defaultdict(list)
    by_fh: dict[tuple, list[int]] = defaultdict(list)
    stage_tot: dict[int, int] = defaultdict(int)
    stage_ok: dict[int, int] = defaultdict(int)
    ftypes: dict[str, int] = defaultdict(int)
    ftypes_auto: dict[str, int] = defaultdict(int)
    episodes = []
    errors = [r.error for r in res if r.error]
    for r in res:
        if r.log is None:
            continue
        log = r.log
        by_h[log.H].append(int(log.success))
        by_fh[(log.family, log.H)].append(int(log.success))
        final = np.asarray(log.meta["final_sat"], dtype=bool)
        for st in range(log.H):
            stage_tot[st] += 1
            stage_ok[st] += int(final.sum() > st)  # episodes that completed more than st subgoals
        a = attribute_oracle(log)
        if a is not None:
            ftypes[a.ftype] += 1
        b = attribute_auto(log)
        if b is not None:
            ftypes_auto[b.ftype] += 1
        episodes.append({"family": log.family, "H": log.H, "seed": log.seed, "success": bool(log.success),
                         "n_sat": int(final.sum()), "steps": log.steps,
                         "oracle": a.to_dict() if a else None, "auto": b.to_dict() if b else None})
    Hs = np.array(sorted(by_h))
    k = np.array([sum(by_h[h]) for h in Hs])
    n = np.array([len(by_h[h]) for h in Hs])
    fit = fit_horizon_decay(Hs, k, n) if len(Hs) >= 2 else None
    table = {int(h): {"k": int(kk), "n": int(nn), "rate": wilson(int(kk), int(nn))} for h, kk, nn in zip(Hs, k, n)}
    fam_table = {f"{f}_H{h}": {"k": sum(v), "n": len(v)} for (f, h), v in sorted(by_fh.items())}
    return {"by_H": table, "by_family_H": fam_table, "fit": fit, "overall": float(k.sum() / max(1, n.sum())),
            "progress": {int(s): stage_ok[s] / max(1, stage_tot[s]) for s in sorted(stage_tot)},
            "failure_types_oracle": dict(ftypes), "failure_types_auto": dict(ftypes_auto),
            "episodes": episodes, "errors": errors[:3]}
