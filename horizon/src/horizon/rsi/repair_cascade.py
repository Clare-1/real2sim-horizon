"""Evidence replay after scene repair, with an explicit cumulative repair cost.

Only observed poses and audited labels are used. No injected-error flags, physical
scene geometry, held_true, or oracle subgoal indices are consulted for decisions.
The recipe is fixed: this is a closed-loop repair controller, not a claim of RSI
of the improver itself (Physical2RSI, 2610.29677).
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import math
from typing import TYPE_CHECKING

import numpy as np

from ..monitor.verifier import Verifier, VerifierConfig
from ..policy.features import state_from_log
from ..real2sim.recon_error import ReconModel

if TYPE_CHECKING:
    from .simreal import Belief, SimRealConfig


def select_cost_subset(shares: dict, costs: dict, budget: float, min_share: float) -> list[str]:
    """Exact subset selection for the three repairs, rather than greedy value/cost."""
    keys = [k for k in ("geometry", "verifier", "graph") if shares.get(k, 0) >= min_share]
    best, best_key = [], (0.0, 0.0)
    for n in range(1, len(keys) + 1):
        for subset in combinations(keys, n):
            cost = sum(costs[k] for k in subset)
            value = sum(shares[k] for k in subset)
            if cost <= budget + 1e-9 and (value, -cost) > best_key:
                best, best_key = list(subset), (value, -cost)
    return best


def robust_geometry_candidate(recon: ReconModel, logs: list, min_episodes: int = 3,
                              tol: float = 0.003) -> tuple[ReconModel, dict]:
    """Give each episode one vote, so long stalled trajectories cannot dominate.

    The lower within-episode quartile reduces the influence of late contacts with
    objects above the support. A cross-episode MAD check rejects inconsistent votes.
    """
    from .simreal import _believed, support_observations

    votes: dict[str, list[float]] = {}
    tops: dict[str, list[float]] = {}
    for log in logs:
        for f in _believed(log).fixtures:
            if f.kind == "plate":
                tops.setdefault(f.name, []).append(f.top_z)
        for name, vals in support_observations([log]).items():
            if vals:
                votes.setdefault(name, []).append(float(np.quantile(vals, 0.25)))
    updated = ReconModel(dict(recon.bias), dict(recon.corrections))
    report = {}
    for name, vals in votes.items():
        est = float(np.median(vals))
        cur = float(np.median(tops[name]))
        mad = float(np.median(np.abs(np.asarray(vals) - est)))
        enough = len(vals) >= min_episodes and mad <= tol and est > 0 and cur > 1e-4
        changed = enough and abs(est - cur) > max(tol, 2 * mad)
        report[name] = {"estimated": est, "believed": cur, "episodes": len(vals),
                        "mad": mad, "changed": bool(changed)}
        if changed:
            updated.corrections[name] = updated.corrections.get(name, 1.0) * est / cur
    return updated, report


def replay_audits(logs: list, old: ReconModel, new: ReconModel, scale: float) -> list:
    """Re-evaluate recorded observations with corrected *belief* geometry.

    A relative reconstruction is applied to the old belief, never to log.scene
    (which contains the privileged physical geometry). Human labels stay fixed.
    The original executed plan stays fixed too: its labels cannot be silently
    reused as labels of a different, repaired plan.
    """
    from .simreal import _believed, _planned

    categories = set(old.bias) | set(old.corrections) | set(new.bias) | set(new.corrections)
    relative = ReconModel({c: new.factor(c) / old.factor(c) for c in categories})
    replayed = []
    for log in logs:
        scene = relative.apply(_believed(log))
        ver = Verifier(scene, VerifierConfig(tol_scale=scale))
        graph = _planned(log)
        pred = np.array([ver.satisfied(state_from_log(log.grip[t], log.obj_obs[t], t), graph)
                         for t in range(log.T)], dtype=bool).reshape(log.sat_ver.shape)
        replayed.append(replace(log, meta={**log.meta, "belief": scene.to_dict()}, sat_ver=pred))
    return replayed


def audit_accuracy(logs: list, scale: float, stride: int) -> float | None:
    """Balanced accuracy on audited keyframes; missing classes are not invented."""
    from .simreal import _believed, _planned

    tp = tn = fp = fn = 0
    for log in logs:
        truth = log.sat_plan if log.sat_plan is not None else log.sat_true
        ver = Verifier(_believed(log), VerifierConfig(tol_scale=scale))
        graph = _planned(log)
        for t in sorted(set(range(0, log.T, stride)) | {log.T - 1}):
            pred = np.asarray(ver.satisfied(state_from_log(log.grip[t], log.obj_obs[t], t), graph))
            label = truth[t]
            tp += int((pred & label).sum())
            tn += int((~pred & ~label).sum())
            fp += int((pred & ~label).sum())
            fn += int((~pred & label).sum())
    rates = ([tp / (tp + fn)] if tp + fn else []) + ([tn / (tn + fp)] if tn + fp else [])
    return float(np.mean(rates)) if rates else None


def split_audits(logs: list) -> tuple[list, list]:
    """Split entire episodes, stratifying by family; no keyframe leakage."""
    by_family: dict[str, list] = {}
    for log in logs:
        by_family.setdefault(log.family, []).append(log)
    fit, check = [], []
    for family in sorted(by_family):
        group = sorted(by_family[family], key=lambda log: (log.seed, log.key))
        if len(group) == 1:
            fit += group
        else:
            n_check = max(1, len(group) // 3)
            check += group[:n_check]
            fit += group[n_check:]
    return fit, check


def cascade_repairs(b: Belief, ev: dict, cfg: SimRealConfig, budget: float = math.inf) -> dict:
    """Repair geometry, replay evidence, then calibrate and correct the plan.

    Attempts (including rejected/no-op fits) cost budget. A repair is committed
    only when it changes the belief; verifier fits must improve a separate set of
    audited episodes. No additional physical rollout or annotation is hidden here.
    """
    from .simreal import audit_test, graph_repair, verifier_recal

    before = ReconModel(dict(b.recon.bias), dict(b.recon.corrections))
    attempts, committed, reports = [], [], {}
    spent = 0.0

    def charge(name: str) -> bool:
        nonlocal spent
        cost = cfg.repair_costs[name]
        if len(attempts) >= cfg.cascade_max_repairs or spent + cost > budget + 1e-9:
            return False
        attempts.append(name)
        spent += cost
        return True

    candidate, geo = robust_geometry_candidate(b.recon, ev["real_logs"], cfg.geometry_min_episodes)
    reports["geometry_report"] = geo
    if any(v["changed"] for v in geo.values()) and charge("geometry"):
        b.recon = candidate
        committed.append("geometry")
    # Reuse audits from a flagged family only when this round corrected all of
    # its observed supports; other scene mismatches remain conservatively excluded.
    repaired_names = {n for n, v in geo.items() if v["changed"] and "geometry" in committed}
    unresolved = []
    from .simreal import _believed
    for family in ev["flagged"]:
        names = {f.name for l in ev["real_logs"] if l.family == family
                 for f in _believed(l).fixtures if f.kind == "plate"}
        if not names or not names <= repaired_names:
            unresolved.append(family)
    usable = [l for l in ev["audited"] if l.family not in unresolved]
    if cfg.cascade_replay:
        usable = replay_audits(usable, before, b.recon, b.judge_scale)
    audit = audit_test(usable)
    reports["replayed_audit"] = audit
    fit, check = split_audits(usable)
    fit_audit, check_audit = audit_test(fit), audit_test(check)
    # Observations collected before scene repair can contain no discriminating
    # verifier evidence. Diagnose that before paying for a fit. Require misreads
    # in separate fit/check episodes so acceptance has independent support.
    eligible = (fit_audit["judge"] > 0 and check_audit["judge"] > 0
                and fit_audit["judge"] + check_audit["judge"] >= cfg.min_evidence)
    reports["calibration_gate"] = {"fit_misreads": fit_audit["judge"],
                                    "check_misreads": check_audit["judge"], "eligible": eligible}
    if eligible and len(fit) >= cfg.min_evidence and check and charge("verifier"):
        candidate_scale, report = verifier_recal(fit, b.judge_scale, cfg.keyframe_stride)
        old_score = audit_accuracy(check, b.judge_scale, cfg.keyframe_stride)
        new_score = audit_accuracy(check, candidate_scale, cfg.keyframe_stride)
        accepted = old_score is not None and new_score is not None and new_score > old_score + cfg.validation_margin
        reports["verifier_report"] = {"chosen": candidate_scale, "accepted": accepted,
                                       "before": old_score, "after": new_score,
                                       "fit_episodes": len(fit), "check_episodes": len(check),
                                       "frames": report.get("frames")}
        if accepted:
            b.judge_scale = candidate_scale
            committed.append("verifier")
    # The original plan's audited labels remain valid after geometry replay.
    if audit["plan"] >= cfg.min_evidence:
        new_graph, report = graph_repair(b.grounding, usable)
        if new_graph != b.grounding and charge("graph"):
            b.grounding = new_graph
            committed.append("graph")
            reports["graph_report"] = report
    return {"chosen": committed, "repair_committed": committed, "repair_attempts": attempts, "repair_cost": spent,
            "cascade_unresolved": unresolved, **reports}
