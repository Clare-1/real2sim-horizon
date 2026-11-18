"""Development repair controller with stable episode splits and replayed audit memory.

No evaluation outcomes or injected-error flags are used for repair decisions.
Check episodes are reused adaptively across rounds: this is a development acceptance
check, not an independent statistical confirmation set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

import numpy as np

from ..monitor.verifier import Verifier, VerifierConfig
from ..policy.features import state_from_log
from ..real2sim.recon_error import ReconModel
from .repair_cascade import replay_audits, robust_geometry_candidate


def select_feasible_greedy(shares, costs, budget, min_share, ratio=False):
    """Fill the remaining budget greedily, skipping unaffordable operators.

    The same evidence threshold and multi-operator allowance as cost_gated are
    used. Stable ties follow geometry/verifier/graph, with no oracle error flags.
    """
    names = [n for n in ("geometry", "verifier", "graph") if shares.get(n, 0) >= min_share]
    names.sort(key=lambda n: shares[n] / costs[n] if ratio else shares[n], reverse=True)
    chosen = []
    for name in names:
        if costs[name] <= budget + 1e-9:
            chosen.append(name)
            budget -= costs[name]
    return chosen


def episode_id(log):
    return f"{log.family}:{log.H}:{log.seed}:{log.key}"


def audit_partition(identity):
    """A fixed ~1/3 check assignment independent of outcomes and batch size."""
    return "check" if int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big") % 3 == 0 else "fit"


@dataclass
class AuditMemory:
    entries: dict = field(default_factory=dict)

    def observe(self, logs, recon, accumulate=True):
        if not accumulate:
            self.entries.clear()
        for log in logs:
            identity = episode_id(log)
            if identity in self.entries:
                # Keep the first observation; do not count repeated tasks as new labels.
                continue
            saved = ReconModel(dict(recon.bias), dict(recon.corrections))
            self.entries[identity] = (log, saved)

    def replay(self, current, scale, excluded=()):
        fit, check = [], []
        for identity, (log, original_recon) in self.entries.items():
            if log.family in excluded:
                continue
            replayed = replay_audits([log], original_recon, current, scale)[0]
            (check if audit_partition(identity) == "check" else fit).append(replayed)
        return fit, check


def keyframe_metrics(logs, scale, stride):
    """Audit-label confusion counts, including successful episodes' misreads."""
    from .simreal import _believed, _planned
    tp = tn = fp = fn = errors = frames = 0
    for log in logs:
        truth = log.sat_plan if log.sat_plan is not None else log.sat_true
        verifier = Verifier(_believed(log), VerifierConfig(tol_scale=scale))
        graph = _planned(log)
        any_error = False
        for t in sorted(set(range(0, log.T, stride)) | {log.T - 1}):
            pred = np.asarray(verifier.satisfied(state_from_log(log.grip[t], log.obj_obs[t], t), graph), dtype=bool)
            label = truth[t]
            tp += int((pred & label).sum())
            tn += int((~pred & ~label).sum())
            fp += int((pred & ~label).sum())
            fn += int((~pred & label).sum())
            any_error |= bool((pred != label).any())
            frames += 1
        errors += int(any_error)
    rates = ([tp / (tp + fn)] if tp + fn else []) + ([tn / (tn + fp)] if tn + fp else [])
    return {"balanced_accuracy": float(np.mean(rates)) if rates else None,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "frames": frames,
            "positive_labels": tp + fn, "negative_labels": tn + fp,
            "error_episodes": errors, "episodes": len(logs)}


def accept_candidate(before, after, margin):
    """Require both classes and improvement on check labels; no new false positives."""
    return (before["positive_labels"] > 0 and before["negative_labels"] > 0
            and before["balanced_accuracy"] is not None and after["balanced_accuracy"] is not None
            and after["balanced_accuracy"] > before["balanced_accuracy"] + margin
            and after["fp"] <= before["fp"])


def memory_repairs(b, ev, cfg, budget=math.inf, memory=None):
    from .simreal import _believed, audit_test, graph_repair, verifier_recal
    if memory is None:
        memory = AuditMemory()
    original = ReconModel(dict(b.recon.bias), dict(b.recon.corrections))
    memory.observe(ev["audited"], original, accumulate=cfg.audit_memory)
    attempts, committed, reports = [], [], {}
    spent = 0.0

    def charge(name):
        nonlocal spent
        cost = cfg.repair_costs[name]
        if len(attempts) >= cfg.cascade_max_repairs or spent + cost > budget + 1e-9:
            return False
        attempts.append(name)
        spent += cost
        return True

    candidate, geo = robust_geometry_candidate(b.recon, ev["real_logs"], cfg.geometry_min_episodes)
    supported = ev["shares"].get("geometry", 0) >= cfg.min_share
    reports["geometry_report"] = geo
    reports["geometry_supported"] = supported
    if supported and any(v["changed"] for v in geo.values()) and charge("geometry"):
        b.recon = candidate
        committed.append("geometry")
    repaired = {n for n, v in geo.items() if v["changed"] and "geometry" in committed}
    unresolved = []
    for family in ev["flagged"]:
        names = {f.name for log in ev["real_logs"] if log.family == family
                 for f in _believed(log).fixtures if f.kind == "plate"}
        if not names or not names <= repaired:
            unresolved.append(family)
    fit, check = memory.replay(b.recon, b.judge_scale, unresolved)
    fit_metrics = keyframe_metrics(fit, b.judge_scale, cfg.keyframe_stride)
    # Only the fit side determines whether there is a candidate worth fitting.
    eligible = len(fit) >= cfg.min_evidence and fit_metrics["error_episodes"] >= cfg.min_evidence and len(check) >= 2
    reports["memory_audit"] = {"accumulate": cfg.audit_memory, "stored": len(memory.entries),
        "fit_ids": [episode_id(log) for log in fit], "check_ids": [episode_id(log) for log in check],
        "fit_metrics": fit_metrics, "eligible": eligible, "unresolved": unresolved}
    if eligible and charge("verifier"):
        scale, fit_report = verifier_recal(fit, b.judge_scale, cfg.keyframe_stride)
        before = keyframe_metrics(check, b.judge_scale, cfg.keyframe_stride)
        after = keyframe_metrics(check, scale, cfg.keyframe_stride)
        supported_candidate = accept_candidate(before, after, cfg.validation_margin)
        accepted = supported_candidate if cfg.reject_verifier_candidate else True
        changed = accepted and scale != b.judge_scale
        reports["verifier_report"] = {"chosen": scale, "accepted": bool(accepted),
            "supported_candidate": bool(supported_candidate), "rejection_enabled": cfg.reject_verifier_candidate,
            "before": before, "after": after, "fit_episodes": len(fit), "check_episodes": len(check),
            "frames": fit_report.get("frames"), "changed": bool(changed)}
        if changed:
            b.judge_scale = scale
            committed.append("verifier")
    # Graph correction uses only this round's labels of its original executed plan.
    usable = replay_audits([log for log in ev["audited"] if log.family not in unresolved], original, b.recon, b.judge_scale)
    if audit_test(usable)["plan"] >= cfg.min_evidence:
        graph, report = graph_repair(b.grounding, usable)
        if graph != b.grounding and charge("graph"):
            b.grounding = graph
            committed.append("graph")
            reports["graph_report"] = report
    return {"chosen": committed, "repair_committed": committed, "repair_attempts": attempts,
            "repair_cost": spent, **reports}
