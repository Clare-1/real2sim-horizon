"""Explicit task banks for development and untouched confirmation evaluation.

Legacy configurations retain their old task bank. New runs validate disjoint
million-seed namespaces before any rollout and save concrete evaluation manifests.
"""
from __future__ import annotations

import hashlib
import json

SEED_SPAN = 1_000_000
FAMILY_IDS = {"pack": 0, "sort": 1, "stack": 2}


def grid_records(base: int, families, H_values, n_per: int, *, stable_ids: bool = True) -> list[dict]:
    if not 1 <= n_per < 1000 or any(not 1 <= H < 100 for H in H_values):
        raise ValueError("Evaluation grid exceeds its seed namespace")
    if len(set(families)) != len(families) or len(set(H_values)) != len(H_values):
        raise ValueError("Evaluation grid contains duplicates")
    records = []
    for fi, family in enumerate(families):
        if family not in FAMILY_IDS:
            raise ValueError(f"Unknown task family: {family}")
        family_id = FAMILY_IDS[family] if stable_ids else fi
        for H in H_values:
            for i in range(n_per):
                task_seed = base + family_id * 100_000 + H * 1000 + i
                records.append({"family": family, "H": H, "task_seed": task_seed,
                                "rollout_seed": task_seed + 17})
    return records


def evaluation_manifest(cfg, dist, n_per: int, curve_n_per: int) -> dict:
    legacy = cfg.evaluation_split == "legacy"
    curve_base = cfg.eval_seed_base if cfg.curve_seed_base is None else cfg.curve_seed_base
    feedback_base = dist.seed_base if cfg.feedback_seed_base is None else cfg.feedback_seed_base
    if not legacy:
        if cfg.curve_seed_base is None or cfg.feedback_seed_base is None:
            raise ValueError("Explicit evaluation requires curve and feedback namespaces")
        banks = {"training": dist.seed_base, "feedback_audit": feedback_base,
                 "development_curve": curve_base, "evaluation": cfg.eval_seed_base}
        if cfg.evaluation_split == "development":
            banks["reserved_confirmation"] = cfg.confirmation_seed_base
        elif cfg.eval_seed_base != cfg.confirmation_seed_base:
            raise ValueError("Confirmation evaluation must use its declared reserved bank")
        items = list(banks.items())
        for i, (name, base) in enumerate(items):
            if not isinstance(base, int) or base < 0:
                raise ValueError(f"Invalid seed namespace: {name}")
            for other, start in items[:i]:
                if max(base, start) < min(base + SEED_SPAN, start + SEED_SPAN):
                    raise ValueError(f"Overlapping task namespaces: {name} / {other}")
            if name not in ("training", "feedback_audit") and max(base, 10_000_000) < min(base + SEED_SPAN, 11_000_000):
                raise ValueError("Explicit evaluation overlaps the previously viewed legacy test bank")
    else:
        banks = {"training": dist.seed_base, "feedback_audit": feedback_base,
                 "development_curve": curve_base, "evaluation": cfg.eval_seed_base}
    manifest = {"split": cfg.evaluation_split, "namespace_span": SEED_SPAN, "banks": banks,
                "evaluation": grid_records(cfg.eval_seed_base, dist.families, dist.H_values, n_per, stable_ids=not legacy),
                "curve": grid_records(curve_base, dist.families, dist.H_values, curve_n_per, stable_ids=not legacy),
                "confirmation_executed": cfg.evaluation_split == "confirmation"}
    manifest["sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest
