"""Twin update for LIBERO: turn first-frame object positions into a layout prior.

No privileged layout parameters are read. The caller passes object xy positions estimated from the
first frame (or a detector); the prior is a per-object mean/std used to resample the twin.
"""
from __future__ import annotations

import copy

import numpy as np


def layout_prior(points: list[np.ndarray]) -> dict:
    """``points``: one (K, 2) array per deployment; returns per-object mean/std."""
    if not points:
        raise ValueError("need at least one first-frame layout")
    arr = np.stack([np.asarray(p, dtype=float).reshape(-1, 2) for p in points], 0)
    return {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(), "n": int(len(arr))}


def apply_layout_prior(scene: dict, prior: dict) -> dict:
    """Write the prior into a scene payload without mutating the input."""
    out = copy.deepcopy(scene)
    out["layout_prior"] = {"mean": prior["mean"], "std": prior["std"], "n": prior.get("n", 0)}
    return out
