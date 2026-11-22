"""Twin calibration from real deployment logs.

This is deliberately small: estimate the support height from placements and apply the delta to the
reconstruction bias. The full robust estimator lives in ``horizon.rsi.simreal.geometry_repair``;
this module is the hardware-side entry point.
"""
from __future__ import annotations

import copy

import numpy as np


def median_support(zs: list[float]) -> float:
    """Robust support height from multiple placement observations."""
    if not zs:
        raise ValueError("need at least one support-height observation")
    return float(np.median(np.asarray(zs, dtype=float)))


def update_recon_bias(recon: dict, plate_delta: float, tray_delta: float) -> dict:
    """Add deltas to the reconstruction bias; returns a new dict (input is not mutated).

    ``recon['bias']['plate']`` / ``['tray']`` follow ``ReconModel.from_dict``. The sign convention
    is the caller's: delta = measured - believed.
    """
    out = copy.deepcopy(recon)
    bias = dict(out.get("bias") or {})
    bias["plate"] = float(bias.get("plate", 0.0)) + float(plate_delta)
    bias["tray"] = float(bias.get("tray", 0.0)) + float(tray_delta)
    out["bias"] = bias
    return out


def placements_from_log(rows: list[dict], obj_index: int = 0, min_z: float = 0.0) -> list[float]:
    """Object-centre heights at successful placement (a simple proxy for support height)."""
    return [float(r["obj_pos"][obj_index][2]) for r in rows
            if r.get("success") and float(r["obj_pos"][obj_index][2]) >= min_z]
