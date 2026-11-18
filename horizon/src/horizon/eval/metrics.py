"""Statistics for the paper's figures: Wilson intervals, horizon-decay fits, bootstrap CIs."""

from __future__ import annotations

import math

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def fit_horizon_decay(H: np.ndarray, k: np.ndarray, n: np.ndarray) -> dict:
    """Binomial MLE of S(H) = S0 * exp(-alpha * H)  (hypothesis H1).

    Returns alpha, S0, the per-stage success p = exp(-alpha) and a pseudo-R^2 of the fit
    on observed proportions.
    """
    from scipy.optimize import minimize

    H = np.asarray(H, float)
    k = np.asarray(k, float)
    n = np.asarray(n, float)

    def nll(th):
        log_s0, alpha = th
        s = np.clip(np.exp(log_s0 - alpha * H), 1e-6, 1 - 1e-6)
        return -np.sum(k * np.log(s) + (n - k) * np.log(1 - s))

    best = None
    for a0 in (0.01, 0.1, 0.3):
        r = minimize(nll, x0=[0.0, a0], method="L-BFGS-B", bounds=[(-3.0, 0.0), (-0.5, 3.0)])
        if best is None or r.fun < best.fun:
            best = r
    log_s0, alpha = best.x
    p = k / np.maximum(n, 1)
    pred = np.exp(log_s0 - alpha * H)
    ss_res = float(np.sum((p - pred) ** 2))
    ss_tot = float(np.sum((p - p.mean()) ** 2)) or 1e-12
    return {"alpha": float(alpha), "S0": float(math.exp(log_s0)), "per_stage": float(math.exp(-alpha)),
            "r2": 1 - ss_res / ss_tot, "nll": float(best.fun)}


def bootstrap_alpha(H: np.ndarray, outcomes: list[np.ndarray], n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% CI of alpha by resampling episodes within each horizon."""
    rng = np.random.default_rng(seed)
    alphas = []
    for _ in range(n_boot):
        ks, ns = [], []
        for o in outcomes:
            s = rng.choice(o, size=len(o), replace=True)
            ks.append(s.sum())
            ns.append(len(s))
        alphas.append(fit_horizon_decay(H, np.array(ks), np.array(ns))["alpha"])
    return float(np.percentile(alphas, 2.5)), float(np.percentile(alphas, 97.5))
