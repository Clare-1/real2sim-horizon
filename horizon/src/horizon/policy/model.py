"""BC policy network: torch for training, a dependency-free numpy forward pass for rollouts.

Rollout workers only need :class:`NumpyMLP`, so evaluation and RSI regeneration scale
across processes without importing torch in each worker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ACT_DIM = 5  # dx, dy, dz, dyaw, gripper (env action)
OUT_DIM = 6  # dx, dy, dz, dyaw (tanh), close logit, pre-shape opening command (tanh, used when not closing)


@dataclass
class TrainConfig:
    hidden: int = 512
    layers: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 2048
    steps: int = 12000
    grip_weight: float = 0.5
    seed: int = 0
    device: str = "auto"
    log_every: int = 2000
    normalization: str = "legacy"  # legacy | constant_safe (zero-variance inputs use unit scale)


def input_normalization(X: np.ndarray, mode: str = "legacy") -> tuple[np.ndarray, np.ndarray]:
    """Fit on training data only; keep historical checkpoints/configs reproducible.

    A padded neighbour slot can be constant during short-task training and become
    occupied on a longer task. Dividing it by the numerical epsilon turns ordinary
    metre-scale coordinates into hundreds of standard deviations. Unit scale for
    exactly constant coordinates avoids this amplification without hiding the new
    obstacle or fitting statistics on evaluation data. It does not learn its effect.
    """
    if mode not in {"legacy", "constant_safe"}:
        raise ValueError(f"unknown input normalization: {mode}")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
        raise ValueError("normalization requires a nonempty finite feature matrix")
    mu, sd = X.mean(0), X.std(0) + 1e-3
    if mode == "constant_safe":
        constant = np.ptp(X, axis=0) == 0
        # Use the exact value: float32 summation of a repeated nonzero value can
        # otherwise leave a spurious training residual in a constant coordinate.
        mu[constant], sd[constant] = X[0, constant], 1.0
    return mu, sd


def _gelu_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


class NumpyMLP:
    """Forward pass matching ``TorchMLP`` (Linear -> LayerNorm -> GELU(tanh)) x L -> Linear."""

    def __init__(self, params: dict):
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items() if isinstance(v, np.ndarray)}
        self.layers = int(params["n_layers"])
        self.mu = self.p["x_mu"]
        self.sd = self.p["x_sd"]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = (np.asarray(x, dtype=np.float32) - self.mu) / self.sd
        for l in range(self.layers):
            h = h @ self.p[f"w{l}"].T + self.p[f"b{l}"]
            m = h.mean(-1, keepdims=True)
            v = ((h - m) ** 2).mean(-1, keepdims=True)
            h = (h - m) / np.sqrt(v + 1e-5) * self.p[f"g{l}"] + self.p[f"be{l}"]
            h = _gelu_np(h)
        out = h @ self.p["wo"].T + self.p["bo"]
        return out

    def act(self, x: np.ndarray) -> np.ndarray:
        out = self(x[None])[0]
        a = np.empty(ACT_DIM, dtype=np.float32)
        a[:4] = np.tanh(out[:4])
        a[4] = 1.0 if out[4] > 0 else float(np.tanh(out[5]))
        return a


def train_bc(X: np.ndarray, Y: np.ndarray, cfg: TrainConfig | None = None, W: np.ndarray | None = None,
             verbose: bool = True) -> tuple[dict, dict]:
    """Behaviour cloning. Returns (numpy params for :class:`NumpyMLP`, training stats)."""
    import torch
    import torch.nn as nn

    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    dev = cfg.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    mu, sd = input_normalization(X, cfg.normalization)
    d_in = X.shape[1]

    class TorchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            mods = []
            d = d_in
            for _ in range(cfg.layers):
                mods += [nn.Linear(d, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(approximate="tanh")]
                d = cfg.hidden
            self.body = nn.Sequential(*mods)
            self.head = nn.Linear(d, OUT_DIM)

        def forward(self, x):
            return self.head(self.body(x))

    net = TorchMLP().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps))))
    Xt = torch.from_numpy((X - mu) / sd).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    Wt = torch.from_numpy(np.asarray(W, dtype=np.float32)).to(dev) if W is not None else None
    n = Xt.shape[0]
    bce = nn.BCEWithLogitsLoss(reduction="none")
    stats = {"loss": [], "n": int(n), "device": dev,
             "normalization": cfg.normalization,
             "constant_input_indices": np.flatnonzero(np.ptp(X, axis=0) == 0).tolist()}
    for step in range(cfg.steps):
        idx = torch.from_numpy(np_rng.integers(0, n, size=min(cfg.batch, n))).to(dev)
        x, y = Xt[idx], Yt[idx]
        out = net(x)
        l_cont = ((torch.tanh(out[:, :4]) - y[:, :4]) ** 2).mean(-1)
        close = (y[:, 4] > 0).float()
        l_grip = bce(out[:, 4], close)
        l_open = (1 - close) * (torch.tanh(out[:, 5]) - y[:, 4]) ** 2
        loss = l_cont + cfg.grip_weight * (l_grip + l_open)
        if Wt is not None:
            w = Wt[idx]
            loss = (loss * w).sum() / w.sum()
        else:
            loss = loss.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            stats["loss"].append((step, float(loss.item())))
            if verbose:
                print(f"    bc step {step:6d} loss {loss.item():.4f}", flush=True)
    params: dict = {"x_mu": mu, "x_sd": sd, "n_layers": cfg.layers}
    net = net.cpu()
    lin = [m for m in net.body if isinstance(m, nn.Linear)]
    ln = [m for m in net.body if isinstance(m, nn.LayerNorm)]
    for l in range(cfg.layers):
        params[f"w{l}"] = lin[l].weight.detach().numpy().copy()
        params[f"b{l}"] = lin[l].bias.detach().numpy().copy()
        params[f"g{l}"] = ln[l].weight.detach().numpy().copy()
        params[f"be{l}"] = ln[l].bias.detach().numpy().copy()
    params["wo"] = net.head.weight.detach().numpy().copy()
    params["bo"] = net.head.bias.detach().numpy().copy()
    # sanity: numpy forward matches torch
    with torch.no_grad():
        xs = torch.from_numpy((X[:64] - mu) / sd)
        ref = net(xs).numpy()
    got = NumpyMLP(params)(X[:64])
    stats["np_torch_maxdiff"] = float(np.abs(ref - got).max())
    return params, stats
