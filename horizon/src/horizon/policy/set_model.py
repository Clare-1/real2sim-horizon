"""Order-aware attention policy for the flat (no task graph) ablation.

A plain MLP over the concatenated 113-d flat features has to learn "find the first
unfinished object" from scratch and fails completely. This model gives the flat baseline
a fair inductive bias without giving it System 2: each subgoal slot becomes a token
(10 features + a learned order embedding), a query built from proprioception scores the
tokens, and the attention-pooled token is decoded into an action together with the
proprioceptive embedding. Progress must still be inferred from the raw slot features.

Layout of the flat feature vector (see :meth:`TaskFeaturizer.flat`):
  [0:9] proprio + travel height + any-held | [9:109] 10 slots x 10 | [109:113] family + H.
"""

from __future__ import annotations

import math

import numpy as np

from .features import FLAT_SLOT, MAX_SUBGOALS
from .model import ACT_DIM, OUT_DIM, TrainConfig, _gelu_np

P_DIM = 9
META_DIM = 4
SLOT_START = P_DIM
SLOT_END = P_DIM + FLAT_SLOT * MAX_SUBGOALS


def split_flat(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """-> proprio+meta (n, 13), slot tokens (n, 10, 10), slot mask (n, 10)."""
    prop = np.concatenate([X[:, :P_DIM], X[:, SLOT_END:]], axis=1)
    tok = X[:, SLOT_START:SLOT_END].reshape(-1, MAX_SUBGOALS, FLAT_SLOT)
    mask = (np.abs(tok).sum(-1) > 0).astype(np.float32)
    return prop, tok, mask, X


def _ln(h: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = h.mean(-1, keepdims=True)
    v = ((h - m) ** 2).mean(-1, keepdims=True)
    return (h - m) / np.sqrt(v + 1e-5) * g + b


class NumpySetPolicy:
    def __init__(self, params: dict):
        self.p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items() if isinstance(v, np.ndarray)}

    def __call__(self, X: np.ndarray) -> np.ndarray:
        p = self.p
        prop, tok, mask, _ = split_flat(np.asarray(X, dtype=np.float32))
        prop = (prop - p["p_mu"]) / p["p_sd"]
        tok = (tok - p["t_mu"]) / p["t_sd"] * mask[..., None]
        hp = _gelu_np(_ln(prop @ p["pw0"].T + p["pb0"], p["pg0"], p["pbe0"]))
        ht = _gelu_np(_ln(tok @ p["tw0"].T + p["tb0"], p["tg0"], p["tbe0"]))
        ht = ht + p["order"][None]
        ht = _gelu_np(_ln(ht @ p["tw1"].T + p["tb1"], p["tg1"], p["tbe1"]))
        q = hp @ p["qw"].T + p["qb"]
        k = ht @ p["kw"].T + p["kb"]
        s = (k * q[:, None, :]).sum(-1) / math.sqrt(k.shape[-1])
        s = np.where(mask > 0, s, -1e9)
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s) * mask
        a = a / np.maximum(a.sum(-1, keepdims=True), 1e-9)
        pooled = (a[:, :, None] * ht).sum(1)
        h = np.concatenate([hp, pooled], axis=1)
        for l in range(2):
            h = _gelu_np(_ln(h @ p[f"hw{l}"].T + p[f"hb{l}"], p[f"hg{l}"], p[f"hbe{l}"]))
        return h @ p["wo"].T + p["bo"]

    def act(self, x: np.ndarray) -> np.ndarray:
        out = self(x[None])[0]
        a = np.empty(ACT_DIM, dtype=np.float32)
        a[:4] = np.tanh(out[:4])
        a[4] = 1.0 if out[4] > 0 else float(np.tanh(out[5]))
        return a


def train_set_policy(X: np.ndarray, Y: np.ndarray, cfg: TrainConfig | None = None, d: int = 192,
                     verbose: bool = True) -> tuple[dict, dict]:
    import torch
    import torch.nn as nn

    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = "cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else ("cpu" if cfg.device == "auto" else cfg.device)
    prop, tok, mask, _ = split_flat(np.asarray(X, dtype=np.float32))
    p_mu, p_sd = prop.mean(0), prop.std(0) + 1e-3
    flat_tok = tok.reshape(-1, FLAT_SLOT)[mask.reshape(-1) > 0]
    t_mu, t_sd = flat_tok.mean(0), flat_tok.std(0) + 1e-3

    def block(i, o):
        return nn.ModuleDict({"lin": nn.Linear(i, o), "ln": nn.LayerNorm(o)})

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.p0 = block(prop.shape[1], d)
            self.t0 = block(FLAT_SLOT, d)
            self.t1 = block(d, d)
            self.order = nn.Parameter(torch.randn(MAX_SUBGOALS, d) * 0.1)
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.h = nn.ModuleList([block(2 * d, cfg.hidden), block(cfg.hidden, cfg.hidden)])
            self.out = nn.Linear(cfg.hidden, OUT_DIM)
            self.act = nn.GELU(approximate="tanh")

        def f(self, b, x):
            return self.act(b["ln"](b["lin"](x)))

        def forward(self, pr, tk, mk):
            hp = self.f(self.p0, pr)
            ht = self.f(self.t0, tk) + self.order[None]
            ht = self.f(self.t1, ht)
            q = self.q(hp)
            k = self.k(ht)
            s = (k * q[:, None, :]).sum(-1) / math.sqrt(d)
            s = s.masked_fill(mk <= 0, -1e9)
            a = torch.softmax(s, -1) * mk
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)
            pooled = (a[..., None] * ht).sum(1)
            h = torch.cat([hp, pooled], 1)
            for b in self.h:
                h = self.f(b, h)
            return self.out(h)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps))))
    P = torch.from_numpy((prop - p_mu) / p_sd).to(dev)
    Tk = torch.from_numpy((tok - t_mu) / t_sd * mask[..., None]).to(dev)
    M = torch.from_numpy(mask).to(dev)
    Yt = torch.from_numpy(np.asarray(Y, dtype=np.float32)).to(dev)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    n = P.shape[0]
    stats = {"loss": [], "n": int(n), "device": dev}
    for step in range(cfg.steps):
        idx = torch.from_numpy(rng.integers(0, n, size=min(cfg.batch, n))).to(dev)
        out = net(P[idx], Tk[idx], M[idx])
        y = Yt[idx]
        close = (y[:, 4] > 0).float()
        loss = ((torch.tanh(out[:, :4]) - y[:, :4]) ** 2).mean(-1) + cfg.grip_weight * (
            bce(out[:, 4], close) + (1 - close) * (torch.tanh(out[:, 5]) - y[:, 4]) ** 2)
        loss = loss.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            stats["loss"].append((step, float(loss.item())))
            if verbose:
                print(f"    set step {step:6d} loss {loss.item():.4f}", flush=True)
    net = net.cpu()
    P_ = {"arch": "set", "p_mu": p_mu, "p_sd": p_sd, "t_mu": t_mu, "t_sd": t_sd,
          "order": net.order.detach().numpy().copy()}

    def put(prefix, b, suffix=""):
        P_[f"{prefix}w{suffix}"] = b["lin"].weight.detach().numpy().copy()
        P_[f"{prefix}b{suffix}"] = b["lin"].bias.detach().numpy().copy()
        P_[f"{prefix}g{suffix}"] = b["ln"].weight.detach().numpy().copy()
        P_[f"{prefix}be{suffix}"] = b["ln"].bias.detach().numpy().copy()

    put("p", net.p0, "0")
    put("t", net.t0, "0")
    put("t", net.t1, "1")
    for l, b in enumerate(net.h):
        put("h", b, str(l))
    P_["qw"], P_["qb"] = net.q.weight.detach().numpy().copy(), net.q.bias.detach().numpy().copy()
    P_["kw"], P_["kb"] = net.k.weight.detach().numpy().copy(), net.k.bias.detach().numpy().copy()
    P_["wo"], P_["bo"] = net.out.weight.detach().numpy().copy(), net.out.bias.detach().numpy().copy()
    with torch.no_grad():
        ref = net(P[:64].cpu(), Tk[:64].cpu(), M[:64].cpu()).numpy()
    got = NumpySetPolicy(P_)(np.asarray(X[:64], dtype=np.float32))
    stats["np_torch_maxdiff"] = float(np.abs(ref - got).max())
    return P_, stats
