"""Why does DAgger collapse? Compare relabelled on-policy samples with expert demos."""

from __future__ import annotations

import numpy as np

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, generate_open_loop
from horizon.datagen.parallel import Runner
from horizon.datagen.rollout import PHASES
from horizon.monitor.attribution import attribute_auto, attribute_oracle
from horizon.policy.dataset import Dataset
from horizon.policy.features import HIER_DIM
from horizon.policy.model import TrainConfig
from horizon.rsi.loop import MethodConfig, eval_jobs, train_policy
from horizon.rsi.strategies import RoundInputs, plan_dagger


def main() -> None:
    W = 8
    dist = TaskDistribution()
    oc = OpenLoopConfig()
    with Runner(W) as r:
        blocks, _ = generate_open_loop(r, dist, 80000, oc, seed=0)
    base = Dataset(blocks)
    params, _ = train_policy(base, "hier", TrainConfig(steps=8000))
    mc = MethodConfig(name="diag", strategy="dagger")
    rng = np.random.default_rng(0)
    with Runner(W, policy={"params": params}) as r:
        res = r.map(eval_jobs(dist, mc, rng, 60, 1))
    logs = [x.log for x in res if x.log is not None]
    print("eval success", np.mean([l.success for l in logs]), "steps per log", np.mean([l.T for l in logs]))
    inp = RoundInputs(logs=logs, auto=[attribute_auto(l) for l in logs], oracle=[attribute_oracle(l) for l in logs],
                      budget=40000, rng=np.random.default_rng(1), dist=dist, open_cfg=oc)
    dag = plan_dagger(inp)[0].samples
    Xe, Ye = base.arrays("hier")
    Xd = np.concatenate([b.x_hier for b in dag])
    Yd = np.concatenate([b.y for b in dag])
    mu, sd = Xe.mean(0), Xe.std(0) + 1e-3
    z = np.abs((Xd - mu) / sd)
    print("dagger samples", len(Xd), "max |z| per dim (top 8):")
    top = np.argsort(-z.max(0))[:8]
    for d in top:
        print(f"   dim {d:2d}: expert std {sd[d]:.4f}  dagger std {Xd[:, d].std():.4f}  max|z| {z[:, d].max():.0f}  "
              f"frac |z|>8: {(z[:, d] > 8).mean():.4f}")
    sd_mix = np.concatenate([Xe, Xd]).std(0) + 1e-3
    print("std inflation from mixing (top 5):", sorted(((sd_mix[d] / sd[d]).round(1), d) for d in range(HIER_DIM))[-5:])
    print("gripper close frac: expert", (Ye[:, 4] > 0).mean().round(3), " dagger", (Yd[:, 4] > 0).mean().round(3))
    print("mean |dpos| label: expert", np.abs(Ye[:, :3]).mean().round(3), " dagger", np.abs(Yd[:, :3]).mean().round(3))
    # phase mix of relabelled states
    from collections import Counter
    ph = Counter()
    for log in logs:
        ok = log.valid & (log.k_true >= 0)
        ph.update(PHASES[p] for p in log.phase[ok])
    tot = sum(ph.values())
    print("expert phase at visited states:", {k: round(v / tot, 3) for k, v in ph.most_common(8)})


if __name__ == "__main__":
    main()


def phase2() -> None:
    """Train base + DAgger, then measure on-policy action bias and failure phases."""
    from collections import Counter

    W = 8
    dist = TaskDistribution()
    oc = OpenLoopConfig()
    with Runner(W) as r:
        blocks, _ = generate_open_loop(r, dist, 80000, oc, seed=0)
    base = Dataset(blocks)
    params0, _ = train_policy(base, "hier", TrainConfig(steps=8000))
    mc = MethodConfig(name="diag", strategy="dagger")
    with Runner(W, policy={"params": params0}) as r:
        logs0 = [x.log for x in r.map(eval_jobs(dist, mc, np.random.default_rng(0), 60, 1)) if x.log is not None]
    inp = RoundInputs(logs=logs0, auto=[None] * len(logs0), oracle=[None] * len(logs0), budget=40000,
                      rng=np.random.default_rng(1), dist=dist, open_cfg=oc)
    dag = plan_dagger(inp)[0].samples
    data = Dataset(list(blocks) + list(dag))
    params1, _ = train_policy(data, "hier", TrainConfig(steps=8000))
    for name, prm in (("base", params0), ("base+dagger", params1)):
        with Runner(W, policy={"params": prm}) as r:
            logs = [x.log for x in r.map(eval_jobs(dist, mc, np.random.default_rng(5), 60, 2)) if x.log is not None]
        ok = np.concatenate([l.valid & (l.k_true >= 0) for l in logs])
        A = np.concatenate([l.action for l in logs])[ok]
        E = np.concatenate([l.expert_action for l in logs])[ok]
        ph = np.concatenate([l.phase for l in logs])[ok]
        print(f"{name}: success {np.mean([l.success for l in logs]):.2f}; signed err (policy-expert) per dim "
              f"{(A[:, :4] - E[:, :4]).mean(0).round(3)}; close-mismatch {((A[:, 4] > 0) != (E[:, 4] > 0)).mean():.3f}")
        fin = Counter(PHASES[l.phase[-1]] for l in logs if not l.success)
        print("   final expert phase of failures:", dict(fin.most_common(6)))
        for p in ("approach", "rise", "descend_grasp", "close"):
            m = ph == [i for i, q in enumerate(PHASES) if q == p][0]
            if m.any():
                print(f"   {p:14s} n={m.sum():6d} signed err {(A[m, :4] - E[m, :4]).mean(0).round(3)} "
                      f"|err| {np.abs(A[m, :3] - E[m, :3]).mean():.3f} grip mismatch {((A[m, 4] > 0) != (E[m, 4] > 0)).mean():.3f}")


if __name__ == "__main__" and False:
    pass
