"""How often do policy rollouts knock objects off the table, and how many relabelled states does that produce?"""

from __future__ import annotations

import numpy as np

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, generate_open_loop
from horizon.datagen.parallel import Runner
from horizon.policy.dataset import Dataset
from horizon.policy.model import TrainConfig
from horizon.rsi.loop import MethodConfig, eval_jobs, train_policy


def main() -> None:
    W = 8
    dist = TaskDistribution()
    with Runner(W) as r:
        blocks, _ = generate_open_loop(r, dist, 80000, OpenLoopConfig(), seed=0)
    params, _ = train_policy(Dataset(blocks), "hier", TrainConfig(steps=8000))
    mc = MethodConfig(name="diag", strategy="dagger")
    with Runner(W, policy={"params": params}) as r:
        logs = [x.log for x in r.map(eval_jobs(dist, mc, np.random.default_rng(0), 120, 1)) if x.log is not None]
    n_states = sum(int((l.valid & (l.k_true >= 0)).sum()) for l in logs)
    fell, after = 0, 0
    zmin = []
    for l in logs:
        z = l.obj_obs[:, :, 2]
        zmin.append(float(z.min()))
        bad = np.nonzero((z < -0.05).any(axis=1))[0]
        if len(bad):
            fell += 1
            ok = l.valid & (l.k_true >= 0)
            after += int(ok[bad[0]:].sum())
    print(f"episodes {len(logs)}, success {np.mean([l.success for l in logs]):.2f}")
    print(f"episodes with an object off the table: {fell}; relabellable states after the fall: {after} "
          f"of {n_states} ({after / max(1, n_states):.1%})")
    print("lowest object z per episode (5 smallest):", sorted(zmin)[:5])


if __name__ == "__main__":
    main()
