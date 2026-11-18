"""Diagnostic: where do spurious support-height observations come from once the scene is already right?

Deploys a policy with the (already corrected) belief and splits the observations by signal:
(1) resting objects at the end, (2) stalled descents while holding an object.
Usage: python scripts/diag/geometry_obs.py results/main_v1/dagger_beta_s0_policy.npz [workers]
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from horizon.datagen.generator import TaskDistribution
from horizon.datagen.parallel import Runner
from horizon.real2sim.recon_error import ReconModel, single_view_bias
from horizon.rsi.loop import load_params
from horizon.rsi.simreal import Belief, SimRealConfig, _believed, deploy_and_attribute


def main() -> None:
    params = load_params(sys.argv[1])
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cfg = SimRealConfig(name="diag", route="routed", judge_scale=1.4, real_episodes=60, twin_episodes=10)
    b = Belief(ReconModel(single_view_bias().bias, {"tray": 3.25, "plate": 3.36}), 1.4, Belief.initial(cfg).grounding)
    with Runner(W, policy={"params": params}) as runner:
        ev = deploy_and_attribute(runner, cfg, b, TaskDistribution(), np.random.default_rng(1))
    rest, stall = defaultdict(list), defaultdict(list)
    for log in ev["real_logs"]:
        sc = _believed(log)
        half = np.array([o.half_height for o in sc.objects])
        P = log.obj_obs[:, :, :3].astype(float)
        G = log.grip.astype(float)
        T = len(G)
        for f in [f for f in sc.fixtures if f.kind == "plate"]:
            inside = (np.abs(P[:, :, 0] - f.pos[0]) < f.size[0] - 0.01) & (np.abs(P[:, :, 1] - f.pos[1]) < f.size[1] - 0.01)
            t = T - 1
            idx = [i for i in range(P.shape[1]) if inside[t, i] and int(log.held_est[t]) != i and G[t, 2] > P[t, i, 2] + half[i] + 0.03]
            if idx:
                i = min(idx, key=lambda j: P[t, j, 2] - half[j])
                rest[f.name].append(float(P[t, i, 2] - half[i]))
            for t in range(0, T - 4, 2):
                i = int(log.held_est[t])
                if i < 0 or not inside[t, i] or log.action[t, 2] > -0.3:
                    continue
                if abs(G[t + 3, 2] - G[t, 2]) < 0.001 and G[t, 2] < 0.2:
                    under = [j for j in range(P.shape[1]) if j != i and inside[t, j]
                             and np.linalg.norm(P[t, j, :2] - P[t, i, :2]) < 0.04 and P[t, j, 2] < P[t, i, 2]]
                    stall[f.name].append((float(P[t, i, 2] - half[i]), log.family, int(log.held_true[t]) == i, len(under) > 0,
                                          round(float(G[t, 2]), 3), round(float(log.action[t, 4]), 2)))
    for name in sorted(set(rest) | set(stall)):
        r = np.array(rest[name])
        print(f"{name}: resting n={len(r)} median {np.median(r) if len(r) else float('nan'):.4f} "
              f"pcts {np.percentile(r, [10, 50, 90]).round(4) if len(r) else '-'}")
        s = stall[name]
        z = np.array([x[0] for x in s])
        print(f"   stalls n={len(s)} median {np.median(z) if len(z) else float('nan'):.4f} pcts {np.percentile(z, [10, 50, 90]).round(4) if len(z) else '-'}")
        hi = [x for x in s if x[0] > 0.03]
        print(f"   stalls with bottom > 3 cm: {len(hi)}; truly held {sum(x[2] for x in hi)}; over another object {sum(x[3] for x in hi)}; "
              f"examples {hi[:6]}")


if __name__ == "__main__":
    main()
