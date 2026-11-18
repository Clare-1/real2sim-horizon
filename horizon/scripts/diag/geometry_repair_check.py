"""Diagnostic: does the geometry repair keep a correct model correct, and fix a biased one?

Usage: python scripts/diag/geometry_repair_check.py results/main_v1/dagger_beta_s0_policy.npz [workers]
"""

from __future__ import annotations

import sys

import numpy as np

from horizon.datagen.generator import TaskDistribution
from horizon.datagen.parallel import Runner
from horizon.real2sim.recon_error import ReconModel, single_view_bias
from horizon.rsi.loop import load_params
from horizon.rsi.simreal import Belief, SimRealConfig, deploy_and_attribute, geometry_repair, support_observations


def main() -> None:
    params = load_params(sys.argv[1])
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cfg = SimRealConfig(name="diag", route="routed", judge_scale=1.4, real_episodes=60, twin_episodes=10)
    for label, corr in (("already correct", {"tray": 3.25, "plate": 3.36}), ("biased", {})):
        b = Belief(ReconModel(single_view_bias().bias, corr), 1.4, Belief.initial(cfg).grounding)
        with Runner(W, policy={"params": params}) as runner:
            ev = deploy_and_attribute(runner, cfg, b, TaskDistribution(), np.random.default_rng(1))
        obs = support_observations(ev["real_logs"])
        print(label, {k: (len(v), round(float(np.median(v)), 4)) for k, v in obs.items()}, flush=True)
        new, rep = geometry_repair(b.recon, ev["real_logs"])
        print("   repair ->", {k: round(v, 2) for k, v in new.corrections.items()},
              {k: (round(v["estimated"], 4), v["n"]) for k, v in rep.items()}, flush=True)


if __name__ == "__main__":
    main()
