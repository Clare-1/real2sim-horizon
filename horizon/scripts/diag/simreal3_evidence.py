"""Validate the three-error evidence chain with a fixed, competent policy (no training in between).

Deploys a trained policy (e.g. main_v1 DAgger) in the deployment world with all three injected
errors (support heights, strict judge, swapped bins), prints the twin test, the audit evidence and
the shares, applies the routed repair, and deploys again - the evidence should move from one error
source to the next and vanish once all three are repaired.
Usage: python scripts/diag/simreal3_evidence.py results/main_v1/dagger_beta_s0_policy.npz [workers] [rounds]
"""

from __future__ import annotations

import sys

import numpy as np

from horizon.datagen.generator import TaskDistribution
from horizon.datagen.parallel import Runner
from horizon.rsi.loop import load_params
from horizon.rsi.simreal import Belief, SimRealConfig, apply_repairs, deploy_and_attribute


def main() -> None:
    params = load_params(sys.argv[1])
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    cfg = SimRealConfig(name="diag", route="routed", judge_scale=0.4, grounding={"confuse": {"green": "blue", "blue": "green"}})
    b = Belief.initial(cfg)
    rng = np.random.default_rng(0)
    with Runner(W, policy={"params": params}) as runner:
        for r in range(1, rounds + 1):
            ev = deploy_and_attribute(runner, cfg, b, TaskDistribution(), rng)
            fam = {f: np.mean([l.success for l in ev["real_logs"] if l.family == f]) for f in ("pack", "sort", "stack")}
            print(f"round {r}: real success {ev['real_success']:.2f} ({', '.join(f'{f} {v:.2f}' for f, v in fam.items())}); "
                  f"judged done {ev['real_judged_done']:.2f}; active {ev['active']}", flush=True)
            print("   twin (real, sim, n):", {f: (round(v["real"], 2), round(v["sim"], 2), v["n"]) for f, v in ev["twin"].items()},
                  "flagged", ev["flagged"])
            print("   audit:", ev["audit"])
            print("   shares:", {k: round(v, 3) for k, v in ev["shares"].items()}, "-> routed choice", ev["chosen"])
            rep = apply_repairs(b, ev["chosen"], ev, cfg)
            print("   repair:", rep, "-> belief", {"corrections": {k: round(v, 2) for k, v in b.recon.corrections.items()},
                                                  "judge": b.judge_scale, "grounding": b.grounding.confuse}, flush=True)


if __name__ == "__main__":
    main()
