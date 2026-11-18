"""Diagnostic: which twin-test signal separates scene errors from the rest when the judge is strict?

Deploys a policy (with the robot's initial, wrong beliefs) on the real scene and in its twin and
compares, per family, judged success and judged progress (fraction of planned subgoals the robot's
own judge confirms at the end).
Usage: python scripts/diag/twin_signal.py results/simreal3/sim_only_s0_policy.npz [workers] [episodes]
"""

from __future__ import annotations

import sys

import numpy as np

from horizon.datagen.generator import TaskDistribution
from horizon.datagen.parallel import Runner
from horizon.rsi.loop import load_params
from horizon.rsi.simreal import Belief, SimRealConfig, _job, judged_done


def main() -> None:
    params = load_params(sys.argv[1])
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    cfg = SimRealConfig(name="diag", judge_scale=0.4, grounding={"confuse": {"green": "blue", "blue": "green"}})
    b = Belief.initial(cfg)
    rng = np.random.default_rng(5)
    tasks = [TaskDistribution().sample(rng) for _ in range(n)]
    out = {}
    with Runner(W, policy={"params": params}) as runner:
        for real in (True, False):
            jobs = [_job(f, H, ts, 1000 + i, cfg, b, real=real, controller="hier", pert=0.1, want_samples=False,
                         keep_log=True) for i, (f, H, ts) in enumerate(tasks)]
            out[real] = [x.log for x in runner.map(jobs)]
    for fam in ("pack", "sort", "stack"):
        row = []
        for real in (True, False):
            logs = [l for l in out[real] if l is not None and l.family == fam]
            succ = np.mean([judged_done(l) for l in logs])
            prog = np.mean([np.asarray(l.sat_ver[-1], bool).mean() for l in logs])
            reach = np.mean([max(0, int(l.k_sel.max())) / l.H if (l.k_sel >= 0).any() else 1.0 for l in logs])
            row.append((succ, prog, reach, len(logs)))
        (rs, rp, rr, nr), (ss, sp, sr, ns) = row
        print(f"{fam}: judged success real {rs:.2f} sim {ss:.2f} (gap {ss - rs:+.2f}) | judged progress real {rp:.2f} "
              f"sim {sp:.2f} (gap {sp - rp:+.2f}) | furthest subgoal real {rr:.2f} sim {sr:.2f} | n={nr}", flush=True)


if __name__ == "__main__":
    main()
