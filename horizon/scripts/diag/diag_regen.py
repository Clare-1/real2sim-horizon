"""Diagnose regeneration throughput: one closed-loop round, per-request success/length/time."""

from __future__ import annotations

import time

import numpy as np

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, generate_open_loop
from horizon.datagen.parallel import Runner
from horizon.monitor.attribution import attribute_auto, attribute_oracle
from horizon.policy.dataset import Dataset
from horizon.policy.model import TrainConfig
from horizon.rsi.loop import MethodConfig, eval_jobs, train_policy
from horizon.rsi.strategies import STRATEGIES, RoundInputs


def main() -> None:
    dist = TaskDistribution()
    oc = OpenLoopConfig()
    with Runner() as r:
        blocks, _ = generate_open_loop(r, dist, 80000, oc, seed=0)
    data = Dataset(blocks)
    params, _ = train_policy(data, "hier", TrainConfig(steps=8000))
    mc = MethodConfig(name="diag", strategy="ours")
    rng = np.random.default_rng(0)
    with Runner(policy={"params": params}) as r:
        res = r.map(eval_jobs(dist, mc, rng, 60, 1))
    logs = [x.log for x in res if x.log is not None]
    auto = [attribute_auto(l) for l in logs]
    orc = [attribute_oracle(l) for l in logs]
    print("eval success", np.mean([l.success for l in logs]))
    for strat in ("ours", "uniform", "embodirsi"):
        inp = RoundInputs(logs=logs, auto=auto, oracle=orc, budget=12000, rng=np.random.default_rng(1), dist=dist,
                          open_cfg=oc)
        reqs = STRATEGIES[strat](inp)
        with Runner() as r:
            for req in reqs:
                if req.make_job is None or req.budget <= 0:
                    continue
                jobs = [req.make_job() for _ in range(44)]
                t0 = time.time()
                out = r.map(jobs)
                ok = [x for x in out if x.success]
                bad = [x for x in out if not x.success]
                print(f"{strat:9s} {req.label:28s} budget={req.budget:6d} ok={len(ok)}/{len(out)} "
                      f"trans/ok={np.mean([x.n_valid for x in ok]) if ok else 0:.0f} "
                      f"steps_bad={np.mean([x.sim_steps / 25 for x in bad]) if bad else 0:.0f} "
                      f"wall={time.time() - t0:.1f}s errors={sum(1 for x in out if x.error)}")
                if any(x.error for x in out):
                    print(next(x.error for x in out if x.error)[:600])


if __name__ == "__main__":
    main()
