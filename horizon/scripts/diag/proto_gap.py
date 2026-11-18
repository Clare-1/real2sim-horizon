"""Prototype: how big is the sim-real gap produced by the injected reconstruction bias?"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, collect, open_loop_job
from horizon.datagen.parallel import Job, Runner
from horizon.eval.evaluate import TEST_SEED_BASE
from horizon.policy.dataset import Dataset
from horizon.policy.model import TrainConfig
from horizon.real2sim.recon_error import DEPLOY_WORLD, single_view_bias
from horizon.rsi.loop import train_policy


def main() -> None:
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    recon = single_view_bias().to_dict()
    dist = TaskDistribution()
    oc = OpenLoopConfig()
    rng = np.random.default_rng(0)

    def make() -> Job:
        j = open_loop_job(dist, oc, rng)
        j.world, j.recon = DEPLOY_WORLD.to_dict(), recon
        return j

    with Runner(W) as r:
        blocks, st = collect(r, make, 80000)
    params, _ = train_policy(Dataset(blocks), "hier", TrainConfig(steps=8000))
    jobs = []
    for cond, rec, real in (("recon-sim", recon, False), ("real|belief=recon", recon, True), ("real|belief=true", {}, True)):
        for fi, fam in enumerate(("pack", "sort", "stack")):
            for H in (2, 4, 6, 8, 10):
                for i in range(12):
                    ts = TEST_SEED_BASE + fi * 100_000 + H * 1000 + i
                    jobs.append(Job(family=fam, H=H, task_seed=ts, seed=ts + 17, controller="hier", recon=rec, real=real,
                                    world=DEPLOY_WORLD.to_dict(),
                                    want_samples=False, keep_log=False, tag={"cond": cond}))
    with Runner(W, policy={"params": params}) as r:
        res = r.map(jobs)
    tab = defaultdict(lambda: [0, 0])
    for x in res:
        if x.error:
            print(x.error[:400])
            break
        tab[(x.job.tag["cond"], x.job.family)][0] += int(x.success)
        tab[(x.job.tag["cond"], x.job.family)][1] += 1
    for cond in ("recon-sim", "real|belief=recon", "real|belief=true"):
        print(cond.ljust(20), " ".join(f"{fam}={tab[(cond, fam)][0]}/{tab[(cond, fam)][1]}" for fam in ("pack", "sort", "stack")))


if __name__ == "__main__":
    main()
