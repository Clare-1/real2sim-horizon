"""Smoke test for data generation, feature extraction and attribution (no torch needed)."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, generate_open_loop
from horizon.datagen.parallel import Job, Runner
from horizon.datagen.rollout import PerturbationConfig
from horizon.monitor.attribution import attribute_auto, attribute_oracle, confusion


def main():
    dist = TaskDistribution(families=("pack", "sort", "stack"), H_values=(2, 4, 6))
    t0 = time.time()
    with Runner() as runner:
        blocks, st = generate_open_loop(runner, dist, 30000, OpenLoopConfig(), seed=0)
        print(f"open-loop: {st.transitions} transitions, kept {st.kept}/{st.attempted}, "
              f"{time.time() - t0:.1f}s wall, errors={len(st.errors)}")
        if st.errors:
            print(st.errors[0])
        prov = Counter()
        for b in blocks:
            prov[b.provenance] += b.n
        print("provenance:", dict(prov), "x_hier", blocks[0].x_hier.shape, "x_flat", blocks[0].x_flat.shape)
        X = np.concatenate([b.x_hier for b in blocks])
        print("feature finite:", bool(np.isfinite(X).all()), "mean|x|", float(np.abs(X).mean()))
        # noisy experts produce failures for exercising the monitor
        rng = np.random.default_rng(1)
        jobs = []
        for _ in range(48):
            fam, H, ts = dist.sample(rng)
            jobs.append(Job(family=fam, H=H, task_seed=ts, seed=int(rng.integers(1 << 30)), controller="expert",
                            expert_noise=0.6, pert=PerturbationConfig.suite(0.3), keep_log=True, want_samples=False,
                            snapshot_every=10, snapshot_events=True, keep_snapshots=True))
        t0 = time.time()
        res = runner.map(jobs)
        print(f"noisy-expert rollouts: {time.time() - t0:.1f}s; errors={sum(1 for r in res if r.error)}")
        for r in res:
            if r.error:
                print(r.error)
                break
        logs = [r.log for r in res if r.log is not None]
        print("success rate:", np.mean([l.success for l in logs]))
        pairs = [(attribute_auto(l), attribute_oracle(l)) for l in logs]
        print("oracle types:", Counter(o.ftype for a, o in pairs if o), "auto types:", Counter(a.ftype for a, o in pairs if a))
        c = confusion(pairs)
        print("attribution agreement: subgoal", round(c["subgoal_acc"], 2), "type", round(c["type_acc"], 2), "n", c["n"])
        print("snapshots per log:", np.mean([len(l.snapshots) for l in logs]))


if __name__ == "__main__":
    main()
