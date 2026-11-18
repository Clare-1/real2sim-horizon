"""Expert under the evaluation perturbation suite: should still succeed nearly always."""
from collections import Counter
from horizon.datagen.parallel import Job, Runner
from horizon.datagen.rollout import PerturbationConfig

if __name__ == "__main__":
    jobs = []
    for fam in ("pack", "sort", "stack"):
        for H in (4, 10):
            for i in range(12):
                jobs.append(Job(family=fam, H=H, task_seed=5000 + i, seed=i, controller="expert",
                                pert=PerturbationConfig.suite(0.3), keep_log=True, want_samples=False))
    with Runner() as r:
        res = r.map(jobs)
    by = {}
    ev = Counter()
    for x in res:
        if x.error:
            print(x.error); break
        by.setdefault((x.job.family, x.job.H), []).append(x.log.success)
        for _, k, _ in x.log.events:
            ev[k] += 1
    for k, v in sorted(by.items()):
        print(k, f"{sum(v)}/{len(v)}")
    print("events:", dict(ev))
