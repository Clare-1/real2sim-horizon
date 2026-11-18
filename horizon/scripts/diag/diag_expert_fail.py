"""Why does the expert fail under heavy perturbation? Print oracle attribution + timing."""
from collections import Counter
from horizon.datagen.parallel import Job, Runner
from horizon.datagen.rollout import PerturbationConfig, PHASES
from horizon.monitor.attribution import attribute_oracle
import numpy as np

if __name__ == "__main__":
    jobs = [Job(family=fam, H=10, task_seed=5000 + i, seed=i, controller="expert", pert=PerturbationConfig.suite(0.3),
                keep_log=True, want_samples=False) for fam in ("pack", "sort", "stack") for i in range(12)]
    with Runner() as r:
        res = r.map(jobs)
    for x in res:
        log = x.log
        if log.success:
            continue
        a = attribute_oracle(log)
        T = log.T
        maxT = log.H * 150 + 50
        ph = Counter(PHASES[p] for p in log.phase[-60:])
        print(f"{log.family} seed={log.seed} steps={log.steps}/{maxT} sat={int(np.asarray(log.meta['final_sat']).sum())}/{log.H} "
              f"held={log.meta['final_held']} attr={a.ftype if a else None}@{a.subgoal if a else None} "
              f"events={[k for _, k, _ in log.events]} last-phases={dict(ph)}")
