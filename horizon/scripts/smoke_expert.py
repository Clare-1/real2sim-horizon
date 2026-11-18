"""Smoke test: the state-based expert should solve every task family at several horizons."""

from __future__ import annotations

import argparse
import collections
import time

from horizon.sim.env import TabletopEnv
from horizon.skills.expert import TaskExpert
from horizon.taskgraph.predicates import PredicateEngine
from horizon.tasks.factory import make_task


def run(family: str, H: int, seed: int, verbose: bool = False) -> dict:
    task = make_task(family, H, seed)
    env = TabletopEnv(task.scene, seed=seed)
    eng = PredicateEngine(task.scene)
    expert = TaskExpert(task.scene, task.graph, eng=eng)
    env.reset()
    t0 = time.perf_counter()
    phases = collections.Counter()
    done_steps = 0
    last = None
    for t in range(H * 150):
        s = env.true_state()
        a, k, phase = expert.act(s)
        phases[phase] += 1
        if verbose and (k, phase) != last:
            print(f"  t={t:4d} sg={k} phase={phase} held={s.held} open={s.grip_open:.3f} tcp={s.grip_pos.round(3)}")
            last = (k, phase)
        if k is None:
            done_steps += 1
            if done_steps > 5:
                break
        env.step(a)
    s = env.true_state()
    ok = eng.task_success(s, task.graph)
    sat = eng.satisfied_vector(s, task.graph)
    return dict(family=family, H=H, seed=seed, success=ok, n_sat=int(sat.sum()), steps=env.t,
                wall=time.perf_counter() - t0, sim_steps=env.sim_steps, phases=dict(phases))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="pack,sort,stack")
    ap.add_argument("--H", default="2,4")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    for fam in args.families.split(","):
        for H in [int(h) for h in args.H.split(",")]:
            for seed in range(args.seeds):
                r = run(fam, H, seed, args.v)
                print(f"{fam:5s} H={H:2d} seed={seed} success={r['success']} sat={r['n_sat']}/{H} "
                      f"steps={r['steps']} wall={r['wall']:.2f}s ({1000 * r['wall'] / max(1, r['steps']):.2f} ms/step)")
                if not r["success"]:
                    print("   phases:", r["phases"])
