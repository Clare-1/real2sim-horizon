"""Where does automatic attribution disagree with the oracle? Replays a saved policy and prints traces."""

from __future__ import annotations

import sys

import numpy as np

from horizon.datagen.generator import TaskDistribution
from horizon.datagen.parallel import Runner
from horizon.monitor.attribution import attribute_auto, attribute_oracle
from horizon.rsi.loop import MethodConfig, eval_jobs, load_params
from horizon.taskgraph.schema import TaskGraph
from horizon.sim.scene import SceneSpec


def held_runs(h: np.ndarray) -> str:
    out, prev, start = [], None, 0
    for t, v in enumerate(list(h) + [None]):
        if v != prev:
            if prev is not None:
                out.append(f"{prev}@{start}-{t}")
            prev, start = v, t
    return " ".join(x for x in out if not x.startswith("-1@"))


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/main/ours_s0_policy.npz"
    params = load_params(path)
    mc = MethodConfig(name="diag", strategy="ours")
    rng = np.random.default_rng(7)
    with Runner(4, policy={"params": params}) as r:
        res = r.map(eval_jobs(TaskDistribution(), mc, rng, 48, 1))
    logs = [x.log for x in res if x.log is not None]
    n_fail = n_dis = 0
    for log in logs:
        a, o = attribute_auto(log), attribute_oracle(log)
        if o is None:
            continue
        n_fail += 1
        if a is not None and a.ftype == o.ftype:
            continue
        n_dis += 1
        g = TaskGraph.from_dict(log.graph)
        sc = SceneSpec.from_dict(log.scene)
        oi = sc.object_index(g.subgoals[o.subgoal].obj)
        print(f"--- {log.key} H={log.H} oracle={o.ftype}@g{o.subgoal}(t{o.t_fail}) auto={a.ftype if a else None}@g{a.subgoal if a else '-'}(t{a.t_fail if a else '-'}) events={[(t, k) for t, k, _ in log.events]}")
        print(f"    target obj index {oi}; held_true runs: {held_runs(log.held_true)}")
        print(f"    held_est  runs: {held_runs(log.held_est)}")
        ks = log.k_sel.astype(int)
        print(f"    k_sel windows for g{o.subgoal}: {[(t, t2) for t, t2 in _win(ks, o.subgoal)]}  k_true windows: {[(t, t2) for t, t2 in _win(log.k_true.astype(int), o.subgoal)]}")
        if n_dis >= 10:
            break
    print(f"failures {n_fail}, type disagreements shown {n_dis}")


def _win(k, j):
    out, s = [], None
    for t, v in enumerate(k):
        if v == j and s is None:
            s = t
        elif v != j and s is not None:
            out.append((s, t))
            s = None
    if s is not None:
        out.append((s, len(k)))
    return out


if __name__ == "__main__":
    main()
