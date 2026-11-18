"""Pilot: open-loop data -> BC -> held-out success vs H. Used to calibrate budget/difficulty."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, generate_open_loop
from horizon.datagen.parallel import Runner
from horizon.eval.evaluate import evaluate
from horizon.policy.dataset import Dataset
from horizon.policy.model import TrainConfig, train_bc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=160000)
    ap.add_argument("--policy", default="hier")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--n-per", type=int, default=10)
    ap.add_argument("--pert", default="0.0,0.15")
    ap.add_argument("--families", default="pack,sort,stack")
    ap.add_argument("--out", default="results/pilot")
    args = ap.parse_args()
    fams = tuple(args.families.split(","))
    dist = TaskDistribution(families=fams, H_values=(2, 4, 6, 8, 10))
    t0 = time.time()
    with Runner() as runner:
        blocks, st = generate_open_loop(runner, dist, args.budget, OpenLoopConfig(), seed=0)
    data = Dataset(blocks)
    t1 = time.time()
    print(f"data: {data.n} transitions, {st.kept}/{st.attempted} demos kept, {t1 - t0:.0f}s; {data.composition()}", flush=True)
    X, Y = data.arrays(args.policy)
    params, stats = train_bc(X, Y, TrainConfig(steps=args.steps, hidden=args.hidden, log_every=1000), verbose=True)
    t2 = time.time()
    print(f"train: {t2 - t1:.0f}s on {stats['device']}, np/torch diff {stats['np_torch_maxdiff']:.2e}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with Runner(policy={"params": params}) as runner:
        for pert in [float(x) for x in args.pert.split(",")]:
            t2 = time.time()
            res = evaluate(params, args.policy, families=fams, n_per=args.n_per, pert_p=pert, runner=runner)
            fit = res["fit"] or {}
            print(f"eval pert={pert}: {time.time() - t2:.0f}s overall {res['overall']:.3f} alpha {fit.get('alpha', 0):.3f} "
                  f"per-stage {fit.get('per_stage', 0):.3f}", flush=True)
            print("  " + " ".join(f"H{h}={v['k']}/{v['n']}" for h, v in res["by_H"].items()))
            print("  by family:", {k: f"{v['k']}/{v['n']}" for k, v in res["by_family_H"].items()})
            print("  failure types oracle:", res["failure_types_oracle"], "auto:", res["failure_types_auto"])
            if res["errors"]:
                print("ERRORS:", res["errors"][0])
            (out / f"pilot_{args.policy}_b{args.budget}_p{pert}.json").write_text(json.dumps(
                {k: v for k, v in res.items() if k != "episodes"}, indent=1, default=str))


if __name__ == "__main__":
    main()
