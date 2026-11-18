"""H5: does the data cost of a new task family fall as the system accumulates task families?

For every target family F and every set P of previously learned families (none, one other, both
others) one multi-task hierarchical policy is trained on [prior_budget transitions of each family in
P] + [b transitions of F] and evaluated on F's held-out test set (same seeds as the main experiment).
C_task = the smallest b at which F reaches the target success (log-linear interpolation between
budget levels). H5 predicts that C_task decreases with |P| (skill reuse across families).

Example:
    python scripts/run_transfer.py --config configs/transfer.yaml --seeds 0
Results: results/<name>/transfer_s<seed>.json; tables and figures via scripts/make_transfer_report.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution, collect, open_loop_job
from horizon.datagen.parallel import Runner
from horizon.eval.evaluate import summarize, test_jobs
from horizon.policy.dataset import Dataset
from horizon.policy.model import TrainConfig
from horizon.rsi.loop import train_policy


def prefix(blocks: list, n: int) -> list:
    """The first n transitions of a pool (pools are generated once; budgets are nested prefixes)."""
    out, total = [], 0
    for b in blocks:
        if total >= n:
            break
        if total + b.n > n:
            b = b.take(n - total)
        out.append(b)
        total += b.n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or f"results/{cfg['name']}")
    out.mkdir(parents=True, exist_ok=True)
    fams = tuple(cfg.get("families", ("pack", "sort", "stack")))
    H_values = tuple(cfg.get("H_values", (2, 4, 6, 8, 10)))
    budgets = [int(b) for b in cfg["budgets"]]
    prior_budget = int(cfg["prior_budget"])
    o = dict(cfg.get("open_loop", {}))
    if "latency_choices" in o:
        o["latency_choices"] = tuple(o["latency_choices"])
    oc = OpenLoopConfig(**o)
    n_per = int(cfg.get("test", {}).get("n_per", 20))
    all_tests = test_jobs("hier", ("pack", "sort", "stack"), H_values, n_per, 0.0)  # main-experiment seeds
    for seed in [int(s) for s in args.seeds.split(",")]:
        path = out / f"transfer_s{seed}.json"
        t0 = time.time()
        tc = TrainConfig(**{**cfg.get("train", {}), "seed": int(cfg.get("train", {}).get("seed", 0)) + seed})
        pool_size = max(prior_budget, max(budgets))
        pools = {}
        with Runner(args.workers) as runner:
            for fi, fam in enumerate(fams):
                rng = np.random.default_rng(1000 * seed + fi)
                dist = TaskDistribution(families=(fam,), H_values=H_values)
                pools[fam], st = collect(runner, lambda: open_loop_job(dist, oc, rng), pool_size)
                print(f"[transfer s{seed}] pool {fam}: {st.transitions} transitions from {st.kept}/{st.attempted} demos",
                      flush=True)
        rows = json.loads(path.read_text())["rows"] if path.exists() else []
        done = {(r["target"], tuple(r["prior"]), r["budget"]) for r in rows}
        for target in fams:
            others = [f for f in fams if f != target]
            priors = [()] + [(f,) for f in others] + [tuple(others)]
            tests = [j for j in all_tests if j.family == target]
            for prior, b in itertools.product(priors, budgets):
                if (target, prior, b) in done:
                    continue
                if not prior and b == 0:
                    rows.append({"target": target, "prior": [], "n_prior": 0, "budget": 0, "k": 0, "n": len(tests),
                                 "success": 0.0, "by_H": {}})
                    continue
                blocks = prefix(pools[target], b) + [x for f in prior for x in prefix(pools[f], prior_budget)]
                params, _ = train_policy(Dataset(blocks), "hier", tc)
                with Runner(args.workers, policy={"params": params}) as runner:
                    summ = summarize(runner.map(tests), H_values)
                k = sum(v["k"] for v in summ["by_H"].values())
                n = sum(v["n"] for v in summ["by_H"].values())
                rows.append({"target": target, "prior": list(prior), "n_prior": len(prior), "budget": b, "k": k, "n": n,
                             "success": k / max(1, n), "by_H": {h: v["k"] / max(1, v["n"]) for h, v in summ["by_H"].items()}})
                print(f"[transfer s{seed}] {target} <- prior {list(prior) or '-'} + {b} new: success {k}/{n}", flush=True)
                path.write_text(json.dumps({"seed": seed, "config": cfg, "train": asdict(tc), "rows": rows}, indent=1))
        path.write_text(json.dumps({"seed": seed, "config": cfg, "train": asdict(tc), "rows": rows,
                                    "wall": time.time() - t0}, indent=1))
        print(f"[transfer s{seed}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
