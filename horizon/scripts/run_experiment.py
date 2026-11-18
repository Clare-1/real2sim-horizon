"""Run the matched-budget comparison: every method x seed -> train -> held-out evaluation.

Example:
    python scripts/run_experiment.py --config configs/mvp.yaml --methods open_loop,ours --seeds 0
Results land in results/<name>/<method>_s<seed>.json; figures via scripts/make_figures.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution
from horizon.datagen.parallel import Runner
from horizon.datagen.rollout import PerturbationConfig
from horizon.eval.evaluate import evaluate
from horizon.policy.model import TrainConfig
from horizon.rsi.loop import MethodConfig, load_params, run_method


def build(cfg: dict, method: str, seed: int) -> tuple[MethodConfig, TaskDistribution, OpenLoopConfig, TrainConfig]:
    base = {k: cfg[k] for k in ("total_budget", "init_frac", "rounds", "eval_episodes", "eval_pert", "nominal_frac", "policy")
            if k in cfg}
    m = {**base, **cfg["methods"][method]}
    open_override = m.pop("open_loop", {})  # per-method override of the open-loop recipe
    if "ops" in m:
        m["ops"] = tuple(m["ops"])
    if "curve" in m:
        m["curve"] = tuple(int(x) for x in m["curve"])
    mc = MethodConfig(name=method, seed=seed, **m)
    d = cfg.get("dist", {})
    dist = TaskDistribution(families=tuple(d.get("families", ("pack", "sort", "stack"))),
                            H_values=tuple(d.get("H_values", (2, 4, 6, 8, 10))))
    o = {**cfg.get("open_loop", {}), **open_override}
    if "recovery_pert" in o:
        o["recovery_pert"] = PerturbationConfig(**o["recovery_pert"])
    if "latency_choices" in o:
        o["latency_choices"] = tuple(o["latency_choices"])
    oc = OpenLoopConfig(**o)
    tc = TrainConfig(**cfg.get("train", {}))
    return mc, dist, oc, tc


def make_test_fn(test: dict, dist: TaskDistribution, workers: int | None):
    """Held-out evaluation under every configured condition (nominal and perturbed)."""
    def fn(params: dict, policy: str) -> dict:
        evals = {}
        with Runner(workers, policy={"params": params}) as runner:
            for p in test.get("pert", [0.0, 0.15]):
                cond = "nominal" if p == 0 else f"pert{p}"
                evals[cond] = evaluate(params, policy, families=tuple(test.get("families", dist.families)),
                                       H_values=tuple(test.get("H_values", dist.H_values)),
                                       n_per=int(test.get("n_per", 30)), pert_p=float(p), runner=runner)
        return evals
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", default=None, help="comma list; default = all methods in config")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="re-evaluate saved policies")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or f"results/{cfg['name']}")
    out.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",") if args.methods else list(cfg["methods"])
    seeds = [int(s) for s in args.seeds.split(",")]
    test = cfg.get("test", {})
    for seed in seeds:
        for method in methods:
            res_path = out / f"{method}_s{seed}.json"
            if args.skip_existing and res_path.exists():
                print(f"skip {res_path}")
                continue
            mc, dist, oc, tc = build(cfg, method, seed)
            t0 = time.time()
            test_fn = make_test_fn(test, dist, args.workers)
            if args.eval_only:
                params = load_params(out / f"{method}_s{seed}_policy.npz")
                train_out = json.loads((out / f"{method}_s{seed}_train.json").read_text())
                evals = test_fn(params, mc.policy)
            else:
                train_out = run_method(mc, dist, oc, tc, workers=args.workers, out_dir=out, test_fn=test_fn)
                train_out.pop("params")
                evals = train_out.pop("final_test")
            for cond, ev in evals.items():
                fit = ev["fit"] or {}
                print(f"  [{method} s{seed}] {cond}: overall {ev['overall']:.3f} "
                      f"alpha {fit.get('alpha', float('nan')):.3f} per-stage {fit.get('per_stage', float('nan')):.3f} "
                      + " ".join(f"H{h}={v['rate'][0]:.2f}" for h, v in ev["by_H"].items()), flush=True)
            res = {"method": method, "seed": seed, "train": train_out, "eval": evals, "wall": time.time() - t0}
            res_path.write_text(json.dumps(res, indent=1, default=str))
            print(f"saved {res_path} ({res['wall']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
