"""Run the sim-real proxy experiment: routed repairs vs. sim-only, random and oracle variants.

Example:
    python scripts/run_simreal.py --config configs/simreal.yaml --seeds 0,1,2
Results: results/<name>/<method>_s<seed>.json; tables via scripts/make_simreal_report.py.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path

import yaml

from horizon.datagen.generator import OpenLoopConfig, TaskDistribution
from horizon.policy.model import TrainConfig
from horizon.rsi.simreal import SimRealConfig, run_simreal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or f"results/{cfg['name']}")
    out.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",") if args.methods else list(cfg["methods"])
    d = cfg.get("dist", {})
    dist = TaskDistribution(families=tuple(d.get("families", ("pack", "sort", "stack"))),
                            H_values=tuple(d.get("H_values", (2, 4, 6, 8, 10))), seed_base=d.get("seed_base", 0))
    o = dict(cfg.get("open_loop", {}))
    if "latency_choices" in o:
        o["latency_choices"] = tuple(o["latency_choices"])
    oc = OpenLoopConfig(**o)
    tc = TrainConfig(**cfg.get("train", {}))
    test = cfg.get("test", {})
    def source_hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted([*Path("src/horizon").rglob("*.py"), Path(__file__)])}

    for seed in [int(s) for s in args.seeds.split(",")]:
        for m in methods:
            sc = SimRealConfig(name=m, seed=seed, **{**cfg.get("common", {}), **cfg["methods"][m]})
            experiment = {"config": asdict(sc), "train": {**asdict(tc), "seed": tc.seed + seed},
                          "distribution": asdict(dist), "open_loop": asdict(oc), "test": test,
                          "source_sha256": source_hashes(), "workers": args.workers,
                          "python": platform.python_version()}
            fingerprint = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
            result_path = out / f"{m}_s{seed}.json"
            if args.skip_existing and result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("experiment", {}).get("fingerprint") != fingerprint:
                    raise ValueError(f"Existing result has different or unrecorded inputs/source: {result_path}; use a new output directory")
                print(f"skip {m}_s{seed}", flush=True)
                continue
            result = run_simreal(sc, dist, oc, tc, test_n_per=int(test.get("n_per", 20)),
                                 curve_n_per=int(test.get("curve_n_per", 8)), workers=args.workers, out_dir=out)
            valid = experiment["source_sha256"] == source_hashes()
            result["experiment"] = {**experiment, "fingerprint": fingerprint, "source_unchanged": valid}
            result_path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
            if not valid:
                raise RuntimeError("Source changed during training; this result is marked invalid and must be rerun")


if __name__ == "__main__":
    main()
