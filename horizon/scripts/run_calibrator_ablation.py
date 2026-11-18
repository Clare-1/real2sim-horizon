"""Table 8: calibrator ablation (advisor 回复学生 §五 实验五 表 8).

Four calibrators on the geometry-bias scenario, three rounds, no policy learning, so the only
thing that changes is the twin. Reports the parameter error per round against the true 20 mm and
counts seeds where the estimate diverges. This is the direct test of the estimator defect that
moved a 19.84 mm belief to 93.54 mm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from horizon.datagen.parallel import Runner
from horizon.harness.artifacts import write_json
from horizon.harness.experiment import collect_recorded, evaluate, nominal_factory, observed_feedback, train_version
from horizon.harness.selector_experiment import _collect, _store
from horizon.real2sim.recon_error import ReconModel
from horizon.rsi.simreal import geometry_repair

TRUE_TOP_MM = 20.0
VARIANTS = ("legacy_repeat", "robust_repeat", "freeze_first", "true_params")


def _believed_mm(recon: ReconModel, store, seed: int) -> float:
    """Believed tray top after applying this reconstruction model to a real task."""
    from horizon.tasks.factory import make_task
    from horizon.real2sim.recon_error import DEPLOY_WORLD
    task = make_task("pack", 4, seed)
    scene = recon.apply(DEPLOY_WORLD.apply(task.scene))
    return max(f.top_z for f in scene.fixtures if f.kind == "plate") * 1000.0


def run(cfg, out: Path, workers: int):
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for scenario in cfg["scenarios"]:
        name = scenario["name"]
        if cfg.get("only_causes") and scenario["cause"] not in cfg["only_causes"]:
            continue
        store_train = _store(out / "tasks_train", cfg, Path(".").resolve(), "full")
        store_deploy = _store(out / "tasks_deploy", cfg, Path(".").resolve(), "full")
        base_recon = ReconModel.from_dict(scenario["conditions"]["recon"])
        for seed in cfg["seeds"]:
            shared = out / "shared" / f"s{seed}"
            shared.mkdir(parents=True, exist_ok=True)
            with Runner(workers) as runner:
                blocks, _, _ = collect_recorded(runner, nominal_factory(store_train, cfg, base_recon, seed),
                                                cfg["init_budget"], base_recon, require_success=True)
            params, _, _ = train_version(list(blocks), cfg, seed, shared / "policy_0.npz")
            for variant in VARIANTS:
                recon = ReconModel.from_dict(base_recon.to_dict())
                if variant == "true_params":
                    recon = ReconModel()
                errors, sr = [], None
                for round_id in range(1, int(cfg.get("rounds", 3)) + 1):
                    if variant in ("legacy_repeat", "robust_repeat", "freeze_first"):
                        use = not (variant == "freeze_first" and round_id > 1)
                        if use:
                            results, _ = _collect(params, cfg, store_deploy, recon, seed, round_id,
                                                  workers, out / "fb" / f"{variant}_s{seed}_r{round_id}")
                            logs = [observed_feedback(r.log) for r in results]
                            recon, _ = geometry_repair(recon, logs, robust=(variant != "legacy_repeat"))
                    errors.append(round(_believed_mm(recon, store_deploy, seed) - TRUE_TOP_MM, 2))
                ev, _ = evaluate(params, cfg, store_deploy, recon, cfg["banks"]["curve"],
                                 cfg["curve_n_per"], workers)
                sr = ev["nominal"]["overall"]
                rec = {"cause": scenario["cause"], "scenario": name, "seed": seed,
                       "variant": variant, "errors_mm": errors,
                       "final_sr": sr}
                recs.append(rec)
                print(f"{variant:14s} s{seed}: err(mm)={errors}  SR={sr:.3f}", flush=True)
    write_json(out / "summary.json", {"records": recs})
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1_matrix.yaml")
    ap.add_argument("--out", default="results/calibrator_ablation_20261112")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run(cfg, Path(args.out), args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
