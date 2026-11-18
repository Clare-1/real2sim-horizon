"""Collect independent failure profiles (feedback only) for diagnosis research.

The selector experiment reuses one feedback bank per scenario, and the feedback seeds do not
depend on the selector, so all selectors see the same profile. Diagnosis needs independent
*deployment instances*, not repeated selectors. This script varies the feedback bank only, reuses
the trained pi0, and writes one profile per (cause, bank).

    python scripts/collect_profiles.py --banks 20 --workers 12
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon.datagen.parallel import Runner  # noqa: E402
from horizon.datagen.rollout import PerturbationConfig  # noqa: E402
from horizon.harness.artifacts import load_params  # noqa: E402
from horizon.harness.experiment import check_results, job_for  # noqa: E402
from horizon.harness.selector_experiment import _profile, _store  # noqa: E402
from horizon.real2sim.recon_error import ReconModel  # noqa: E402


def scenario_cfg(base: dict, scenario: dict) -> dict:
    cfg = copy.deepcopy(base)
    if scenario.get("open_loop"):
        cfg["open_loop"] = {**base.get("open_loop", {}), **scenario["open_loop"]}
    for key in ("feedback_pert", "eval_pert", "eval_mode", "friction_range"):
        if scenario.get(key) is not None:
            cfg[key] = scenario[key]
    if scenario.get("world"):
        cfg["world"] = {**base.get("world", {}), **scenario["world"]}
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/minimal_validation_4causes.yaml")
    ap.add_argument("--results", default="results/four_causes_20261118_rule")
    ap.add_argument("--out", default="results/diagnosis_profiles_20261118")
    ap.add_argument("--banks", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    base = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    results = (ROOT / args.results).resolve()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in base["scenarios"]:
        name = scenario["name"]
        policy = results / "shared" / name / "s0" / "policy_0.npz"
        if not policy.exists():
            print(f"skip {name}: no pi0 at {policy}", flush=True)
            continue
        cfg = scenario_cfg(base, scenario)
        store = _store(results / "tasks_deploy" / name, cfg,
                       Path(args.project_root).resolve(), "full")
        recon = ReconModel.from_dict(scenario.get("conditions", {}).get("recon", {}))
        params = load_params(policy)
        with Runner(args.workers, policy={"params": params}) as runner:
            for bank in range(args.banks):
                jobs = []
                for i in range(args.episodes):
                    fam, H = base["families"][i % len(base["families"])], base["H_values"][0]
                    ts = (base["banks"]["feedback"] + bank * 1_000_000
                          + 0 * 100_000 + 1 * 10_000 + H * 100 + i)
                    jobs.append(job_for(store, fam, H, ts, ts + 31, cfg, recon, real=True,
                                        controller="hier",
                                        pert=PerturbationConfig.suite(cfg.get("feedback_pert", 0.1)),
                                        want_samples=False, keep_log=True, provenance="diagnosis"))
                res = runner.map(jobs)
                check_results(res)
                prof, _ = _profile(res, recon)
                rows.append({
                    "cause": scenario["cause"], "scenario": name, "bank_index": bank,
                    "feedback_pert": cfg.get("feedback_pert", 0.1),
                    "sr": prof.sr, "n_episodes": prof.n_episodes,
                    "profile": {"residual": prof.calibration_residual,
                                "concentration": prof.concentration,
                                "per_subgoal": prof.per_subgoal_failure,
                                "failure_types": dict(prof.failure_types),
                                "sr": prof.sr,
                                "perturbed_failure_rate": prof.perturbed_failure_rate,
                                "perturbation_failure_rate": prof.perturbation_failure_rate,
                                "perturbation_events": prof.perturbation_events},
                })
                print(f"{name} bank{bank}: sr={prof.sr:.3f} "
                      f"res={prof.calibration_residual} conc={prof.concentration:.2f}", flush=True)
    path = out / "profiles.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    print(f"[written] {path}  rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
