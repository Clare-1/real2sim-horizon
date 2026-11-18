"""Measure the recovery signal on already-trained policies, before trusting v2.

Why this exists
---------------
``_profile()`` used to read ``log.meta["perturbed"]`` (never written) and count event kind
``"recover"`` (never emitted), so ``perturbed_failure_rate`` and ``recovery_triggers`` were both
always 0. The rule's recovery branch could therefore never fire, which is why every non-oracle
selector picks ``dagger_round`` on the recovery scenario.

``FailureProfile`` now also carries the measured fields
(``perturbation_events``, ``perturbation_failure_rate``). This probe checks — on the policies that
already exist — whether that measured signal actually separates the four scenarios. Do not enable
``recovery_signal: v2`` before this table shows separation.

    python scripts/probe_recovery_signal.py --results results/four_causes_20261118 --workers 6
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon.datagen.parallel import Runner  # noqa: E402
from horizon.datagen.rollout import PERT_KINDS, PerturbationConfig  # noqa: E402
from horizon.harness.artifacts import load_params  # noqa: E402
from horizon.harness.experiment import check_results, job_for  # noqa: E402
from horizon.harness.selector_experiment import _store  # noqa: E402
from horizon.real2sim.recon_error import ReconModel  # noqa: E402


def scenario_cfg(base: dict, scenario: dict) -> dict:
    """The same per-scenario overrides the runner applies (kept in sync by intent)."""
    cfg = copy.deepcopy(base)
    if scenario.get("open_loop"):
        cfg["open_loop"] = {**base.get("open_loop", {}), **scenario["open_loop"]}
    for key in ("feedback_pert", "eval_pert", "eval_mode", "friction_range"):
        if scenario.get(key) is not None:
            cfg[key] = scenario[key]
    if scenario.get("world"):
        cfg["world"] = {**base.get("world", {}), **scenario["world"]}
    return cfg


def find_policy(results: Path, name: str) -> Path | None:
    for cand in (results / "runs" / name / "bank0" / "fixed" / "s0" / "policy_3.npz",
                 results / "runs" / name / "bank0" / "fixed" / "s0" / "policy_2.npz",
                 results / "shared" / name / "s0" / "policy_0.npz"):
        if cand.exists():
            return cand
    return None


def _ftypes(log) -> list[str]:
    from horizon.monitor.attribution import attribute_auto
    att = attribute_auto(log)
    return [att.ftype] if att is not None else []


def measure(results, episodes: int) -> dict:
    logs = [r.log for r in results]
    failed = [r for r in results if not r.log.success]
    pert_eps = [r for r in results if r.job.pert.any()]
    pert_fail = [r for r in pert_eps if not r.log.success]
    events = [e for log in logs for e in log.events]
    return {
        "episodes": episodes,
        "sr": 1.0 - len(failed) / len(results) if results else 0.0,
        "armed_episodes": len(pert_eps),
        "armed_rate": len(pert_eps) / len(results) if results else 0.0,
        "perturbation_events": sum(1 for e in events if e[1] in PERT_KINDS),
        "perturbed_failure_rate": len(pert_fail) / len(pert_eps) if pert_eps else 0.0,
        "perturbed_given_failure": (sum(1 for r in failed if r.job.pert.any()) / len(failed)
                                    if failed else 0.0),
        "legacy_perturbed_failure_rate": sum(
            bool(log.meta.get("perturbed")) for log in logs if "perturbed" in log.meta) / len(logs),
        "legacy_recovery_triggers": sum(1 for e in events if e[1] == "recover"),
        "failure_types": dict(Counter(k for r in pert_fail for k in _ftypes(r.log))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/minimal_validation_4causes.yaml")
    ap.add_argument("--results", default="results/four_causes_20261118")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    base = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    results_dir = (ROOT / args.results).resolve()
    rows = []
    for scenario in base["scenarios"]:
        name = scenario["name"]
        policy = find_policy(results_dir, name)
        if policy is None:
            print(f"skip {name}: no trained policy yet", flush=True)
            continue
        cfg = scenario_cfg(base, scenario)
        store = _store(results_dir / "tasks_deploy" / name, cfg,
                       Path(args.project_root).resolve(), "full")
        recon = ReconModel.from_dict(scenario.get("conditions", {}).get("recon", {}))
        jobs = []
        for i in range(args.episodes):
            fam, H = base["families"][i % len(base["families"])], base["H_values"][0]
            # same seeds as the runner's round-1 feedback (_collect)
            ts = base["banks"]["feedback"] + 1 * 10_000 + H * 100 + i
            jobs.append(job_for(store, fam, H, ts, ts + 31, cfg, recon, real=True,
                                controller="hier",
                                pert=PerturbationConfig.suite(cfg.get("feedback_pert", 0.1)),
                                want_samples=False, keep_log=True, provenance="probe"))
        params = load_params(policy)
        with Runner(args.workers, policy={"params": params}) as runner:
            out = runner.map(jobs)
        check_results(out)
        row = measure(out, args.episodes)
        try:
            shown = str(policy.relative_to(ROOT))
        except ValueError:
            shown = str(policy)
        row.update(scenario=name, cause=scenario["cause"], policy=shown)
        rows.append(row)
        print(f"{name:10s} sr={row['sr']:.3f} armed={row['armed_rate']:.2f} "
              f"events={row['perturbation_events']} "
              f"P(fail|armed)={row['perturbed_failure_rate']:.3f} "
              f"legacy={row['legacy_perturbed_failure_rate']:.3f}/"
              f"{row['legacy_recovery_triggers']}", flush=True)

    if not rows:
        print("no policies found")
        return 1
    lines = ["# 恢复信号探针（v2 指标）", "",
             "`legacy` = 当前批次实际写入的值；v2 = 从轨迹事件实测的值。", "",
             "| scenario | policy | SR | armed eps | events | **P(fail\\|armed)** | P(armed\\|fail) "
             "| legacy rate/triggers |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | `{r['policy']}` | {r['sr']:.3f} | "
                     f"{r['armed_episodes']} | {r['perturbation_events']} | "
                     f"**{r['perturbed_failure_rate']:.3f}** | "
                     f"{r['perturbed_given_failure']:.3f} | "
                     f"{r['legacy_perturbed_failure_rate']:.3f} / "
                     f"{r['legacy_recovery_triggers']} |")
    lines += ["", "失败类型（armed 且失败）：", ""]
    for r in rows:
        lines.append(f"- {r['scenario']}: {r['failure_types']}")
    text = "\n".join(lines)
    dest = results_dir / "probe_recovery_signal.md"
    dest.write_text(text + "\n", encoding="utf-8")
    (results_dir / "probe_recovery_signal.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + text)
    print(f"\n[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
