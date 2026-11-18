"""Exactly 12 expert feasibility rollouts on six existing transfer task artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


HORIZONS = (6, 8, 10)
BANK = 68_000_000
MODES = (("nominal", 0.0), ("perturbed", 0.15))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class ExistingTasks:
    """Read exactly the preselected six files; never generate or replace a task."""

    def __init__(self, root):
        self.root = root
        self.allowed = {("pack", H, BANK + H * 1000 + i) for H in HORIZONS for i in (0, 1)}

    def get(self, family, H, seed):
        from horizon.tasks.serialization import task_from_payload

        require((family, H, seed) in self.allowed, "Task is outside the fixed six-task scope")
        key = f"{family}_H{H}_s{seed}"
        path = self.root / f"{key}.json"
        sidecar = self.root / f"{key}.identity.json"
        require(path.is_file() and sidecar.is_file(), f"Existing transfer task missing: {key}")
        sha = file_hash(path)
        require(read_json(sidecar) == {"key": key, "sha256": sha}, f"Task identity mismatch: {key}")
        payload = read_json(path)
        task_from_payload(payload, expected_family=family, expected_H=H, expected_seed=seed)
        return payload, {"path": str(path), "sha256": sha, "key": key,
                         "family": family, "H": H, "task_seed": seed,
                         "identity_path": str(sidecar), "identity_sha256": file_hash(sidecar)}


def plan_text(plan):
    tasks = "\n".join(f"- `{r['key']}` — `{r['sha256']}`" for r in plan["tasks"])
    return f"""# Fixed expert feasibility check

This post-outcome development diagnostic follows the first uniformly zero learned
H=6/8/10 transfer scores. It does not alter or stop the full transfer matrix.
The task selection is fixed before any expert outcome: existing transfer artifacts
at H=6/8/10 and offsets i=0,1, nominal and p=0.15, exactly 6 × 2 = 12 episodes.
No task replacement, retry, threshold change, training, calibration, or new task bank.

Use the E1 frozen source, original privileged ExpertController with zero expert
noise, identity ReconModel, real=True, original world configuration and scoring.
The expert uses simulator true state; this is a feasibility control, not a fair
observation-matched learned-policy baseline. Original default action/environment,
expert, simulator and scoring configurations are recorded in definition.json.
Rollout seeds equal task_seed+17, matching transfer. Max steps remain H*180+60,
done patience is 8, and workers=2. Perturbed mode uses slip=.15, displace=.15,
knock=.075, traj=target=0. All outcomes and actual costs are retained.

Source experiment fingerprint: `{plan['source_experiment_fingerprint']}`
Transfer definition fingerprint: `{plan['transfer_definition_fingerprint']}`
Expert-check fingerprint: `{plan['fingerprint']}`

{tasks}

Success demonstrates executability only for these six task identities and their
two sampled rollout conditions. It does not establish feasibility of all 60
transfer tasks, an expert success probability, real-world transfer, or the cause
of learned-policy failure. Failure also does not prove that a task is impossible.
No reserved confirmation task is used. A started marker prevents accidental
repeat execution; completed results are checksum-verified on subsequent calls.
"""


def readme_text(result):
    lines = ["# Expert feasibility diagnostic", "",
             "This fixed post-outcome check used the original privileged expert, identity geometry,",
             "the E1 world and scoring, and six existing transfer tasks (i=0,1 at H=6/8/10).",
             "No training, calibration, task replacement, or threshold adjustment occurred.", "",
             "| H | Nominal success | Perturbed success (p=.15) |", "|---|---:|---:|"]
    for H in HORIZONS:
        cells = []
        for mode, _ in MODES:
            eps = [e for e in result["episodes"] if e["H"] == H and e["mode"] == mode]
            cells.append(f"{sum(e['success'] is True for e in eps)}/{len(eps)}")
        lines.append(f"| {H} | {cells[0]} | {cells[1]} |")
    ledger = result["ledger"]
    lines.extend(["", f"Status: {result['status']}. Worker errors: {ledger['worker_errors']}.",
                  f"Actual episodes: {ledger['episodes']}; physics steps: {ledger['sim_steps']:,}; "
                  f"sum of worker rollout-loop time: {ledger['worker_seconds']:.3f} s; "
                  f"pool wall time including startup/shutdown: {ledger['pool_wall_seconds']:.3f} s.",
                  "Worker seconds are not elapsed time and exclude environment construction; this run",
                  "shared the machine with the main transfer experiment, so this is not a speed benchmark.", "",
                  "Task/identity hashes, full job configuration, final scorer fields, perturbation events,",
                  "and individual costs are in results.json; recorded commanded actions are in actions.npz.",
                  "The frozen definition contains source hashes and action/environment/expert/scorer defaults.", "",
                  "The expert has privileged simulator state. Success supports executability only for",
                  "these six tasks and sampled conditions, not all 60 transfer tasks or learned-policy",
                  "generalization. Failure does not prove impossibility. This diagnostic is development",
                  "evidence and cannot establish physical transfer or agentic-learning advantage.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    source = base / "results/harness_e1_20261026"
    transfer = base / "results/harness_transfer_20261026"
    out = base / "results/harness_transfer_expert_20261026"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "source/src"))

    import mujoco
    import numpy as np
    import horizon.datagen.parallel as parallel
    from horizon.datagen.rollout import rollout
    from horizon.harness.artifacts import write_json
    from horizon.harness.experiment import eval_jobs, source_hashes
    from horizon.real2sim.recon_error import ReconModel
    from horizon.sim.mjcf import SimConfig
    from horizon.skills.expert import ExpertConfig
    from horizon.taskgraph.predicates import Tolerances

    require(Path(parallel.__file__).resolve().is_relative_to(source / "source/src"), "Wrong active source")
    manifest = read_json(source / "experiment.json")
    require(manifest["fingerprint"] == digest({k: v for k, v in manifest.items() if k != "fingerprint"}),
            "Source manifest fingerprint mismatch")
    require(source_hashes() == manifest["source_sha256"], "Frozen source hashes differ")
    transfer_plan = read_json(transfer / "evaluation_definition.json")
    require(transfer_plan["fingerprint"] == digest({k: v for k, v in transfer_plan.items() if k != "fingerprint"}),
            "Transfer definition fingerprint mismatch")
    require(transfer_plan["source_experiment_fingerprint"] == manifest["fingerprint"], "Wrong source experiment")
    require(transfer_plan["source_sha256"] == manifest["source_sha256"], "Transfer source differs")
    require(transfer_plan["evaluation_bank"] == BANK and transfer_plan["H_values"] == list(HORIZONS),
            "Unexpected transfer task bank")
    cfg = transfer_plan["evaluation_config"]
    require(cfg["families"] == ["pack"] and cfg["H_values"] == list(HORIZONS), "Wrong transfer task grid")
    require(cfg["world"] == manifest["config"]["world"], "World differs from E1")
    tasks = ExistingTasks(transfer / "tasks")
    batches = []
    for mode, p in MODES:
        jobs = eval_jobs(tasks, cfg, ReconModel(), BANK, 2, p)
        for job in jobs:
            job.controller = "expert"
            require(job.real and job.recon == {"bias": {}, "corrections": {}}, "Wrong identity/real setting")
        batches.append((mode, jobs))
    job_records = [{"mode": mode, **{k: v for k, v in asdict(j).items() if k != "task_payload"}}
                   for mode, jobs in batches for j in jobs]
    require(len(job_records) == 12, "Expected exactly 12 episodes")
    inputs = {"schema_version": 1, "source_experiment_fingerprint": manifest["fingerprint"],
              "transfer_definition_fingerprint": transfer_plan["fingerprint"],
              "source_sha256": manifest["source_sha256"], "script_sha256": file_hash(__file__),
              "tasks": [j.tag["task_artifact"] for j in batches[0][1]], "jobs": job_records,
              "world": cfg["world"], "workers": 2, "training_steps": 0, "calibration_attempts": 0,
              "expert_config": asdict(ExpertConfig()), "sim_config": asdict(SimConfig()),
              "scoring_tolerances": asdict(Tolerances()), "expert_privileged": True,
              "action_order": ["delta_x", "delta_y", "delta_z", "delta_yaw", "gripper"],
              "action_clip": [-1.0, 1.0], "rollout_max_steps": {str(H): H * 180 + 60 for H in HORIZONS},
              "done_patience": rollout.__kwdefaults__["done_patience"],
              "stop_when": "done", "scoring": "unchanged log.success = PredicateEngine.task_success on true final state"}
    plan = {**inputs, "fingerprint": digest(inputs)}
    out.mkdir(parents=True, exist_ok=True)
    definition_path = out / "definition.json"
    if definition_path.exists():
        require(read_json(definition_path) == plan, "Expert-check definition changed")
    else:
        require(not (out / "STARTED.json").exists(), "Started run has no definition")
        write_json(definition_path, plan)
    planned_text = plan_text(plan)
    plan_path = out / "PLAN.md"
    if plan_path.exists():
        require(plan_path.read_text(encoding="utf-8") == planned_text, "Saved plan text changed")
    else:
        plan_path.write_text(planned_text, encoding="utf-8")
    if args.plan_only:
        print(f"Planned exactly 12 episodes: {plan['fingerprint']}", flush=True)
        return
    result_path = out / "results.json"
    if result_path.exists():
        saved = read_json(result_path)
        require(saved["definition_fingerprint"] == plan["fingerprint"], "Saved result definition changed")
        require(saved["record_sha256"] == digest({k: v for k, v in saved.items() if k != "record_sha256"}),
                "Saved result checksum changed")
        require(saved["actions"]["sha256"] == file_hash(saved["actions"]["path"]), "Actions changed")
        (out / "README.md").write_text(readme_text(saved), encoding="utf-8")
        print("Existing 12-episode result verified; no new rollout.", flush=True)
        return
    with (out / "STARTED.json").open("x", encoding="utf-8") as handle:
        json.dump({"definition_fingerprint": plan["fingerprint"], "planned_episodes": 12}, handle)
    episodes, actions = [], {}
    started = time.perf_counter()
    with parallel.Runner(workers=2) as runner:
        for mode, jobs in batches:
            results = runner.map(jobs)
            require(len(results) == 6, "Worker result count differs")
            for r in results:
                log = r.log
                action_key = f"{mode}_H{r.job.H}_s{r.job.task_seed}"
                if log is not None:
                    actions[action_key] = log.action
                episodes.append({"mode": mode, "family": r.job.family, "H": r.job.H,
                                 "task_seed": r.job.task_seed, "rollout_seed": r.job.seed,
                                 "task_artifact": r.job.tag["task_artifact"], "error": r.error,
                                 "success": bool(log.success) if log is not None else None,
                                 "judged_success": bool(log.success_ver) if log is not None else None,
                                 "steps": log.steps if log is not None else None,
                                 "progress": float(np.mean(log.meta["final_sat"])) if log is not None else None,
                                 "final_sat": log.meta["final_sat"] if log is not None else None,
                                 "final_held": log.meta["final_held"] if log is not None else None,
                                 "events": log.events if log is not None else None,
                                 "sim_steps": r.sim_steps, "worker_seconds": r.wall,
                                 "action_key": action_key if log is not None else None})
            print(f"{mode}: {sum(e['success'] is True for e in episodes if e['mode'] == mode)}/6", flush=True)
    pool_wall = time.perf_counter() - started
    require(source_hashes() == plan["source_sha256"], "Source changed during check")
    for ref in plan["tasks"]:
        _, current = tasks.get(ref["family"], ref["H"], ref["task_seed"])
        require(current == ref, "Task changed during check")
    np.savez_compressed(out / "actions.npz", **actions)
    errors = sum(e["error"] is not None or e["success"] is None for e in episodes)
    require(Counter((e["mode"], e["H"]) for e in episodes) ==
            Counter({(mode, H): 2 for mode, _ in MODES for H in HORIZONS}), "Wrong final result grid")
    result = {"schema_version": 1, "status": "complete" if not errors else "completed_with_errors",
              "definition_fingerprint": plan["fingerprint"], "episodes": episodes,
              "actions": {"path": str(out / "actions.npz"), "sha256": file_hash(out / "actions.npz")},
              "training_steps": 0, "calibration_attempts": 0,
              "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                          "numpy": np.__version__, "mujoco": mujoco.__version__},
              "ledger": {"episodes": len(episodes), "worker_errors": errors,
                         "sim_steps": sum(e["sim_steps"] for e in episodes),
                         "worker_seconds": sum(e["worker_seconds"] for e in episodes),
                         "pool_wall_seconds": pool_wall,
                         "failed_worker_cost_may_be_unavailable": bool(errors)}}
    write_json(result_path, {**result, "record_sha256": digest(result)})
    (out / "README.md").write_text(readme_text(result), encoding="utf-8")
    print(f"Saved all 12 outcomes: {result_path}", flush=True)


if __name__ == "__main__":
    main()
