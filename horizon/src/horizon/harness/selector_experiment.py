"""Selector-driven minimal validation: which learning operation to run next?

This is the runner described in ``最小验证规格_2026-11-12.md``: each scenario hides one failure
cause, every round the policy is evaluated on the deployment distribution, the deployment
feedback is compressed into a :class:`FailureProfile`, and a selector chooses one learning
operation. Costs are charged for every round (deployment episodes and new labels).

It reuses the E1 building blocks so that budgets, task artifacts and training are identical to
the existing harness. Selection truth (the injected cause) never enters the profile.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from ..datagen.generator import TaskDistribution
from ..datagen.parallel import Runner
from ..datagen.rollout import PERT_KINDS, PerturbationConfig
from ..monitor.attribution import attribute_auto
from ..monitor.verifier import VerifierConfig
from ..policy.dataset import Dataset
from ..real2sim.recon_error import ReconModel
from ..rsi.selector import (CAUSE_ACTION, Budget, FailureProfile, FixedRecipeSelector, HistoryEntry,
                            Observation, OracleSelector, RulePlannerSelector)
from ..rsi.simreal import geometry_repair
from ..tasks.families import RandomizationConfig
from .artifacts import canonical_hash, load_blocks, save_blocks, save_policy, write_json
from .experiment import (TaskStore, check_results, collect_recorded, dagger_factory, evaluate,
                         job_for, nominal_factory, nominal_recovery_factory, observed_feedback,
                         rollout_cost, train_version)

SELECTORS = {"fixed": FixedRecipeSelector, "rule": RulePlannerSelector}
WIRED = {"random": "RandomSelector", "round_robin": "RoundRobinSelector", "bandit": "UCBBanditSelector"}


def make_selector(name: str, cause: str, cfg: dict | None = None):
    if name == "oracle":
        return OracleSelector(true_cause=cause)
    if name == "llm":
        from ..rsi.llm_planner import LLMPlannerSelector
        return LLMPlannerSelector()
    if name in WIRED:
        from ..rsi import selector as _sel
        return getattr(_sel, WIRED[name])()
    if name == "rule":
        return RulePlannerSelector(recovery_signal=(cfg or {}).get("recovery_signal", "v1"))
    return SELECTORS[name]()


def _store(root: Path, cfg: dict, project_root: Path, prior: str) -> TaskStore:
    local = copy.deepcopy(cfg)
    rand = dict(local.get("randomization", {}))
    rand["layout_prior"] = prior
    local["randomization"] = rand
    return TaskStore(root, local, project_root)


def _profile(results, recon) -> tuple[FailureProfile, dict]:
    """Compress deployment feedback into what a selector is allowed to see."""
    logs = [r.log for r in results]
    per_subgoal: dict[str, int] = {}
    ftypes: dict[str, int] = {}
    failed = 0
    for log in logs:
        att = attribute_auto(log)
        if att is None:
            continue
        failed += 1
        key = f"subgoal{att.subgoal}"
        per_subgoal[key] = per_subgoal.get(key, 0) + 1
        ftypes[att.ftype] = ftypes.get(att.ftype, 0) + 1
    obs = [observed_feedback(log) for log in logs]
    _, report = geometry_repair(recon, obs, robust=True)
    residual, spread = None, None
    for entry in report.values():
        r = abs(float(entry["estimated"]) - float(entry["believed"]))
        residual = r if residual is None else max(residual, r)
        s = float(entry.get("spread", 0.0))
        spread = s if spread is None else max(spread, s)
    pert = [bool(log.meta.get("perturbed")) for log in logs if "perturbed" in log.meta]
    # V2: measure perturbation pressure from the log itself. The legacy `meta["perturbed"]` key is
    # never written by the rollout, so `pert` is always empty and the legacy rate is always 0.0;
    # `recovery_triggers` looks for an event kind ("recover") that the rollout never emits either.
    pert_logs = [log for log in logs if any(e[1] in PERT_KINDS for e in log.events)]
    pert_events = sum(1 for log in logs for e in log.events if e[1] in PERT_KINDS)
    pert_failed = sum(1 for log in pert_logs if not log.success)
    recovery = sum(kind == "recover" for log in logs for _, kind, _ in log.events)
    prof = FailureProfile(
        n_episodes=len(logs),
        per_subgoal_failure=per_subgoal,
        calibration_residual=residual,
        calibration_spread=spread,
        recovery_triggers=int(recovery),
        perturbed_failure_rate=(sum(pert) / len(pert)) if pert else 0.0,
        sr=1.0 - failed / len(logs) if logs else 0.0,
        failure_types=ftypes,
        perturbation_events=pert_events,
        perturbation_failure_rate=(pert_failed / len(pert_logs)) if pert_logs else 0.0,
    )
    return prof, report


def _collect(results_obs, cfg, store, recon, seed, round_id, workers, out, bank_base=None):
    """Raw feedback episodes kept for attribution; only counters leave this function."""
    cells = []
    for family in cfg["families"]:
        for H in cfg["H_values"]:
            cells.append((family, H))
    base = cfg["banks"]["feedback"] if bank_base is None else bank_base
    jobs = []
    for i in range(int(cfg["feedback_episodes"])):
        fam, H = cells[i % len(cells)]
        ts = base + seed * 100_000 + round_id * 10_000 + H * 100 + i
        jobs.append(job_for(store, fam, H, ts, ts + 31, cfg, recon, real=True, controller="hier",
                            pert=PerturbationConfig.suite(cfg.get("feedback_pert", 0.1)),
                            want_samples=False, keep_log=True, provenance="feedback"))
    with Runner(workers, policy={"params": results_obs}) as runner:
        results = runner.map(jobs)
    check_results(results)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "feedback.json", {"episodes": len(results),
                                       "success": [bool(r.log.success) for r in results]})
    return results, rollout_cost(results)




def randomized_physics_factory(store, cfg, recon, seed, round_id, lo, hi):
    """Domain randomisation over friction: every demo is generated under a different physical world.

    This is the advisor's alternative for a dynamics gap (“calibrate dynamics or randomise”):
    there is no friction estimator, so the operation that can help is to widen the physical
    distribution the policy is trained on.
    """
    from ..datagen.generator import OpenLoopConfig, TaskDistribution, open_loop_job
    from ..monitor.verifier import VerifierConfig
    dist = TaskDistribution(tuple(cfg["families"]), tuple(cfg["H_values"]),
                            seed_base=cfg["banks"]["training"])
    oc = OpenLoopConfig(**cfg.get("open_loop", {}))
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, 733]))
    def make():
        j = open_loop_job(dist, oc, rng)
        j.world, j.recon = cfg["world"], recon.to_dict()
        j.observation_mode = cfg.get("training_observation_mode", "legacy")
        j.verifier_cfg = VerifierConfig(tol_scale=1.0)
        payload, ref = store.get(j.family, j.H, j.task_seed)
        payload = copy.deepcopy(payload)
        f = float(rng.uniform(lo, hi))
        payload["scene"]["physics"] = {**payload["scene"].get("physics", {}), "friction_scale": f}
        j.task_payload = payload
        j.tag = {"task_artifact": ref, "twin_sha256": canonical_hash(recon.to_dict())}
        return j
    return make



def randomized_physics_factory(store, cfg, recon, seed, round_id, lo, hi):
    """Domain randomisation over friction: every demo is generated under a different physical world.

    This is the advisor's alternative for a dynamics gap (“calibrate dynamics or randomise”):
    there is no friction estimator, so the operation that can help is to widen the physical
    distribution the policy is trained on.
    """
    from ..datagen.generator import OpenLoopConfig, TaskDistribution, open_loop_job
    from ..monitor.verifier import VerifierConfig
    dist = TaskDistribution(tuple(cfg["families"]), tuple(cfg["H_values"]),
                            seed_base=cfg["banks"]["training"])
    oc = OpenLoopConfig(**cfg.get("open_loop", {}))
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, 733]))
    def make():
        j = open_loop_job(dist, oc, rng)
        j.world, j.recon = cfg["world"], recon.to_dict()
        j.observation_mode = cfg.get("training_observation_mode", "legacy")
        j.verifier_cfg = VerifierConfig(tol_scale=1.0)
        payload, ref = store.get(j.family, j.H, j.task_seed)
        payload = copy.deepcopy(payload)
        f = float(rng.uniform(lo, hi))
        payload["scene"]["physics"] = {**payload["scene"].get("physics", {}), "friction_scale": f}
        j.task_payload = payload
        j.tag = {"task_artifact": ref, "twin_sha256": canonical_hash(recon.to_dict())}
        return j
    return make

def _act(action, target, cfg, store_train, store_deploy, recon, seed, round_id, workers, params):
    """Execute one learning operation. Returns (recon, new_blocks, unit_cost, note)."""
    if action == "calibrate_geometry":
        return recon, [], 1, "committed calibration", _empty_cost()
    if action == "recalibrate_verifier":
        return recon, [], 0, "not implemented in this suite", _empty_cost()
    if action == "stop":
        return recon, [], 0, "stopped", _empty_cost()
    if action == "dagger_round":
        factory, note = dagger_factory(store_deploy, cfg, recon, seed, round_id), "dagger"
    elif action == "regenerate_targeted":
        # resample where the deployment distribution actually puts objects (observed layout)
        factory, note = nominal_recovery_factory(store_deploy, cfg, recon, seed, round_id), "targeted"
    elif action == "regenerate_uniform":
        # resample from the training prior, i.e. the region the original data already covered
        factory, note = nominal_recovery_factory(store_train, cfg, recon, seed, round_id), "uniform"
    elif action == "randomize_physics":
        lo, hi = cfg.get("friction_range", (0.4, 1.0))
        factory = randomized_physics_factory(store_train, cfg, recon, seed, round_id, lo, hi)
        note = "randomize"
    elif action == "randomize_physics":
        lo, hi = cfg.get("friction_range", (0.4, 1.0))
        factory = randomized_physics_factory(store_train, cfg, recon, seed, round_id, lo, hi)
        note = "randomize"
    elif action == "add_recovery_data":
        local = copy.deepcopy(cfg)
        local.setdefault("open_loop", {})["recovery_frac"] = 1.0
        factory, note = nominal_recovery_factory(store_deploy, local, recon, seed, round_id), "recovery"
    else:  # defensive: never silently pick an operation
        raise ValueError(f"unhandled action: {action}")
    with Runner(workers, policy={"params": params}) as runner:
        blocks, ledger, _ = collect_recorded(runner, factory, cfg["round_budget"], recon,
                                             require_success=False)
    cost = _empty_cost()
    for key in ("episodes", "sim_steps", "worker_seconds", "accepted_labels", "available_labels"):
        cost[key] = cost.get(key, 0) + ledger.get(key, 0)
    return recon, blocks, 4, note, cost


def _empty_cost() -> dict:
    """Costs are charged for every completed job, including over-budget and failed batches."""
    return {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0,
            "accepted_labels": 0, "available_labels": 0, "eval_episodes": 0}


def run_selector_experiment(cfg, out, project_root, workers=4):
    """Run every (scenario, seed, selector) cell.

    Per-scenario overrides (open_loop / feedback_pert / eval_mode / eval_pert / world /
    friction_range) are applied to a *copy*. The base config is never rebound, otherwise a later
    scenario would silently inherit the previous one's overrides.
    """
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = canonical_hash({"config": cfg})
    write_json(out / "experiment.json", {"config": cfg, "fingerprint": fingerprint})
    base_cfg = copy.deepcopy(cfg)
    records = []
    for scenario in base_cfg["scenarios"]:
        name, cause = scenario["name"], scenario["cause"]
        prior = scenario.get("train_randomization", {}).get("layout_prior", "full")
        for seed in base_cfg["seeds"]:
            cfg_scn = copy.deepcopy(base_cfg)
            if scenario.get("open_loop"):
                cfg_scn["open_loop"] = {**base_cfg.get("open_loop", {}), **scenario["open_loop"]}
            for key in ("feedback_pert", "eval_pert", "eval_mode", "friction_range"):
                if scenario.get(key) is not None:
                    cfg_scn[key] = scenario[key]
            if scenario.get("world"):
                cfg_scn["world"] = {**base_cfg.get("world", {}), **scenario["world"]}
            shared = out / "shared" / name / f"s{seed}"
            shared.mkdir(parents=True, exist_ok=True)
            # Record the effective per-scenario config so an override cannot leak silently.
            write_json(shared / "effective_config.json", {
                k: cfg_scn.get(k) for k in
                ("open_loop", "feedback_pert", "eval_mode", "eval_pert", "world", "friction_range",
                 "init_budget", "round_budget", "rounds", "curve_n_per")})
            store_train = _store(out / "tasks_train" / name, cfg_scn, project_root, prior)
            store_deploy = _store(out / "tasks_deploy" / name, cfg_scn, project_root, "full")
            recon0 = ReconModel.from_dict(scenario.get("conditions", {}).get("recon", {}))
            data_path = shared / "data.npz"
            if data_path.exists():
                blocks = load_blocks({"path": str(data_path), "sha256": ""})
                print(f"INIT {name}/s{seed}: reused {sum(b.n for b in blocks)} labels", flush=True)
            else:
                with Runner(workers) as runner:
                    blocks, ledger, _ = collect_recorded(
                        runner, nominal_factory(store_train, cfg_scn, recon0, seed),
                        cfg_scn["init_budget"], recon0, require_success=True)
                save_blocks(data_path, blocks)
                write_json(shared / "collection.json", {"ledger": ledger})
                print(f"INIT {name}/s{seed}: {sum(b.n for b in blocks)} labels", flush=True)
            base_blocks = list(blocks)
            pi0, _, _ = train_version(base_blocks, cfg_scn, seed, shared / "policy_0.npz")
            print(f"INIT {name}/s{seed}: {sum(b.n for b in base_blocks)} labels", flush=True)
            for bank_index in range(int(cfg_scn.get("scenario_banks", 1))):
                curve_bank = cfg_scn["banks"]["curve"] + bank_index * 1_000_000
                feedback_bank = cfg_scn["banks"]["feedback"] + bank_index * 1_000_000
                for sel_name in cfg_scn["selectors"]:
                    records.append(_run_one(out, cfg_scn, name, cause, seed, sel_name, store_train,
                                            store_deploy, recon0, base_blocks, pi0, workers,
                                            bank_index, curve_bank, feedback_bank))
    write_json(out / "summary.json", {"fingerprint": fingerprint, "records": records})
    write_json(out / "completion.json", {
        "complete": True, "fingerprint": fingerprint,
        "expected_runs": (len(cfg["scenarios"]) * len(cfg["seeds"]) *
                          int(cfg.get("scenario_banks", 1)) * len(cfg["selectors"])),
        "runs": len(records)})
    return records


def _run_one(out, cfg, name, cause, seed, sel_name, store_train, store_deploy, recon0, base, pi0,
             workers, bank_index=0, curve_bank=None, feedback_bank=None):
    """One selector on one scenario instance/seed, starting from the shared D0 and pi0."""
    curve_bank = cfg["banks"]["curve"] if curve_bank is None else curve_bank
    feedback_bank = cfg["banks"]["feedback"] if feedback_bank is None else feedback_bank
    d = out / "runs" / name / f"bank{bank_index}" / sel_name / f"s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "record.json").exists():
        return json.loads((d / "record.json").read_text(encoding="utf-8"))
    selector = make_selector(sel_name, cause, cfg)
    recon = ReconModel.from_dict(recon0.to_dict())
    data, params = Dataset(list(base)), pi0
    history, decisions = [], []
    first_action, deploy, labels, calib = None, 0, 0, 0
    total_cost = _empty_cost()
    sr_now = 0.0
    for round_id in range(1, int(cfg["rounds"]) + 1):
        ev, ev_cost = evaluate(params, cfg, store_deploy, recon, curve_bank, cfg["curve_n_per"], workers)
        total_cost["eval_episodes"] += int(ev_cost.get("episodes", 0))
        sr_before = ev[cfg.get("eval_mode", "nominal")]["overall"]
        results, fb_cost = _collect(params, cfg, store_deploy, recon, seed, round_id, workers,
                                    d / f"round_{round_id}", bank_base=feedback_bank)
        total_cost["episodes"] += int(fb_cost.get("episodes", 0))
        total_cost["sim_steps"] += int(fb_cost.get("sim_steps", 0))
        total_cost["worker_seconds"] += float(fb_cost.get("worker_seconds", 0.0))
        prof, _ = _profile(results, recon)
        obs = Observation(
            prof,
            Budget(deploy_spent=deploy, deploy_total=int(cfg.get("deploy_budget", 10**9)),
                   labels_spent=labels, labels_total=int(cfg.get("label_budget", 10**9)),
                   rounds_done=round_id - 1, rounds_total=int(cfg["rounds"])),
            tuple(history))
        dec = selector.decide(obs)
        if first_action is None:
            first_action = dec.action
        deploy += int(cfg["feedback_episodes"])
        if dec.action == "stop":
            sr_now = sr_before
            decisions.append({**dec.__dict__, "round": round_id, "sr_before": sr_before,
                              "sr_after": sr_before, "profile": _prof_dict(prof)})
            break
        if dec.action == "calibrate_geometry":
            recon, _ = geometry_repair(recon, [observed_feedback(r.log) for r in results], robust=True)
            calib += 1
            added = []
        else:
            recon, added, _, _, act_cost = _act(dec.action, dec.target, cfg, store_train, store_deploy,
                                                recon, seed, round_id, workers, params)
            for key in ("episodes", "sim_steps", "worker_seconds", "accepted_labels", "available_labels"):
                total_cost[key] += act_cost.get(key, 0)
        if added:
            data.add(added)
            labels += sum(b.n for b in added)
            params, _, _ = train_version(list(data.blocks), cfg, seed, d / f"policy_{round_id}.npz")
        ev2, _ = evaluate(params, cfg, store_deploy, recon, curve_bank, cfg["curve_n_per"], workers)
        sr_now = ev2[cfg.get("eval_mode", "nominal")]["overall"]
        history.append(HistoryEntry(dec.action, sr_before, sr_now, dec.target, dec.est_cost))
        decisions.append({**dec.__dict__, "round": round_id, "sr_before": sr_before, "sr_after": sr_now,
                          "profile": _prof_dict(prof)})
        print(f"  {name}/{sel_name}/s{seed} r{round_id}: {dec.action} "
              f"SR {sr_before:.3f} -> {sr_now:.3f}", flush=True)
    record = {"scenario": name, "cause": cause, "seed": seed, "selector": sel_name,
              "bank_index": bank_index,
              "first_action": first_action,
              "first_step_correct": first_action == CAUSE_ACTION.get(cause),
              "final_sr": sr_now, "deploy_episodes": deploy, "new_labels": labels,
              "calibrations": calib, "cost": total_cost, "decisions": decisions}
    write_json(d / "record.json", record)
    return record


def _prof_dict(prof):
    """Everything the selector saw, so a recorded run can be audited afterwards."""
    return {"sr": prof.sr, "residual": prof.calibration_residual,
            "concentration": prof.concentration, "per_subgoal": prof.per_subgoal_failure,
            "failure_types": dict(prof.failure_types),
            "perturbed_failure_rate": prof.perturbed_failure_rate,
            "perturbation_failure_rate": prof.perturbation_failure_rate,
            "perturbation_events": prof.perturbation_events,
            "recovery_triggers": prof.recovery_triggers}


def run_selector_experiment_from_config(config, out, project_root, workers=4):
    return run_selector_experiment(config, out, project_root, workers)
