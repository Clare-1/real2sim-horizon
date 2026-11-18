"""E0 input-to-learning integration and E1 twin-update x weight-learning experiment.

This is a fixed experimental controller, not a model planner. Evaluation truth never
enters calibration or collection. All jobs execute serialized task artifacts; the
procedural sampler is one producer of those artifacts, not a hidden worker fallback.
"""
from __future__ import annotations

import copy
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..datagen.generator import OpenLoopConfig, TaskDistribution, open_loop_job
from ..datagen.parallel import Job, Runner
from ..datagen.rollout import PerturbationConfig
from ..monitor.verifier import VerifierConfig
from ..policy.dataset import Dataset
from ..policy.model import TrainConfig
from ..real2sim.mjcf_import import import_mjcf
from ..real2sim.recon_error import ReconModel
from ..rsi.loop import train_policy
from ..rsi.simreal import geometry_repair
from ..taskgraph.compiler import TemplateCompiler
from ..taskgraph.feasibility import check
from ..tasks.factory import TaskInstance, make_task
from ..tasks.families import RandomizationConfig
from ..tasks.serialization import task_from_payload, task_to_payload
from .artifacts import canonical_hash, file_hash, load_blocks, load_policy, save_blocks, save_policy, write_json

ARMS = {"S00": (False, False), "S01": (False, True),
        "S10": (True, False), "S11": (True, True)}


def validate_config(cfg: dict) -> None:
    if type(cfg.get("reuse_pre_evaluation", False)) is not bool:
        raise ValueError("reuse_pre_evaluation must be a boolean")
    if cfg.get("training_observation_mode", "legacy") not in ("legacy", "observed"):
        raise ValueError("training_observation_mode must be legacy or observed")
    if cfg.get("calibration_schedule", "every_round") not in ("every_round", "first_round"):
        raise ValueError("calibration_schedule must be every_round or first_round")
    if cfg.get("learning_collection", "dagger") not in ("dagger", "nominal_recovery"):
        raise ValueError("learning_collection must be dagger or nominal_recovery")
    for name in ("init_budget", "round_budget", "rounds", "feedback_episodes", "curve_n_per", "final_n_per"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not cfg["arms"] or set(cfg["arms"]) - ARMS.keys():
        raise ValueError("unknown or empty experimental arms")
    if len(set(cfg["arms"])) != len(cfg["arms"]) or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("duplicate arms or seeds")
    banks = cfg["banks"]
    if set(banks) != {"training", "feedback", "curve", "final"}:
        raise ValueError("four disjoint banks are required")
    # Previously used namespaces, including untouched confirmation, are unavailable.
    protected = [(0, 4_000_000), (10_000_000, 11_000_000), (20_000_000, 22_000_000),
                 (30_000_000, 32_000_000), (40_000_000, 41_000_000)]
    for name, start in banks.items():
        if type(start) is not int or start < 0:
            raise ValueError(f"invalid {name} bank")
        if any(max(start, lo) < min(start + 1_000_000, hi) for lo, hi in protected):
            raise ValueError(f"{name} overlaps a historical/reserved bank")
    for a, b in itertools.combinations(banks.values(), 2):
        if abs(a - b) < 1_000_000:
            raise ValueError("overlapping task banks")
    if cfg["rounds"] >= 10 or any(not 0 <= s < 9 for s in cfg["seeds"]):
        raise ValueError("round/seed count exceeds feedback namespace layout")
    if cfg["feedback_episodes"] >= 100 or max(cfg["curve_n_per"], cfg["final_n_per"]) >= 1000:
        raise ValueError("task count exceeds namespace layout")
    if len(cfg["families"]) != len(set(cfg["families"])) or not set(cfg["families"]) <= {"pack", "sort", "stack"}:
        raise ValueError("unknown or duplicate task family")
    if not cfg["H_values"] or any(type(h) is not int or not 1 <= h <= 10 for h in cfg["H_values"]):
        raise ValueError("H must be in 1..10")
    if len(set(cfg["H_values"])) != len(cfg["H_values"]):
        raise ValueError("duplicate H")
    if cfg.get("judge_scale", 1.0) != 1.0:
        raise ValueError("E1 fixes judge scale at 1.0")
    if cfg.get("source", {}).get("kind", "procedural") not in ("procedural", "mjcf_fixture"):
        raise ValueError("unsupported task source")


def source_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*.py"))}
    script = root.parent.parent / "scripts" / "run_harness.py"
    if script.exists():
        hashes["scripts/run_harness.py"] = file_hash(script)
    return hashes


def assert_source(expected):
    if source_hashes() != expected:
        raise RuntimeError("source changed; refusing to commit a resumable result")


def artifact_ref(path):
    return {"path": str(Path(path).resolve()), "sha256": file_hash(path)}


def verify_artifacts(value, seen=None):
    """Verify referenced files, including task references inside JSON manifests."""
    seen = set() if seen is None else seen
    if isinstance(value, list):
        for item in value:
            verify_artifacts(item, seen)
    elif isinstance(value, dict):
        pairs = [(value.get("path"), value.get("sha256")),
                 (value.get("metadata_path"), value.get("metadata_sha256"))]
        for path, expected in pairs:
            if path is None or expected is None or (path, expected) in seen:
                continue
            seen.add((path, expected))
            if file_hash(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path}")
            if Path(path).suffix == ".json":
                verify_artifacts(json.loads(Path(path).read_text(encoding="utf-8")), seen)
        for item in value.values():
            if isinstance(item, (dict, list)):
                verify_artifacts(item, seen)


class TaskStore:
    def __init__(self, root: Path, cfg: dict, project_root: Path):
        self.root, self.cfg, self.project_root = root, cfg, project_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.source = cfg.get("source", {"kind": "procedural"})
        self.template = None
        self.input_sha = None
        # Optional placement override; absent means the historical full-support sampler.
        self.rcfg = RandomizationConfig(**cfg.get("randomization", {}))
        if self.source["kind"] == "mjcf_fixture":
            src = (project_root / self.source["path"]).resolve()
            self.input_sha = file_hash(src)
            scene = import_mjcf(src)
            graph = TemplateCompiler().compile(scene, self.source["instruction"])
            self.template = TaskInstance(graph.family, graph.H, 0, scene, graph,
                                         graph.instruction, check(scene, graph))

    def get(self, family: str, H: int, seed: int) -> tuple[dict, dict]:
        key = f"{family}_H{H}_s{seed}"
        if key in self.cache:
            return self.cache[key]
        path = self.root / f"{key}.json"
        identity_path = self.root / f"{key}.identity.json"
        if path.exists():
            if not identity_path.exists():
                raise ValueError(f"task artifact has no original identity record: {key}")
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if file_hash(path) != identity["sha256"]:
                raise ValueError(f"task artifact changed since creation: {key}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            task_from_payload(payload, expected_family=family, expected_H=H, expected_seed=seed)
        else:
            if identity_path.exists():
                raise ValueError(f"previously materialized task artifact is missing: {key}")
            if self.template is None:
                task = make_task(family, H, seed, self.rcfg)
                task.scene.provenance = "harness:procedural-export"
            else:
                task = copy.deepcopy(self.template)
                if (family, H) != (task.family, task.H):
                    raise ValueError("fixture family/H mismatch")
                task.seed = seed
                # E0 uses one source fixture. Layout variants are explicitly small,
                # and must not be interpreted as independent reconstructed scenes.
                rng = np.random.default_rng(seed)
                for obj in task.scene.objects:
                    obj.pos = (obj.pos[0] + float(rng.uniform(-0.008, 0.008)),
                               obj.pos[1] + float(rng.uniform(-0.008, 0.008)), obj.pos[2])
                task.scene.scene_id = f"fixture_{seed}"
                task.scene.provenance = f"synthetic-fixture:{self.input_sha}"
                task.report = check(task.scene, task.graph)
            if not task.report.ok:
                raise ValueError(f"infeasible task artifact: {key}: {task.report}")
            payload = task_to_payload(task)
            write_json(path, payload)
            write_json(identity_path, {"key": key, "sha256": file_hash(path)})
        ref = {"path": str(path.resolve()), "sha256": file_hash(path), "key": key,
               "family": family, "H": H, "task_seed": seed}
        self.cache[key] = payload, ref
        return payload, ref


def job_for(store: TaskStore, family: str, H: int, task_seed: int, rollout_seed: int,
            cfg: dict, recon: ReconModel, *, real: bool, **kwargs) -> Job:
    payload, ref = store.get(family, H, task_seed)
    return Job(family, H, task_seed, rollout_seed, task_payload=payload,
               observation_mode=cfg.get("training_observation_mode", "legacy"),
               world=cfg["world"], recon=recon.to_dict(), real=real,
               verifier_cfg=VerifierConfig(tol_scale=1.0),
               tag={"task_artifact": ref, "twin_sha256": canonical_hash(recon.to_dict())}, **kwargs)


def eval_jobs(store, cfg, recon, bank, n_per, pert):
    jobs = []
    for family in cfg["families"]:
        fi = {"pack": 0, "sort": 1, "stack": 2}[family]
        for H in cfg["H_values"]:
            for i in range(n_per):
                ts = bank + fi * 100_000 + H * 1000 + i
                jobs.append(job_for(store, family, H, ts, ts + 17, cfg, recon, real=True,
                                    controller="hier", pert=PerturbationConfig.suite(pert),
                                    want_samples=False, keep_log=True, provenance="development"))
    return jobs


def check_results(results):
    errors = [r.error for r in results if r.error]
    if errors:
        raise RuntimeError(f"{len(errors)} worker errors; no silent exclusion:\n" + "\n".join(errors[:3]))


def rollout_cost(results) -> dict:
    return {"episodes": len(results), "sim_steps": sum(r.sim_steps for r in results),
            "worker_seconds": float(sum(r.wall for r in results))}


def evaluate(params, cfg, store, recon, bank, n_per, workers):
    output, ledger = {}, {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}
    with Runner(workers, policy={"params": params}) as runner:
        eval_pert = float(cfg.get("eval_pert", 0.15))
        for mode, pert in (("nominal", 0.0), ("perturbed", eval_pert)):
            results = runner.map(eval_jobs(store, cfg, recon, bank, n_per, pert))
            check_results(results)
            episodes = []
            for r in results:
                log = r.log
                if log is None:
                    raise RuntimeError("evaluation lost an episode log")
                final_sat = log.meta["final_sat"]
                episodes.append({"family": r.job.family, "H": r.job.H,
                                 "task_seed": r.job.task_seed, "rollout_seed": r.job.seed,
                                 "task_sha256": r.job.tag["task_artifact"]["sha256"],
                                 "task_artifact": r.job.tag["task_artifact"],
                                 "success": bool(log.success), "judged_success": bool(log.success_ver),
                                 "steps": log.steps, "progress": float(np.mean(final_sat)),
                                 "final_satisfied": int(sum(final_sat)),
                                 "max_satisfied": max(int(sum(final_sat)), int(np.max(np.sum(log.sat_true, axis=1)))),
                                 "first_unsatisfied_index": next((i for i, sat in enumerate(final_sat) if not sat), len(final_sat))})
            output[mode] = {"overall": float(np.mean([e["success"] for e in episodes])), "episodes": episodes}
            for k, v in rollout_cost(results).items():
                ledger[k] += v
    return output, ledger


def observed_feedback(log):
    """Allowlist geometry signals; log.grip retains v4's ideal encoder assumption.

    Object poses are noisy. This does not reproduce the actor's 0.5 mm noisy
    proprioception stream; the distinction is explicit in the experiment protocol.
    No task success, privileged held identity or oracle predicate enters the tool.
    """
    believed = copy.deepcopy(log.meta.get("belief", log.scene))
    return SimpleNamespace(scene=believed, meta={"belief": believed}, obj_obs=log.obj_obs,
                           grip=log.grip, held_est=log.held_est, action=log.action)


def feedback(params, cfg, store, recon, seed, round_id, workers, out):
    cells = list(itertools.product(cfg["families"], cfg["H_values"]))
    jobs = []
    for i in range(cfg["feedback_episodes"]):
        fam, H = cells[i % len(cells)]
        ts = cfg["banks"]["feedback"] + seed * 100_000 + round_id * 10_000 + H * 100 + i
        jobs.append(job_for(store, fam, H, ts, ts + 31, cfg, recon, real=True, controller="hier",
                            pert=PerturbationConfig.suite(cfg.get("feedback_pert", 0.1)),
                            want_samples=False, keep_log=True, provenance="feedback"))
    with Runner(workers, policy={"params": params}) as runner:
        results = runner.map(jobs)
    check_results(results)
    observed = [observed_feedback(r.log) for r in results]
    # Save exactly the tool-visible signals and their scene; no scorer labels here.
    arrays, records = {}, []
    for i, (result, obs) in enumerate(zip(results, observed)):
        for name in ("obj_obs", "grip", "held_est", "action"):
            arrays[f"{i}_{name}"] = getattr(obs, name)
        records.append({"task": result.job.tag["task_artifact"], "rollout_seed": result.job.seed,
                        "belief": obs.scene, "prefix": str(i)})
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "observations.npz", **arrays)
    write_json(out / "manifest.json", {"episodes": records, "signals": list(arrays),
                                       "observations": artifact_ref(out / "observations.npz"),
                                       "gripper_channel": "ideal_encoder_v4", "object_channel": "noisy_observation"})
    return observed, rollout_cost(results)


def collect_recorded(runner, make_job, budget, recon, *, require_success, batch=None):
    """Match labels exactly while charging every completed job in an overshoot batch."""
    batch = batch or max(8, runner.workers * 2)
    blocks, jobs_record, total = [], [], 0
    ledger = {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0,
              "accepted_labels": 0, "available_labels": 0, "successful_episodes": 0}
    for _ in range(250):
        if total >= budget:
            break
        results = runner.map([make_job() for _ in range(batch)])
        check_results(results)
        for k, v in rollout_cost(results).items():
            ledger[k] += v
        for r in results:
            usable = (not require_success or r.success) and r.samples is not None and r.samples.n > 0
            ledger["successful_episodes"] += int(r.success)
            available = r.samples.n if usable else 0
            ledger["available_labels"] += available
            take = min(available, budget - total)
            jobs_record.append({"task": r.job.tag["task_artifact"], "rollout_seed": r.job.seed,
                                "controller": r.job.controller, "beta": r.job.beta,
                                "observation_mode": r.job.observation_mode,
                                "perturbation": asdict(r.job.pert), "latency": r.job.env_cfg.latency,
                                "provenance": r.job.provenance, "expert_noise": r.job.expert_noise,
                                "max_samples": r.job.max_samples,
                                "success": bool(r.success), "available_labels": available,
                                "accepted_labels": take, "sim_steps": r.sim_steps})
            if take:
                block = r.samples.take(take)
                block.meta = {**block.meta, "task_sha256": r.job.tag["task_artifact"]["sha256"],
                              "twin_sha256": canonical_hash(recon.to_dict()), "rollout_seed": r.job.seed}
                blocks.append(block)
                total += take
        if ledger["episodes"] >= 2 * batch and total == 0:
            raise RuntimeError("no usable training data after two batches")
    if total != budget:
        raise RuntimeError(f"collection shortfall {total}/{budget}; no unrecorded refill")
    ledger["accepted_labels"] = total
    return blocks, ledger, jobs_record


def nominal_factory(store, cfg, recon, seed):
    dist = TaskDistribution(tuple(cfg["families"]), tuple(cfg["H_values"]), seed_base=cfg["banks"]["training"])
    oc = OpenLoopConfig(**cfg.get("open_loop", {}))
    rng = np.random.default_rng(seed)
    def make():
        j = open_loop_job(dist, oc, rng)
        j.world, j.recon = cfg["world"], recon.to_dict()
        j.observation_mode = cfg.get("training_observation_mode", "legacy")
        j.verifier_cfg = VerifierConfig(tol_scale=1.0)
        j.task_payload, ref = store.get(j.family, j.H, j.task_seed)
        j.tag = {"task_artifact": ref, "twin_sha256": canonical_hash(recon.to_dict())}
        return j
    return make


def nominal_recovery_factory(store, cfg, recon, seed, round_id):
    """Fresh expert demonstrations under the current twin, with the initial recovery recipe."""
    dist = TaskDistribution(tuple(cfg["families"]), tuple(cfg["H_values"]), seed_base=cfg["banks"]["training"])
    oc = OpenLoopConfig(**cfg.get("open_loop", {}))
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, 252]))
    def make():
        j = open_loop_job(dist, oc, rng)
        j.provenance = f"nominal_recovery_r{round_id}"
        j.world, j.recon = cfg["world"], recon.to_dict()
        j.observation_mode = cfg.get("training_observation_mode", "legacy")
        j.verifier_cfg = VerifierConfig(tol_scale=1.0)
        j.task_payload, ref = store.get(j.family, j.H, j.task_seed)
        j.tag = {"task_artifact": ref, "twin_sha256": canonical_hash(recon.to_dict())}
        return j
    return make


def dagger_factory(store, cfg, recon, seed, round_id):
    dist = TaskDistribution(tuple(cfg["families"]), tuple(cfg["H_values"]), seed_base=cfg["banks"]["training"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_id, 251]))
    def make():
        fam, H, ts = dist.sample(rng)
        return job_for(store, fam, H, ts, int(rng.integers(1 << 30)), cfg, recon, real=False,
                       controller="mixed", learner="hier", beta=0.5 ** round_id,
                       pert=PerturbationConfig.suite(cfg.get("collection_pert", 0.1)),
                       label_expert=True, max_samples=300, only_success=False,
                       provenance=f"dagger_r{round_id}")
    return make


def train_version(blocks, cfg, seed, path):
    tcfg = TrainConfig(**{**cfg["train"], "seed": cfg["train"].get("seed", 0) + seed})
    import torch
    torch.set_num_threads(int(cfg.get("torch_threads", 4)))
    if tcfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    params, stats = train_policy(Dataset(blocks), "hier", tcfg)
    if tcfg.device == "cuda":
        torch.cuda.synchronize()
    stats.update({"wall_seconds": time.perf_counter() - started, "steps": tcfg.steps,
                  "sample_presentations": tcfg.steps * min(tcfg.batch, sum(b.n for b in blocks)),
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if tcfg.device == "cuda" else 0})
    record = save_policy(path, params)
    write_json(path.with_suffix(".train.json"), stats)
    return params, record, stats


def load_or_create_initial(root, cfg, condition, seed, store, workers, experiment):
    fingerprint = experiment["fingerprint"]
    directory = root / "shared" / condition / f"s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "initial.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["fingerprint"] != fingerprint:
            raise ValueError("initial cache fingerprint mismatch")
        verify_artifacts(record)
        return record, load_blocks(record["dataset"]), load_policy(record["policy"])
    recon = ReconModel.from_dict(cfg["conditions"][condition]["recon"])
    with Runner(workers) as runner:
        blocks, ledger, jobs = collect_recorded(runner, nominal_factory(store, cfg, recon, seed),
                                                cfg["init_budget"], recon, require_success=True)
    data_ref = save_blocks(directory / "data.npz", blocks)
    write_json(directory / "collection.json", {"jobs": jobs, "ledger": ledger})
    params, policy_ref, train_stats = train_version(blocks, cfg, seed, directory / "policy.npz")
    record = {"fingerprint": fingerprint, "dataset": data_ref, "policy": policy_ref,
              "data_sha256": data_ref["sha256"], "policy_sha256": policy_ref["sha256"],
              "n": cfg["init_budget"], "train_steps": cfg["train"]["steps"],
              "train_stats": train_stats, "collection_ledger": ledger, "recon": recon.to_dict(),
              "collection_record": artifact_ref(directory / "collection.json")}
    verify_artifacts(record)
    assert_source(experiment["source_sha256"])
    write_json(record_path, record)
    return record, blocks, params


def curve_evaluation_key(fingerprint, policy_sha256, recon):
    """Experiment identity binds source, task banks, modes, world, counts and workers."""
    return canonical_hash({"experiment": fingerprint, "policy": policy_sha256,
                           "recon": recon, "evaluation": "curve"})


def previous_curve_evaluation(history, cfg, experiment, policy_sha256, recon):
    """Reuse only the preceding round's identical system, including after resume."""
    if not cfg.get("reuse_pre_evaluation", False) or not history:
        return None
    last = history[-1]
    expected = curve_evaluation_key(experiment["fingerprint"], policy_sha256, recon)
    if (last.get("post_train_evaluation_key") != expected or
            last["policy_after"] != policy_sha256 or last["recon_after"] != recon):
        raise ValueError("previous evaluation does not match current policy/twin/experiment")
    return copy.deepcopy(last["post_train"])


def run_arm(root, cfg, condition, seed, arm, initial, blocks, params, store, workers, experiment):
    output_path = root / "runs" / condition / f"{arm}_s{seed}.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing["experiment"]["fingerprint"] != experiment["fingerprint"]:
            raise ValueError(f"result fingerprint mismatch: {output_path}")
        if existing.get("status") == "complete":
            verify_artifacts(existing)
            assert_source(experiment["source_sha256"])
            print(f"skip {condition}/{arm}/s{seed}", flush=True)
            return existing
    directory = root / "artifacts" / condition / f"{arm}_s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    update_twin, learn = ARMS[arm]
    recon = ReconModel.from_dict(initial["recon"])
    policy_ref = initial["policy"]
    data = Dataset(list(blocks))
    history = []
    started = time.perf_counter()
    for round_id in range(1, cfg["rounds"] + 1):
        round_dir = directory / f"round_{round_id}"
        record_path = round_dir / "round.json"
        if record_path.exists():
            saved = json.loads(record_path.read_text(encoding="utf-8"))
            if saved["fingerprint"] != experiment["fingerprint"]:
                raise ValueError("round fingerprint mismatch")
            verify_artifacts(saved)
            if saved.get("new_dataset"):
                data.add(load_blocks(saved["new_dataset"]))
            recon = ReconModel.from_dict(saved["recon_after"])
            policy_ref = saved["policy_record"]
            params = load_policy(policy_ref)
            history.append(saved)
            continue
        round_dir.mkdir(parents=True, exist_ok=True)
        n_before, recon_before, policy_before = data.n, recon.to_dict(), policy_ref["sha256"]
        # No scores are used to accept the update or select data. These are
        # independently recorded development measurements for the factorial.
        pre = previous_curve_evaluation(history, cfg, experiment, policy_before, recon_before)
        pre_reused = pre is not None
        if pre_reused:
            pre_cost = {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}
        else:
            pre, pre_cost = evaluate(params, cfg, store, recon, cfg["banks"]["curve"], cfg["curve_n_per"], workers)
        obs, feedback_cost = feedback(params, cfg, store, recon, seed, round_id, workers, round_dir / "feedback")
        geometry_report = {}
        calibration_started = time.perf_counter()
        calibration_due = update_twin and (cfg.get("calibration_schedule", "every_round") == "every_round" or round_id == 1)
        if calibration_due:
            recon, geometry_report = geometry_repair(recon, obs)
        calibration_seconds = time.perf_counter() - calibration_started if calibration_due else 0.0
        changed = canonical_hash(recon_before) != canonical_hash(recon.to_dict())
        if changed:
            post_cal, cal_cost = evaluate(params, cfg, store, recon, cfg["banks"]["curve"], cfg["curve_n_per"], workers)
        else:
            post_cal, cal_cost = copy.deepcopy(pre), {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}
        new_ref, train_stats, collection_cost = None, None, {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0, "accepted_labels": 0}
        if learn:
            use_dagger = cfg.get("learning_collection", "dagger") == "dagger"
            factory = dagger_factory if use_dagger else nominal_recovery_factory
            with Runner(workers, policy={"params": params} if use_dagger else None) as runner:
                new, collection_cost, jobs = collect_recorded(
                    runner, factory(store, cfg, recon, seed, round_id), cfg["round_budget"],
                    recon, require_success=not use_dagger)
            new_ref = save_blocks(round_dir / "data.npz", new)
            write_json(round_dir / "collection.json", {"jobs": jobs, "ledger": collection_cost})
            data.add(new)
            params, policy_ref, train_stats = train_version(data.blocks, cfg, seed, round_dir / "policy.npz")
            post_train, train_eval_cost = evaluate(params, cfg, store, recon, cfg["banks"]["curve"], cfg["curve_n_per"], workers)
        else:
            post_train, train_eval_cost = copy.deepcopy(post_cal), {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}
        record = {"round": round_id, "fingerprint": experiment["fingerprint"],
                  "n_before": n_before, "n_after": data.n, "pre": pre, "post_cal": post_cal,
                  "post_train": post_train, "recon_before": recon_before, "recon_after": recon.to_dict(),
                  "policy_before": policy_before, "policy_after": policy_ref["sha256"],
                  "policy_record": policy_ref, "new_dataset": new_ref, "geometry_report": geometry_report,
                  "collection_record": artifact_ref(round_dir / "collection.json") if learn else None,
                  "feedback_record": artifact_ref(round_dir / "feedback" / "manifest.json"),
                  "calibration_attempted": calibration_due, "calibration_changed": changed,
                  "reused_equal_state_evaluation": {"pre": pre_reused, "post_cal": not changed, "post_train": not learn},
                  "pre_evaluation_source_round": history[-1]["round"] if pre_reused else None,
                  "post_train_evaluation_key": curve_evaluation_key(experiment["fingerprint"], policy_ref["sha256"], recon.to_dict()),
                  "train_stats": train_stats,
                  "ledger": {"feedback": feedback_cost, "collection": collection_cost,
                             "pre_evaluation": pre_cost, "calibration_evaluation": cal_cost,
                             "trained_evaluation": train_eval_cost,
                             "calibration_seconds": calibration_seconds,
                             "optimizer_steps": cfg["train"]["steps"] if learn else 0}}
        verify_artifacts(record)
        assert_source(experiment["source_sha256"])
        write_json(record_path, record)
        history.append(record)
        print(f"{condition}/{arm}/s{seed} r{round_id}: n={data.n}; "
              f"SR {pre['nominal']['overall']:.3f} -> {post_cal['nominal']['overall']:.3f} -> "
              f"{post_train['nominal']['overall']:.3f}; geometry_changed={changed}", flush=True)
    final, final_cost = evaluate(params, cfg, store, recon, cfg["banks"]["final"], cfg["final_n_per"], workers)
    ledger = {"initial_training_steps_logical": initial["train_steps"],
              "additional_training_steps": sum(r["ledger"]["optimizer_steps"] for r in history),
              "initial_labels": initial["n"], "additional_labels": data.n - initial["n"],
              "feedback_episodes": sum(r["ledger"]["feedback"]["episodes"] for r in history),
              "feedback_sim_steps": sum(r["ledger"]["feedback"]["sim_steps"] for r in history),
              "collection_sim_steps": sum(r["ledger"]["collection"]["sim_steps"] for r in history),
              "final_evaluation": final_cost, "wall_seconds_this_invocation": time.perf_counter() - started,
              "initial_cost_shared_physically": True, "audit_labels": 0, "twin_replays": 0,
              "geometry_attempts": sum(r["calibration_attempted"] for r in history)}
    result = {"schema_version": 1, "status": "complete", "condition": condition, "arm": arm, "seed": seed,
              "experiment": experiment, "initial": initial, "history": history, "final": final,
              "final_recon": recon.to_dict(), "final_policy": policy_ref, "ledger": ledger,
              "limits": ["same-MuJoCo deployment proxy", "development only; no confirmation",
                         "fixed controller, not a model planner", "full-task SR, not partial-completion Score",
                         "geometry uses noisy object poses and v4 ideal gripper encoders"]}
    verify_artifacts(result)
    assert_source(experiment["source_sha256"])
    write_json(output_path, result)
    print(f"DONE {condition}/{arm}/s{seed}: final SR="
          f"{final['nominal']['overall']:.3f}/{final['perturbed']['overall']:.3f}", flush=True)
    return result


def run_experiment(cfg, out, project_root, workers=4):
    validate_config(cfg)
    out, project_root = Path(out).resolve(), Path(project_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    source_input = cfg.get("source", {}).get("path")
    inputs = {"config": cfg, "source_sha256": hashes, "workers": workers,
              "source_input_sha256": file_hash(project_root / source_input) if source_input else None}
    experiment = {**inputs, "fingerprint": canonical_hash(inputs)}
    manifest_path = out / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8"))["fingerprint"] != experiment["fingerprint"]:
        raise ValueError("output belongs to different source/config; choose a new directory")
    write_json(manifest_path, experiment)
    store = TaskStore(out / "tasks", cfg, project_root)
    for condition in cfg["conditions"]:
        for seed in cfg["seeds"]:
            initial, blocks, params = load_or_create_initial(out, cfg, condition, seed, store, workers, experiment)
            print(f"INIT {condition}/s{seed}: {initial['n']} labels; policy {initial['policy_sha256'][:12]}", flush=True)
            for arm in cfg["arms"]:
                run_arm(out, cfg, condition, seed, arm, initial, blocks, params, store, workers, experiment)
                assert_source(hashes)
    write_json(out / "completion.json", {"complete": True, "fingerprint": experiment["fingerprint"],
                                         "expected_runs": len(cfg["conditions"]) * len(cfg["seeds"]) * len(cfg["arms"]),
                                         "confirmation_executed": False})
