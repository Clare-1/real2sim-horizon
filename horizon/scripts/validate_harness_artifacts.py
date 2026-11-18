"""Independently validate saved harness artifacts; never run training or simulation.

Usage: python scripts/validate_harness_artifacts.py RESULTS [--allow-partial]
Only artifact_validation.json is written in RESULTS. Running experiments are read
at their committed initial/round/result boundaries; unfinished work is not counted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

ARMS = {"S00": (False, False), "S01": (False, True), "S10": (True, False), "S11": (True, True)}
FIELDS = ("x_hier", "x_flat", "y", "k", "stage")


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical(value)  # reject NaN/Inf, including Python JSON's permissive input forms
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_round_calibration(cfg, arm, round_id, record, current_recon):
    """Check the declared schedule independently of the runner implementation."""
    schedule = cfg.get("calibration_schedule", "every_round")
    require(schedule in ("every_round", "first_round"), "Unknown calibration_schedule")
    due = ARMS[arm][0] and (schedule == "every_round" or round_id == 1)
    require(record["calibration_attempted"] is due, "Calibration attempt differs from schedule")
    if not due:
        require(record["recon_after"] == current_recon, "Unscheduled calibration changed reconstruction")
        require(record["geometry_report"] == {} and record["ledger"]["calibration_seconds"] == 0,
                "Unscheduled calibration reported observations or charged time")
    changed = canonical(current_recon) != canonical(record["recon_after"])
    require(record["calibration_changed"] is changed, "Calibration changed flag mismatch")
    return changed


def validate_pre_evaluation(cfg, fingerprint, record, history):
    """Verify cache identity, contents and zero physical cost independently."""
    enabled = cfg.get("reuse_pre_evaluation", False)
    require(type(enabled) is bool, "Invalid reuse_pre_evaluation flag")
    reused = enabled and bool(history)
    require(record["reused_equal_state_evaluation"].get("pre", False) is reused,
            "Pre-evaluation reuse flag differs from configuration/history")
    if enabled:
        key = canonical({"experiment": fingerprint, "policy": record["policy_after"],
                         "recon": record["recon_after"], "evaluation": "curve"})
        require(record.get("post_train_evaluation_key") == key, "Cached post-training evaluation key mismatch")
    if reused:
        previous = history[-1]
        expected = canonical({"experiment": fingerprint, "policy": record["policy_before"],
                              "recon": record["recon_before"], "evaluation": "curve"})
        require(previous.get("post_train_evaluation_key") == expected and
                previous["policy_after"] == record["policy_before"] and
                previous["recon_after"] == record["recon_before"], "Cached system identity mismatch")
        require(record.get("pre_evaluation_source_round") == previous["round"], "Wrong cache source round")
        require(record["pre"] == previous["post_train"], "Cached evaluation contents changed")
        require(record["ledger"]["pre_evaluation"] == {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0},
                "Cached pre-evaluation charged new work")
    else:
        require(record.get("pre_evaluation_source_round") is None, "Unexpected cache source round")
    return reused


class Validator:
    def __init__(self, root, partial):
        self.root, self.partial = Path(root).resolve(), partial
        self.project = Path(__file__).resolve().parents[1]
        self.exp = read_json(self.root / "experiment.json")
        self.cfg, self.fingerprint = self.exp["config"], self.exp["fingerprint"]
        self.verified, self.json_seen, self.datasets, self.policies = {}, set(), {}, {}
        self.tasks, self.feedback_pairs = {}, {}
        self.initials, self.round_count, self.completed = [], 0, []
        self.pending = []
        self.costs = defaultdict(lambda: {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0})
        self.training = {"initial_calls": 0, "additional_calls": 0, "initial_optimizer_steps": 0,
                         "additional_optimizer_steps": 0, "initial_labels": 0, "additional_labels": 0,
                         "gpu_training_wall_seconds": 0.0, "cpu_training_wall_seconds": 0.0,
                         "peak_gpu_allocated_bytes_max": 0, "calibration_seconds": 0.0}
        self.source_root = None

    def file(self, path, expected):
        path = Path(path).resolve()
        if path not in self.verified:
            self.verified[path] = digest(path)
        require(self.verified[path] == expected, f"Hash mismatch: {path}")
        return path

    def refs(self, value):
        if isinstance(value, list):
            for item in value:
                self.refs(item)
        elif isinstance(value, dict):
            for pk, hk in (("path", "sha256"), ("metadata_path", "metadata_sha256")):
                if pk in value and hk in value:
                    path = self.file(value[pk], value[hk])
                    if path.suffix == ".json" and path not in self.json_seen:
                        self.json_seen.add(path)
                        self.refs(read_json(path))
            if {"family", "H", "task_seed", "key", "path", "sha256"} <= value.keys():
                self.task(value)
            for item in value.values():
                if isinstance(item, (dict, list)):
                    self.refs(item)

    def task(self, ref):
        path = self.file(ref["path"], ref["sha256"])
        if path in self.tasks:
            require(self.tasks[path] == (ref["key"], ref["family"], ref["H"], ref["task_seed"]),
                    f"Conflicting task identity: {path}")
            return
        require(path.parent == self.root / "tasks", f"Task outside this experiment: {path}")
        identity = read_json(path.with_name(path.stem + ".identity.json"))
        require(identity == {"key": ref["key"], "sha256": ref["sha256"]}, f"Task identity sidecar mismatch: {path}")
        payload = read_json(path)
        require((payload["family"], payload["H"], payload["seed"]) ==
                (ref["family"], ref["H"], ref["task_seed"]), f"Task payload identity mismatch: {path}")
        require(ref["key"] == f"{ref['family']}_H{ref['H']}_s{ref['task_seed']}", f"Invalid task key: {path}")
        require(payload["graph"]["family"] == payload["family"] and
                len(payload["graph"]["subgoals"]) == payload["H"], f"Task graph identity mismatch: {path}")
        self.tasks[path] = (ref["key"], ref["family"], ref["H"], ref["task_seed"])

    def source(self):
        require(self.cfg.get("calibration_schedule", "every_round") in ("every_round", "first_round"),
                "Unknown calibration_schedule")
        require(self.cfg.get("learning_collection", "dagger") in ("dagger", "nominal_recovery"),
                "Unknown learning_collection")
        inputs = {k: v for k, v in self.exp.items() if k != "fingerprint"}
        require(canonical(inputs) == self.fingerprint, "Experiment fingerprint does not match its contents")
        self.source_root = self.root / "source" if (self.root / "source" / "src" / "horizon").exists() else self.project
        package = self.source_root / "src" / "horizon"
        actual = {p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob("*.py"))}
        actual["scripts/run_harness.py"] = digest(self.source_root / "scripts" / "run_harness.py")
        require(actual == self.exp["source_sha256"], f"Source snapshot/hash manifest mismatch: {self.source_root}")
        if self.source_root != self.project:
            configs = list((self.source_root / "configs").glob("*.yaml"))
            require(any(canonical(yaml.safe_load(p.read_text(encoding="utf-8"))) == canonical(self.cfg)
                        for p in configs), "No frozen configuration matches experiment.config")
        if self.exp.get("source_input_sha256") is not None:
            self.file(self.project / self.cfg["source"]["path"], self.exp["source_input_sha256"])
        banks = self.cfg["banks"]
        for a, b in itertools.combinations(banks.values(), 2):
            require(abs(a - b) >= 1_000_000, "Task bank namespaces overlap")
        require(not any(max(base, 40_000_000) < min(base + 1_000_000, 41_000_000)
                        for base in banks.values()), "Reserved confirmation bank was used")

    def dataset(self, ref):
        self.refs(ref)
        key = ref["sha256"], ref["metadata_sha256"]
        if key in self.datasets:
            items = self.datasets[key]
            require(ref["n"] == sum(item["n"] for item in items) and ref["blocks"] == len(items),
                    "Cached dataset counts disagree")
            return items
        meta = read_json(ref["metadata_path"])
        require(meta["version"] == 1 and meta["n"] == ref["n"] and len(meta["blocks"]) == ref["blocks"],
                f"Dataset sidecar counts disagree: {ref['path']}")
        items = meta["blocks"]
        with np.load(ref["path"], allow_pickle=False) as archive:
            require(set(archive.files) == {f"block{i}_{field}" for i in range(len(items)) for field in FIELDS},
                    f"Unexpected dataset arrays: {ref['path']}")
            for i, item in enumerate(items):
                require(type(item["n"]) is int and item["n"] > 0, "Empty or invalid saved sample block")
                for field in FIELDS:
                    array = archive[f"block{i}_{field}"]
                    require(not array.dtype.hasobject and np.isfinite(array).all(), "Invalid training array")
                    require(array.ndim == (1 if field in ("k", "stage") else 2) and array.shape[0] == item["n"],
                            f"Wrong shape in {ref['path']} / block{i}_{field}")
        require(sum(item["n"] for item in items) == ref["n"], "Actual dataset n disagrees with budget record")
        self.datasets[key] = items
        return items

    def policy(self, ref):
        self.refs(ref)
        if ref["sha256"] not in self.policies:
            weights = hashlib.sha256()
            with np.load(ref["path"], allow_pickle=False) as archive:
                require(str(archive["arch"]) == "mlp" and int(archive["n_layers"]) == self.cfg["train"]["layers"],
                        f"Policy architecture mismatch: {ref['path']}")
                require("x_sd" in archive.files and np.all(archive["x_sd"] > 0),
                        f"Policy normalization scales must be positive: {ref['path']}")
                parameter_names = set(archive.files) - {"arch", "n_layers", "x_mu", "x_sd"}
                require({"w0", "b0", "wo", "bo"} <= parameter_names, "Policy has no saved learned weights")
                for name in sorted(archive.files):
                    array = archive[name]
                    require(not array.dtype.hasobject, "Policy contains object array")
                    if array.dtype.kind in "fciub":
                        require(np.isfinite(array).all(), "Policy contains non-finite parameters")
                    if name in parameter_names:
                        weights.update(name.encode())
                        weights.update(str((array.shape, array.dtype.str)).encode())
                        weights.update(array.tobytes())
            self.policies[ref["sha256"]] = weights.hexdigest()
        return self.policies[ref["sha256"]]

    def collection(self, ref, ledger, dataset, recon, round_id=None):
        self.refs(ref)
        collection = read_json(ref["path"])
        require(collection["ledger"] == ledger, "Collection ledgers disagree")
        jobs = collection["jobs"]
        require(len(jobs) == ledger["episodes"], "Completed collection jobs are not fully counted")
        for field in ("sim_steps", "available_labels", "accepted_labels"):
            require(sum(job[field] for job in jobs) == ledger[field], f"Collection {field} sum mismatch")
        require(sum(bool(job["success"]) for job in jobs) == ledger["successful_episodes"], "Collection success count mismatch")
        items = self.dataset(dataset)
        require(dataset["n"] == ledger["accepted_labels"], "Dataset labels do not equal charged accepted labels")
        accepted = [job for job in jobs if job["accepted_labels"]]
        require(len(accepted) == len(items), "Saved block count does not match accepted jobs")
        remaining = dataset["n"]
        for job in jobs:
            require(0 <= job["accepted_labels"] <= job["available_labels"], "Invalid label truncation")
            require(job["accepted_labels"] == min(job["available_labels"], remaining), "Collection order/truncation changed")
            remaining -= job["accepted_labels"]
            base = self.cfg["banks"]["training"]
            require(base <= job["task"]["task_seed"] < base + 1_000_000, "Training task outside training bank")
            if round_id is not None:
                collection_kind = self.cfg.get("learning_collection", "dagger")
                if collection_kind == "dagger":
                    require(job["controller"] == "mixed" and job["beta"] == 0.5 ** round_id,
                            "DAgger beta/controller mismatch")
                    require(job.get("max_samples", 300) == 300 and job["available_labels"] <= 300,
                            "DAgger per-episode label cap differs")
                else:
                    require(collection_kind == "nominal_recovery", "Unknown learning_collection")
                    require(job["controller"] == "expert" and job["beta"] == 0.0,
                            "Nominal/recovery collection must execute only the expert")
                    require(job.get("provenance") == f"nominal_recovery_r{round_id}" and job["max_samples"] is None,
                            "Nominal/recovery collection recipe differs")
                    oc = self.cfg.get("open_loop", {})
                    require(job["expert_noise"] == oc.get("expert_noise", 0.0) and
                            job["latency"] in oc.get("latency_choices", [0, 0, 1]),
                            "Nominal/recovery noise or latency differs")
                    recovery = oc.get("recovery_pert", {"slip": 0.15, "displace": 0.15, "knock": 0.1,
                                                         "traj": 0.25, "target": 0.25})
                    nominal = {key: 0.0 for key in ("slip", "displace", "knock", "traj", "target")}
                    fraction = oc.get("recovery_frac", 0.3)
                    require((fraction < 1 and job["perturbation"] == nominal) or
                            (fraction > 0 and job["perturbation"] == recovery),
                            "Nominal/recovery perturbation recipe differs")
                    require(job["success"] or (job["available_labels"] == job["accepted_labels"] == 0),
                            "Nominal/recovery collection retained a failed episode")
        for item, job in zip(items, accepted):
            observation_mode = self.cfg.get("training_observation_mode", "legacy")
            require(observation_mode in ("legacy", "observed"), "Unknown training observation mode")
            require(item["meta"].get("observation_mode", "legacy") == observation_mode,
                    "Dataset observation mode differs from config")
            channel = "ideal_encoder" if observation_mode == "legacy" else "actor_observation"
            require(item["meta"].get("gripper_channel", "ideal_encoder") == channel and
                    job.get("observation_mode", "legacy") == observation_mode,
                    "Training observation channel differs from config")
            require(item["n"] == job["accepted_labels"] and item["key"] == job["task"]["key"], "Block/job identity mismatch")
            require(item["meta"]["task_sha256"] == job["task"]["sha256"] and
                    item["meta"]["rollout_seed"] == job["rollout_seed"], "Block provenance mismatch")
            require(item["meta"]["twin_sha256"] == canonical(recon), "Training data twin is not recon_after")
            if round_id is not None:
                prefix = "dagger" if self.cfg.get("learning_collection", "dagger") == "dagger" else "nominal_recovery"
                require(item["provenance"] == f"{prefix}_r{round_id}", "Wrong round dataset provenance")

    def add_cost(self, name, cost):
        for key in ("episodes", "sim_steps", "worker_seconds"):
            value = cost[key]
            require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, f"Invalid {name}/{key}")
            self.costs[name][key] += value

    def training_cost(self, stats, n, initial=False):
        steps = self.cfg["train"]["steps"]
        require(stats["n"] == n and stats["steps"] == steps, "Training n/step budget mismatch")
        require(stats["sample_presentations"] == steps * min(self.cfg["train"]["batch"], n), "Training sample presentations mismatch")
        require(stats["device"] == self.cfg["train"]["device"], "Training device differs from config")
        require(stats.get("normalization", "legacy") == self.cfg["train"].get("normalization", "legacy"),
                "Training normalization differs from config")
        label = "initial" if initial else "additional"
        self.training[label + "_calls"] += 1
        self.training[label + "_optimizer_steps"] += steps
        seconds = stats["wall_seconds"]
        require(math.isfinite(seconds) and seconds >= 0, "Invalid training wall time")
        self.training[("gpu" if str(stats["device"]).startswith("cuda") else "cpu") + "_training_wall_seconds"] += seconds
        self.training["peak_gpu_allocated_bytes_max"] = max(self.training["peak_gpu_allocated_bytes_max"], stats["peak_gpu_allocated_bytes"])

    def evaluation(self, evaluation, cost, bank, n_per, reused=False):
        expected = []
        for family in self.cfg["families"]:
            for H in self.cfg["H_values"]:
                for i in range(n_per):
                    ts = bank + {"pack": 0, "sort": 1, "stack": 2}[family] * 100_000 + H * 1000 + i
                    expected.append((family, H, ts, ts + 17))
        for mode in ("nominal", "perturbed"):
            episodes = evaluation[mode]["episodes"]
            require([(e["family"], e["H"], e["task_seed"], e["rollout_seed"]) for e in episodes] == expected,
                    "Evaluation task identities/order differ from declared bank")
            require(math.isclose(evaluation[mode]["overall"], sum(e["success"] for e in episodes) / len(episodes), abs_tol=1e-12),
                    "Evaluation SR arithmetic mismatch")
            for e in episodes:
                require(type(e["success"]) is bool and 0 <= e["progress"] <= 1, "Invalid success/progress score")
                key = f"{e['family']}_H{e['H']}_s{e['task_seed']}"
                self.task({"path": str(self.root / "tasks" / (key + ".json")), "sha256": e["task_sha256"],
                           "key": key, "family": e["family"], "H": e["H"], "task_seed": e["task_seed"]})
        require(cost["episodes"] == (0 if reused else 2 * len(expected)), "Evaluation rollout accounting mismatch")
        if reused:
            require(cost["sim_steps"] == 0 and cost["worker_seconds"] == 0, "Reused evaluation charged new work")

    def feedback(self, record, condition, seed, round_id):
        ref = record["feedback_record"]
        self.refs(ref)
        manifest = read_json(ref["path"])
        cells = list(itertools.product(self.cfg["families"], self.cfg["H_values"]))
        identities = []
        require(manifest["gripper_channel"] == "ideal_encoder_v4" and manifest["object_channel"] == "noisy_observation",
                "Feedback channel assumptions differ from the frozen protocol")
        with np.load(manifest["observations"]["path"], allow_pickle=False) as archive:
            expected_arrays = set()
            require(len(manifest["episodes"]) == self.cfg["feedback_episodes"], "Feedback episode count mismatch")
            for i, item in enumerate(manifest["episodes"]):
                family, H = cells[i % len(cells)]
                ts = self.cfg["banks"]["feedback"] + seed * 100_000 + round_id * 10_000 + H * 100 + i
                task = item["task"]
                identity = (task["family"], task["H"], task["task_seed"], item["rollout_seed"], task["sha256"])
                require(identity[:4] == (family, H, ts, ts + 31), "Feedback identity differs from declared paired schedule")
                identities.append(identity)
                prefix = item["prefix"]
                require(prefix == str(i), "Feedback array prefix does not match its episode")
                arrays = {field: archive[f"{prefix}_{field}"] for field in ("obj_obs", "grip", "held_est", "action")}
                expected_arrays.update(f"{prefix}_{field}" for field in arrays)
                T = len(arrays["grip"])
                require(T > 0 and arrays["grip"].shape == (T, 6) and arrays["action"].shape == (T, 5)
                        and arrays["held_est"].shape == (T,) and arrays["obj_obs"].shape == (T, len(item["belief"]["objects"]), 7),
                        "Feedback observation shape mismatch")
                require(all(np.isfinite(a).all() for a in arrays.values()), "Nonfinite feedback signals")
            require(set(archive.files) == expected_arrays == set(manifest["signals"]), "Unexpected feedback signals")
        group = condition, seed, round_id
        if group in self.feedback_pairs:
            require(self.feedback_pairs[group] == identities, "Arms do not share feedback tasks/seeds/hashes")
        else:
            self.feedback_pairs[group] = identities
        require(record["ledger"]["feedback"]["episodes"] == len(identities), "Feedback resource count mismatch")

    def initial(self, condition, seed):
        path = self.root / "shared" / condition / f"s{seed}" / "initial.json"
        if not path.exists():
            require(self.partial, f"Missing initial record: {path}")
            require(not any((self.root / "runs" / condition).glob(f"*_s{seed}.json")) and
                    not any((self.root / "artifacts" / condition).glob(f"*_s{seed}/round_*/round.json")),
                    f"Committed runs/rounds exist without their shared initial record: {path}")
            self.pending.append(str(path.relative_to(self.root)))
            return None
        record = read_json(path)
        require(record["fingerprint"] == self.fingerprint, "Initial fingerprint mismatch")
        self.refs(record)
        require(record["n"] == record["dataset"]["n"] == self.cfg["init_budget"], "Initial label budget mismatch")
        require(record["data_sha256"] == record["dataset"]["sha256"] and record["policy_sha256"] == record["policy"]["sha256"],
                "Initial hash aliases disagree")
        expected = self.cfg["conditions"][condition]["recon"]
        require(record["recon"] == {"bias": expected.get("bias", {}), "corrections": expected.get("corrections", {})},
                "Initial reconstruction differs from condition")
        self.policy(record["policy"])
        require(read_json(Path(record["policy"]["path"]).with_suffix(".train.json")) == record["train_stats"],
                "Initial training stats sidecar differs from the committed record")
        self.collection(record["collection_record"], record["collection_ledger"], record["dataset"], record["recon"])
        self.training_cost(record["train_stats"], record["n"], initial=True)
        require(record["train_steps"] == self.cfg["train"]["steps"], "Initial step count mismatch")
        self.training["initial_labels"] += record["n"]
        self.add_cost("initial_collection", record["collection_ledger"])
        self.initials.append({"condition": condition, "seed": seed, "data_sha256": record["data_sha256"],
                              "policy_sha256": record["policy_sha256"]})
        return record

    def arm(self, condition, seed, arm, initial):
        path = self.root / "runs" / condition / f"{arm}_s{seed}.json"
        result = read_json(path) if path.exists() else None
        _, learn = ARMS[arm]
        if result is not None:
            require(result["status"] == "complete" and (result["condition"], result["seed"], result["arm"]) ==
                    (condition, seed, arm), "Run identity/status mismatch")
            require(result["experiment"] == self.exp and result["initial"] == initial, "Arms do not share the same experiment/D0/pi0")
            self.refs(result)
            require(len(result["history"]) == self.cfg["rounds"], "Completed run has missing rounds")
        else:
            require(self.partial, f"Missing run: {path}")
            self.pending.append(str(path.relative_to(self.root)))
        current_policy, current_recon, n = initial["policy"], initial["recon"], initial["n"]
        history, missing = [], False
        directory = self.root / "artifacts" / condition / f"{arm}_s{seed}"
        for r in range(1, self.cfg["rounds"] + 1):
            round_dir = directory / f"round_{r}"
            round_path = round_dir / "round.json"
            if not round_path.exists():
                require(result is None, f"Completed run missing {round_path}")
                missing = True
                continue
            require(not missing, "Committed round sequence has a gap")
            record = read_json(round_path)
            self.refs(record)
            if result:
                require(record == result["history"][r - 1], "Saved round differs from completed history")
            require(record["round"] == r and record["fingerprint"] == self.fingerprint, "Round identity/fingerprint mismatch")
            require(record["n_before"] == n and record["policy_before"] == current_policy["sha256"] and
                    record["recon_before"] == current_recon, "Round chain does not continue previous data/policy/twin")
            require(record["policy_after"] == record["policy_record"]["sha256"], "Policy record/hash mismatch")
            changed = validate_round_calibration(self.cfg, arm, r, record, current_recon)
            self.feedback(record, condition, seed, r)
            before_weights = self.policy(current_policy)
            after_weights = self.policy(record["policy_record"])
            if learn:
                require(record["new_dataset"] is not None and record["new_dataset"]["n"] == self.cfg["round_budget"],
                        "Learning round label budget mismatch")
                require(Path(record["policy_record"]["path"]).resolve() == (round_dir / "policy.npz").resolve() and
                        Path(record["new_dataset"]["path"]).resolve() == (round_dir / "data.npz").resolve(),
                        "Learning artifacts not saved under their round")
                require(before_weights != after_weights, "Learning round did not change actual learned weight arrays")
                require(read_json(Path(record["policy_record"]["path"]).with_suffix(".train.json")) == record["train_stats"],
                        "Round training stats sidecar differs from the committed record")
                n += self.cfg["round_budget"]
                self.collection(record["collection_record"], record["ledger"]["collection"], record["new_dataset"], record["recon_after"], r)
                self.training_cost(record["train_stats"], n)
                self.training["additional_labels"] += self.cfg["round_budget"]
            else:
                require(record["new_dataset"] is None and record["train_stats"] is None and record["collection_record"] is None,
                        "Frozen policy arm collected/trained new data")
                require(not any((round_dir / filename).exists() for filename in ("data.npz", "policy.npz", "policy.train.json")),
                        "Frozen policy arm has a new data/policy/training artifact")
                require(record["policy_record"] == current_policy and after_weights == before_weights,
                        "Frozen policy arm changed its policy")
                require(record["ledger"]["collection"]["accepted_labels"] == 0 and
                        record["ledger"]["collection"]["episodes"] == 0, "Frozen policy arm charged collection")
            require(record["n_after"] == n and record["ledger"]["optimizer_steps"] == (self.cfg["train"]["steps"] if learn else 0),
                    "Round data/optimizer totals mismatch")
            pre_reused = validate_pre_evaluation(self.cfg, self.fingerprint, record, history)
            self.evaluation(record["pre"], record["ledger"]["pre_evaluation"], self.cfg["banks"]["curve"], self.cfg["curve_n_per"], pre_reused)
            self.evaluation(record["post_cal"], record["ledger"]["calibration_evaluation"], self.cfg["banks"]["curve"], self.cfg["curve_n_per"], not changed)
            self.evaluation(record["post_train"], record["ledger"]["trained_evaluation"], self.cfg["banks"]["curve"], self.cfg["curve_n_per"], not learn)
            if not changed:
                require(record["pre"] == record["post_cal"], "Unchanged twin reused a different evaluation")
            if not learn:
                require(record["post_train"] == record["post_cal"], "Frozen policy reused a different evaluation")
            for label, ledger_key in (("feedback", "feedback"), ("learning_collection", "collection"),
                                      ("development_evaluation", "pre_evaluation"), ("development_evaluation", "calibration_evaluation"),
                                      ("development_evaluation", "trained_evaluation")):
                self.add_cost(label, record["ledger"][ledger_key])
            calibration_seconds = record["ledger"]["calibration_seconds"]
            require(math.isfinite(calibration_seconds) and calibration_seconds >= 0, "Invalid calibration duration")
            self.training["calibration_seconds"] += calibration_seconds
            current_policy, current_recon = record["policy_record"], record["recon_after"]
            history.append(record)
            self.round_count += 1
        if result:
            require(result["final_policy"] == current_policy and result["final_recon"] == current_recon, "Final system differs from last round")
            ledger = result["ledger"]
            require(ledger["initial_labels"] == initial["n"] and ledger["additional_labels"] == n - initial["n"], "Run label totals mismatch")
            require(ledger["initial_training_steps_logical"] == initial["train_steps"] and
                    ledger["additional_training_steps"] == sum(x["ledger"]["optimizer_steps"] for x in history), "Run training totals mismatch")
            for field, category, key in (("feedback_episodes", "feedback", "episodes"),
                                          ("feedback_sim_steps", "feedback", "sim_steps"), ("collection_sim_steps", "collection", "sim_steps")):
                require(ledger[field] == sum(x["ledger"][category][key] for x in history), f"Run {field} mismatch")
            require(ledger["audit_labels"] == ledger["twin_replays"] == 0 and ledger["initial_cost_shared_physically"] is True,
                    "Run resource protocol mismatch")
            require(ledger["geometry_attempts"] == sum(h["calibration_attempted"] for h in history),
                    "Geometry attempt total differs from scheduled rounds")
            self.evaluation(result["final"], ledger["final_evaluation"], self.cfg["banks"]["final"], self.cfg["final_n_per"])
            self.add_cost("final_evaluation", ledger["final_evaluation"])
            self.completed.append({"condition": condition, "seed": seed, "arm": arm})

    def run(self):
        self.source()
        for condition in self.cfg["conditions"]:
            for seed in self.cfg["seeds"]:
                initial = self.initial(condition, seed)
                if initial is not None:
                    for arm in self.cfg["arms"]:
                        self.arm(condition, seed, arm, initial)
        expected = len(self.cfg["conditions"]) * len(self.cfg["seeds"]) * len(self.cfg["arms"])
        completion_path = self.root / "completion.json"
        complete = len(self.completed) == expected and completion_path.exists()
        if completion_path.exists():
            completion = read_json(completion_path)
            require(completion["complete"] is True and completion["expected_runs"] == expected and
                    completion["fingerprint"] == self.fingerprint and completion["confirmation_executed"] is False,
                    "Completion manifest mismatch")
            require(len(self.completed) == expected, "Completion claims missing runs are complete")
        require(complete or self.partial, "Matrix is incomplete; use --allow-partial for committed artifacts")
        # Also check materialized but unused overshoot tasks after a complete run.
        if complete:
            for path in (self.root / "tasks").glob("*.json"):
                if path.name.endswith(".identity.json"):
                    require(path.with_name(path.name.replace(".identity.json", ".json")).exists(), "Dangling task identity")
                else:
                    payload = read_json(path)
                    self.task({"path": str(path), "sha256": digest(path), "key": path.stem,
                               "family": payload["family"], "H": payload["H"], "task_seed": payload["seed"]})
        return {"passed": True, "complete": complete, "partial": not complete, "expected_runs": expected,
                "validated_runs": len(self.completed), "validated_rounds": self.round_count,
                "validated_initial_states": len(self.initials), "verified_files": len(self.verified),
                "verified_tasks": len(self.tasks), "verified_dataset_versions": len(self.datasets),
                "verified_policy_versions": len(self.policies), "shared_initial_states": self.initials,
                "completed_runs": self.completed, "pending": self.pending,
                "source_root": str(self.source_root), "experiment_fingerprint": self.fingerprint,
                "physical_resources_committed": {"training": self.training, "rollouts": dict(self.costs)},
                "checks": ["source/config/fingerprint", "recursive artifact hashes and task identities", "actual NPZ counts and finite arrays",
                           "shared D0/pi0", "sequential policy/data/twin versions", "round data matches recon_after",
                           "frozen arm invariants and actual learned weight changes", "paired feedback and evaluation task identities",
                           "exact label budgets and full collection batch accounting", "shared initial resources counted once"],
                "limits": ["Counts cover committed initial/round/result records; uncommitted interrupted work is not reconstructible here",
                           "Worker seconds are cumulative process time, not elapsed experiment time",
                           "GPU training wall time is synchronized training-call duration, not a CUDA-event kernel timer",
                           "Feedback/evaluation physics-step totals lack per-episode counters; their stored totals are checked for consistency/nonnegativity",
                           "Artifact validity does not establish an algorithmic or real-world performance advantage"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    report = {"passed": False}
    try:
        report = Validator(args.root, args.allow_partial).run()
    except Exception as exc:
        report = {"passed": False, "complete": False, "error_type": type(exc).__name__, "error": str(exc)}
    report.update(validator_sha256=digest(__file__), validation_started_utc=started,
                  root=str(args.root.resolve()), allow_partial=args.allow_partial)
    output = args.root / "artifact_validation.json"
    with tempfile.NamedTemporaryFile(dir=args.root, mode="w", encoding="utf-8", delete=False, suffix=".tmp") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, output)
    print(json.dumps({k: report[k] for k in ("passed", "complete", "validated_runs", "validated_rounds", "error") if k in report}, ensure_ascii=False))
    print(str(output.resolve()))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
