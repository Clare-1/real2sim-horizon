"""Protocol and budget tests for the factorial learning runner."""
from types import SimpleNamespace
import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import yaml
from pathlib import Path

from horizon.harness.experiment import (TaskStore, artifact_ref, assert_source, collect_recorded,
                                        observed_feedback, validate_config, verify_artifacts)
from horizon.policy.dataset import SampleBlock
from horizon.real2sim.recon_error import ReconModel


def block(n=3):
    return SampleBlock("test", "pack", 2, "test", np.zeros((n, 4), np.float32),
                       np.zeros((n, 5), np.float32), np.zeros((n, 5), np.float32),
                       np.zeros(n, np.int16), np.zeros(n, np.int16))


def result(error=None):
    from horizon.datagen.parallel import Job
    job = Job("pack", 2, 1, 2, tag={"task_artifact": {"sha256": "test"}})
    return SimpleNamespace(error=error, job=job, samples=block(), success=True, sim_steps=25, wall=1.5)


class BatchRunner:
    workers = 1

    def __init__(self, results):
        self.results = results

    def map(self, jobs):
        return self.results


def test_exact_label_budget_charges_whole_completed_batch():
    blocks, ledger, jobs = collect_recorded(BatchRunner([result() for _ in range(4)]), lambda: None,
                                           4, ReconModel(), require_success=False, batch=4)
    assert [b.n for b in blocks] == [3, 1]
    assert ledger["episodes"] == 4 and ledger["sim_steps"] == 100
    assert ledger["worker_seconds"] == 6 and ledger["available_labels"] == 12
    assert [r["accepted_labels"] for r in jobs] == [3, 1, 0, 0]


def test_worker_failure_after_label_budget_is_not_silently_ignored():
    with pytest.raises(RuntimeError, match="late error"):
        collect_recorded(BatchRunner([result(), result("late error")]), lambda: None,
                         2, ReconModel(), require_success=False, batch=2)


def test_geometry_feedback_has_no_scoring_or_snapshot_fields():
    log = SimpleNamespace(scene={"hidden": "physical"}, meta={"belief": {"source": "estimated"}},
                          obj_obs=np.zeros((2, 1, 7)), grip=np.zeros((2, 5)),
                          held_est=np.array([-1, -1]), action=np.zeros((2, 5)),
                          success=True, sat_true=np.ones((2, 1)), snapshots={"secret": 1})
    obs = observed_feedback(log)
    assert set(vars(obs)) == {"scene", "meta", "obj_obs", "grip", "held_est", "action"}
    assert obs.scene == {"source": "estimated"}
    assert "physical" not in str(vars(obs))


def config():
    return yaml.safe_load((Path(__file__).parents[1] / "configs/harness_e1.yaml").read_text())


def test_protocol_disallows_confirmation_and_overlap():
    cfg = config()
    validate_config(cfg)
    cfg["banks"]["final"] = 40_000_000
    with pytest.raises(ValueError, match="historical/reserved"):
        validate_config(cfg)
    cfg = config()
    cfg["banks"]["final"] = cfg["banks"]["curve"]
    with pytest.raises(ValueError, match="overlapping"):
        validate_config(cfg)


def test_factorial_keeps_judge_fixed():
    cfg = config()
    cfg["judge_scale"] = 0.4
    with pytest.raises(ValueError, match="judge"):
        validate_config(cfg)


def test_task_identity_cannot_silently_change_between_resumes(tmp_path):
    import json
    cfg = config()
    store = TaskStore(tmp_path / "tasks", cfg, tmp_path)
    payload, ref = store.get("pack", 2, 54_000_009)
    payload["scene"]["scene_id"] = "changed-after-initialization"
    Path(ref["path"]).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="changed since creation"):
        TaskStore(tmp_path / "tasks", cfg, tmp_path).get("pack", 2, 54_000_009)


def test_recursive_artifact_check_catches_modified_task(tmp_path):
    import json
    task = tmp_path / "task.json"
    task.write_text('{"name":"original"}')
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({"jobs": [{"task": artifact_ref(task)}]}))
    record = {"collection_record": artifact_ref(manifest)}
    verify_artifacts(record)
    task.write_text('{"name":"changed"}')
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_artifacts(record)


def test_source_drift_prevents_commit(monkeypatch):
    import horizon.harness.experiment as experiment
    monkeypatch.setattr(experiment, "source_hashes", lambda: {"code.py": "changed"})
    with pytest.raises(RuntimeError, match="refusing to commit"):
        assert_source({"code.py": "original"})


def test_observation_allowlist_preserves_v4_geometry_estimate():
    from horizon.datagen.parallel import Job, run_serial
    from horizon.rsi.simreal import geometry_repair
    recon = ReconModel(bias={"tray": 0.3})
    results = run_serial([Job("pack", 2, s, s, controller="expert", real=True,
                              world={"tray_height": 0.02}, recon=recon.to_dict(),
                              keep_log=True, want_samples=False) for s in range(3)])
    assert all(r.error is None for r in results)
    raw = [r.log for r in results]
    a, report_a = geometry_repair(recon, raw)
    b, report_b = geometry_repair(recon, [observed_feedback(log) for log in raw])
    assert a.to_dict() == b.to_dict() and report_a == report_b
    assert "tray" in report_a and b.corrections


@pytest.mark.parametrize("field,value", [("calibration_schedule", "best_round"),
                                         ("learning_collection", "unmixed"),
                                         ("calibration_schedule", None),
                                         ("learning_collection", None)])
def test_unknown_protocol_options_are_rejected(field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(cfg)


@pytest.mark.parametrize("recovery_fraction", [0.0, 1.0])
def test_nominal_recovery_factory_executes_fresh_expert_jobs(tmp_path, recovery_fraction):
    from horizon.harness.experiment import nominal_recovery_factory
    from horizon.datagen.generator import OpenLoopConfig
    cfg = config()
    cfg["open_loop"]["recovery_frac"] = recovery_fraction
    store = TaskStore(tmp_path / "tasks", cfg, tmp_path)
    recon = ReconModel(bias={"tray": 0.3}, corrections={"tray": 3.2})
    factory = nominal_recovery_factory(store, cfg, recon, 1, 2)
    jobs = [factory() for _ in range(4)]
    repeated = nominal_recovery_factory(store, cfg, recon, 1, 2)
    assert [asdict(j) for j in jobs] == [asdict(repeated()) for _ in range(4)]
    other_round = nominal_recovery_factory(store, cfg, recon, 1, 3)()
    assert (jobs[0].task_seed, jobs[0].seed) != (other_round.task_seed, other_round.seed)
    expected_pert = asdict(OpenLoopConfig().recovery_pert) if recovery_fraction else {
        key: 0.0 for key in ("slip", "displace", "knock", "traj", "target")}
    for job in jobs:
        assert job.controller == "expert" and job.beta == 0.0 and not job.real
        assert job.only_success and job.max_samples is None and job.task_payload is not None
        assert job.world == cfg["world"] and job.recon == recon.to_dict()
        assert job.provenance == "nominal_recovery_r2" and asdict(job.pert) == expected_pert


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("arm,collection,schedule,expected_calls", [
    ("S10", "dagger", None, 3),       # Legacy default still calibrates every round.
    ("S10", "dagger", "first_round", 1),
    ("S11", "dagger", "first_round", 1),
    ("S11", "nominal_recovery", "first_round", 1),
    ("S01", "nominal_recovery", "first_round", 0),
    ("S00", "nominal_recovery", "first_round", 0),
])
def test_round_execution_respects_schedule_collection_and_resume(tmp_path, monkeypatch, arm, collection, schedule, expected_calls, reuse):
    """Exercise the real round state machine and persistence without physics or optimization."""
    import horizon.harness.experiment as ex
    from horizon.harness.artifacts import save_policy, write_json
    cfg = config()
    cfg.update(init_budget=2, round_budget=4, rounds=3, feedback_episodes=1, learning_collection=collection)
    cfg["reuse_pre_evaluation"] = reuse
    cfg["train"]["steps"] = 4
    if schedule is not None:
        cfg["calibration_schedule"] = schedule
    counts = {"calibrate": 0, "train": 0, "evaluate": 0}
    policies_sent = []

    class FakeRunner:
        def __init__(self, workers, policy=None):
            self.workers = workers
            policies_sent.append(policy)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, jobs):
            outputs = []
            for i, job in enumerate(jobs):
                sample = block(3)
                sample.key = job.tag["task_artifact"]["key"]
                sample.family, sample.H, sample.provenance = job.family, job.H, job.provenance
                outputs.append(SimpleNamespace(job=job, samples=sample, success=i > 0,
                                               error=None, sim_steps=25, wall=0.1))
            return outputs

    def fake_evaluate(*args):
        counts["evaluate"] += 1
        return {m: {"overall": 0.5, "episodes": []} for m in ("nominal", "perturbed")}, {
            "episodes": 2, "sim_steps": 50, "worker_seconds": 0.1}

    def fake_feedback(params, cfg, store, recon, seed, round_id, workers, directory):
        write_json(directory / "manifest.json", {"episodes": []})
        return [], {"episodes": 1, "sim_steps": 25, "worker_seconds": 0.1}

    def fake_calibrate(recon, logs):
        counts["calibrate"] += 1
        updated = ReconModel(dict(recon.bias), {"tray": float(counts["calibrate"])})
        return updated, {"tray": {"estimated": 0.02, "n": 6}}

    def fake_train(blocks, cfg, seed, path):
        counts["train"] += 1
        params = {"arch": "mlp", "n_layers": 1, "w0": np.array([counts["train"]], np.float32)}
        return params, save_policy(path, params), {"n": sum(b.n for b in blocks)}

    monkeypatch.setattr(ex, "Runner", FakeRunner)
    monkeypatch.setattr(ex, "evaluate", fake_evaluate)
    monkeypatch.setattr(ex, "feedback", fake_feedback)
    monkeypatch.setattr(ex, "geometry_repair", fake_calibrate)
    monkeypatch.setattr(ex, "train_version", fake_train)
    monkeypatch.setattr(ex, "assert_source", lambda expected: None)
    params = {"arch": "mlp", "n_layers": 1, "w0": np.zeros(1, np.float32)}
    initial = {"n": 2, "train_steps": 4, "policy": save_policy(tmp_path / "initial.npz", params),
               "recon": ReconModel().to_dict()}
    store = TaskStore(tmp_path / "tasks", cfg, tmp_path)
    experiment = {"fingerprint": "synthetic", "source_sha256": {}}
    result = ex.run_arm(tmp_path, cfg, "healthy", 0, arm, initial, [block(2)], params, store, 1, experiment)
    learn = arm in ("S01", "S11")
    assert counts["calibrate"] == result["ledger"]["geometry_attempts"] == expected_calls
    assert counts["train"] == (3 if learn else 0)
    assert counts["evaluate"] == 4 + expected_calls + (3 if learn else 0) - (2 if reuse else 0)
    assert result["ledger"]["additional_labels"] == (12 if learn else 0)
    for r in result["history"]:
        expected_reuse = reuse and r["round"] > 1
        assert r["reused_equal_state_evaluation"]["pre"] is expected_reuse
        assert r["ledger"]["pre_evaluation"]["episodes"] == (0 if expected_reuse else 2)
        if expected_reuse:
            assert r["pre"] == result["history"][r["round"] - 2]["post_train"]
        if not r["calibration_attempted"]:
            assert r["recon_before"] == r["recon_after"]
            assert r["geometry_report"] == {} and r["ledger"]["calibration_seconds"] == 0
            assert r["pre"] == r["post_cal"] and r["ledger"]["calibration_evaluation"]["episodes"] == 0
        if learn:
            saved = json.loads(Path(r["collection_record"]["path"]).read_text())
            jobs = saved["jobs"]
            assert sum(j["accepted_labels"] for j in jobs) == 4
            if collection == "nominal_recovery":
                assert all(j["controller"] == "expert" for j in jobs)
                assert jobs[0]["available_labels"] == jobs[0]["accepted_labels"] == 0
                assert all(p is None for p in policies_sent)
            else:
                assert all(j["controller"] == "mixed" and j["beta"] == 0.5 ** r["round"] for j in jobs)
                assert jobs[0]["accepted_labels"] == 3
    before = copy.deepcopy(counts)
    # Completed rounds are reloaded after interruption before the final commit;
    # calibration and training must not run a second time.
    (tmp_path / "runs/healthy" / f"{arm}_s0.json").unlink()
    resumed = ex.run_arm(tmp_path, cfg, "healthy", 0, arm, initial, [block(2)], params, store, 1, experiment)
    assert counts == {**before, "evaluate": before["evaluate"] + 1}
    assert resumed["final_recon"] == result["final_recon"]
    assert resumed["final_policy"] == result["final_policy"]


@pytest.mark.parametrize("final_sat,expected_final,expected_max,expected_first", [
    ([True, False, True], 2, 3, 1),
    ([True, True, True], 3, 3, 3),  # Progress can improve on the final, unlogged post-action state.
])
def test_evaluation_predicate_diagnostics_are_distinct_from_full_success(monkeypatch, final_sat, expected_final, expected_max, expected_first):
    import horizon.harness.experiment as ex
    job = SimpleNamespace(family="pack", H=3, task_seed=1, seed=18,
                          tag={"task_artifact": {"sha256": "task"}})
    log = SimpleNamespace(success=False, success_ver=True, steps=4,
                          meta={"final_sat": final_sat},
                          sat_true=np.array([[True, True, False], [True, True, final_sat[1] is False]]))
    output = SimpleNamespace(job=job, log=log, error=None, sim_steps=100, wall=0.1)

    class FakeRunner:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, jobs): return [output]

    monkeypatch.setattr(ex, "Runner", FakeRunner)
    monkeypatch.setattr(ex, "eval_jobs", lambda *args: [job])
    ev, ledger = ex.evaluate({}, {}, None, ReconModel(), 0, 1, 1)
    for mode in ("nominal", "perturbed"):
        e = ev[mode]["episodes"][0]
        assert e["success"] is False and e["final_satisfied"] == expected_final and e["max_satisfied"] == expected_max
        assert e["first_unsatisfied_index"] == expected_first and ev[mode]["overall"] == 0
        assert e["progress"] == pytest.approx(expected_final / 3)
    assert ledger["episodes"] == 2


def test_previous_evaluation_reuse_checks_complete_identity_and_copies():
    from horizon.harness.experiment import curve_evaluation_key, previous_curve_evaluation
    recon = ReconModel().to_dict()
    original = {"nominal": {"overall": 0.5, "episodes": [True, False]}}
    previous = {"round": 1, "policy_after": "pi", "recon_after": recon,
                "post_train_evaluation_key": curve_evaluation_key("experiment", "pi", recon),
                "post_train": original}
    cfg, exp = {"reuse_pre_evaluation": True}, {"fingerprint": "experiment"}
    copied = previous_curve_evaluation([previous], cfg, exp, "pi", recon)
    copied["nominal"]["episodes"].append(False)
    assert len(original["nominal"]["episodes"]) == 2
    assert previous_curve_evaluation([], cfg, exp, "pi", recon) is None
    assert previous_curve_evaluation([previous], {}, exp, "different", recon) is None
    for fingerprint, policy, belief in [("changed-config", "pi", recon), ("experiment", "new", recon),
                                         ("experiment", "pi", {"bias": {"tray": 0.3}, "corrections": {}})]:
        with pytest.raises(ValueError, match="does not match"):
            previous_curve_evaluation([previous], cfg, {"fingerprint": fingerprint}, policy, belief)


@pytest.mark.parametrize("factory_name", ["nominal_factory", "nominal_recovery_factory", "dagger_factory"])
def test_all_training_factories_respect_observation_mode(tmp_path, factory_name):
    import horizon.harness.experiment as ex
    cfg = config()
    cfg["training_observation_mode"] = "observed"
    store = TaskStore(tmp_path / "tasks", cfg, tmp_path)
    factory = getattr(ex, factory_name)
    args = (store, cfg, ReconModel(), 0)
    job = factory(*args, *(() if factory_name == "nominal_factory" else (1,)))()
    assert job.observation_mode == "observed"
