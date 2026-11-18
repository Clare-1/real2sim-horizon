"""Persistence invariants using synthetic arrays; no optimizer or training run."""
from __future__ import annotations

import copy
import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from horizon.harness.artifacts import (
    canonical_hash, file_hash, load_blocks, load_policy, save_blocks, save_policy, write_json,
)
from horizon.policy.dataset import SampleBlock
from horizon.rsi.loop import load_params


def sample_block(key: str, n: int) -> SampleBlock:
    return SampleBlock(
        key, "pack", 2, "synthetic-fixture",
        np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        np.arange(n * 4, dtype=np.float32).reshape(n, 4),
        np.ones((n, 5), dtype=np.float32), np.zeros(n, dtype=np.int16), np.ones(n, dtype=np.int16),
        {"twin_version": "孪生-1", "source": {"seed": 21, "audited": False}, "costs": [1, 2.5, None]},
    )


def policy_params() -> dict:
    return {"arch": "mlp", "n_layers": 1,
            "x_mu": np.zeros(3, dtype=np.float32), "x_sd": np.ones(3, dtype=np.float32),
            "w0": np.ones((4, 3), dtype=np.float32), "b0": np.zeros(4, dtype=np.float32),
            "g0": np.ones(4, dtype=np.float32), "be0": np.zeros(4, dtype=np.float32),
            "wo": np.ones((6, 4), dtype=np.float32), "bo": np.zeros(6, dtype=np.float32)}


def test_canonical_hash_is_order_independent_and_finite():
    assert canonical_hash({"b": [2, 3], "a": "中文"}) == canonical_hash({"a": "中文", "b": [2, 3]})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            canonical_hash({"value": value})


def test_invalid_json_does_not_replace_existing_file(tmp_path):
    path = tmp_path / "nested" / "record.json"
    write_json(path, {"message": "中文", "n": 4})
    before = path.read_bytes()
    assert json.loads(before) == {"message": "中文", "n": 4}
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert path.read_bytes() == before
    assert list(path.parent.glob("*.tmp")) == []


def test_blocks_roundtrip_preserves_arrays_dtypes_and_nested_meta(tmp_path):
    original = [sample_block("first", 3), sample_block("second", 2)]
    record = save_blocks(tmp_path / "blocks.npz", original)
    assert record["n"] == 5 and record["blocks"] == 2
    assert Path(record["path"]).is_absolute() and Path(record["metadata_path"]).is_absolute()
    assert file_hash(record["path"]) == record["sha256"]
    restored = load_blocks(record)
    for source, target in zip(original, restored):
        assert (source.key, source.family, source.H, source.provenance, source.meta) == (
            target.key, target.family, target.H, target.provenance, target.meta)
        for field in ("x_hier", "x_flat", "y", "k", "stage"):
            a, b = getattr(source, field), getattr(target, field)
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)
    restored[0].meta["source"]["seed"] = 99
    assert original[0].meta["source"]["seed"] == 21
    with np.load(record["path"], allow_pickle=False) as archive:
        assert len(archive.files) == 10
        assert all(not archive[name].dtype.hasobject for name in archive.files)


def test_empty_blocks_roundtrip(tmp_path):
    record = save_blocks(tmp_path / "empty.npz", [])
    assert record["n"] == 0 and record["blocks"] == 0
    assert load_blocks(record) == []


@pytest.mark.parametrize("field", ["path", "metadata_path"])
def test_block_artifact_tampering_is_detected_before_loading(tmp_path, field):
    record = save_blocks(tmp_path / "blocks.npz", [sample_block("first", 3)])
    with Path(record[field]).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_blocks(record)


def test_manifest_count_mismatch_is_detected(tmp_path):
    record = save_blocks(tmp_path / "blocks.npz", [sample_block("first", 3)])
    with pytest.raises(ValueError, match="counts disagree"):
        load_blocks({**record, "n": 4})


def test_object_arrays_and_inconsistent_shapes_are_rejected(tmp_path):
    block = sample_block("first", 3)
    block.y = np.array([[object()] * 5] * 3, dtype=object)
    with pytest.raises(ValueError, match="object dtype"):
        save_blocks(tmp_path / "bad.npz", [block])
    block = sample_block("first", 3)
    block.stage = np.zeros(2, dtype=np.int16)
    with pytest.raises(ValueError, match="shape"):
        save_blocks(tmp_path / "bad.npz", [block])
    assert not (tmp_path / "bad.npz").exists()


def test_policy_roundtrip_matches_existing_loader_and_detects_changed_weights(tmp_path):
    params = policy_params()
    before = save_policy(tmp_path / "before.npz", params)
    loaded = load_policy(before)
    legacy = load_params(before["path"])
    assert loaded["arch"] == legacy["arch"] == "mlp"
    assert loaded["n_layers"] == legacy["n_layers"] == 1
    for name, value in params.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, loaded[name])
            np.testing.assert_array_equal(legacy[name], loaded[name])
    # Synthetic stand-in for weights before/after an update, without running training.
    updated = copy.deepcopy(params)
    updated["w0"][0, 0] += 0.25
    after = save_policy(tmp_path / "after.npz", updated)
    assert before["sha256"] != after["sha256"]
    assert load_policy(after)["w0"][0, 0] == pytest.approx(1.25)
    assert load_policy(before)["w0"][0, 0] == pytest.approx(1.0)


def test_policy_hash_is_checked(tmp_path):
    record = save_policy(tmp_path / "policy.npz", policy_params())
    with Path(record["path"]).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_policy(record)


def test_policy_object_array_is_rejected(tmp_path):
    params = policy_params()
    params["w0"] = np.array([object()], dtype=object)
    with pytest.raises(ValueError, match="object dtype"):
        save_policy(tmp_path / "bad.npz", params)


def test_set_policy_arch_without_n_layers_is_supported(tmp_path):
    record = save_policy(tmp_path / "set.npz", {"arch": "set", "wo": np.ones((6, 4), dtype=np.float32)})
    assert load_policy(record)["arch"] == "set"


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mutation", ["attempt", "reconstruction", "report", "time", "changed"])
def test_artifact_validator_rejects_work_after_first_calibration(mutation):
    validator = load_script("validate_harness_artifacts")
    recon = {"bias": {}, "corrections": {"tray": 3.3}}
    record = {"calibration_attempted": False, "calibration_changed": False,
              "recon_after": copy.deepcopy(recon), "geometry_report": {},
              "ledger": {"calibration_seconds": 0.0}}
    cfg = {"calibration_schedule": "first_round"}
    assert validator.validate_round_calibration(cfg, "S11", 2, record, recon) is False
    if mutation == "attempt": record["calibration_attempted"] = True
    if mutation == "reconstruction": record["recon_after"]["corrections"]["tray"] = 8.0
    if mutation == "report": record["geometry_report"] = {"tray": {"estimated": 0.1}}
    if mutation == "time": record["ledger"]["calibration_seconds"] = 0.5
    if mutation == "changed": record["calibration_changed"] = True
    with pytest.raises(ValueError):
        validator.validate_round_calibration(cfg, "S11", 2, record, recon)


def test_artifact_validator_keeps_legacy_calibration_schedule():
    validator = load_script("validate_harness_artifacts")
    recon = {"bias": {}, "corrections": {}}
    record = {"calibration_attempted": True, "calibration_changed": False,
              "recon_after": recon, "geometry_report": {}, "ledger": {"calibration_seconds": 0.01}}
    assert validator.validate_round_calibration({}, "S10", 3, record, recon) is False
    with pytest.raises(ValueError, match="schedule"):
        validator.validate_round_calibration({"calibration_schedule": "first_round"}, "S10", 3, record, recon)


@pytest.mark.parametrize("mutation", [None, "controller", "failed_episode", "perturbation", "provenance"])
def test_nominal_recovery_artifact_permissions(tmp_path, mutation):
    module = load_script("validate_harness_artifacts")
    validator = object.__new__(module.Validator)
    validator.cfg = {"banks": {"training": 70_000_000}, "learning_collection": "nominal_recovery"}
    recon = {"bias": {}, "corrections": {}}
    job = {"task": {"key": "pack_H2_s70000000", "task_seed": 70_000_000, "sha256": "task"},
           "rollout_seed": 42, "controller": "expert", "beta": 0.0, "max_samples": None,
           "provenance": "nominal_recovery_r1", "expert_noise": 0.0, "latency": 1,
           "perturbation": {k: 0.0 for k in ("slip", "displace", "knock", "traj", "target")},
           "success": True, "sim_steps": 100, "available_labels": 3, "accepted_labels": 3}
    item = {"n": 3, "key": job["task"]["key"], "provenance": "nominal_recovery_r1",
            "meta": {"task_sha256": "task", "rollout_seed": 42, "twin_sha256": module.canonical(recon)}}
    if mutation == "controller": job["controller"] = "mixed"
    if mutation == "failed_episode": job["success"] = False
    if mutation == "perturbation": job["perturbation"]["slip"] = 0.9
    if mutation == "provenance": item["provenance"] = "dagger_r1"
    ledger = {"episodes": 1, "sim_steps": 100, "available_labels": 3,
              "accepted_labels": 3, "successful_episodes": int(job["success"])}
    path = tmp_path / "collection.json"
    write_json(path, {"jobs": [job], "ledger": ledger})
    validator.refs = lambda ref: None
    validator.dataset = lambda ref: [item]
    if mutation is None:
        validator.collection({"path": str(path)}, ledger, {"n": 3}, recon, 1)
    else:
        with pytest.raises(ValueError):
            validator.collection({"path": str(path)}, ledger, {"n": 3}, recon, 1)


def test_report_separates_families_and_names_the_collection_recipe():
    report = load_script("report_harness")
    cfg = {"families": ["pack", "sort"], "H_values": [2], "rounds": 1,
           "learning_collection": "nominal_recovery", "calibration_schedule": "first_round"}
    episodes = [{"family": "pack", "H": 2, "success": True, "progress": 1.0},
                {"family": "sort", "H": 2, "success": False, "progress": 0.5}]
    ev = {mode: {"overall": 0.5, "episodes": episodes} for mode in ("nominal", "perturbed")}
    run = {"history": [{"round": 1, "pre": ev, "post_cal": ev, "post_train": ev}], "final": ev}
    rows = report.family_horizon_metrics({("healthy", "S01", 0): run}, cfg)
    final = [r for r in rows if r["phase"] == "final" and r["mode"] == "nominal"]
    assert [(r["family"], r["success"], r["n"]) for r in final] == [("pack", 1.0, 1), ("sort", 0.0, 1)]
    assert all(r["mean_final_satisfied"] is None for r in rows)  # Old records remain readable.
    summary = {"config": cfg, "validation": {"partial": False, "complete_runs": 1, "expected_runs": 1},
               "final": [], "effects": [], "omitted_factorial_contrasts": []}
    text = report.render_markdown(summary)
    assert "| S01 | Fixed | Fresh nominal/recovery expert demonstrations + BC |" in text
    assert "first round only; then frozen" in text and "family_horizon_metrics.csv" in text


@pytest.mark.parametrize("recorded", [None, "legacy", "constant_safe"])
def test_artifact_validator_checks_training_normalization(tmp_path, recorded):
    module = load_script("validate_harness_artifacts")
    write_json(tmp_path / "experiment.json", {"config": {"train": {
        "steps": 4, "batch": 8, "device": "cuda", "normalization": "constant_safe"}}, "fingerprint": "fixture"})
    validator = module.Validator(tmp_path, False)
    stats = {"n": 8, "steps": 4, "sample_presentations": 32, "device": "cuda",
             "wall_seconds": 0.1, "peak_gpu_allocated_bytes": 16}
    if recorded is not None:
        stats["normalization"] = recorded
    if recorded == "constant_safe":
        validator.training_cost(stats, 8)
    else:
        with pytest.raises(ValueError, match="normalization"):
            validator.training_cost(stats, 8)


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_artifact_validator_rejects_nonpositive_normalization_scale(tmp_path, scale):
    module = load_script("validate_harness_artifacts")
    write_json(tmp_path / "experiment.json", {"config": {"train": {"layers": 1}}, "fingerprint": "fixture"})
    validator = module.Validator(tmp_path, False)
    params = policy_params()
    params["x_sd"][0] = scale
    ref = save_policy(tmp_path / "policy.npz", params)
    with pytest.raises(ValueError, match="normalization scales"):
        validator.policy(ref)
