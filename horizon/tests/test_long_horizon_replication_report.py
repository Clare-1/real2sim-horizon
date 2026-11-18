"""Replication reporting invariants using synthetic artifacts, never rollouts."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location("replication_report_test", SCRIPTS / "report_long_horizon_replication.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": report.digest(path)}


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def evaluations(successes=1, bank=87_000_000, n_per=2):
    rows = []
    for fi, family in enumerate(report.FAMILIES):
        for H in report.HORIZONS:
            for i in range(n_per):
                task_seed = bank + fi * 100_000 + H * 1000 + i
                task_hash = sha(str(task_seed))
                ok = i < successes
                rows.append({"family": family, "H": H, "task_seed": task_seed, "rollout_seed": task_seed + 17,
                             "task_sha256": task_hash, "task_artifact": {"path": f"task_{task_seed}.json", "sha256": task_hash},
                             "success": ok, "progress": 1.0 if ok else 0.0,
                             "final_satisfied": H if ok else 0, "max_satisfied": H if ok else 0,
                             "first_unsatisfied_index": H if ok else 0})
    return {mode: {"episodes": deepcopy(rows), "overall": successes / n_per} for mode in report.MODES}


def feedback_ref(tmp_path, arm, seed, content=b"same-observations"):
    folder = tmp_path / arm / f"s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    observations = folder / "observations.npz"
    observations.write_bytes(content)
    return write_json(folder / "manifest.json", {
        "episodes": [{"task": {"path": str(folder / "task.json"), "key": "shared", "sha256": sha("task")},
                      "rollout_seed": 42, "belief": {"objects": []}, "prefix": "0"}],
        "observations": {"path": str(observations), "sha256": report.digest(observations)},
        "signals": ["0_action"], "gripper_channel": "ideal_encoder_v4", "object_channel": "noisy_observation"})


@pytest.fixture
def systems(tmp_path):
    output = {}
    for seed in report.SEEDS:
        for method in report.SPECS:
            is_healthy = method.startswith("healthy")
            initial = {"data_sha256": sha(f"data:{is_healthy}:{seed}"), "policy_sha256": sha(f"policy:{is_healthy}:{seed}"),
                       "n": 80_000, "train_steps": 8_000}
            n_success = {"healthy_dagger": (2, 1, 0), "healthy_nominal": (1, 0, 1)}.get(method, (1, 1, 1))[seed]
            first = {"round": 1, "recon_before": {"bias": {}, "corrections": {}},
                     "recon_after": {"bias": {}, "corrections": {"tray": 0.01}},
                     "feedback_record": feedback_ref(tmp_path, method, seed), "n_before": 80_000, "n_after": 120_000}
            first.update({phase: evaluations(bank=86_000_000, n_per=4) for phase in report.PHASES})
            history = []
            for round_id in (1, 2, 3):
                h = deepcopy(first)
                h.update(round=round_id, reused_equal_state_evaluation={"pre": round_id > 1},
                         pre_evaluation_source_round=round_id - 1 if round_id > 1 else None,
                         ledger={"pre_evaluation": {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}
                                 if round_id > 1 else {"episodes": 120, "sim_steps": 1200, "worker_seconds": 1.0}},
                         n_before=80_000 if method == "bias_S10" else 80_000 + (round_id - 1) * 40_000,
                         n_after=80_000 if method == "bias_S10" else 80_000 + round_id * 40_000)
                history.append(h)
            output[(method, seed)] = {"initial": initial, "history": history, "final": evaluations(n_success)}
    return output


def test_seed_statistics_are_unweighted_sample_sd_not_pooled_episode_sd():
    rows = [{"method": "a", "seed": seed, "rate": rate, "episodes": n}
            for seed, rate, n in ((0, 0.0, 1), (1, 0.5, 2), (2, 1.0, 100))]
    result, = report.aggregate_seed_rows(rows, ("method",), ("rate", "episodes"))
    assert result["rate"] == {"values": [0.0, 0.5, 1.0], "mean": 0.5, "sample_sd": 0.5}
    assert result["rate"]["mean"] != pytest.approx(101 / 103)
    assert result["n_seeds"] == 3 and result["seeds"] == [0, 1, 2]


@pytest.mark.parametrize("seeds", [(0, 1), (0, 1, 1), (0, 1, 2, 3)])
def test_aggregate_rejects_missing_duplicate_or_extra_seeds(seeds):
    with pytest.raises(ValueError, match="Missing/duplicate seed"):
        report.aggregate_seed_rows([{"seed": s, "group": "a", "rate": 0.5} for s in seeds], ("group",), ("rate",))


def test_initial_and_feedback_pairing_is_per_seed_and_ignores_storage_paths(systems):
    inputs = report.Inputs()
    pairs = report.validate_pairing(systems, inputs)
    assert [p["seed"] for p in pairs] == [0, 1, 2]
    assert len({p["healthy_policy_sha256"] for p in pairs}) == 3
    for seed in report.SEEDS:
        a = systems[("bias_S10", seed)]["history"][0]["feedback_record"]
        b = systems[("bias_S11", seed)]["history"][0]["feedback_record"]
        assert a["sha256"] != b["sha256"]
    inputs.unchanged()


@pytest.mark.parametrize("fault,pattern", [
    ("missing", "Exactly 15"), ("healthy_initial", "Healthy initial"), ("bias_initial", "Bias initial"),
    ("belief", "first belief"), ("feedback", "first feedback"), ("task", "Unpaired shared"),
    ("seed_schedule", "Unpaired shared"), ("duplicate_task", "Unpaired shared")])
def test_pairing_rejects_confounding_or_unpaired_systems(systems, tmp_path, fault, pattern):
    if fault == "missing":
        systems.pop(("bias_S11", 2))
    elif fault == "healthy_initial":
        systems[("healthy_nominal", 1)]["initial"]["policy_sha256"] = sha("different")
    elif fault == "bias_initial":
        systems[("bias_S10", 2)]["initial"]["data_sha256"] = sha("different")
    elif fault == "belief":
        systems[("bias_S11", 1)]["history"][0]["recon_after"]["corrections"]["tray"] = 0.05
    elif fault == "feedback":
        systems[("bias_S11", 2)]["history"][0]["feedback_record"] = feedback_ref(tmp_path, "changed", 2, b"changed")
    else:
        rows = systems[("healthy_dagger", 2)]["final"]["perturbed"]["episodes"]
        if fault == "task":
            rows[0]["task_sha256"] = sha("different")
        elif fault == "seed_schedule":
            rows[0]["rollout_seed"] += 1
        else:
            rows[1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match=pattern):
        report.validate_pairing(systems, report.Inputs())


def test_paired_delta_keeps_seed_and_family_horizon_cells(systems):
    rows = []
    for seed in report.SEEDS:
        rows.extend({**r, "seed": seed} for r in report.paired_contrasts({m: systems[(m, seed)] for m in report.SPECS}))
    aggregated = report.aggregate_seed_rows(rows, ("left", "right", "mode", "family", "H"), ("delta_pp",))
    cell = next(r for r in aggregated if (r["left"], r["mode"], r["family"], r["H"]) ==
                ("healthy_dagger", "nominal", "sort", 8))
    assert cell["delta_pp"]["values"] == [50.0, 50.0, -50.0]
    assert cell["delta_pp"]["mean"] == pytest.approx(100 / 6)
    assert cell["delta_pp"]["sample_sd"] == pytest.approx(100 / np.sqrt(3))
    per_seed = [r for r in rows if (r["left"], r["mode"], r["family"], r["H"]) ==
                ("healthy_dagger", "nominal", "sort", 8)]
    assert [(r["wins"], r["ties"], r["losses"]) for r in per_seed] == [(1, 1, 0), (1, 1, 0), (0, 1, 1)]


def preflight_files(root):
    for name, expected in report.EXPECTED.items():
        suffix = name.removeprefix("fullH_")
        config = yaml.safe_load((SCRIPTS.parent / "configs" / f"harness_long_horizon_replication_{suffix}.yaml").read_text())
        fingerprint = sha(name)
        write_json(root / name / "experiment.json", {"fingerprint": fingerprint, "config": config, "source_sha256": {"source.py": sha("source")}})
        write_json(root / name / "completion.json", {"complete": True, "confirmation_executed": False,
                                                    "expected_runs": len(expected), "fingerprint": fingerprint})
        write_json(root / name / "artifact_validation.json", {"passed": True, "complete": True, "partial": False,
                   "expected_runs": len(expected), "validated_runs": len(expected), "validated_rounds": len(expected) * 3,
                   "experiment_fingerprint": fingerprint, "pending": []})


def test_preflight_accepts_exact_prepared_protocol(tmp_path):
    preflight_files(tmp_path)
    assert set(report.preflight(tmp_path, report.Inputs())) == set(report.EXPECTED)


def test_incomplete_third_matrix_is_rejected_before_any_scores_or_deep_validation(tmp_path, monkeypatch):
    preflight_files(tmp_path)
    path = tmp_path / "fullH_bias" / "artifact_validation.json"
    value = json.loads(path.read_text())
    value["complete"] = False
    write_json(path, value)
    monkeypatch.setattr(report, "revalidate", lambda *args: pytest.fail("Read artifacts before all gates passed"))
    monkeypatch.setattr(report, "load_and_validate", lambda *args, **kwargs: pytest.fail("Read scores before all gates passed"))
    with pytest.raises(ValueError, match="Incomplete/failed"):
        report.load_inputs(tmp_path)
    assert not (tmp_path / "comparison.json").exists()


@pytest.mark.parametrize("fault,pattern", [("seeds", "three-seed"), ("bank", "banks"),
                                         ("count", "run count"), ("protocol", "Non-treatment"),
                                         ("observations", "observation mode"), ("reuse", "reuse is not enabled")])
def test_preflight_rejects_wrong_protocol(tmp_path, fault, pattern):
    preflight_files(tmp_path)
    path = tmp_path / "fullH_nominal" / ("artifact_validation.json" if fault == "count" else "experiment.json")
    value = json.loads(path.read_text())
    if fault == "count": value["validated_runs"] = 2
    elif fault == "seeds": value["config"]["seeds"] = [0]
    elif fault == "bank": value["config"]["banks"]["final"] = 77_000_000
    elif fault == "observations": value["config"]["training_observation_mode"] = "observed"
    elif fault == "reuse": value["config"]["reuse_pre_evaluation"] = False
    else: value["config"]["collection_pert"] = 0.4
    write_json(path, value)
    with pytest.raises(ValueError, match=pattern):
        report.preflight(tmp_path, report.Inputs())


def test_saved_validation_cannot_hide_changed_physical_costs(tmp_path, monkeypatch):
    fresh = {"passed": True, "complete": True, "physical_resources_committed": {"episodes": 10}}
    class FakeValidator:
        def __init__(self, *args): pass
        def run(self): return fresh
    monkeypatch.setattr(report, "TrackedValidator", FakeValidator)
    saved = {**deepcopy(fresh), "validator_sha256": report.digest(SCRIPTS / "validate_harness_artifacts.py")}
    saved["physical_resources_committed"]["episodes"] = 9
    with pytest.raises(ValueError, match="stale.*physical_resources"):
        report.revalidate(tmp_path, {"artifact_validation": saved}, report.Inputs())


def test_physical_costs_charge_separate_directories_and_peak_is_max():
    groups = {name: {"fresh_artifact_validation": {"physical_resources_committed": {
        "training": {"initial_calls": 3, "additional_calls": calls, "initial_optimizer_steps": 24_000,
                     "additional_optimizer_steps": calls * 8_000, "peak_gpu_allocated_bytes_max": peak},
        "rollouts": {"initial_collection": {"episodes": 12, "sim_steps": 1200, "worker_seconds": 1.0}}}}}
        for name, calls, peak in (("dagger", 9, 100), ("nominal", 9, 200), ("bias", 18, 150))}
    result = report.physical_totals(groups)
    assert result["training"]["initial_calls"] == 9
    assert result["training"]["additional_calls"] == 36
    assert result["training"]["peak_gpu_allocated_bytes_max"] == 200
    assert result["all_rollouts"] == {"episodes": 36, "sim_steps": 3600, "worker_seconds": 3.0}


def test_boundary_reuse_counts_actual_records_and_rejects_any_physical_cost(systems):
    reuse = report.boundary_reuse_summary(systems)
    assert reuse["reused_boundaries"] == len(reuse["records"]) == 30
    assert reuse["avoided_rollouts"] == 3600
    assert reuse["actual_cost"] == {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}
    cost = systems[("bias_S11", 2)]["history"][1]["ledger"]["pre_evaluation"]
    for key in ("episodes", "sim_steps", "worker_seconds"):
        cost[key] = 1
        with pytest.raises(ValueError, match="charged physical pre-evaluation cost"):
            report.boundary_reuse_summary(systems)
        cost[key] = 0


def test_actual_npz_coverage_retains_zero_cells_and_distinguishes_stage_from_k(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(path, block0_stage=np.array([0, 1, 1], dtype=np.int16), block0_k=np.array([1, 0, 0], dtype=np.int16))
    meta = write_json(tmp_path / "data.meta.json", {"version": 1, "n": 3, "blocks": [
        {"family": "pack", "H": 2, "n": 3, "key": "one", "provenance": "expert", "meta": {"rollout_seed": 12}}]})
    ref = {"path": str(path), "sha256": report.digest(path), "metadata_path": meta["path"], "metadata_sha256": meta["sha256"], "blocks": 1, "n": 3}
    run = {"initial": {"dataset": ref}, "history": [{"round": 1, "n_after": 3, "new_dataset": None}]}
    inputs = report.Inputs()
    coverage = report.training_coverage("method", run, inputs, {})
    pack = [r for r in coverage if (r["data_phase"], r["family"], r["H"]) == ("cumulative", "pack", 2)]
    assert [(r["labels"], r["k_labels"]) for r in pack] == [(1, 2), (2, 1)]
    zero = next(r for r in coverage if (r["data_phase"], r["family"], r["H"], r["stage"]) == ("cumulative", "stack", 10, 9))
    assert zero["labels"] == zero["contributing_episodes"] == zero["family_H_labels"] == 0
    path.write_bytes(b"changed after coverage")
    with pytest.raises(ValueError, match="changed before report commit"):
        inputs.unchanged()


def fake_plots(summary, folder):
    for name in report.OUTPUT_NAMES[:2]:
        (folder / name).write_bytes(b"plot")


def test_publisher_uses_workspace_siblings_and_publishes_results_last(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "make_plots", fake_plots)
    monkeypatch.setattr(report, "markdown", lambda summary: "new report")
    real_replace, replacements = report.os.replace, []
    def replace(source, target):
        replacements.append((Path(source), Path(target)))
        return real_replace(source, target)
    monkeypatch.setattr(report.os, "replace", replace)
    report.publish(tmp_path, {"complete": True}, report.Inputs())
    assert tuple(target.name for _, target in replacements) == report.OUTPUT_NAMES
    assert all(source.parent == tmp_path and source.suffix == ".publish" for source, _ in replacements)
    assert (tmp_path / "RESULTS.md").read_text() == "new report"
    assert not list(tmp_path.glob("*.pdf")) and not list(tmp_path.glob(".*.publish"))


def test_changed_input_during_plotting_does_not_replace_previous_report(tmp_path, monkeypatch):
    original = tmp_path / "input.json"
    original.write_text("original")
    (tmp_path / "RESULTS.md").write_text("previous report")
    inputs = report.Inputs()
    inputs.track(original)
    def changed(summary, folder):
        fake_plots(summary, folder)
        original.write_text("changed")
    monkeypatch.setattr(report, "make_plots", changed)
    monkeypatch.setattr(report, "markdown", lambda summary: "new report")
    with pytest.raises(ValueError, match="changed before report commit"):
        report.publish(tmp_path, {}, inputs)
    assert (tmp_path / "RESULTS.md").read_text() == "previous report"
    assert not (tmp_path / "comparison.json").exists()


def test_cli_missing_matrix_fails_without_publishing(tmp_path):
    assert report.main([str(tmp_path)]) == 2
    assert not any((tmp_path / name).exists() for name in report.OUTPUT_NAMES)


def test_synthetic_summary_markdown_and_actual_png_svg_export(tmp_path, systems, monkeypatch):
    """Exercise report assembly and render backends without any experiment data."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setattr(report, "training_coverage", lambda method, run, inputs, cache: [
        {"method": method, "data_phase": "cumulative", "family": "pack", "H": 2, "stage": 1,
         "k_index": 1, "labels": 3, "k_labels": 2, "family_H_labels": 5, "contributing_episodes": 1,
         "family_H_blocks": 1, "provenance_labels": {"expert": 5}}])
    def resources(method, run):
        frozen = method == "bias_S10"
        return {"method": method, "initial_labels": 80_000, "additional_labels": 0 if frozen else 120_000,
                "initial_optimizer_steps_logical": 8_000, "additional_optimizer_steps": 0 if frozen else 24_000,
                "additional_collection_physical": {"episodes": 0 if frozen else 2, "sim_steps": 100},
                "evaluation_physical": {"episodes": 600, "sim_steps": 6000}, "additional_training_wall_seconds": 1.0}
    monkeypatch.setattr(report, "resource_summary", resources)
    preflight_files(tmp_path)
    groups = report.preflight(tmp_path, report.Inputs())
    for group in groups.values():
        group["fresh_artifact_validation"] = {"physical_resources_committed": {
            "training": {"initial_calls": 3, "additional_calls": 9, "initial_optimizer_steps": 24_000,
                         "additional_optimizer_steps": 72_000},
            "rollouts": {"evaluation": {"episodes": 600, "sim_steps": 6000, "worker_seconds": 1.0}}}}
    inputs = report.Inputs()
    pairs = report.validate_pairing(systems, inputs)
    summary = report.summarize(groups, systems, inputs, pairs)
    assert len(summary["final_metrics"]) == 840
    assert len(summary["final_summary"]) == 280
    assert len(summary["paired_contrasts"]) == 504
    assert all(r["n_seeds"] == 3 for r in summary["paired_summary"])
    summary["input_files_sha256"] = {str(path): value for path, value in inputs.hashes.items()}
    report.publish(tmp_path, summary, inputs)
    assert (tmp_path / "long_horizon_replication.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = (tmp_path / "long_horizon_replication.svg").read_text(encoding="utf-8")
    assert "<svg" in svg and "sample SD (not CI)" in svg
    markdown = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "15/15" in markdown and "900" in markdown and "seed 2" in markdown
    assert "样本标准差" in markdown and "900 个独立任务" in markdown
    assert "30 个轮次边界" in markdown and "3,600" in markdown and "不能解释为学习或样本效率优势" in markdown
    assert not list(tmp_path.glob("*.pdf"))
    saved = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert saved["seeds"] == [0, 1, 2]
