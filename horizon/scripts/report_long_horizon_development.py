"""Report the five frozen full-H development systems; never run experiments.

Usage: python scripts/report_long_horizon_development.py RESULTS_ROOT

Requires complete fullH_dagger, fullH_nominal and fullH_bias matrices, each
with a passing independent artifact_validation.json. Validation/pairing errors
are fatal. The separate normalization diagnostic is deliberately never loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import uuid

import numpy as np

from report_harness import canonical, identity, load_and_validate, require


FAMILIES = ("pack", "sort", "stack")
HORIZONS = (2, 4, 6, 8, 10)
MODES = ("nominal", "perturbed")
SPECS = {
    "healthy_dagger": ("fullH_dagger", "healthy", "S01", "健康：DAgger", "Healthy DAgger"),
    "healthy_nominal": ("fullH_nominal", "healthy", "S01", "健康：名义＋恢复专家", "Healthy nominal+recovery"),
    "bias_S01": ("fullH_bias", "height_bias", "S01", "高度偏差：固定 twin＋DAgger", "Biased twin + DAgger"),
    "bias_S10": ("fullH_bias", "height_bias", "S10", "高度偏差：一次校准＋冻结策略", "Once calibrated, frozen policy"),
    "bias_S11": ("fullH_bias", "height_bias", "S11", "高度偏差：一次校准＋DAgger", "Once calibrated + DAgger"),
}
COLORS = {
    "healthy_dagger": "#2878b5", "healthy_nominal": "#777777",
    "bias_S01": "#bb5566", "bias_S10": "#dd9900", "bias_S11": "#228877",
}
EXPECTED = {
    "fullH_dagger": {("healthy", "S01", 0)},
    "fullH_nominal": {("healthy", "S01", 0)},
    "fullH_bias": {("height_bias", a, 0) for a in ("S01", "S10", "S11")},
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical(value)  # reject NaN/Inf accepted by Python's permissive reader
    return value


class Inputs:
    """Track consumed files so a concurrent change cannot silently mix results."""

    def __init__(self):
        self.hashes = {}

    def track(self, path, expected=None):
        path = Path(path).resolve()
        actual = digest(path)
        require(expected is None or actual == expected, f"Artifact hash mismatch: {path}")
        require(path not in self.hashes or self.hashes[path] == actual, f"Input changed while reading: {path}")
        self.hashes[path] = actual
        return path

    def json(self, path, expected=None):
        return read_json(self.track(path, expected))

    def unchanged(self):
        for path, expected in self.hashes.items():
            require(digest(path) == expected, f"Input changed before report commit: {path}")


def checked_integer(value, label, lower=0, upper=None):
    require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
            f"Expected integer {label}, got {value!r}")
    value = int(value)
    require(value >= lower and (upper is None or value <= upper), f"Out of range {label}: {value}")
    return value


def check_terminal(episode, label):
    H = checked_integer(episode["H"], f"{label}/H", 1)
    prefix = checked_integer(episode.get("first_unsatisfied_index"), f"{label}/first_unsatisfied_index", 0, H)
    final = checked_integer(episode.get("final_satisfied"), f"{label}/final_satisfied", 0, H)
    maximum = checked_integer(episode.get("max_satisfied"), f"{label}/max_satisfied", 0, H)
    require(prefix <= final <= maximum, f"Inconsistent terminal/progress counts: {label}")
    require(bool(episode["success"]) == (prefix == H), f"Full success differs from terminal prefix: {label}")
    require(math.isclose(float(episode["progress"]), final / H, abs_tol=1e-12),
            f"Progress disagrees with final satisfied count: {label}")
    require(isinstance(episode.get("task_sha256"), str) and len(episode["task_sha256"]) == 64,
            f"Missing task hash: {label}")
    require(episode["task_artifact"]["sha256"] == episode["task_sha256"], f"Task reference hash mismatch: {label}")


def load_inputs(root):
    inputs, groups, systems = Inputs(), {}, {}
    for folder_name, expected in EXPECTED.items():
        folder = root / folder_name
        artifact = inputs.json(folder / "artifact_validation.json")
        require(artifact.get("passed") is True and artifact.get("complete") is True and
                artifact.get("partial") is False, f"Independent artifact validation is not complete/passed: {folder}")
        require(artifact.get("expected_runs") == len(expected) and artifact.get("validated_runs") == len(expected),
                f"Independent validator has the wrong run count: {folder}")
        require(not artifact.get("error") and not artifact.get("pending"), f"Validator contains errors/pending work: {folder}")
        experiment = inputs.json(folder / "experiment.json")
        completion = inputs.json(folder / "completion.json")
        require(completion.get("complete") is True and completion.get("confirmation_executed") is False,
                f"Incomplete run or confirmation bank used: {folder}")
        require(completion.get("expected_runs") == len(expected), f"Wrong completion run count: {folder}")
        require(artifact.get("experiment_fingerprint") == completion.get("fingerprint") == experiment.get("fingerprint"),
                f"Validator/completion/experiment fingerprints disagree: {folder}")
        for path in sorted((folder / "runs").glob("*/*_s*.json")):
            inputs.track(path)
        runs, cfg, basic_validation = load_and_validate(folder, allow_partial=False)
        require(set(runs) == expected, f"Expected exactly {sorted(expected)}, got {sorted(runs)} in {folder}")
        require(canonical(cfg) == canonical(experiment["config"]), f"Result config differs from experiment: {folder}")
        require(cfg["seeds"] == [0] and set(cfg["families"]) == set(FAMILIES) and
                tuple(sorted(cfg["H_values"])) == HORIZONS, f"Not the single-seed three-family full-H protocol: {folder}")
        require(cfg.get("calibration_schedule") == "first_round", f"Expected first-round-only calibration: {folder}")
        require(cfg["train"].get("normalization") == "constant_safe", f"Expected constant_safe normalization: {folder}")
        expected_collection = "nominal_recovery" if folder_name == "fullH_nominal" else "dagger"
        require(cfg.get("learning_collection") == expected_collection, f"Wrong collection mode in {folder}")
        require(artifact.get("validated_rounds") == len(expected) * cfg["rounds"], f"Wrong validated round count: {folder}")
        groups[folder_name] = {"config": cfg, "experiment": experiment, "artifact_validation": artifact,
                               "record_validation": basic_validation}
        for key, run in runs.items():
            policy_refs = [run["initial"]["policy"], run["final_policy"],
                           *[h["policy_record"] for h in run["history"]]]
            for ref in policy_refs:
                inputs.track(ref["path"], ref["sha256"])
            for method, spec in SPECS.items():
                if spec[:3] == (folder_name, key[0], key[1]):
                    require(method not in systems, f"Duplicate method {method}")
                    systems[method] = run
            for r in [run["final"], *[h[p] for h in run["history"] for p in ("pre", "post_cal", "post_train")]]:
                for mode in MODES:
                    for episode in r[mode]["episodes"]:
                        check_terminal(episode, f"{folder_name}/{key}/{mode}/{identity(episode)}")
                        inputs.track(episode["task_artifact"]["path"], episode["task_sha256"])
    require(set(systems) == set(SPECS), "All five systems must be complete")
    common_cfg = None
    source_versions = set()
    for folder_name, group in groups.items():
        cfg = {k: v for k, v in group["config"].items() if k not in {"name", "arms", "conditions", "learning_collection"}}
        if common_cfg is None:
            common_cfg = canonical(cfg)
        require(canonical(cfg) == common_cfg, f"Non-treatment configuration mismatch: {folder_name}")
        source_versions.add(canonical(group["experiment"]["source_sha256"]))
    require(len(source_versions) == 1, "The three matrices used different source versions")

    a, b = [systems[k]["initial"] for k in ("healthy_dagger", "healthy_nominal")]
    for field in ("data_sha256", "policy_sha256", "n", "train_steps"):
        require(a[field] == b[field], f"Healthy initial mismatch ({field}); comparison is not matched")
    require(canonical(groups["fullH_dagger"]["config"]["conditions"]) ==
            canonical(groups["fullH_nominal"]["config"]["conditions"]), "Healthy conditions differ")
    require(canonical(systems["bias_S10"]["history"][0]["recon_after"]) ==
            canonical(systems["bias_S11"]["history"][0]["recon_after"]),
            "Bias S10/S11 first calibration differs; frozen-vs-learning contrast is confounded")

    # Raw task hashes must match even across condition folders; belief is a treatment.
    reference = systems["healthy_dagger"]
    paired_checks = []
    for method, run in systems.items():
        evaluations = [("final", run["final"], reference["final"])]
        for h, ref_h in zip(run["history"], reference["history"], strict=True):
            evaluations.extend((f"round{h['round']}/{p}", h[p], ref_h[p]) for p in ("pre", "post_cal", "post_train"))
        for label, value, ref_value in evaluations:
            for mode in MODES:
                keys = {identity(e) for e in value[mode]["episodes"]}
                ref_keys = {identity(e) for e in ref_value[mode]["episodes"]}
                require(keys == ref_keys, f"Unpaired task hashes/rollout seeds: {method}/{label}/{mode}; "
                        f"missing={len(ref_keys - keys)}, extra={len(keys - ref_keys)}")
        paired_checks.append(method)
    return inputs, groups, systems, paired_checks


def selections(episodes):
    """Full grid, family summaries, horizon summaries and a long-H summary."""
    for family in ("all", *FAMILIES):
        family_rows = [e for e in episodes if family == "all" or e["family"] == family]
        for H in ("all", "H>=8", *HORIZONS):
            selected = [e for e in family_rows if H == "all" or (H == "H>=8" and e["H"] >= 8) or e["H"] == H]
            require(bool(selected), f"Empty family/H group: {family}/{H}")
            yield family, H, selected


def metrics(method, mode, episodes):
    rows = []
    for family, H, selected in selections(episodes):
        n, k = len(selected), sum(int(e["success"]) for e in selected)
        rows.append({"method": method, "mode": mode, "family": family, "H": H, "successes": k,
                     "episodes": n, "success_rate": k / n,
                     "terminal_prefix_fraction": float(np.mean([e["first_unsatisfied_index"] / e["H"] for e in selected])),
                     "terminal_satisfied_fraction": float(np.mean([e["final_satisfied"] / e["H"] for e in selected])),
                     "max_simultaneous_satisfied_fraction": float(np.mean([e["max_satisfied"] / e["H"] for e in selected]))})
    return rows


def prefix_metrics(method, mode, episodes):
    rows = []
    for family in ("all", *FAMILIES):
        for H in HORIZONS:
            selected = [e for e in episodes if e["H"] == H and (family == "all" or e["family"] == family)]
            for j in range(H + 1):
                k = sum(e["first_unsatisfied_index"] >= j for e in selected)
                rows.append({"method": method, "mode": mode, "family": family, "H": H,
                             "prefix_length": j, "surviving": k, "episodes": len(selected),
                             "survival": k / len(selected)})
    return rows


def paired_contrasts(systems):
    rows = []
    comparisons = (("healthy_dagger", "healthy_nominal", "健康 DAgger − 名义恢复"),
                   ("bias_S11", "bias_S01", "偏差：校准＋学习 − 固定 twin 学习"),
                   ("bias_S11", "bias_S10", "偏差：校准＋学习 − 校准冻结策略"))
    for left, right, label in comparisons:
        for mode in MODES:
            right_by_id = {identity(e): e for e in systems[right]["final"][mode]["episodes"]}
            for family, H, selected in selections(systems[left]["final"][mode]["episodes"]):
                differences = [int(e["success"]) - int(right_by_id[identity(e)]["success"]) for e in selected]
                rows.append({"left": left, "right": right, "label": label, "mode": mode,
                             "family": family, "H": H, "episodes": len(selected),
                             "delta_pp": 100 * sum(differences) / len(differences),
                             "wins": differences.count(1), "ties": differences.count(0), "losses": differences.count(-1)})
    return rows


def dataset_coverage(ref, inputs):
    meta = inputs.json(ref["metadata_path"], ref["metadata_sha256"])
    path = inputs.track(ref["path"], ref["sha256"])
    require(meta["version"] == 1 and len(meta["blocks"]) == ref["blocks"] and meta["n"] == ref["n"],
            f"Dataset metadata mismatch: {path}")
    output, total = [], 0
    with np.load(path, allow_pickle=False) as archive:
        for index, block in enumerate(meta["blocks"]):
            H, n = int(block["H"]), int(block["n"])
            require(block["family"] in FAMILIES and H in HORIZONS, f"Unexpected training task in {path}")
            stage, k = archive[f"block{index}_stage"], archive[f"block{index}_k"]
            require(stage.shape == k.shape == (n,), f"Stage/k shape mismatch in {path}, block {index}")
            for values, name in ((stage, "stage"), (k, "k")):
                require(np.issubdtype(values.dtype, np.integer) and bool(np.all((values >= 0) & (values < H))),
                        f"Invalid {name} labels in {path}, block {index}")
            episode_key = (block["family"], H, block["key"], block["meta"].get("rollout_seed"))
            output.append({"family": block["family"], "H": H, "n": n, "episode_key": episode_key,
                           "provenance": block["provenance"], "stage": Counter(map(int, stage)),
                           "k": Counter(map(int, k))})
            total += n
    require(total == ref["n"], f"Training sample total mismatch in {path}")
    return output


def training_coverage(method, run, inputs, cache):
    refs = [("initial", run["initial"]["dataset"])]
    refs.extend((f"round{h['round']}", h["new_dataset"]) for h in run["history"] if h["new_dataset"] is not None)
    staged, all_blocks = {}, []
    for phase, ref in refs:
        cache_key = (ref["sha256"], ref["metadata_sha256"])
        # Verify every referenced path, even when identical bytes were loaded before.
        inputs.track(ref["path"], ref["sha256"])
        inputs.track(ref["metadata_path"], ref["metadata_sha256"])
        if cache_key not in cache:
            cache[cache_key] = dataset_coverage(ref, inputs)
        staged[phase] = cache[cache_key]
        all_blocks.extend(staged[phase])
    staged["cumulative"] = all_blocks
    require(sum(b["n"] for b in all_blocks) == run["history"][-1]["n_after"], f"Cumulative data mismatch: {method}")
    result = []
    for phase, blocks in staged.items():
        for family in FAMILIES:
            for H in HORIZONS:
                selected = [b for b in blocks if b["family"] == family and b["H"] == H]
                stage_counts, k_counts = Counter(), Counter()
                provenance = Counter()
                for block in selected:
                    stage_counts.update(block["stage"])
                    k_counts.update(block["k"])
                    provenance[block["provenance"]] += block["n"]
                for stage in range(H):
                    result.append({"method": method, "data_phase": phase, "family": family, "H": H,
                                   "stage": stage, "labels": stage_counts[stage],
                                   "k_index": stage, "k_labels": k_counts[stage],
                                   "family_H_labels": sum(b["n"] for b in selected),
                                   "contributing_episodes": len({b["episode_key"] for b in selected if b["stage"][stage]}),
                                   "family_H_blocks": len(selected), "provenance_labels": dict(provenance)})
    return result


def add_costs(items):
    return {key: sum(item[key] for item in items) for key in ("episodes", "sim_steps", "worker_seconds")}


def resource_summary(method, run):
    history = run["history"]
    collection = add_costs([h["ledger"]["collection"] for h in history])
    feedback = add_costs([h["ledger"]["feedback"] for h in history])
    evaluations = add_costs([h["ledger"][p] for h in history
                             for p in ("pre_evaluation", "calibration_evaluation", "trained_evaluation")]
                            + [run["ledger"]["final_evaluation"]])
    initial_train = run["initial"]["train_stats"]
    additional_train = [h["train_stats"] for h in history if h["train_stats"] is not None]
    return {"method": method, "initial_labels": run["initial"]["n"],
            "additional_labels": run["ledger"]["additional_labels"], "final_labels": history[-1]["n_after"],
            "initial_optimizer_steps_logical": run["initial"]["train_steps"],
            "additional_optimizer_steps": sum(h["ledger"]["optimizer_steps"] for h in history),
            "initial_collection_logical": run["initial"]["collection_ledger"],
            "additional_collection_physical": collection, "feedback_physical": feedback,
            "evaluation_physical": evaluations,
            "post_initial_total_physical": add_costs([collection, feedback, evaluations]),
            "additional_available_labels": sum(h["ledger"]["collection"].get("available_labels", 0) for h in history),
            "geometry_attempts": run["ledger"]["geometry_attempts"],
            "initial_training_wall_seconds_logical": initial_train["wall_seconds"],
            "additional_training_wall_seconds": sum(t["wall_seconds"] for t in additional_train),
            "peak_gpu_allocated_bytes_max": max(t.get("peak_gpu_allocated_bytes", 0)
                                                 for t in [initial_train, *additional_train]),
            "calibration_seconds": sum(h["ledger"]["calibration_seconds"] for h in history)}


def summarize(groups, systems, inputs, checks):
    final, prefix, curves, coverage, cache, resources = [], [], [], [], {}, []
    for method, run in systems.items():
        for mode in MODES:
            episodes = run["final"][mode]["episodes"]
            final.extend(metrics(method, mode, episodes))
            prefix.extend(prefix_metrics(method, mode, episodes))
            for h in run["history"]:
                for phase in ("pre", "post_cal", "post_train"):
                    for row in metrics(method, mode, h[phase][mode]["episodes"]):
                        curves.append({**row, "round": h["round"], "phase": phase,
                                       "training_labels": h["n_after"] if phase == "post_train" else h["n_before"],
                                       "task_bank": "curve_development"})
        coverage.extend(training_coverage(method, run, inputs, cache))
        resources.append(resource_summary(method, run))
    return {
        "schema_version": 1, "complete": True,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "scope": "single-seed full-H mechanism development; same-MuJoCo deployment proxy",
        "seeds": [0], "families": list(FAMILIES), "horizons": list(HORIZONS),
        "methods": {key: {"folder": s[0], "condition": s[1], "arm": s[2], "label": s[3], "plot_label": s[4]}
                    for key, s in SPECS.items()},
        "validation": {"passed": True, "complete_systems": len(systems), "paired_systems": checks,
                       "healthy_initial_data_sha256": systems["healthy_dagger"]["initial"]["data_sha256"],
                       "healthy_initial_policy_sha256": systems["healthy_dagger"]["initial"]["policy_sha256"],
                       "checks": ["three independent complete artifact validations", "all five configured systems",
                                  "same healthy D0/policy bytes", "same source and non-treatment configuration",
                                  "task hash and rollout seed pairing across methods/phases",
                                  "same first-round calibration for bias S10/S11", "terminal prefix consistency",
                                  "hashed cumulative NPZ stage/k coverage", "no normalization diagnostic loaded"]},
        "experiments": {name: {"fingerprint": group["experiment"]["fingerprint"], "config": group["config"],
                               "physical_resources_committed": group["artifact_validation"]["physical_resources_committed"]}
                        for name, group in groups.items()},
        "final_metrics": final, "paired_contrasts": paired_contrasts(systems),
        "terminal_prefix_survival": prefix, "curve_metrics": curves,
        "training_stage_coverage": coverage, "system_resources": resources,
        "limits": ["Only training seed 0; no significance or population confidence claims",
                   "Final tasks are shared across methods, not independent training repetitions",
                   "Terminal prefix survival is not historical stage reach/survival",
                   "S10 is a frozen-policy mechanism control with fewer labels/optimizer steps",
                   "Curves use a different developer bank from final evaluation; they are not concatenated",
                   "No complete four-arm factorial in the bias condition; no interaction estimated",
                   "Normalization diagnostic, physical robots and recursive improver transfer are excluded",
                   "Resource totals cover committed artifacts; interrupted uncommitted work may be absent"]}


def percentage(row):
    return f"{100 * row['success_rate']:.1f}% ({row['successes']}/{row['episodes']})"


def markdown(summary):
    lookup = {(r["method"], r["mode"], r["family"], r["H"]): r for r in summary["final_metrics"]}
    lines = ["# 长程学习机制开发结果", "", f"生成时间：{summary['created_at']}。",
             "", "**5 个系统全部完成；三个目录的独立 artifact 校验均通过。仅训练种子 0，属于开发筛选，不宣称统计显著性、稳健领先、真机迁移或改进器递归学习。**",
             "", "训练和评测均覆盖 pack/sort/stack × H=2/4/6/8/10；当前采用 constant_safe 归一化。健康条件比较同预算 DAgger 与名义＋恢复专家追加数据。高度偏差条件比较固定 twin 学习、首轮校准冻结策略、首轮校准后学习。",
             "", "健康两组 D0 与初始策略的文件哈希相同；所有末次任务按原始 task SHA256 和 rollout seed 配对。S10/S11 首轮校准结果相同。任何配对或完整性检查失败均停止生成本报告。",
             "", "## 完整任务成功率", "",
             "SR 是完整任务评分谓词的成功比例，不是局部进度 Score。各格任务数相同，all 为三族×五种 H 的等权汇总；H≥8 仅汇总 H8/10。", "",
             "| 系统 | 名义：全部 H | 扰动：全部 H | 名义：H≥8 | 扰动：H≥8 |", "|---|---:|---:|---:|---:|"]
    for method, spec in SPECS.items():
        values = [percentage(lookup[(method, mode, "all", H)]) for H, mode in
                  (("all", "nominal"), ("all", "perturbed"), ("H>=8", "nominal"), ("H>=8", "perturbed"))]
        lines.append(f"| {spec[3]} | " + " | ".join(values) + " |")
    lines += ["", "## 按任务族与阶段数", "", "保留所有 family×H 单元格；sort 的几何偏差影响与 pack/stack 不同，不用总分掩盖。"]
    for mode in MODES:
        lines += ["", f"### {'名义' if mode == 'nominal' else '扰动 p=0.15'}", "",
                  "| 系统 | 任务族 | H2 | H4 | H6 | H8 | H10 |", "|---|---|---:|---:|---:|---:|---:|"]
        for method, spec in SPECS.items():
            for family in FAMILIES:
                values = [percentage(lookup[(method, mode, family, H)]) for H in HORIZONS]
                lines.append(f"| {spec[3]} | {family} | " + " | ".join(values) + " |")
    lines += ["", "## 配对差值", "", "差值为左减右；胜/平/负按同一任务与 rollout seed 计数，不是独立训练种子。S11−S10 包含新增训练资源，不能解读为同计算量效率优势。没有 S00，因此不估计完整 2×2 交互项。", "",
              "| 对照 | 模式 | 范围 | Δ 百分点 | 胜 / 平 / 负 |", "|---|---|---|---:|---:|"]
    for r in summary["paired_contrasts"]:
        if r["family"] == "all" and r["H"] in ("all", "H>=8"):
            lines.append(f"| {r['label']} | {r['mode']} | {r['H']} | {r['delta_pp']:+.2f} | {r['wins']} / {r['ties']} / {r['losses']} |")
    lines += ["", "## 终态前缀与学习覆盖", "",
              "令 L 为记录的 first_unsatisfied_index，满足 0≤L≤H；终态前缀曲线为 P(L≥j)。它表示 episode 结束时，前 j 个任务谓词仍全部满足，不是轨迹历史上曾达到 j 的概率，也不是因果阶段 hazard。图展示 H10 以保持各 j 的分母不变；全部 family×H 曲线在 comparison.json。",
              "", "训练覆盖由 D0 加每轮 new_dataset 的实际 NPZ k/stage 数组统计。下表只列最后一阶段；完整逐阶段、每轮、初始和累计覆盖均保存在 comparison.json。零覆盖仍报告为 0。标签数不等于独立 episode 数，stage 为从 0 开始的拓扑位置，k 为子目标索引。", "",
              "| 系统 | 任务族 | H | 累计标签 | 最后阶段标签 | 最后阶段贡献 episode 数 |", "|---|---|---:|---:|---:|---:|"]
    for r in summary["training_stage_coverage"]:
        if r["data_phase"] == "cumulative" and r["H"] >= 8 and r["stage"] == r["H"] - 1:
            lines.append(f"| {SPECS[r['method']][3]} | {r['family']} | {r['H']} | {r['family_H_labels']} | {r['labels']} | {r['contributing_episodes']} |")
    lines += ["", "## 训练预算与实际执行成本", "",
              "同样保留标签数不代表相同 rollout 成本。采集账目包含完成但未保留全部标签的 rollout；评测账目按实际执行计数，不将复用的测量重复收费。下表的追加成本不含共享初始数据生成。S10 无新增标签和优化步骤。", "",
              "| 系统 | 初始＋新增标签 | 初始＋新增优化步数 | 追加采集 episodes / 物理步 | 反馈 episodes / 物理步 | 实际评测 episodes / 物理步 | 追加训练秒 | 峰值 GPU MiB |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary["system_resources"]:
        c, f, e = [r[key] for key in ("additional_collection_physical", "feedback_physical", "evaluation_physical")]
        lines.append(f"| {SPECS[r['method']][3]} | {r['initial_labels']} + {r['additional_labels']} | "
                     f"{r['initial_optimizer_steps_logical']} + {r['additional_optimizer_steps']} | "
                     f"{c['episodes']} / {c['sim_steps']} | {f['episodes']} / {f['sim_steps']} | "
                     f"{e['episodes']} / {e['sim_steps']} | {r['additional_training_wall_seconds']:.2f} | "
                     f"{r['peak_gpu_allocated_bytes_max'] / 1024**2:.1f} |")
    lines += ["", "按目录的实际总量以独立 artifact validator 为准：同一 bias 目录的三组只收费一次共享初始训练；healthy 两个目录即使初始数据/策略哈希相同，实际重复生成和训练也分别收费，不能按哈希跨目录抵消。", "",
              "| 目录 | 初始训练调用 | 新增训练调用 | 初始＋新增优化步数 | 所有实际 rollout episodes | 所有物理步 |", "|---|---:|---:|---:|---:|---:|"]
    for name, group in summary["experiments"].items():
        physical = group["physical_resources_committed"]
        t, rollouts = physical["training"], list(physical["rollouts"].values())
        lines.append(f"| {name} | {t['initial_calls']} | {t['additional_calls']} | "
                     f"{t['initial_optimizer_steps']} + {t['additional_optimizer_steps']} | "
                     f"{sum(r['episodes'] for r in rollouts)} | {sum(r['sim_steps'] for r in rollouts)} |")
    lines += ["", "worker_seconds 是进程累计时间，不是端到端墙钟时间。上述总量覆盖已提交产物；中断且未提交的工作可能未计入，不能据此保证整个实验零额外成本。", "",
              "## 曲线与边界", "", "![长程开发曲线](long_horizon_curves.png)", "",
              "图前两行是末次开发 bank 的完整任务 SR—H；第三行是 H10 的终态前缀；第四行是独立 curve bank 在训练预算上的轨迹。curve 和 final 的任务数量/身份不同，不拼成一条学习曲线，也不选择最高轮次作为最终成绩。",
              "", "单独的 horizon_normalization_20261028 是冻结模型的归一化诊断，本脚本没有读取或混入。全 H 训练、constant_safe 以及当前任务 bank 与旧 E1 H2/4 外推协议不同，本表不能直接把跨实验差值归因于某一单项改动。",
              "", "本次未拟合指数 α、未进行显著性检验、未打开确认库。后续需要独立训练种子和预先固定的复测协议。完整结构化数据：[comparison.json](comparison.json)；矢量图：[SVG](long_horizon_curves.svg)。", ""]
    return "\n".join(lines)


def make_plots(summary, folder):
    import matplotlib
    matplotlib.use("Agg")
    # Preserve text in the vector artifact without importing the PDF font backend.
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(13, 16), layout="constrained")
    for col, mode in enumerate(MODES):
        for row, methods in ((0, ("healthy_dagger", "healthy_nominal")),
                             (1, ("bias_S01", "bias_S10", "bias_S11"))):
            ax = axes[row, col]
            for method in methods:
                values = sorted((r for r in summary["final_metrics"] if r["method"] == method and r["mode"] == mode
                                 and r["family"] == "all" and isinstance(r["H"], int)), key=lambda r: r["H"])
                ax.plot([r["H"] for r in values], [r["success_rate"] for r in values], "o-",
                        label=SPECS[method][4], color=COLORS[method])
            ax.set(title=f"{'Healthy' if row == 0 else 'Height bias'} / {mode}: final developer tasks",
                   xlabel="Subgoals H", ylabel="Full-task success", xticks=HORIZONS)
        for method in SPECS:
            values = [r for r in summary["terminal_prefix_survival"] if r["method"] == method and r["mode"] == mode
                      and r["family"] == "all" and r["H"] == 10]
            axes[2, col].plot([r["prefix_length"] for r in values], [r["survival"] for r in values], "o-",
                              label=SPECS[method][4], color=COLORS[method], markersize=3)
            first = [r for r in summary["curve_metrics"] if r["method"] == method and r["mode"] == mode
                     and r["family"] == "all" and r["H"] == "all" and r["round"] == 1 and r["phase"] == "pre"]
            after = sorted((r for r in summary["curve_metrics"] if r["method"] == method and r["mode"] == mode
                            and r["family"] == "all" and r["H"] == "all" and r["phase"] == "post_train"),
                           key=lambda r: r["round"])
            points = first + after
            axes[3, col].plot([r["training_labels"] / 1000 for r in points],
                              [r["success_rate"] for r in points], "o-", label=SPECS[method][4], color=COLORS[method])
        axes[2, col].set(title=f"H=10 / {mode}: terminal prefix (not ever-reached)",
                         xlabel="Prefix length j", ylabel="P(terminal prefix >= j)", xticks=range(0, 11, 2))
        axes[3, col].set(title=f"{mode}: separate curve developer bank", xlabel="Retained training labels (thousands)",
                         ylabel="Full-task success")
    for ax in axes.flat:
        ax.set_ylim(-0.02, 1.03)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Full-H development / training seed 0 / same-MuJoCo proxy / no significance claim", fontsize=13)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"long_horizon_curves.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Parent containing the three fullH_* result directories")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        inputs, groups, systems, checks = load_inputs(root)
        inputs.track(__file__)
        inputs.track(Path(__file__).with_name("report_harness.py"))
        summary = summarize(groups, systems, inputs, checks)
        summary["input_files_sha256"] = {str(p): h for p, h in sorted(inputs.hashes.items())}
        with tempfile.TemporaryDirectory(prefix=".long-horizon-report-", dir=root) as temporary:
            staging = Path(temporary)
            make_plots(summary, staging)
            (staging / "RESULTS.md").write_text(markdown(summary), encoding="utf-8")
            (staging / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                                        encoding="utf-8")
            inputs.unchanged()
            for name in ("long_horizon_curves.png", "long_horizon_curves.svg", "comparison.json", "RESULTS.md"):
                # TemporaryDirectory is private on Windows. Moving its files
                # directly would retain that ACL and hide deliverables from the
                # workspace owner/app viewer. Create a sibling under the normal
                # workspace directory, then atomically replace the published file.
                published = root / f".{name}.{uuid.uuid4().hex}.publish"
                try:
                    with published.open("xb") as stream:
                        stream.write((staging / name).read_bytes())
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(published, root / name)
                finally:
                    published.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"汇总失败，未发布新的最终报告：{exc}\n若目录已有旧报告，它不代表本次校验通过。", file=sys.stderr)
        return 2
    print(f"已汇总 5/5 个系统：{root / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
