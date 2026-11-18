"""Report the complete three-seed full-H replication; never run experiments.

Usage: python scripts/report_long_horizon_replication.py RESULTS_ROOT
Requires fullH_dagger, fullH_nominal and fullH_bias, all 15 systems, and passing
artifact_validation.json files. Revalidates actual artifacts read-only before
producing descriptive seed means/sample SD, paired deltas, coverage and costs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, stdev
import sys
import tempfile
import uuid

from report_harness import canonical, identity, load_and_validate, require
from report_long_horizon_development import (
    COLORS, FAMILIES, HORIZONS, MODES, SPECS, Inputs, check_terminal, digest,
    metrics, paired_contrasts, prefix_metrics, resource_summary, training_coverage,
)
from validate_harness_artifacts import Validator


SEEDS = (0, 1, 2)
EXPECTED = {
    "fullH_dagger": {("healthy", "S01", seed) for seed in SEEDS},
    "fullH_nominal": {("healthy", "S01", seed) for seed in SEEDS},
    "fullH_bias": {("height_bias", arm, seed) for arm in ("S01", "S10", "S11") for seed in SEEDS},
}
PHASES = ("pre", "post_cal", "post_train")
OUTPUT_NAMES = ("long_horizon_replication.png", "long_horizon_replication.svg", "comparison.json", "RESULTS.md")


def preflight(root, inputs):
    """Gate all three matrices before reading any episode scores."""
    groups = {}
    for name, expected in EXPECTED.items():
        folder = root / name
        artifact = inputs.json(folder / "artifact_validation.json")
        experiment = inputs.json(folder / "experiment.json")
        completion = inputs.json(folder / "completion.json")
        require(artifact.get("passed") is True and artifact.get("complete") is True and
                artifact.get("partial") is False and not artifact.get("error") and not artifact.get("pending"),
                f"Incomplete/failed independent artifact validation: {name}")
        require(artifact.get("expected_runs") == artifact.get("validated_runs") == len(expected),
                f"Wrong independent run count: {name}")
        require(completion.get("complete") is True and completion.get("confirmation_executed") is False and
                completion.get("expected_runs") == len(expected), f"Incomplete matrix or wrong completion: {name}")
        require(artifact.get("experiment_fingerprint") == completion.get("fingerprint") == experiment.get("fingerprint"),
                f"Experiment/completion/validator fingerprint mismatch: {name}")
        cfg = experiment["config"]
        require(cfg["seeds"] == list(SEEDS) and cfg["families"] == list(FAMILIES) and
                cfg["H_values"] == list(HORIZONS), f"Wrong three-seed full-H grid: {name}")
        require({(c, a, s) for c in cfg["conditions"] for a in cfg["arms"] for s in cfg["seeds"]} == expected,
                f"Wrong configured systems: {name}")
        require(cfg.get("calibration_schedule") == "first_round" and cfg["train"].get("normalization") == "constant_safe",
                f"Wrong calibration/normalization protocol: {name}")
        require(cfg.get("training_observation_mode") == "legacy", f"Wrong replication observation mode: {name}")
        require(cfg.get("reuse_pre_evaluation") is True, f"Boundary pre-evaluation reuse is not enabled: {name}")
        require(cfg.get("learning_collection") == ("nominal_recovery" if name == "fullH_nominal" else "dagger"),
                f"Wrong learning collection: {name}")
        require(cfg["banks"] == {"training": 84_000_000, "feedback": 85_000_000,
                                 "curve": 86_000_000, "final": 87_000_000}, f"Wrong replication banks: {name}")
        require(cfg["rounds"] == 3 and cfg["curve_n_per"] == 4 and cfg["final_n_per"] == 20 and
                cfg["init_budget"] == 80_000 and cfg["round_budget"] == 40_000 and
                cfg["feedback_episodes"] == 30 and cfg["train"]["steps"] == 8_000,
                f"Wrong replication budgets: {name}")
        require(artifact.get("validated_rounds") == len(expected) * cfg["rounds"], f"Wrong validated rounds: {name}")
        groups[name] = {"config": cfg, "experiment": experiment, "artifact_validation": artifact}
    common = None
    for name, group in groups.items():
        cfg = {k: v for k, v in group["config"].items() if k not in {"name", "arms", "conditions", "learning_collection"}}
        if common is None:
            common = canonical(cfg)
        require(canonical(cfg) == common, f"Non-treatment configuration mismatch: {name}")
    require(len({canonical(g["experiment"]["source_sha256"]) for g in groups.values()}) == 1,
            "Different frozen source versions")
    require(groups["fullH_dagger"]["config"]["conditions"] == groups["fullH_nominal"]["config"]["conditions"],
            "Healthy conditions differ")
    return groups


class TrackedValidator(Validator):
    """Use the independent validator without writing its report or running code."""

    def __init__(self, root, inputs):
        self.inputs = inputs
        # Include records and sidecars that Validator reads without file().
        for path in sorted(Path(root).rglob("*.json")):
            inputs.track(path)
        super().__init__(root, partial=False)

    def file(self, path, expected):
        path = super().file(path, expected)
        # Hash once here; Inputs.unchanged() rechecks before publishing.
        if path not in self.inputs.hashes:
            self.inputs.track(path, expected)
        else:
            require(self.inputs.hashes[path] == expected, f"Conflicting input hash: {path}")
        return path

    def source(self):
        super().source()
        require(self.source_root == self.root / "source", "A frozen source snapshot is required")
        for relative, expected in self.exp["source_sha256"].items():
            path = self.source_root / relative if relative.startswith("scripts/") else self.source_root / "src/horizon" / relative
            self.inputs.track(path, expected)
        for path in sorted((self.source_root / "configs").glob("*.yaml")):
            self.inputs.track(path)


def revalidate(folder, group, inputs):
    validator_path = Path(__file__).with_name("validate_harness_artifacts.py")
    saved = group["artifact_validation"]
    require(saved.get("validator_sha256") == digest(validator_path), "Saved validator version differs; validate again")
    inputs.track(validator_path, saved["validator_sha256"])
    current = TrackedValidator(folder, inputs).run()
    for key, value in current.items():
        require(canonical(saved.get(key)) == canonical(value), f"Saved artifact validation is stale: {folder}/{key}")
    return current


def evaluation_items(run):
    yield "final", run["final"]
    for h in run["history"]:
        for phase in PHASES:
            yield f"round{h['round']}/{phase}", h[phase]


def feedback_signature(record, inputs):
    ref = record["feedback_record"]
    manifest = inputs.json(ref["path"], ref["sha256"])
    observations = manifest["observations"]
    inputs.track(observations["path"], observations["sha256"])
    # Storage paths differ by arm; compare actual signal bytes and episode data.
    return canonical({"observations_sha256": observations["sha256"],
                      "signals": manifest["signals"], "gripper_channel": manifest["gripper_channel"],
                      "object_channel": manifest["object_channel"],
                      "episodes": [{**item, "task": {k: v for k, v in item["task"].items() if k != "path"}}
                                   for item in manifest["episodes"]]})


def validate_pairing(systems, inputs):
    require(set(systems) == {(method, seed) for method in SPECS for seed in SEEDS},
            "Exactly 15 systems (five methods x three seeds) are required")
    initial_pairs = []
    for seed in SEEDS:
        healthy_a, healthy_b = [systems[(method, seed)]["initial"] for method in ("healthy_dagger", "healthy_nominal")]
        for field in ("data_sha256", "policy_sha256", "n", "train_steps"):
            require(healthy_a[field] == healthy_b[field], f"Healthy initial {field} differs at seed {seed}")
        bias_runs = [systems[(method, seed)] for method in ("bias_S01", "bias_S10", "bias_S11")]
        for field in ("data_sha256", "policy_sha256", "n", "train_steps"):
            require(len({canonical(run["initial"][field]) for run in bias_runs}) == 1,
                    f"Bias initial {field} differs at seed {seed}")
        h10, h11 = [systems[(method, seed)]["history"][0] for method in ("bias_S10", "bias_S11")]
        require(canonical(h10["recon_after"]) == canonical(h11["recon_after"]), f"S10/S11 first belief differs at seed {seed}")
        require(canonical(h10["recon_before"]) == canonical(h11["recon_before"]), f"S10/S11 initial belief differs at seed {seed}")
        require(feedback_signature(h10, inputs) == feedback_signature(h11, inputs),
                f"S10/S11 first feedback differs at seed {seed}")
        initial_pairs.append({"seed": seed, "healthy_data_sha256": healthy_a["data_sha256"],
                              "healthy_policy_sha256": healthy_a["policy_sha256"],
                              "bias_data_sha256": bias_runs[0]["initial"]["data_sha256"],
                              "bias_policy_sha256": bias_runs[0]["initial"]["policy_sha256"],
                              "first_calibrated_belief": h10["recon_after"]})
    # The evaluation seed schedule intentionally does not depend on training seed.
    reference = dict(evaluation_items(systems[("healthy_dagger", SEEDS[0])]))
    for (method, seed), run in systems.items():
        evaluations = dict(evaluation_items(run))
        require(set(evaluations) == set(reference), f"Evaluation phases differ: {method}/s{seed}")
        for label, value in evaluations.items():
            for mode in MODES:
                keys = [identity(e) for e in value[mode]["episodes"]]
                expected = [identity(e) for e in reference[label][mode]["episodes"]]
                require(len(keys) == len(set(keys)) and set(keys) == set(expected),
                        f"Unpaired shared task hashes/rollout seeds: {method}/s{seed}/{label}/{mode}")
    return initial_pairs


def load_inputs(root):
    inputs = Inputs()
    groups = preflight(root, inputs)
    systems = {}
    for name, group in groups.items():
        folder = root / name
        group["fresh_artifact_validation"] = revalidate(folder, group, inputs)
        runs, cfg, validation = load_and_validate(folder, allow_partial=False)
        require(set(runs) == EXPECTED[name], f"Missing or unexpected systems: {name}")
        require(canonical(cfg) == canonical(group["config"]), f"Manifest/run configuration differs: {name}")
        require(validation["fingerprint"] == group["experiment"]["fingerprint"], f"Run fingerprint differs: {name}")
        group["record_validation"] = validation
        for (condition, arm, seed), run in runs.items():
            method = next(key for key, spec in SPECS.items() if spec[:3] == (name, condition, arm))
            require((method, seed) not in systems, f"Duplicate system: {method}/s{seed}")
            systems[(method, seed)] = run
            for label, value in evaluation_items(run):
                for mode in MODES:
                    for episode in value[mode]["episodes"]:
                        check_terminal(episode, f"{method}/s{seed}/{label}/{mode}/{identity(episode)}")
    pairs = validate_pairing(systems, inputs)
    return inputs, groups, systems, pairs


def aggregate_seed_rows(rows, keys, values):
    """Every metric is first computed per seed; never pool tasks to estimate SD."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, members in grouped.items():
        members.sort(key=lambda row: row["seed"])
        require([r["seed"] for r in members] == list(SEEDS), f"Missing/duplicate seed in summary group: {key}")
        result = {**dict(zip(keys, key)), "seeds": list(SEEDS), "n_seeds": len(SEEDS)}
        for field in values:
            numbers = [float(r[field]) for r in members]
            result[field] = {"values": numbers, "mean": mean(numbers), "sample_sd": stdev(numbers)}
        output.append(result)
    return output


def physical_totals(groups):
    training, rollouts = {}, defaultdict(lambda: {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0})
    for group in groups.values():
        cost = group["fresh_artifact_validation"]["physical_resources_committed"]
        for key, value in cost["training"].items():
            training[key] = max(training.get(key, 0), value) if key.endswith("_max") else training.get(key, 0) + value
        for kind, value in cost["rollouts"].items():
            for key in rollouts[kind]:
                rollouts[kind][key] += value[key]
    return {"training": training, "rollouts": dict(rollouts),
            "all_rollouts": {key: sum(value[key] for value in rollouts.values())
                             for key in ("episodes", "sim_steps", "worker_seconds")},
            "accounting": "Sum physical costs across directories; shared initial counted once per condition/seed within a directory. Identical hashes in separate directories are charged separately."}


def boundary_reuse_summary(systems):
    """Count actual saved boundary records separately from learning efficiency."""
    records = []
    zero_cost = {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}
    for (method, seed), run in systems.items():
        previous = None
        for h in run["history"]:
            reused = h["reused_equal_state_evaluation"]["pre"]
            require(reused is (previous is not None), f"Unexpected boundary reuse flag: {method}/s{seed}/r{h['round']}")
            if reused:
                require(h["pre_evaluation_source_round"] == previous["round"] and h["pre"] == previous["post_train"],
                        f"Boundary reuse contents/source differ: {method}/s{seed}/r{h['round']}")
                cost = h["ledger"]["pre_evaluation"]
                require(cost == zero_cost, f"Reused boundary charged physical pre-evaluation cost: {method}/s{seed}/r{h['round']}")
                episodes = sum(len(h["pre"][mode]["episodes"]) for mode in MODES)
                require(episodes == 120, f"Reused boundary episode count differs: {method}/s{seed}/r{h['round']}")
                records.append({"method": method, "seed": seed, "round": h["round"],
                                "source_round": h["pre_evaluation_source_round"],
                                "avoided_rollouts": episodes, "actual_cost": dict(cost)})
            previous = h
    avoided = sum(r["avoided_rollouts"] for r in records)
    require(len(records) == 30 and avoided == 3600, "Expected 30 reused boundaries / 3600 avoided rollouts")
    return {"reused_boundaries": len(records), "avoided_rollouts": avoided,
            "actual_cost": {key: sum(r["actual_cost"][key] for r in records) for key in zero_cost},
            "records": records,
            "interpretation": "Reuses unchanged boundary evaluation records; computational bookkeeping optimization, not evidence of learning/sample-efficiency advantage. Avoided physics steps or wall time are not estimated."}


def summarize(groups, systems, inputs, pairs):
    final, prefix, curves, coverage, resources, contrasts, cache = [], [], [], [], [], [], {}
    for seed in SEEDS:
        seed_systems = {method: systems[(method, seed)] for method in SPECS}
        contrasts.extend({**row, "seed": seed} for row in paired_contrasts(seed_systems))
        for method, run in seed_systems.items():
            for mode in MODES:
                final.extend({**row, "seed": seed} for row in metrics(method, mode, run["final"][mode]["episodes"]))
                prefix.extend({**row, "seed": seed} for row in prefix_metrics(method, mode, run["final"][mode]["episodes"]))
                for h in run["history"]:
                    for phase in PHASES:
                        curves.extend({**row, "seed": seed, "round": h["round"], "phase": phase,
                                       "training_labels": h["n_after"] if phase == "post_train" else h["n_before"],
                                       "task_bank": "curve_development"}
                                      for row in metrics(method, mode, h[phase][mode]["episodes"]))
            coverage.extend({**row, "seed": seed} for row in training_coverage(method, run, inputs, cache))
            resources.append({**resource_summary(method, run), "seed": seed})
    metric_keys = ("method", "mode", "family", "H")
    metric_values = ("success_rate", "terminal_prefix_fraction", "terminal_satisfied_fraction", "max_simultaneous_satisfied_fraction", "successes", "episodes")
    return {
        "schema_version": 1, "complete": True, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Three-seed full-H development replication; same-MuJoCo deployment proxy",
        "seeds": list(SEEDS), "families": list(FAMILIES), "horizons": list(HORIZONS),
        "methods": {key: {"folder": spec[0], "condition": spec[1], "arm": spec[2], "label": spec[3], "plot_label": spec[4]}
                    for key, spec in SPECS.items()},
        "validation": {"passed": True, "complete_systems": len(systems), "initial_pairs": pairs,
                       "checks": ["All three completion and independent-validation gates before score reads",
                                  "Fresh read-only validation of actual artifacts and source/config hashes",
                                  "Exactly five methods x seeds 0/1/2; per-seed identical healthy initial data/policy bytes",
                                  "Per-seed shared bias initial bytes and identical S10/S11 first feedback/belief",
                                  "Shared task hashes and rollout seeds across every method and training seed",
                                  "Actual NPZ stage/k coverage and committed physical cost accounting",
                                  "All tracked input hashes rechecked before publication"]},
        "experiments": {name: {"fingerprint": group["experiment"]["fingerprint"], "config": group["config"],
                               "physical_resources_committed": group["fresh_artifact_validation"]["physical_resources_committed"]}
                        for name, group in groups.items()},
        "final_metrics": final, "final_summary": aggregate_seed_rows(final, metric_keys, metric_values),
        "paired_contrasts": contrasts,
        "paired_summary": aggregate_seed_rows(contrasts, ("left", "right", "label", "mode", "family", "H"),
                                               ("delta_pp", "wins", "ties", "losses", "episodes")),
        "terminal_prefix_survival": prefix,
        "terminal_prefix_summary": aggregate_seed_rows(prefix, (*metric_keys, "prefix_length"), ("survival", "surviving", "episodes")),
        "curve_metrics": curves,
        "curve_summary": aggregate_seed_rows(curves, (*metric_keys, "round", "phase", "training_labels"), metric_values),
        "training_stage_coverage": coverage,
        "coverage_summary": aggregate_seed_rows(coverage, ("method", "data_phase", "family", "H", "stage", "k_index"),
                                                 ("labels", "k_labels", "family_H_labels", "contributing_episodes", "family_H_blocks")),
        "system_resources": resources, "physical_resources_total": physical_totals(groups),
        "boundary_evaluation_reuse": boundary_reuse_summary(systems),
        "limits": ["Means/sample SD describe only three training seeds; SD is not a confidence interval or significance test",
                   "The same 300 final tasks and rollout seeds are shared across training seeds and methods; these are not independent evaluation task banks",
                   "No reserved confirmation bank, real robot, independent simulation engine, agent-decision advantage or recursive-improver transfer",
                   "S10 has fewer labels/optimizer steps; S11-S10 is not a compute-matched efficiency comparison",
                   "No full four-arm factorial; no interaction estimate",
                   "Terminal prefix means all earlier predicates hold at episode end, not historical reach or causal hazard",
                   "Curve and final banks are distinct; final round is fixed rather than chosen by best score",
                   "Physical totals cover committed artifacts; interrupted uncommitted work may be absent"]}


def stats_text(value, scale=1, signed=False):
    sign = "+" if signed else ""
    return f"{value['mean'] * scale:{sign}.2f} ± {value['sample_sd'] * scale:.2f}"


def markdown(summary):
    lookup = {(r["method"], r["mode"], r["family"], r["H"]): r for r in summary["final_summary"]}
    lines = ["# 三种子长程机制复现", "", f"生成时间：{summary['created_at']}。", "",
             "15/15 个系统完成；三个目录保存的独立校验与本次只读重校验均通过。训练种子为 0、1、2；所有统计先逐种子计算，再报告均值 ± 样本标准差（n−1 分母）。标准差不是置信区间，本报告不做显著性宣称。", "",
             "训练和评测均覆盖 pack/sort/stack × H2/4/6/8/10。每个种子的最终评测均为 300 个共享任务 × 名义/扰动两模式；同一批任务与 rollout seed 跨训练种子、方法配对，不把 900 次同任务评估当作 900 个独立任务。", "",
             "每个种子的健康两组 D0/初始策略字节相同；偏差三组共享各自初始状态，S10/S11 首轮反馈与校准后 belief 相同。S10 使用更少标签与训练步数；没有完整四臂因子设计，因此不估计交互项。", "",
             "## 完整任务成功率", "", "单位：百分比，三种子均值 ± 样本标准差；SR 为完整任务真值谓词成功率。", "",
             "| 系统 | 名义：全部 H | 扰动：全部 H | 名义：H≥8 | 扰动：H≥8 |", "|---|---:|---:|---:|---:|"]
    for method, spec in SPECS.items():
        cells = [stats_text(lookup[(method, mode, "all", H)]["success_rate"], 100)
                 for H, mode in (("all", "nominal"), ("all", "perturbed"), ("H>=8", "nominal"), ("H>=8", "perturbed"))]
        lines.append(f"| {spec[3]} | " + " | ".join(cells) + " |")
    lines += ["", "## 任务族与 H", "", "保留所有 family×H 单元格，避免总分掩盖 sort 与 pack/stack 的差异。"]
    for mode in MODES:
        lines += ["", f"### {mode}", "", "| 系统 | 任务族 | H2 | H4 | H6 | H8 | H10 |", "|---|---|---:|---:|---:|---:|---:|"]
        for method, spec in SPECS.items():
            for family in FAMILIES:
                cells = [stats_text(lookup[(method, mode, family, H)]["success_rate"], 100) for H in HORIZONS]
                lines.append(f"| {spec[3]} | {family} | " + " | ".join(cells) + " |")
    lines += ["", "## 逐种子配对差", "", "左减右，单位为百分点；先在同一训练种子的相同任务/rollout seed 上求差，再跨种子汇总。逐 seed 的全部 family×H 胜/平/负和差值在 comparison.json。", "",
              "| 对照 | 模式 | 范围 | seed 0 | seed 1 | seed 2 | 均值 ± 样本 SD |", "|---|---|---|---:|---:|---:|---:|"]
    for row in summary["paired_summary"]:
        if row["family"] == "all" and row["H"] in ("all", "H>=8"):
            values = " | ".join(f"{v:+.2f}" for v in row["delta_pp"]["values"])
            lines.append(f"| {row['label']} | {row['mode']} | {row['H']} | {values} | {stats_text(row['delta_pp'], signed=True)} |")
    lines += ["", "## 实际训练覆盖与成本", "",
              "comparison.json 保留每个 seed 的初始、每轮及累计 NPZ stage/k 标签和贡献 episode 数，包括零覆盖，并给出跨 seed 描述统计。终态前缀曲线表示 P(结束时前 j 个谓词均满足)，不表示历史上曾到达 j 的概率。", "",
              "| 系统 | seed | 初始＋新增标签 | 初始＋新增优化步 | 追加采集 episodes / 物理步 | 实际评测 episodes / 物理步 | 追加训练秒 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary["system_resources"]:
        collection, evaluation = row["additional_collection_physical"], row["evaluation_physical"]
        lines.append(f"| {SPECS[row['method']][3]} | {row['seed']} | {row['initial_labels']} + {row['additional_labels']} | "
                     f"{row['initial_optimizer_steps_logical']} + {row['additional_optimizer_steps']} | "
                     f"{collection['episodes']} / {collection['sim_steps']} | {evaluation['episodes']} / {evaluation['sim_steps']} | "
                     f"{row['additional_training_wall_seconds']:.2f} |")
    total = summary["physical_resources_total"]
    training, rollouts = total["training"], total["all_rollouts"]
    reuse = summary["boundary_evaluation_reuse"]
    lines += ["", f"按三个目录物理账目合计：初始训练 {training['initial_calls']} 次、追加训练 {training['additional_calls']} 次；"
              f"优化 {training['initial_optimizer_steps'] + training['additional_optimizer_steps']:,} 步；"
              f"rollout {rollouts['episodes']:,} 次、物理步 {rollouts['sim_steps']:,}。", "",
              "同一 bias 目录每个 seed 的共享初始生成/训练只收费一次；两个 healthy 目录实际重复生成和训练，即使哈希相同也分别收费。worker_seconds 为进程累计时间，不是端到端墙钟时间。完整反馈、采集、评测与 GPU 资源账目在 JSON；中断而未提交的工作可能未计入。", "",
              f"边界评估复用独立统计：从保存记录核得 {reuse['reused_boundaries']} 个轮次边界，复用 {reuse['avoided_rollouts']:,} 次 rollout 的评测结果；"
              "这些边界的实际新增 pre-evaluation episodes、物理步与 worker_seconds 均经核验为 0。每次复用内容与上一轮 post_train 完全一致。"
              "这属于消除重复评测的执行优化，不能解释为学习或样本效率优势；不推算节省的物理步或墙钟时间。", "",
              "## 曲线与解释范围", "", "![三种子长程复现](long_horizon_replication.png)", "",
              "曲线为逐种子统计的均值，误差线为样本 SD。前两行使用 final bank；第三行是 H10 终态前缀；第四行使用独立 curve bank，不能与 final 点拼接或选择最高轮次作为最终结果。S10 的训练标签量不随轮次增加。", "",
              "本报告只支持三种子、固定开发任务库和同 MuJoCo 代理环境内的复现描述，不能证明真机迁移、独立仿真引擎泛化、agent 决策优势或改进器递归学习。未读取确认库，也未混入旧单种子开发结果或归一化诊断。", "",
              "结构化逐种子数据与输入哈希：[comparison.json](comparison.json)；矢量图：[SVG](long_horizon_replication.svg)。", ""]
    return "\n".join(lines)


def make_plots(summary, folder):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    def draw(ax, method, rows, x_key, field, scale=1):
        ax.errorbar([r[x_key] / scale for r in rows], [r[field]["mean"] for r in rows],
                    yerr=[r[field]["sample_sd"] for r in rows], fmt="o-", capsize=3,
                    label=SPECS[method][4], color=COLORS[method], markersize=3)

    fig, axes = plt.subplots(4, 2, figsize=(13, 16), layout="constrained")
    for col, mode in enumerate(MODES):
        for row, methods in ((0, ("healthy_dagger", "healthy_nominal")), (1, ("bias_S01", "bias_S10", "bias_S11"))):
            for method in methods:
                values = sorted((r for r in summary["final_summary"] if r["method"] == method and r["mode"] == mode
                                 and r["family"] == "all" and isinstance(r["H"], int)), key=lambda r: r["H"])
                draw(axes[row, col], method, values, "H", "success_rate")
            axes[row, col].set(title=f"{'Healthy' if row == 0 else 'Height bias'} / {mode}: final tasks",
                               xlabel="Subgoals H", ylabel="Full-task success", xticks=HORIZONS)
        for method in SPECS:
            values = sorted((r for r in summary["terminal_prefix_summary"] if r["method"] == method and r["mode"] == mode
                             and r["family"] == "all" and r["H"] == 10), key=lambda r: r["prefix_length"])
            draw(axes[2, col], method, values, "prefix_length", "survival")
            values = [r for r in summary["curve_summary"] if r["method"] == method and r["mode"] == mode
                      and r["family"] == "all" and r["H"] == "all" and
                      ((r["round"] == 1 and r["phase"] == "pre") or r["phase"] == "post_train")]
            values.sort(key=lambda r: (r["round"], r["phase"] != "pre"))
            draw(axes[3, col], method, values, "training_labels", "success_rate", 1000)
        axes[2, col].set(title=f"H=10 / {mode}: terminal prefix", xlabel="Prefix length j",
                         ylabel="P(terminal prefix >= j)", xticks=range(0, 11, 2))
        axes[3, col].set(title=f"{mode}: separate curve bank", xlabel="Retained labels (thousands)", ylabel="Full-task success")
    for ax in axes.flat:
        low, high = ax.get_ylim()
        ax.set_ylim(min(-0.02, low), max(1.03, high))
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Full-H replication / 3 training seeds / mean +/- sample SD (not CI) / same-MuJoCo proxy", fontsize=12)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"long_horizon_replication.{extension}", dpi=180)
    plt.close(fig)


def publish(root, summary, inputs):
    """Create published files under the workspace ACL; publish RESULTS last."""
    with tempfile.TemporaryDirectory(prefix=".replication-report-", dir=root) as temporary:
        staging = Path(temporary)
        make_plots(summary, staging)
        (staging / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (staging / "RESULTS.md").write_text(markdown(summary), encoding="utf-8")
        inputs.unchanged()
        # Copy all files out of the private temporary-directory ACL before any
        # replacement, so an ACL/copy failure cannot publish a new success marker.
        ready = []
        try:
            for name in OUTPUT_NAMES:
                sibling = root / f".{name}.{uuid.uuid4().hex}.publish"
                ready.append((name, sibling))
                with sibling.open("xb") as stream:
                    stream.write((staging / name).read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
            inputs.unchanged()
            for name, sibling in ready:
                os.replace(sibling, root / name)
        finally:
            for _, sibling in ready:
                sibling.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        inputs, groups, systems, pairs = load_inputs(root)
        for name in ("report_long_horizon_replication.py", "report_long_horizon_development.py", "report_harness.py", "validate_harness_artifacts.py"):
            inputs.track(Path(__file__).with_name(name))
        summary = summarize(groups, systems, inputs, pairs)
        summary["input_files_sha256"] = {str(path): sha for path, sha in sorted(inputs.hashes.items())}
        publish(root, summary, inputs)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(f"汇总失败：{exc}\n未成功发布本次完整报告；目录中的旧报告不代表本次校验通过。", file=sys.stderr)
        return 2
    print(f"已汇总 15/15 个系统，三个训练种子：{root / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
