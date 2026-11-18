"""Observe the frozen legacy MixedController's stopping behavior in 90 rollouts.

No training, controller patch, counterfactual continuation or source mutation.
Run --plan-only first, then run without arguments. Outputs are isolated under
results/mixed_termination_20261028; completed results are verified on rerun.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time


BANK = 88_000_000
FAMILIES = ("pack", "sort", "stack")
HORIZONS = (2, 4, 6, 8, 10)
N_PER = 2
WORKERS = 2
OFF_TABLE_Z = -0.05


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    digest(value)
    return value


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def longest_run(values):
    largest = current = 0
    for value in values:
        current = current + 1 if value else 0
        largest = max(largest, current)
    return largest


def trailing_run(values):
    result = 0
    for value in reversed(values):
        if not value:
            break
        result += 1
    return result


def analyze(log, job, done_patience):
    import numpy as np

    T, steps = int(log.T), int(log.steps)
    require(T > 0, "Empty episode log")
    learner_done = log.k_sel < 0
    oracle_active = log.k_true >= 0
    conflict = learner_done & oracle_active
    trailing = trailing_run(learner_done)
    done_stop = trailing >= done_patience
    if done_stop:
        require(steps == T - 1, "Legacy done stop should log but not execute its final command")
        reason = "learner_done_patience"
    else:
        require(steps == T == job.max_steps, "Unexpected legacy termination; do not infer a reason")
        reason = "time_limit"
    observed_fallen = (log.obj_obs[:, :, 2] < OFF_TABLE_Z).any(axis=1)
    fallen_indices = np.flatnonzero(observed_fallen)
    first_fallen = int(fallen_indices[0]) if len(fallen_indices) else None
    eligible = log.valid & oracle_active
    if first_fallen is not None:
        eligible[first_fallen:] = False
    # Frozen MixedController draws exactly once from seed+13 at every act call.
    # Reconstruct the branch schedule; this does not alter or wrap the controller.
    expert_branch = np.random.default_rng(job.seed + 13).random(T) < job.beta
    terminal = slice(max(0, T - done_patience), T)
    final_sat = list(map(bool, log.meta["final_sat"]))
    strict = bool(done_stop and T >= done_patience and np.all(conflict[terminal]))
    statistics = {
        "stop_reason": reason, "command_rows": T, "executed_control_steps": steps,
        "last_command_unexecuted": done_stop, "max_steps": job.max_steps,
        "success": bool(log.success), "judged_success": bool(log.success_ver),
        "terminal_done_streak": trailing, "false_done_steps": int(conflict.sum()),
        "max_false_done_streak": longest_run(conflict),
        "strict_false_done_patience_stop": strict,
        "done_stop_oracle_active_at_last_row": bool(done_stop and oracle_active[-1]),
        "done_stop_oracle_active_any_terminal_row": bool(done_stop and np.any(oracle_active[terminal])),
        "done_stop_but_task_failed": bool(done_stop and not log.success),
        "time_limit": reason == "time_limit",
        "observed_off_table": first_fallen is not None, "first_observed_off_table_row": first_fallen,
        "final_satisfied": int(sum(final_sat)), "final_sat": final_sat,
        "first_unsatisfied_index": next((i for i, value in enumerate(final_sat) if not value), job.H),
        "max_simultaneous_satisfied": int(np.max(np.sum(log.sat_true, axis=1))),
        "final_held": int(log.meta["final_held"]),
        "eligible_labels_before_cap": int(eligible.sum()),
        "would_retain_labels_at_original_cap": min(int(eligible.sum()), int(job.max_samples)),
        "actual_retained_training_labels": 0,
        "expert_branch_decisions_reconstructed": int(expert_branch.sum()),
        "executed_expert_branch_commands_reconstructed": int(expert_branch[:steps].sum()),
        "expert_branch_on_false_done_rows_reconstructed": int((expert_branch & conflict).sum()),
        "terminal_expert_branch_decisions_reconstructed": int(expert_branch[terminal].sum()),
        "events": log.events,
    }
    arrays = {"k_sel": log.k_sel, "k_true": log.k_true, "sat_true": log.sat_true,
              "sat_ver": log.sat_ver, "action": log.action, "expert_action": log.expert_action,
              "valid": log.valid, "obj_observed_z": log.obj_obs[:, :, 2], "grip": log.grip,
              "expert_branch_reconstructed": expert_branch}
    return statistics, arrays


def summary_rows(episodes):
    output = []
    for round_id in (1, 2, 3):
        for family in ("all", *FAMILIES):
            for H in ("all", *HORIZONS):
                selected = [e for e in episodes if e["round"] == round_id and
                            (family == "all" or e["family"] == family) and (H == "all" or e["H"] == H)]
                if not selected:
                    continue
                complete = [e for e in selected if not e.get("error")]
                output.append({"round": round_id, "beta": 0.5 ** round_id, "family": family, "H": H,
                               "planned_episodes": len(selected), "complete_episodes": len(complete),
                               "worker_errors": len(selected) - len(complete),
                               **{name: sum(bool(e.get(name)) for e in complete) for name in
                                  ("success", "strict_false_done_patience_stop", "done_stop_oracle_active_at_last_row",
                                   "done_stop_but_task_failed", "time_limit", "observed_off_table")},
                               "done_patience_stops": sum(e["stop_reason"] == "learner_done_patience" for e in complete),
                               "sim_steps": sum(e["sim_steps"] for e in selected),
                               "worker_seconds": sum(e["worker_seconds"] for e in selected)})
    return output


def plan_text(plan):
    return f"""# 旧 MixedController 终止机制诊断：冻结计划

日期：2026-10-28。定义指纹：`{plan['fingerprint']}`。

只运行旧 fullH_dagger 的 seed 0。分别加载第 1/2/3 轮 DAgger 采集前的 checkpoint，
配对 β=0.5/0.25/0.125。三组复用新 88M 开发 bank 的 30 个任务：
pack/sort/stack × H=2/4/6/8/10 × 每格 2 个，共 90 次 rollout。
保留原 collection 扰动 p={plan['collection_pert']}，不是无扰动策略评测。
workers=2；当前轮的 checkpoint 与 β 同时变化，因此跨轮差异不能归因于 β 单独变化。

只导入 `{plan['source_root']}` 的冻结源码，保持旧 MixedController/rollout 不变。
不训练、不修 controller、不反事实续跑、不改主实验、不挑任务或增加种子。
source、script、checkpoint、任务和 job 配置在 definition.json 中冻结并存哈希。
本计划先于诊断执行保存；没有提前查看诊断结果。

主要事件：旧 rollout 因 k_sel<0 连续 {plan['done_patience']} 行终止，且相同的全部
{plan['done_patience']} 行 k_true>=0。另记录最后一行 oracle active、终止但任务失败、
time-limit、各族/H 的完整成功率和终态进度。它检验旧机制是否出现，不估计修复后的收益。

掉桌指标遵循旧数据筛选条件：观测物体中心 z<{OFF_TABLE_Z} m。
旧 rollout 不因掉桌直接结束，故掉桌是可与 time-limit/done-stop 重叠的状态标志，不是第三种互斥终止原因。
旧循环会记录最后一条命令再检查 done；done-stop 的最后命令未执行，T=steps+1。
分支标志由旧 MixedController 的 seed+13 单次随机抽样序列重建，明确标为 reconstructed。

只保留诊断日志，实际新增训练标签/优化步骤均为 0。would_retain_labels 是原 300 条 cap 下的
可保留数量诊断，不是实际采集训练数据。失败或零事件均完整保存，不静默排除 worker error。
成本报告包含实际 sim_steps（含环境 settle）、worker 累计时间与每组 pool 墙钟时间。
worker error 若发生在部分执行后，旧 worker 返回的 0 成本不能证明未消耗资源，单独标未知成本。

输出：逐 episode JSON、紧凑 NPZ 轨迹、results.json、README.md。若完整结果已存在只校验并复用；
若中断且无完整结果，STARTED 标志阻止无记账的自动重跑。不得将这份单 checkpoint 系列诊断称为新三种子实验。
"""


def report_text(result):
    lines = ["# 旧 MixedController 终止诊断结果", "",
             f"状态：{result['status']}。定义指纹：`{result['definition_fingerprint']}`。",
             "", "固定 90 次诊断，seed 0 的三个采集前策略分别配原轮次 β。不训练、不修改旧控制器，",
             "不把该结果当修复后收益或 β 的独立消融。所有结果均保留。", "",
             "| 轮次 / β | 完成 / 计划 | 完整成功 | 连续8行冲突并done终止 | done终止末行oracle仍active | done终止但失败 | time-limit | 观测掉桌 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["summary"]:
        if row["family"] == "all" and row["H"] == "all":
            lines.append(f"| {row['round']} / {row['beta']} | {row['complete_episodes']} / {row['planned_episodes']} | "
                         f"{row['success']} | {row['strict_false_done_patience_stop']} | "
                         f"{row['done_stop_oracle_active_at_last_row']} | {row['done_stop_but_task_failed']} | "
                         f"{row['time_limit']} | {row['observed_off_table']} |")
    lines += ["", "## 每个任务族与 H", "",
              "| 轮次 | 任务族 | H | 完整成功 / 完成 | 连续8行冲突终止 | time-limit | 观测掉桌 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["summary"]:
        if row["family"] != "all" and row["H"] != "all":
            lines.append(f"| {row['round']} | {row['family']} | {row['H']} | {row['success']} / {row['complete_episodes']} | "
                         f"{row['strict_false_done_patience_stop']} | {row['time_limit']} | {row['observed_off_table']} |")
    cost = result["ledger"]
    lines += ["", "## 成本与解释边界", "",
              f"实际返回 {cost['episodes']} 个 episode；worker error {cost['worker_errors']}；记录物理步 {cost['sim_steps']:,}；",
              f"worker 累计时间 {cost['worker_seconds']:.3f} 秒；三组 pool 墙钟总计 {cost['pool_wall_seconds']:.3f} 秒。",
              "新增训练标签和优化步骤均为 0。worker 累计时间不等于墙钟时间；若有 worker error，其未记录的部分成本未知。",
              "", "连续冲突事件直接由 k_sel/k_true 轨迹和末尾 done streak 计算。它表明旧终止条件在该轨迹上发生，",
              "不证明继续执行一定能成功。没有观察到事件的单元格同样保留，不能将其当作不存在机制风险的证明。",
              "掉桌是观测状态标志，会与终止类型重叠；不是新增环境终止规则。",
              "", "每个 episode 的输入任务、策略、随机种子、判定、事件和成本在 episodes/*.json；对应紧凑轨迹在 traces/*.npz。",
              "原源码、checkpoint 和任务哈希均已执行后复核。完整矩阵与汇总在 [results.json](results.json)。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    source = base / "results/long_horizon_dev_20261028/fullH_dagger"
    frozen = source / "source/src"
    out = base / "results/mixed_termination_20261028"
    sys.dont_write_bytecode = True
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, str(frozen))

    import numpy as np
    import mujoco
    import horizon.datagen.parallel as parallel
    from horizon.datagen.rollout import PerturbationConfig, rollout
    from horizon.harness.artifacts import load_policy, write_json
    from horizon.harness.experiment import TaskStore, job_for, source_hashes
    from horizon.real2sim.recon_error import ReconModel

    require(Path(parallel.__file__).resolve().is_relative_to(frozen), "Wrong active source")
    experiment = read_json(source / "experiment.json")
    require(experiment["fingerprint"] == digest({k: v for k, v in experiment.items() if k != "fingerprint"}),
            "Source experiment fingerprint mismatch")
    require(source_hashes() == experiment["source_sha256"], "Frozen source drift")
    validation = read_json(source / "artifact_validation.json")
    require(validation.get("passed") is True and validation.get("complete") is True and
            validation.get("experiment_fingerprint") == experiment["fingerprint"], "Source artifacts not completely validated")
    run_path = source / "runs/healthy/S01_s0.json"
    run = read_json(run_path)
    cfg = experiment["config"]
    require(run["status"] == "complete" and run["seed"] == 0 and run["arm"] == "S01" and run["condition"] == "healthy",
            "Unexpected source run")
    require(run["experiment"]["fingerprint"] == experiment["fingerprint"], "Source run fingerprint mismatch")
    require(cfg["families"] == list(FAMILIES) and cfg["H_values"] == list(HORIZONS) and cfg["seeds"] == [0],
            "Source task distribution/seed differs")
    require(cfg["learning_collection"] == "dagger" and len(run["history"]) == 3, "Expected three DAgger rounds")
    done_patience = rollout.__kwdefaults__["done_patience"]
    require(done_patience == 8, "Unexpected legacy done patience")
    out.mkdir(parents=True, exist_ok=True)
    store = TaskStore(out / "tasks", cfg, base)
    batches, checkpoints = [], []
    for round_id, ref in enumerate([run["initial"]["policy"], run["history"][0]["policy_record"],
                                     run["history"][1]["policy_record"]], 1):
        require(file_hash(ref["path"]) == ref["sha256"] == run["history"][round_id - 1]["policy_before"],
                f"Wrong checkpoint before round {round_id}")
        recon = ReconModel.from_dict(run["history"][round_id - 1]["recon_before"])
        require(recon.to_dict() == {"bias": {}, "corrections": {}}, "Expected healthy twin")
        jobs = []
        for fi, family in enumerate(FAMILIES):
            for H in HORIZONS:
                for i in range(N_PER):
                    task_seed = BANK + fi * 100_000 + H * 1000 + i
                    jobs.append(job_for(store, family, H, task_seed, task_seed + 17, cfg, recon, real=False,
                                        controller="mixed", learner="hier", beta=0.5 ** round_id,
                                        pert=PerturbationConfig.suite(cfg.get("collection_pert", 0.1)),
                                        label_expert=True, max_samples=300, want_samples=False, keep_log=True,
                                        only_success=False, max_steps=H * 180 + 60,
                                        provenance=f"mixed_termination_diagnostic_r{round_id}"))
        batches.append((round_id, ref, jobs))
        checkpoints.append({"round": round_id, "beta": 0.5 ** round_id, "policy": ref,
                            "label": f"checkpoint before original DAgger collection round {round_id}"})
    inputs = {"schema_version": 1, "source_root": str(frozen), "source_experiment_fingerprint": experiment["fingerprint"],
              "source_sha256": experiment["source_sha256"], "script_sha256": file_hash(__file__),
              "source_run": {"path": str(run_path), "sha256": file_hash(run_path)},
              "source_validation": {"path": str(source / "artifact_validation.json"),
                                    "sha256": file_hash(source / "artifact_validation.json")},
              "evaluation_bank": BANK, "families": list(FAMILIES), "H_values": list(HORIZONS), "n_per": N_PER,
              "planned_episodes": 90, "unique_task_identities": 30, "workers": WORKERS,
              "collection_pert": cfg.get("collection_pert", 0.1), "done_patience": done_patience,
              "training_steps": 0, "calibration_attempts": 0, "actual_retained_training_labels": 0,
              "world": cfg["world"], "checkpoints": checkpoints,
              "tasks": [job.tag["task_artifact"] for job in batches[0][2]],
              "jobs": [{"round": round_id, **{k: v for k, v in asdict(job).items() if k != "task_payload"}}
                       for round_id, _, jobs in batches for job in jobs],
              "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                          "numpy": np.__version__, "mujoco": mujoco.__version__},
              "definitions": {"strict_false_done_patience_stop": "last eight k_sel<0 and k_true>=0; done stop before final command",
                              "off_table": "any observed object center z < -0.05m; not a rollout stop rule",
                              "expert_branch": "reconstructed default_rng(job.seed+13).random(T) < beta"}}
    plan = {**inputs, "fingerprint": digest(inputs)}
    definition = out / "definition.json"
    if definition.exists():
        require(read_json(definition) == plan, "Frozen diagnostic definition differs; do not overwrite")
    else:
        require(not (out / "STARTED.json").exists(), "Started run lacks definition")
        write_json(definition, plan)
    text = plan_text(plan)
    plan_path = out / "PLAN.md"
    if plan_path.exists():
        require(plan_path.read_text(encoding="utf-8") == text, "Plan text drift")
    else:
        plan_path.write_text(text, encoding="utf-8")
    if args.plan_only:
        print(f"Frozen 90-episode plan: {plan['fingerprint']}", flush=True)
        return 0
    result_path = out / "results.json"
    if result_path.exists():
        result = read_json(result_path)
        require(result["definition_fingerprint"] == plan["fingerprint"] and
                result["record_sha256"] == digest({k: v for k, v in result.items() if k != "record_sha256"}),
                "Saved result checksum/definition mismatch")
        for episode in result["episodes"]:
            require(file_hash(episode["record"]["path"]) == episode["record"]["sha256"], "Episode JSON changed")
            if episode.get("trace"):
                require(file_hash(episode["trace"]["path"]) == episode["trace"]["sha256"], "Episode trace changed")
        (out / "README.md").write_text(report_text(result), encoding="utf-8")
        print("Verified existing diagnostic; no new rollout.", flush=True)
        return 0 if result["status"] == "complete" else 2
    with (out / "STARTED.json").open("x", encoding="utf-8") as stream:
        json.dump({"definition_fingerprint": plan["fingerprint"], "planned_episodes": 90}, stream)
    (out / "episodes").mkdir(exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)
    (out / "source").mkdir(exist_ok=True)
    shutil.copyfile(__file__, out / "source/diagnose_mixed_termination.py")
    episodes, pool_wall = [], 0.0
    for round_id, ref, jobs in batches:
        params = load_policy(ref)
        started = time.perf_counter()
        with parallel.Runner(workers=WORKERS, policy={"params": params}) as runner:
            results = runner.map(jobs)
        elapsed = time.perf_counter() - started
        pool_wall += elapsed
        require(len(results) == 30, "Worker batch size differs")
        for result in results:
            job, log = result.job, result.log
            name = f"r{round_id}_{job.family}_H{job.H}_s{job.task_seed}"
            episode = {"round": round_id, "beta": job.beta, "family": job.family, "H": job.H,
                       "task_seed": job.task_seed, "rollout_seed": job.seed, "task_artifact": job.tag["task_artifact"],
                       "policy": ref, "error": result.error, "sim_steps": int(result.sim_steps),
                       "worker_seconds": float(result.wall), "trace": None}
            if log is None and not result.error:
                episode["error"] = "Worker returned no log without an error"
            if log is not None:
                stats, arrays = analyze(log, job, done_patience)
                episode.update(stats)
                trace_path = out / "traces" / f"{name}.npz"
                np.savez_compressed(trace_path, **arrays)
                episode["trace"] = {"path": str(trace_path), "sha256": file_hash(trace_path)}
            episode_path = out / "episodes" / f"{name}.json"
            write_json(episode_path, episode)
            episodes.append({**episode, "record": {"path": str(episode_path), "sha256": file_hash(episode_path)}})
        write_json(out / f"round{round_id}.json", {"round": round_id, "definition_fingerprint": plan["fingerprint"],
                                                 "episodes": episodes[-30:], "pool_wall_seconds": elapsed})
        count = sum(e.get("strict_false_done_patience_stop", False) for e in episodes[-30:])
        print(f"round {round_id} beta={0.5**round_id}: 30 returned; strict false-done stops={count}; pool {elapsed:.2f}s", flush=True)
    require(source_hashes() == plan["source_sha256"] and file_hash(__file__) == plan["script_sha256"],
            "Source/script changed during diagnostic")
    require(file_hash(run_path) == plan["source_run"]["sha256"], "Source run changed")
    for checkpoint in checkpoints:
        require(file_hash(checkpoint["policy"]["path"]) == checkpoint["policy"]["sha256"], "Checkpoint changed")
    for ref in plan["tasks"]:
        require(file_hash(ref["path"]) == ref["sha256"], "Task changed during diagnostic")
    expected = Counter({(r, f, H): N_PER for r in (1, 2, 3) for f in FAMILIES for H in HORIZONS})
    require(Counter((e["round"], e["family"], e["H"]) for e in episodes) == expected, "Incomplete diagnostic grid")
    errors = sum(bool(e["error"]) for e in episodes)
    output = {"schema_version": 1, "status": "complete" if not errors else "complete_with_errors",
              "definition_fingerprint": plan["fingerprint"], "episodes": episodes, "summary": summary_rows(episodes),
              "ledger": {"episodes": len(episodes), "worker_errors": errors,
                         "sim_steps": sum(e["sim_steps"] for e in episodes),
                         "worker_seconds": sum(e["worker_seconds"] for e in episodes),
                         "pool_wall_seconds": pool_wall, "training_steps": 0, "retained_training_labels": 0,
                         "unknown_partial_error_cost_episodes": errors}}
    output["record_sha256"] = digest(output)
    write_json(result_path, output)
    (out / "README.md").write_text(report_text(output), encoding="utf-8")
    print(f"Saved {output['status']}: {result_path}", flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
