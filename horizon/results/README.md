# 结果快照说明

本目录是论文实验结果的**轻量快照**，不是原始运行目录。原始结果（策略权重、任务 payload、逐 episode rollout）体积为 GB 级，未纳入仓库。

## 目录内容

| 目录 | 内容 | 用途 |
|---|---|---|
| `four_causes_20261118/` | `experiment.json`、`shared/*/*/effective_config.json`、每 run 的 `record.json` + `feedback.json`、`report.md`、`probe_recovery_signal*.{md,json}` | 当前四类原因 × 6 选择器批次（48/48；rule 列已用当前代码重跑） |
| `mixed_grid_20261118/` | 4 种混合条件 × 6 选择器的 48 runs + `mixed_report.md` | Gap 1：混合原因与 gap recovery |
| `operation_sweep_20261118/` | 40 条反事实 `(profile, action, ΔSR)` + `value_model.json` | Gap 2：learned value 的负结果 |
| `four_causes_learned_20261118/` | linucb / learned_value 的 16 runs | Gap 2：learned selector |
| `four_causes_multilabel_20261118/` | multilabel selector 的 8 runs（v1 模型） | Gap 2：8/8 首步 |
| `four_causes_multilabel_v2_20261118/` | 用完整混合样本重训后的 8 runs | v2 复验 |
| `diagnosability_bank_20261118/` | 按 bank 分组留一的 5 类混淆矩阵与 recall | learned diagnoser 的可信版本 |
| `diagnosability_multilabel_20261118/` | 3 个二分类的 recall/precision + 混合成分命中率 | 多标签接口（mixed 0% → 任一 100%） |
| `fig_R1_recipe_cause_20261121/` | 8 配方 × 4 原因的最终 SR 热力图 | 导师 §4.4 的 R1 |
| `diagnosis_profiles_20261118/` | 4 类 × 20 个独立 feedback profile（只跑 feedback） | 去重后的诊断数据 |
| `fig3_action_20261118/` | 原因 × 第一步操作热图（总图 + 分选择器） | 图 3 |
| `fig4_trajectory_20261118/` | 典型场景逐轮 SR 与操作链 | 附录图 4 |
| `friction_calibration_20261118/friction_sweep.json` | 物体摩擦 1.0 → 0.01 的 SR 扫描 | 动力学不可表达的负结果 |
| `minimal_validation_20261112/stage{1,2}` | `summary.json` / `completion.json` / `experiment.json` | 表 2/3 的历史协议结果 |
| `minimal_validation_mixed_20261116/` | `summary.json` | 混合原因对照 |
| `budget_split_20261116/` | `summary.json` | 预算拆分 |
| `twin_ablation_mixedfix_20261116/`、`verify_ablation_20261116/`、`verify_pred_20261116/` | `summary.json` | 孪生消融与核验消融 |
| `bias_response_20261116/bias_response.json` | 偏差档响应曲线 | 表 9 的数据 |
| `long_horizon_replication_20261028/` | README / RESULTS / 审计文档 | 长程复验结论（`comparison.json` 18 MB 未纳入） |
| `paper_figures_20261116/` | 图 1/3/7 与表 | 历史图件 |
| `diagnosability5_20261117/` | 五类可诊断性（已作废，保留为记录） | 审计证据 |

## 未纳入的内容与再生成

| 未纳入 | 原因 | 如何再生成 |
|---|---|---|
| `runs/**/policy_*.npz`、`shared/*/policy_0.npz` | 每个 2 MB，纯确定性权重 | 用同一 config + seed 重跑 |
| `tasks_train/`、`tasks_deploy/`（约 46 MB） | 由 bank 种子确定性生成的 JSON payload | `run_minimal_validation.py` 会自动重建（已存在则复用） |
| `minimal_validation_20261112` 的逐 run 记录（993 MB） | 体量 | 只保留 `summary.json`；需要逐 run 时重跑 |
| 其他历史目录的原始 rollout | 体量 | 按对应 `configs/*.yaml` 重跑 |

## 一致性校验

每个运行目录都有 `experiment.json` 的 `fingerprint`；四类批次完成后还会写 `completion.json`（含 `expected_runs` 与实际 `runs`）。逐场景生效配置在 `shared/<scenario>/s<seed>/effective_config.json`，用于检查场景覆盖是否泄漏（这是 2026-11-18 修掉的一个整批作废 bug，详见仓库根目录的 `重跑风险审计_2026-11-18.md`）。
