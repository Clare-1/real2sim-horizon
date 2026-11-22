# 论文图表/实验覆盖审计（2026-11-22）

**结论：三条线代码侧完善；论文的图表实验仍有缺项。** 不是“只差真机/LIBERO/LLM”这三块外部资源。
图例：✅ 当前协议已出；🟡 有旧协议/部分；🔴 未做；⛔ 外部阻塞；⏭️ 可不跑。

## 实验一：要不要更新 twin（表 1 / 图 1 / 图 2 / 表 A1-A2）

| 项 | 状态 | 现有 | 缺 |
|---|---|---|---|
| 表 1（8 方法 × 高度/位置/摩擦 × pack/sort/stack × H2–10） | 🟡 | **8 条 pipeline 已全部实现**：`open_loop`(200k)、`always_dagger`、`domain_randomize`、`deploy_dagger`、`mixed_train`、`fixed`、`calibrate_alternate`、`true_params`；`configs/experiment1_table1_pilot.yaml` 已启动（3 条件 × 2 banks × 8 方法 = 48 格） | sort/stack；H2–10；位置/摩擦列（无估计器 / 平台不可表达） |
| 图 1（SR vs 累计部署） | ✅ | `results/fig1_sr_cost_20261122/`（8 selectors × 4 causes，离线） | 无 |
| 图 2（SR vs H） | 🔴 | `h_extrapolation_20261116` 是旧编码，H6/8/10 = 0 | 新编码 + H6/8/10 |
| 表 A1/A2 | 🔴 | — | 扰动模式 / 按任务族拆分 |

**平台限制**：高度偏差可注入可修；位置偏差可注入但 `geometry_repair` 只估高度（无修复算子）；摩擦 1.0→0.01 SR 不变。导师表 1 的三列在本平台只有高度有信息量。

## 实验二：按失败原因选操作（表 2 / 表 3 / 图 3 / 图 4）

| 项 | 状态 | 现有 | 缺 |
|---|---|---|---|
| 表 2（5 causes + mixed × 8 selectors） | 🟡 | paper8：4 causes × 8 selectors × 9 cells + CI；mixed grid 独立 48 runs | 动力学列（不可表达）、mixed 并入主表、LLM/LLM+经验行（无 key） |
| 表 3（AUC / oracle-90% / regret / 首步 / 健康花费 / API） | ✅ | `paper8_report.md` 新增 AUC/oracle-90/regret/健康花费（离线算） | 仅 API 花费（无 key） |
| 图 3（原因 × 第一步操作） | ✅ | `results/fig3_action_four_causes_paper8_20261120/` | 无 |
| 图 4（典型场景逐轮） | 🟡 | `make_fig4_trajectory.py` 已有，sources 含 four_causes | 用 paper8 重新生成一张当前版本 |
| dev 30 / test 60 场景划分 | 🔴 | 只有 10 banks 的 held-out 分 bank 检查 | 正式 30/60 划分 + 规则阈值只在 dev 上调 |

## 实验三：LIBERO（表 4 / 图 5）

| 项 | 状态 | 说明 |
|---|---|---|
| 表 4 / 图 5 | 🔴 | 代码有 HDF5/ImageBC/官方 env 适配器/DPAdapter；**无数据、无权重、无 BDDL、未跑** |

## 实验四：真机（表 5/6/7 / 图 6）

| 项 | 状态 | 说明 |
|---|---|---|
| 表 5/6/7 / 图 6 | 🔴 | 代码有接口/mock/Verifier 成功判定/EpisodeLog；**无硬件、无位姿源、未跑** |

## 实验五：消融（表 8/9/10/11 / 图 7）

| 项 | 状态 | 现有 | 缺 |
|---|---|---|---|
| 表 8 校准器 | ✅ | `calibrator_ablation_20261112/TABLE8.md`（legacy/robust/freeze/true/predictive_gate） | 若要当前协议可重跑，不阻塞 |
| 表 9 偏差档 | 🟡 | `bias_response_20261116`（4 档） | 当前 4-cause 协议下的 3 档 |
| 表 10 agent 消融 | 🔴 | — | 无 LLM key；LLM/无经验/去工具/便宜模型 |
| 表 11 H 外推 | 🟡 | `h_extrapolation_20261116` 是旧编码（H6/8/10=0） | 新编码训练 H2/4 与 H2/4/6 |
| 图 7 部署预算 8/24/48 | 🟡 | `fig7_budget_sweep.png` 是旧 2-cause/3-selector | 当前 4-cause/8-selector 的预算扫描（需新跑） |

## R1 / R2 / 理论

| 项 | 状态 | 说明 |
|---|---|---|
| 图 R1 配方×原因 | ✅ | 8 selectors × 4 causes（`fig_R1_recipe_cause_20261121`） |
| 图 R2 主导原因占比 vs 遗憾 | ✅ | 10-bank 每类均值算的期望遗憾（`fig_R2_regret_20261122`） |
| 命题 1/2/推论 | 🟡 | 写在 `导师意见回应_2026-11-21.md`，尚未进论文正文/证明文件 |

## 可以跳过 / 不能跑

- ⏭️ **最小验证（10×3）**：导师明确说“不进论文”；stage-1 已 PASS，stage-2 到 235/240，可跑完留档，也可直接停。
- ⏭️ **动力学原因**：平台不可表达（摩擦/质量/延迟扫描），写 limitation，不跑。
- ⏭️ **RoboDojo / RoboCasa365**：可选、外部。
- ⛔ **LLM rows、真机、LIBERO**：外部 key/硬件/数据。

## 论文前必须补的（按优先级）

1. **实验一扩表**：8 条 pipeline 已实现，pilot（`experiment1_table1_20261122`）在跑；之后补 sort/stack、H2–10。
2. **dev/test 30/60 划分**：规则阈值只在 dev 上调，test 只报数。
3. **表 11 新编码 H 外推**：`check_h_extrapolation.py` 已能跑，补 H=6/8/10。
4. **图 2（SR vs H）与图 7（8/24/48）**：需要新跑；先确认队列。
5. **mixed 并入表 2**：把 48-run mixed grid 作为第 5 列。
6. **LLM / 真机 / LIBERO**：外部。

> 已离线补齐：表 3 的 AUC/oracle-90/regret/健康花费（`report_paper8.py`）与图 1（`make_fig1_cost.py`）。
