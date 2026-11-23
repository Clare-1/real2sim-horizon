# 组会汇报材料（2026-11-23）

一句话：**失败原因不同，"下一步该做什么"就不同；而且"判断修没修对"和"选哪一步"是两个独立决策，前者的默认做法（用策略在 twin 里的成功率验收）会把正确的修正拒掉。**

所有数字来自 `horizon/results/` 下的批次，本文件夹只是汇总，不改任何数。带 🔶 的是筛查级（1 seed 或 1 bank），不能当主表。

## 一、准备讲的五张图（`figures/`）

| 图 | 文件 | 要说的结论 |
|---|---|---|
| 配方 × 原因热图 | `paper_fig1_heatmap.pdf`、`figR1_recipe_cause.png` | 8 条固定配方里**没有一条在两列都最好**：首轮校准让高度偏差 .417→.750（+33.3 pp），却让覆盖不足 .933→.717（−21.7 pp）。这张是全文的入口。 |
| 选择器 × 原因 | `figC_paired_vs_rule.png` | 均值列会骗人：UCB 在几何/覆盖/健康三列都**显著为正**（+6/+6/+5），恢复上 **−8.3**；LinUCB 恢复上 **−54.4**。所以结论必须按原因分列。 |
| 核验闸门（新增，**已进论文正文**） | `figD_verification.png` | 左：9 个几何修正的留出高度误差全部落在对角线下方；1 个"对本来正确的 twin"的候选（2.1→6.5 mm）被拒。右：无闸门 .717、**策略闸门 .450（拒 7/9）**、预测闸门 .717（拒 0/9）。 |
| 混合原因各臂（新增） | `figA_mixed_arms.png` | 规则与"门打开版规则"**逐格相同**（.72/.90/.57）；API 计划器几何+覆盖上 .85 但三类混合掉到 .30；知原因参考 .89。 |
| 顺序干预（新增，1 bank） | `figB_order.png` | 同两个操作换顺序：C→T .95 vs T→C .85；单独定向 .25；C→U 也是 .95。**顺序非交换，且"先校准"是前置条件。** |

另有 7 张历史图：`fig1_sr_vs_cost`（SR–成本）、`fig2_sr_vs_h`（H 外推，仍是 0，**不建议放主图**）、`fig3_first_action*`（第一步操作）、`fig4_trajectory`（逐轮 SR）、`fig7_budget_sweep`（部署预算）、`figR2_regret`（主导原因占比 vs 遗憾）。

## 二、表格（`tables/`，15 个）

| 文件 | 内容 | 口径 |
|---|---|---|
| `T1_recipes_x_causes.md` | 8 配方 × 3 原因，72 格，含配对差 | 3 banks × 1 seed |
| `T2_full_matrix.md` | 纯原因 4 列 + 混合列 + 全部均值 | 288 + 48 格 |
| `T2b_probe_signals.md` | A2/A/C/D 探针（策略无关诊断） | 3 banks × 10 eps |
| `T3_selectors_288.md` | 8 选择器 × 4 原因 + 效率（AUC / oracle-90% / regret） | 3 banks × 3 seeds |
| `T4_paired_vs_rule.md` | 每个选择器对 rule 的分原因配对差 | 9 格/项 |
| `T5_ablation_and_llm.md` | 表 9（偏差档探针）、表 10（LLM 消融 32 格） | 筛查 |
| `T6_llm_vs_rule_mixed.md` | 混合原因 24 格 + 门打开对照 6 格 + 逐格决策链 | 2 banks × 1 seed 🔶 |
| `T7_order_partial.md` | 顺序实验（**14/36 在跑**） | 1 seed，bank 逐步补齐 🔶 |
| `T8_calibrator_ablation.md` | 校准器消融（含发散种子数） | 2 seeds |
| `fig4_chains.md` | 典型场景逐轮 SR 与操作链 | — |
| `fig1/fig2/fig7/figR1/figR2_numbers.md` | 各图的原始数字 | — |

## 三、这一轮新增的两条硬结论

1. **`rule_fix` 与 `rule` 完全相同**（ΔSR 0.0 pp，区间 [0,0]，n=6）。查机制：门只在 6 格里的 **1 格**绑住过，且那一格结果不变（.95 vs .95）；真正卡住规则的是分支 5 的 `concentration > 0.5`——第一次定向再生成之后失败就不再集中，所以第三轮永远走不到"再定向"。**原先"标签门是结构性盲点"的解释作废。**
2. **三类混合第 3 轮真正缺的是"补恢复数据"**：知原因参考 .50→.80，规则退到 DAgger .50→.65，API 又做一次定向 .50→.20。nominal 评测下扰动压力 0.1 < 阈值 0.5，两者都看不到——这是**可观测性限制**，不是推理能力。

## 四、复现

```bash
cd ../horizon
PYTHONPATH=src .venv-linux/bin/python scripts/report_experiment1.py      # T1
PYTHONPATH=src .venv-linux/bin/python scripts/report_paper8.py results/four_causes_paper8_20261120   # T3
PYTHONPATH=src .venv-linux/bin/python scripts/paired_diffs.py results/four_causes_paper8_20261120    # T4
PYTHONPATH=src .venv-linux/bin/python scripts/report_llm_mixed.py        # T6
PYTHONPATH=src .venv-linux/bin/python scripts/report_conflict_order.py results/experiment_conflict_order_20261123
PYTHONPATH=src .venv-linux/bin/python scripts/make_group_figs.py ../组会汇报_2026-11-23/figures
bash ../组会汇报_2026-11-23/refresh.sh    # 一键重跑全部表/图并覆盖本文件夹
```

## 五、汇报时不要说的话

- 不说"选择器显著打赢固定配方"：rule − fixed 是 +13.5 pp，但 **bandit − rule = +2.1 [−1.1,5.1]、oracle − rule = +1.0 [−2.2,4.0]**，后两个区间都跨零。正确说法是"**收益来自做出选择，而不是来自更聪明的选择器**"。
- 不说"LLM 会处理规则没写到的原因"：混合原因上 LLM .667 < 规则 .733，且失败原因是不可观测，不是不会推理。
- 不说"H 外推成立"：图 2 的 H=6/8/10 仍是 0.000，论文没声称，别在会上被问出来才发现。
- 不说"动力学是第五类原因"：摩擦 1.0→0.01 无变化、质量噪声内、延迟 0/1/2 = .750/.850/.000（二值）。

## 六、还差什么

见 `PENDING.md`（在跑的批次和它们会补上哪一格）与 `RSS审稿风险清单.md`（审稿人最可能打的点、已有证据、缺的那件东西）。