# Breaking the Horizon — Subgoal-Attributed Agentic Real2Sim2Real

一个基于 MuJoCo 的 **real2sim2real 学习闭环**：用部署反馈归因失败子目标，再由选择器决定下一轮做哪种学习操作（校准孪生体 / 补数据 / DAgger / 停止）。本仓库是论文实验代码、配置、协议文档与当前结果的快照。

> 平台定位：同一 MuJoCo 引擎下的“真机代理”（部署配置代理），**不是**物理机器人，也不含跨模拟器验证。所有“真机”表述均为此代理。

---

## 1. 核心问题

给定一个与真实部署存在偏差的孪生体，系统能否**只根据部署反馈**判断该修什么，并在同一预算下比固定配方更好？

- **选择器**：`fixed` / `rule` / `oracle` / `random` / `round_robin` / `bandit` / `linucb` / `multilabel`（`learned_value` 作为负结果保留）
- **操作集**：`calibrate_geometry`、`regenerate_uniform`、`regenerate_targeted`、`add_recovery_data`、`dagger_round`、`stop`
- **可分级原因（四类）**：几何偏差 / 覆盖不足 / 恢复不足 / 健康
- **动力学原因**：本平台**不可表达**（见 §5 的扫描证据），论文中写作 limitation，不硬凑第五类

## 2. 仓库结构

```
.
├── README.md                      # 本文件
├── 重跑风险审计_2026-11-18.md      # 逐表逐图对照导师意见的审计（含自我更正）
├── 实验进展_2026-11-16.md          # 最近一轮实验进展
├── 最小验证规格_2026-11-12.md      # 选择器实验协议（预算/指标/场景）
├── v7_论文主张与验证路线.md          # 主张与验证路线
├── 导师意见回应_2026-11-21.md       # 对导师意见的逐条回应（主张/最小验证/计划/故事线/相关工作）
├── 论文批次结果_2026-11-21.md       # 288-run 8 选择器主表 + 95% CI
├── 汇报_2026-11-17/                # 组会材料、图、表、数据
└── horizon/                        # 代码（Python 包 + 脚本 + 配置 + 测试）
    ├── src/horizon/                # 任务图、数据生成、策略、监控归因、RSI、选择器
    ├── scripts/                    # 运行与报表脚本
    ├── configs/                    # 实验配置（YAML）
    ├── tests/                      # pytest
    └── results/                    # 结果快照（见 results/README.md）
```

`horizon/README.md` 中有更细的模块—论文方案对应表；本文件中的相对链接（`../*.md`）在仓库根目录可正常解析。

## 3. 当前状态（2026-11-23）

- 筛查批次 `results/four_causes_20261118`：**48/48 完成**（4 causes × 2 banks × 6 selectors，1 seed）。
- 论文批次 `results/four_causes_paper8_20261120`：**288/288 完成**（4 causes × 3 banks × 3 seeds × 8 selectors，2026-11-21 03:22）；最终表见 §5.7。
- **实验一 8 条固定流程**：`results/experiment1_table1_20261122` **48/48 完成**（8 方法 × 3 条件 × 2 banks，2026-11-22 22:08）；表见 §5.9。
- **导师最小验证 stage-2**：`results/minimal_validation_advisor_20261121` **240/240 完成**（2 causes × 10 banks × 3 seeds × 4 selectors）。
- **LLM 已进主表**：`results/four_causes_llm_20261122` 8 格，真实端点（minara.cloud / `openai/gpt-6.2-astra`），23 次调用 / **$0.044**（§9.2）。
- 完整完成度对照：**外部阻塞 3 块 + 内部缺口 11 项**，见 §8；论文图表覆盖见 §10。

| 项 | 值 |
|---|---|
| 筛查批次 | 48/48，rule 首步 6/8；每格 1–2 样本，仅供方向 |
| 论文批次 | 288/288，9 cells/格；`configs/minimal_validation_4causes_paper8.yaml` |
| 论文 multilabel | `multilabel_model_heldout_paper.json`（训练排除 profile banks 0–2 与 four_causes） |
| 协议预算 | 初始化 80k / 每轮 40k / 8000 步训练 / 24 反馈 episode / 3 轮 |

最终表（2 样本/格，**筛查级，不能直接上主表**）：

| 选择器 | 几何 | 覆盖 | 恢复 | 健康 | 平均 |
|---|---:|---:|---:|---:|---:|
| fixed | 0.775 | 0.675 | 0.850 | 0.975 | 0.819 |
| rule | 0.900 | 0.875 | **1.000** | 0.875 | 0.912 |
| random | 0.450 | 0.925 | 1.000 | 0.950 | 0.831 |
| round_robin | 0.875 | 0.825 | 0.650 | 0.850 | 0.800 |
| bandit | 0.925 | 1.000 | 0.850 | 1.000 | **0.944** |
| oracle | 0.925 | 0.950 | 0.875 | 0.875 | 0.906 |

> **2026-11-18 13:00 更新**：rule 列已换成当前代码（删掉旧动力学分支、stop 分支加 SR 下限 0.85、concentration 边界改为 `<= 0.5`）。两个必须讲清楚的 trade-off：
> 1. **恢复 final 1.000 但首步 0/2**——不是选对了 `add_recovery_data`，而是 3 轮 DAgger 兜底；恢复作为“原因”不可诊断。探针显示即使把 feedback 扰动从 0.5 提到 0.7，`P(fail|armed)` 仍只有 0.292（低于覆盖 0.458），所以没有启用 v2。
> 2. **healthy 首步 stop 2/2，final 0.875 vs fixed 0.975**——用 0.10 SR 换 0 标签（fixed 花 80k）。成本敏感的表必须同时报 SR 和标签。
>
> 详见审计 §12。历史 note：11:30 之前的 rule 行来自旧版内存模块（含 `slip/drop → randomize_physics` 分支），已在 13:00 被替换。

> 读数注意：筛查批次每格 1–2 个样本（95% CI 约 ±6–8 pp），只用于方向；论文批次是 `configs/minimal_validation_4causes_paper8.yaml`（4 causes × 3 banks × 3 seeds × 8 selectors = **288 runs**，约 19 h），已完成。
>
> **导师最小验证：10 banks 预注册 PASS**——bandit 几何 +12.5 pp、覆盖 +23.5 pp（均显著）；rule 覆盖 +20.5 pp（显著）、几何 +5.0 pp（不显著）。1 seed；详见 §5.8 与 [`导师意见回应_2026-11-21.md`](导师意见回应_2026-11-21.md)。

## 4. 快速开始

```bash
cd horizon
python3.12 -m venv .venv
.venv/bin/pip install -e .[dev,train]      # 或按 pyproject 安装 mujoco/numpy/scipy/matplotlib/pyyaml/tqdm/torch

PYTHONPATH=src .venv/bin/python -m pytest -q tests/
PYTHONPATH=src .venv/bin/python scripts/run_minimal_validation.py \
    --config configs/minimal_validation_4causes.yaml \
    --out results/four_causes_20261118 --workers 12
```

报告与图（**可在批处理运行中执行**，读到多少算多少）：

```bash
cd horizon
.venv/bin/python scripts/report_four_causes.py          # 进度 + SR/首步/首步分布/配对差/成本
.venv/bin/python scripts/make_fig3_action.py            # 原因 × 第一步操作（默认单图）
.venv/bin/python scripts/make_fig3_action.py --per-selector
.venv/bin/python scripts/make_fig4_trajectory.py        # 典型场景逐轮轨迹
.venv/bin/python scripts/make_all_tables.py             # 全部表格 → ../实验表格汇总_2026-11-12.md
.venv/bin/python scripts/probe_recovery_signal.py       # 在已有策略上实测恢复信号（先看分离度）
```

## 5. 关键结论与负结果

### 5.1 四类原因 + 动力学不可表达

| 旋钮 | 扫描 | 结果 |
|---|---|---|
| 物体摩擦 | 1.0 → 0.01（100×） | SR 恒为 0.750 / 0.700 |
| 手指摩擦 | 1.5 → 0.1（15×） | 0.750 → 0.700 后**平到底** |
| 物体质量 | 1× → 8× | 0.70–0.80（噪声内） |
| 控制延迟 | 0 → 4 步 | 0.750 / 0.850 / **0.000** / 0.000 / 0.000（二值，无中间档） |

原因：夹爪是力闭合夹持，竖直搬运时摩擦与自重不起决定作用。**“MJCF 里的 μ 变了”只是机制证据，不是因果证据**；必须先扫 SR 再谈原因是否成立。

### 5.2 已修掉的整批作废 bug

`run_selector_experiment` 曾写 `cfg = cfg_scn`，让 recovery 场景的 `eval_mode/eval_pert/recovery_frac` 泄漏到其后的 healthy 场景，且日志看不出来。现已重写为基配置只读、每场景深拷贝，并逐场景写 `effective_config.json` 便于审计。

### 5.3 其他限制

- **恢复信号链在 v1 里是断的（2026-11-18 新发现）**：`perturbed_failure_rate` 读的是从未写入的 `log.meta["perturbed"]`，`recovery_triggers` 找的是从不产生的事件类型，因此恢复列对非 oracle 选择器结构性为 0。已加 v2（从 `log.events` 实测 P(fail | 扰动)）与探针；探针显示 `feedback_pert=0.5` 时恢复场景的扰动失败率反而最低（0.25，对比覆盖 0.46、几何 0.83），所以 **216-run 之前必须把 recovery 的 `feedback_pert` 提到 0.7 并重测分离度**，再启用 `recovery_signal: v2`。详见审计第九节。
- `randomize_physics` 动作保留，但**没有对应的真实原因**，在实验中永远不会是正确的第一步；规则也不选它。
- 位置偏差可行但没有修复算子（`geometry_repair` 只估高度），只能当“未知原因”。
- 反馈扰动 (0.1/0.5) 低于评测扰动 (0.7)，偏保守，不影响方向判断——但正是这一点让恢复不足在反馈里不可见。

详细证据与自我更正见 [`重跑风险审计_2026-11-18.md`](重跑风险审计_2026-11-18.md) 第八节。

### 5.4 learned diagnoser 的可信数字（Phase 1）

按 bank 分组留一（同一 bank 的多个 selector 看到同一 profile，旧脚本把它们当独立样本，88.9% 被放大），并补 4 类 × 20 个独立 feedback profile 后：**184 个独立 profile / 156 组**，几何 100%、覆盖 93%、健康 92%、恢复 100%，单标签 **mixed 6%**（18 个里 16 个判成覆盖）。

→ 纯原因可分；混合原因不可靠。论文主张应是“单原因探针 + 混合压力测试 + 弃权”，而不是“分类器能分四类”。详见审计 §13。

**多标签修正**：把单标签换成 3 个二分类后（阈值 0.3，按混合 recall 选），mixed 从 6% 变成“任一成分命中 100%、两成分全中 72%”（n=18，bank-grouped LOO；exact-set 82%）。论文接口应写成多标签后验，而不是单原因分类；`MultiLabelSelector` 已把预测集合映射为操作序列。

### 5.5 Gap 1：混合原因（48 runs）

| 条件 | fixed | rule | bandit | oracle | rule gap | bandit gap |
|---|---:|---:|---:|---:|---:|---:|
| mixed_gc | 0.650 | 0.725 | 0.925 | 0.925 | +27% | **+100%** |
| mixed_gr | 0.775 | 0.725 | 0.800 | 0.925 | −33% | **+17%** |
| mixed_cr | 0.550 | 0.900 | 0.925 | 0.950 | +88% | **+94%** |
| mixed_gcr | 0.575 | 0.575 | 0.800 | 0.800 | 0% | **+100%** |

gap = (SR − fixed)/(oracle − fixed)。bandit 在四个条件全部 ≥ rule；mixed_gr 上 rule 甚至负收益。**混合场景是 exploring/learned selector 的空间，规则不是天花板。**

### 5.6 Gap 2：learned selectors

| 选择器 | 几何 | 覆盖 | 恢复 | 健康 | 平均 | 首步对 |
|---|---:|---:|---:|---:|---:|---:|
| bandit | 0.925 | 1.000 | 0.850 | 1.000 | **0.944** | 2/8 |
| rule | 0.900 | 0.875 | 1.000 | 0.875 | 0.912 | 6/8 |
| multilabel (held-out) | 0.900 | 0.925 | 0.800 | 0.875 | 0.875 | 7/8 |
| oracle | 0.925 | 0.950 | 0.875 | 0.875 | 0.906 | 8/8 |
| linucb | 0.925 | 0.950 | 0.425 | 0.925 | 0.806 | 2/8 |
| learned_value | 0.350 | 1.000 | 0.875 | 0.950 | 0.794 | 2/8 |
| fixed | 0.775 | 0.675 | 0.850 | 0.975 | 0.819 | 2/8 |
| random | 0.450 | 0.925 | 1.000 | 0.950 | 0.831 | 0/8 |
| round_robin | 0.875 | 0.825 | 0.650 | 0.850 | 0.800 | 2/8 |

> learned 行来自独立目录（`four_causes_learned/multilabel*`），仍是 1 seed × 2 banks 的筛查级；论文批次用 3 seeds × 3 banks 重跑。

- **multilabel（held-out 诊断 + 序列）首步 7/8，平均 0.875**：learned 诊断接近 oracle（8/8），但没有超过 rule 的最终 SR（0.912）。in-sample v1 的 8/8/0.906 偏高；v2 与 held-out 一致（7/8、0.875）。
- bandit 最终 SR 最高但首步 2/8（探索型）。
- learned_value 是负结果：40 条反事实样本的 value model R²=0.138、排序 3/8，首步总选 recovery；且该评估仍是 in-sample。写进 limitations。

### 5.7 论文批次结果（288/288，9 cells/格）

| 选择器 | 几何 | 覆盖 | 恢复 | 健康 | 平均 | 首步对 |
|---|---:|---:|---:|---:|---:|---:|
| bandit | 0.961 | 0.983 | 0.817 | 0.978 | **0.935** | 9/36 |
| oracle | 0.939 | 0.972 | 0.939 | 0.844 | 0.924 | 36/36 |
| rule | 0.900 | 0.928 | 0.900 | 0.928 | 0.914 | 21/36 |
| round_robin | 0.894 | 0.872 | 0.822 | 0.939 | 0.882 | 9/36 |
| multilabel | 0.828 | 0.928 | 0.828 | 0.883 | 0.867 | **28/36** |
| linucb | 0.939 | 0.917 | 0.356 | 0.961 | 0.793 | 9/36 |
| fixed | 0.772 | 0.600 | 0.833 | 0.911 | 0.779 | 9/36 |
| random | 0.300 | 0.939 | 0.867 | 0.972 | 0.769 | 0/36 |

- **bandit 最终 SR 最高（0.935）**；oracle 0.924；rule 0.914 是强 baseline。
- **multilabel 首步 28/36（78%）**，显著高于 rule 的 21/36（58%），接近 oracle；标签成本最低（55.6k）。但最终 SR 0.867 低于 rule/bandit——诊断对不等于序列最优，false positive 会浪费一轮。
- **oracle 的健康 SR 反而比 fixed/bandit 低 6.7 pp**：正确动作是 stop，停止放弃了 DAgger 的提升。**oracle 是诊断正确性上界，不是 SR 上界**。
- 完整 CI / 配对差 / 成本见 [`论文批次结果_2026-11-21.md`](论文批次结果_2026-11-21.md)。

```bash
cd horizon
PYTHONPATH=src .venv-linux/bin/python scripts/run_minimal_validation.py \
  --config configs/minimal_validation_4causes_paper8.yaml \
  --out results/four_causes_paper8_20261120 --workers 12
```

### 5.8 导师意见与最小验证（go/no-go）

- 逐条回应：[`导师意见回应_2026-11-21.md`](导师意见回应_2026-11-21.md)（主张、最小验证、修订计划、故事线、相关工作、7 条冲突）。
- 图 R1（8 配方 × 4 原因，无一行通吃）：[`horizon/results/fig_R1_recipe_cause_20261121/fig_R1_recipe_cause.png`](horizon/results/fig_R1_recipe_cause_20261121/fig_R1_recipe_cause.png)。

| 选择器 | 几何（10 banks） | 覆盖（10 banks） | 首步对（几何/覆盖） | 标签/格 |
|---|---:|---:|---:|---:|
| fixed | 0.835 [0.790, 0.880] | 0.720 [0.680, 0.765] | 10/10, 0/10 | 80,000 |
| rule | 0.885 [0.825, 0.940] | 0.925 [0.885, 0.965] | 10/10, 9/10 | 81,111 |
| bandit | 0.960 [0.925, 0.990] | 0.955 [0.920, 0.985] | 10/10, 0/10 | 77,778 |
| oracle | 0.965 [0.940, 0.990] | 0.985 [0.960, 1.000] | 10/10, 10/10 | 74,444 |

- **PASS**：bandit 两类都显著高于 fixed（几何 +12.5 pp、覆盖 +23.5 pp）；rule 覆盖 +20.5 pp 显著，几何 +5.0 pp（CI 重叠，不显著）。
- 这是 **10 banks × 1 seed** 的 stage-1 预注册；论文主表仍是 paper8 的 **3 seeds × 3 banks**（§5.7）。通过后再补 seeds 1–2 即完全对齐导师规格。
- 与 3-bank 表的差异：10 banks 的 fixed 几何更高（0.835 vs 0.772），所以 rule 的几何优势从 +12.8 缩到 +5.0；覆盖优势稳定（+32.8 → +20.5）。
- 仍需说明：无 LLM key（bandit 作为 learned/exploring 代理）；规则阈值未做 dev/test 划分；oracle 是诊断正确性上界，不是 SR 上界。
- **图 R2**（用 10-bank 每类均值算期望遗憾，不新跑）：fixed 13.0/26.5 pp、rule 8.0/6.0 pp、bandit 0.5/3.0 pp（几何/覆盖）；fixed−rule gap 从覆盖 20.5 pp 降到几何 5.0 pp。链接：[`results/fig_R2_regret_20261122/fig_R2_regret.png`](horizon/results/fig_R2_regret_20261122/fig_R2_regret.png)。
- **dev/test bank 检查**（seed 0，banks 0–2 = dev，3–9 = held-out）：fixed 0.871/0.721、rule 0.907/0.929、bandit 0.979/0.936（几何/覆盖）。排序保持；rule 的几何优势从 +8.3（dev）缩到 +3.6（test），覆盖仍 +20.8 pp。注意 rule 阈值是硬编码的，这不是完整的调参/测试分离。
- **stage-2 完成**：`results/minimal_validation_advisor_20261121` **240/240**（2 causes × 10 banks × 3 seeds × 4 selectors），结论与 stage-1 一致（详见审计 §18/§21）。

### 5.9 实验一：8 条固定流程（导师表 1 版式）

数据源 `results/experiment1_table1_20261122`（48 格 = 8 方法 × 3 条件 × 2 banks）；最后一列是相对方法 2 的按条件配对 bootstrap 95% CI。

| # | 方法 | 用反馈 | 用 twin | 校准 twin | 高度偏差 | 覆盖不足 | 健康 | 平均 | 标签/格 |
|---|---|---|---|:---:|---:|---:|---:|---:|---:|
| 1 | 开环生成 200k | ✗ | ✓ | ✗ | 0.350 | 0.050 | 0.950 | 0.450 | 0 |
| 2 | 固定 twin + DAgger | ✓ | ✓ | ✗ | 0.425 | **0.975** | 0.925 | 0.775 | 100,000 |
| 3 | 领域随机化 + DAgger | ✓ | ✓ | ✗ | 0.525 | 0.350 | **1.000** | 0.625 | 120,000 |
| 4 | 部署环境 DAgger | ✓ | ✗ | — | 0.575 | 0.875 | 0.975 | 0.808 | 113,333 |
| 5 | 仿真 + 部署混合 | ✓ | ✓ | ✗ | 0.225 | 0.850 | 0.975 | 0.683 | 120,000 |
| 6 | 首轮校准后冻结 | ✓ | ✓ | ✓ | 0.775 | 0.675 | 0.975 | 0.808 | 80,000 |
| 7 | 重复校准 + DAgger | ✓ | ✓ | ✓ | **0.800** | 0.225 | 0.950 | 0.658 | 40,000 |
| 8 | 真实参数当 twin（上界） | ✓ | ✓ | 真值 | **0.950** | **0.975** | 0.925 | **0.950** | 86,667 |

**没有一条固定配方通吃**（导师“配方 × 原因错配”的直接证据）：

- 高度偏差：8 (0.950) > 7 (0.800) > 6 (0.775) > 4 (0.575) > 3 (0.525) > 2 (0.425) > 1 (0.350) > 5 (0.225)；
- 覆盖不足：2 ≈ 8 (0.975) > 4 (0.875) > 5 (0.850) > 6 (0.675) > 3 (0.350) > 7 (0.225) > 1 (0.050)；
- 健康：3 (1.000) > 4/5/6 (0.975) > 1 (0.950) > 2/8 (0.925)。

三条关键读数：

1. **校准类方法在几何强、在覆盖弱**：方法 6 几何 0.775 / 覆盖 0.675；方法 7 几何 0.800 / 覆盖 **0.225**——校准吃掉了覆盖需要的标签预算。
2. **方法 2 覆盖最强、几何最弱**：固定 twin 把部署失败全归因成“数据不够”，对几何偏差无能为力（0.425）。
3. **方法 5（混合训练）几何最差（0.225）**：twin 与部署数据混在一起，校准信号被稀释；方法 1（开环 200k）几何 0.350 / 覆盖 0.050——数据多但不用反馈，两个条件都差。

口径：只有 2 banks × 1 seed，CI 很宽（只有方法 1 显著更差、方法 8 显著更好）；进论文前需扩到 3–5 banks（约 120–200 格），见审计 §24。

### 5.10 表 11 / 图 2：H 外推（训练 H=2,4 → 测 H=2..10）

| 训练集 | 模式 | H=2 | H=4 | H=6 | H=8 | H=10 |
|---|---|---:|---:|---:|---:|---:|
| {2,4} | nominal | 0.950 | 0.900 | **0.000** | **0.000** | **0.000** |
| {2,4} | perturbed | 0.950 | 0.900 | **0.000** | **0.000** | **0.000** |
| {2,4,6,8,10} | nominal | 0.800 | 0.800 | 0.500 | 0.900 | 0.750 |
| {2,4,6,8,10} | perturbed | 0.750 | 0.800 | 0.450 | 0.800 | 0.700 |

- **外推失败是真的**：同一评测路径下，全 H 训练模型在 H≥6 得 0.45–0.90；只训 H2/4 的模型是精确 0.000。
- **不是任务不可解**：脚本专家在 H=6/8/10 各 **3/3** 成功（步数 506/683/860）。
- **旧解释是错的**：此前把 H≥6 的 0 归因于“旧观测编码”，但当前特征已是连续 `k/10`、`H/10`，**仍然全 0**——应写成“策略不能在 H 上外推”。
- **全 H 训练有代价**：同 80k 标签预算下 H=2 从 0.950 掉到 0.800、H=4 从 0.900 掉到 0.800（预算被 5 个 H 摊薄）。
- 口径：1 seed、20 episodes/点。图：`results/fig2_sr_vs_h_20261123/fig2_sr_vs_h.png`；审计 §25。

### 5.11 操作冲突与顺序：真正值钱的地方（2026-11-23）

"规则赢固定配方"有一个廉价解释：固定配方把第一轮预算花在校准上。若"校准 + 定向再生成"每轮都做就能两类通吃，选择问题就不存在。所以必须证明**操作之间确实冲突**。从 440 个已有决策里抽出的证据（`scripts/analyze_operation_conflicts.py`）：

**边际 ΔSR（百分点，括号内 n）**

| 原因 | calibrate | targeted | uniform | recovery data | dagger |
|---|---:|---:|---:|---:|---:|
| coverage | +0.0 (48) | **+59.8 (92)** | −0.0 (9) | +10.8 (13) | +20.4 (92) |
| geometry | **+40.4 (81)** | +13.0 (51) | +16.9 (34) | +2.5 (12) | +3.6 (92) |
| recovery | +0.0 (44) | +4.7 (48) | +7.8 (9) | **+27.7 (52)** | +23.8 (92) |
| healthy | +0.0 (49) | +6.1 (44) | +8.9 (9) | +3.7 (15) | +4.2 (73) |

**同一操作，先校准 vs 不先校准**

| 原因 | 操作 | 先校准 | 未校准 |
|---|---|---:|---:|
| coverage | targeted | **+81.9 (31)** | +48.5 (61) |
| geometry | targeted | **+18.2 (42)** | **−11.1 (9)** |
| geometry | dagger | +1.1 (58) | +7.9 (34) |
| recovery | recovery data | **+42.5 (12)** | +23.3 (40) |
| mixed_gc | targeted | **+68.5 (10)** | +2.5 (2) |

四条结论：

1. **顺序非交换**：几何上 targeted 单独做 **−11.1 pp**，先校准变 **+18.2 pp**（符号翻转）；覆盖上 +48.5 → +81.9。
2. **校准是前置条件型操作**：非几何三类直接 ΔSR 是 **+0.0**（n=44–49）——固定配方第一轮给校准，在覆盖上纯亏。
3. **DAgger 的价值随 twin 变准而下降**（几何 +7.9 → +1.1），所以"多打 DAgger"不能替代"先修 twin"。
4. **uniform 与 targeted 冲突**：uniform 几何 +16.9、覆盖 −0.0，覆盖要的是 targeted。

口径：以上来自混合选择器的日志，不是预注册序列（几何"未校准 targeted" n=9）。已排队预注册顺序实验：同一操作不同顺序，2 原因 × 8 序列 × 2 banks = **32 runs**（`configs/experiment_conflict_order.yaml`，`rsi/sequences.py` 的 `seq:` 选择器）。

**诊断信号必须与策略无关**（第二个主张）：策略已在错的 twin 上优化，twin 内部自检天然循环——证据是 recovery 的 legacy 信号恒 0、`P(fail|armed)` 0.29 < 覆盖 0.458、反馈驱动的 linucb 在 recovery 上 −47.8 pp。因此诊断改用策略无关量：`expert_SR(twin)` vs `expert_SR(deployment)`（twin 审计）、`expert_SR − policy_SR`（知识与数据之别）、仅扰动差距（恢复类）。详见审计 §26.5。

### 5.12 策略无关诊断：四类实测可分（2026-11-23）

**主张**：策略是在错的 twin 上优化出来的，twin 内部的自检天然循环。所以诊断信号必须与策略无关。

**探针**（`scripts/probe_policy_free_signals.py`，不训练、不调 LLM）：

| 量 | 含义 |
|---|---|
| A2 | 专家用**真值**规划、在真值里执行（任务可解吗） |
| A | 专家用 **twin** 规划、在**真值**里执行（这个 twin 能让最优者活下来吗） |
| B | 专家用 twin 规划、在 twin 里执行（自洽，恒 1） |
| C | 策略在真值里执行 |
| D | 策略在 twin 里执行（**自检**） |

`twin_cost = A2 − A`、`knowledge_gap = A − C`、`selfcheck_bias = D − C`。

| 原因 | A2 | A | B | C | D | twin_cost | knowledge_gap | selfcheck_bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| geometry | 1.00 | **0.00** | 1.00 | 0.30 | **0.80** | **+1.00** | −0.30 | **+0.50** |
| coverage | 1.00 | 1.00 | 1.00 | 0.10 | 0.10 | +0.00 | **+0.90** | +0.00 |
| recovery | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | +0.00 | +0.00 | +0.00 |
| healthy | 1.00 | 1.00 | 1.00 | 0.70 | 0.70 | +0.00 | +0.30 | +0.00 |

（**3 banks × 10 episodes**，nominal；Wilson 95% 区间示例：geometry 的 A = 0.00 [0.00,0.28]、D = 0.80 [0.49,0.94]；coverage 的 C = 0.10 [0.02,0.40]。三个 bank 上数值逐位相同，见 `results/policy_free_signals_paper_20261123/signals.md`。）

扰动下标：`perturbation_drop = policy_SR(nominal) − policy_SR(perturbed)`：**recovery 场景 1.00**（1.00 → 0.00）、healthy 0.30（0.70 → 0.40）、coverage −0.10、geometry 0.30。

四条读数：

1. **twin 错被干净识别**：几何上 A2=1.00、A=0.00——**最优执行者在错的 twin 上一件事也做不成**；其余三类 twin_cost=0。
2. **自检偏差只在 twin 错时出现**：几何 D=0.80 vs C=0.30（**+0.50**），其余三类 **0.00**。这就是"twin 内成功率是反的"的直接测量。
3. **恢复类只能靠扰动**：nominal 下 C=1.00，perturbed 下 0.00，而专家恒 1.00 → drop 1.00（healthy 只有 0.30）。
4. **几何的 knowledge_gap 是负的（−0.30）**：策略比"用错 twin 的专家"更抗偏差 → 用策略自身成功率做诊断，会把 twin 错误掩盖成"策略还行"。

**判定规则**（阈值取在实测分离的中点，bank 0 定、banks 1–2 校验；三个 bank 数值相同，规则稳定）：

```
if twin_cost > 0.3:              calibrate_geometry      # 实测 1.00 vs 0.00
elif perturbation_drop > 0.5:    add_recovery_data       # 实测 1.00 vs 0.30
elif knowledge_gap > 0.5:        regenerate_targeted     # 实测 0.90
elif knowledge_gap <= 0.35:      stop                    # 实测 0.30
```

实现：`rsi/policy_free.py`（`diagnose` + `PolicyFreeSelector`，无探针则硬失败）；每轮重算探针 → 混合原因自动"先修主因再诊断"（`harness/policy_free_probe.py`）。4 个单测把上表直接编码成断言。

## 6. 数据与可复现性

为控制仓库体积，**只保留元数据、逐次 `record.json`、`effective_config.json`、图表和小型 `summary.json`**；策略权重 `.npz`、任务 payload（`tasks_*`）与原始大规模结果目录未纳入。它们由固定种子确定性生成，重跑同一配置即可复现。完整清单与再生成命令见 [`horizon/results/README.md`](horizon/results/README.md)。

## 7. 引用与致谢

本仓库是研究原型，尚未附论文。第三方基线代码（如 LiLo-VLA、openpi、mimicgen）未复制进本仓库，只保留引用与笔记。

## 8. 与导师完整方案的完成度（2026-11-21）

**结论：外部阻塞只有 3 块；但方案还有约 10 项内部缺口，不能说“只差真机/LIBERO/LLM”。**

### 8.1 外部阻塞（这台机器做不了的 3 块）

| 块 | 现状 | 缺什么 |
|---|---|---|
| LLM agent / LLM+经验 | **已跑通真实调用并进主表**（8 格，23 次，$0.044，平均 SR 0.931） | 论文级网格（3 seeds × 3 banks ≈ 24 格 / $0.14）；LLM+经验变体 |
| 真机 | 0 | 机器人、相机、AprilTag/位姿链路 |
| LIBERO / 外部 benchmark | 0 | checkout、Diffusion Policy 管线、磁盘 |

### 8.2 内部未完成（能在这台机器上做，不能算外部阻塞）

| # | 缺口 | 现状 | 补法 | 机时 |
|---|---|---|---|---|
| 1 | 最小验证规模 | ✅ **已完成 240/240**（10 banks × 3 seeds × 4 selectors） | — | 0 |
| 2 | dev/test 划分 | ✅ **已完成**（banks 0–2 dev / 3–9 held-out；dev 调参无改善 → 保留先验阈值） | 正式 30/60 划分（换 bank 采样） | 0 + 重跑 |
| 3 | 实验一完整矩阵 | **pack/H4 × 8 pipelines 已跑完（48 格）**；sort/stack 与 H2–10 未跑 | 3 families × H2–10 × 8 pipelines × 3 seeds | 数十 h |
| 4 | 8 条固定流程 baseline | **✅ 已实现并跑完**（`open_loop`/`always_dagger`/`domain_randomize`/`deploy_dagger`/`mixed_train`/`fixed`/`calibrate_alternate`/`true_params`，48 格） | 扩到 3–5 banks | ~3 h |
| 5 | 图 R2 + 命题 | ✅ **图 R2 已出**；命题文字未写 | 命题用 ε/Δ/D/κ（纯写作） | 0 |
| 6 | 混合原因并入主表 | ✅ **已并入表 2**（`results/table2_full.md`：混合列 gap recovery bandit 86% / rule 36%） | 混合网格扩到 3 banks | ~2 h |
| 7 | 实验五消融 | 🟡 **表 11/图 2 已出**、表 8 有、表 10 在跑（32 格）；**图 7 预算扫描在跑**（fb8/fb48） | 表 9 三档偏差（当前协议） | 数 h |
| 8 | 判定器偏差原因 | `recalibrate_verifier` 未实现 | 实现重校准动作 | 1 天 |
| 9 | 位置偏差 | `lateral_offset` 可注入，但 `geometry_repair` 只估高度 | 实现位置估计器，否则只能当“未知原因” | 1–2 天 |
| 10 | 动力学原因 | 摩擦/质量/延迟扫描证明不可表达 | 写 limitation（已有 4 组扫描） | 0 |
| 11 | 主张口径 | learned agent SR 0.867 < rule 0.914 | 把“选择效应”与“learned 诊断”分开写 | 写作 |

**实验一的三类偏差在本平台上并不等价**：
- 高度偏差：可注入、可修复（`geometry_repair`），是唯一能完整构成“修 twin”实验的原因；
- 位置偏差：`lateral_offset` 可注入，但没有位置估计器 → 所有“校准 twin”的方法都修不了，只能当负结果/未知原因；
- 摩擦偏差：可注入，但 1.0→0.01 扫描 SR 不变（力闭合夹持），不是可分级原因。

所以导师表 1 的三列里，真正能支撑“要不要更新 twin”的只有高度列；位置/摩擦要么补估计器，要么写成 limitation。

### 8.3 已完成、可以作为论文主结果的

- **288-run 8 选择器主表 + 95% CI**（§5.7、`论文批次结果_2026-11-21.md`）；
- **实验一 8 条固定流程 × 3 条件（48 格）**：没有一条配方通吃——高度偏差方法 8 (0.950) > 方法 7 (0.800) > 方法 6 (0.775) > 方法 2 (0.425)；覆盖不足方法 2 ≈ 方法 8 (0.975) > 方法 4 (0.875) > 方法 7 (0.225)；健康方法 3 (1.000) 最高。详见 `results/experiment1_table1_20261122/table1_report.md`；
- **LLM 进主表**：8 格，平均 SR **0.931**（仅次于 bandit 0.944），恢复列 **1.000** 全场最高，23 次调用 / $0.044（`results/llm_in_main_table.md`）；
- **图 R1 配方 × 原因**：没有固定配方通吃（`results/fig_R1_recipe_cause_20261121/`）；
- **多标签诊断**：184 profiles，exact 82%，mixed 任一 100% / 全中 72%；
- **混合网格 48 runs**：bandit gap +100/+17/+94/+100 vs rule +27/−33/+88/0；
- **负结果**：动力学不可表达、恢复信号死、value model R²=0.138、oracle 是诊断上界不是 SR 上界；
- rule 的 SR 下限、recovery 探针、held-out 无泄漏修正。

## 9. 外部三条线 handoff（LLM / 真机 / LIBERO）

**状态：代码侧已闭环；外部只剩 key（LLM）、硬件驱动/位姿源（真机）、LIBERO 包/数据/DP 权重。逐项证据见 [`DEPLOYMENT_AUDIT.md`](DEPLOYMENT_AUDIT.md)。**

### 9.1 权重（`weights/`，16 MB）

| 文件 | 内容 |
|---|---|
| `weights/mujoco_4causes/{geometry,coverage,recovery,healthy}_pi0.npz` | 初始策略 π0 |
| `weights/mujoco_4causes/{scenario}_fixed_final.npz` | fixed 配方第 3 轮（校准 + DAgger） |
| `weights/models/multilabel_model_heldout_paper.json` | 多标签诊断（阈值 0.3，训练排除 banks 0–2） |
| `weights/models/value_model.json` | 40 条反事实的负结果基线 |
| `weights/SHA256SUMS.txt` / `weights/README.md` | 校验和 + 加载说明 |

注意：这些是**状态策略**（输入 `TaskFeaturizer.hier` 特征），不是图像策略；LIBERO 权重不随仓库分发。

### 9.2 三条线

| 线 | 已就绪 | 只差 |
|---|---|---|
| **LLM** | `LLMPlannerSelector`（token/成本记账、无 key 硬失败、`client=` 注入、retry/backoff、`max_calls/max_usd`、`memory`）、`scripts/llm_pilot.py`（`--mock` / `--probe` / `--per-cause`）、`scripts/report_llm_pilot.py`、`scripts/report_llm_grid.py`、7 tests；**真实端点已验证**：`openai/gpt-6.2-astra` @ `https://minara.cloud/v1`；**LLM 已进主表**（`results/four_causes_llm_20261122`，8 格，23 次调用 / $0.044，平均 SR **0.931**，恢复列 **1.000**） | 论文级网格（3 seeds × 3 banks ≈ 24 格 / $0.14） |
| **真机** | `horizon/real/` 接口/mock/JSONL/日志/校准；**task payload → TaskInstance → HierarchicalController → TaskFeaturizer → NumpyMLP 端到端**（dim=55，7 steps）；**Verifier 成功判定 + EpisodeLog 构建**；示例 payload + 状态权重 | 机器人驱动、AprilTag/位姿源 + 外参、夹爪映射、安全限位；EpisodeLog phase/valid 标注 |
| **LIBERO** | `horizon/libero/` HDF5 读取、布局先验、`MockLiberoEnv`、**`LiberoEnvAdapter`**、**`DPAdapter`**、`ImageBC`、`train_libero.py`/`eval_libero.py --bddl`、6 tests | **libero 包/数据 + DP 权重 + 新布局演示生成** |

### 9.3 最短路径

```bash
git clone https://github.com/Clare-1/real2sim-horizon.git
cd real2sim-horizon/horizon
# LLM（凭据放在仓库外：~/.config/horizon/llm.env，chmod 600）
PYTHONPATH=src python scripts/llm_pilot.py --provider openai --env-file ~/.config/horizon/llm.env \
  --probe                                  # 1 次调用验证端点/模型/key
PYTHONPATH=src python scripts/llm_pilot.py --provider openai --env-file ~/.config/horizon/llm.env \
  --results results/four_causes_paper8_20261120 --per-cause 3 --max-usd 0.5
PYTHONPATH=src python scripts/report_llm_pilot.py     # LLM vs 脚本选择器同表
# LLM 进主表（4 原因 × 2 banks = 8 格，约 23 次调用 / $0.044）
PYTHONPATH=src python scripts/run_minimal_validation.py \
  --config configs/experiment_llm_grid.yaml --out results/four_causes_llm_20261122
PYTHONPATH=src python scripts/report_llm_grid.py       # → results/llm_in_main_table.md
# 真机（先 mock 自测）
PYTHONPATH=src python scripts/run_real_deploy.py --robot mock --out /tmp/deploy.jsonl
# LIBERO
PYTHONPATH=src python scripts/train_libero.py --dataset /path/libero_10.hdf5 --out weights/libero/bc.pt
```

### 9.4 测试

```bash
cd horizon
PYTHONPATH=src python -m pytest -q tests/test_llm_planner.py tests/test_real_interfaces.py tests/test_libero_scaffold.py
# 12 passed, 1 skipped
```

完整说明见 [`HANDOFF_external.md`](HANDOFF_external.md)，权重清单见 [`weights/README.md`](weights/README.md)。

## 10. 论文图表覆盖（2026-11-23）

- 完整矩阵：[`PAPER_COVERAGE_AUDIT.md`](PAPER_COVERAGE_AUDIT.md)。
- **已出（当前协议）**：paper8 主表（4 causes × 8 selectors + CI）、**表 1**（实验一 8 条 pipeline × 3 条件，§5.9）、**表 2 完整版**（纯原因 + 混合列，`results/table2_full.md`）、**表 3**（AUC / oracle-90% 部署 / regret / 健康花费）、**表 11 / 图 2**（H 外推：训 H2/4 在 H≥6 = 0.000，全 H 对照 0.45–0.90，专家各 3/3，`results/fig2_sr_vs_h_20261123/`）、**图 1**（SR–成本）、**图 3**（原因 × 首步）、图 4、R1、R2。
- **进行中**：**图 7**（部署预算 8/24/48）——`results/budget_sweep_20261123/fb8`、`fb48` 在跑，24/round 取 paper8 同 bank；出图 `make_fig7_budget.py --fb8 … --fb24 … --fb48 …`。
- **LLM 行已可用**：8 格平均 SR 0.931、恢复列 1.000、$0.044（§9.2、`results/llm_in_main_table.md`）；首步 12/20 vs rule 6/8（`results/llm_vs_rule_20261122.md`）。
- **旧协议 / 部分**：表 8/9/10。
- **未做（可跑）**：实验一扩到 3–5 banks、sort/stack family、位置列（无目标位姿观测）、`recalibrate_verifier`。
- **外部**：LIBERO（表 4/图 5）、真机（表 5/6/7/图 6）。
- **可不跑**：最小验证（导师明确不进论文；stage-1 PASS + stage-2 240/240 仅留档）、动力学原因（平台不可表达）、RoboDojo/RoboCasa。
