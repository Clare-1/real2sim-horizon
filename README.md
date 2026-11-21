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

## 3. 当前状态（2026-11-20）

- 筛查批次 `results/four_causes_20261118`：**48/48 完成**（4 causes × 2 banks × 6 selectors，1 seed）。
- 论文批次 `results/four_causes_paper8_20261120`：**288/288 完成**（4 causes × 3 banks × 3 seeds × 8 selectors，2026-11-21 03:22）；最终表见 §5.7。

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
> **导师最小验证：PASS**——rule 几何 +12.8 pp、覆盖 +32.8 pp；multilabel +5.6 / +32.8 pp；详见 §5.8 与 [`导师意见回应_2026-11-21.md`](导师意见回应_2026-11-21.md)。

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

| 选择器 | 几何 SR | 覆盖 SR | 首步对（几何/覆盖） | 标签/格 |
|---|---:|---:|---:|---:|
| fixed | 0.772 | 0.600 | 9/9, 0/9 | 80,000 |
| rule | 0.900 | 0.928 | 9/9, 9/9 | 81,111 |
| multilabel | 0.828 | 0.928 | 9/9, 9/9 | **55,556** |
| bandit | 0.961 | 0.983 | 9/9, 0/9 | 77,778 |
| oracle | 0.939 | 0.972 | 9/9, 9/9 | 74,444 |

- 过关标准（开跑前写死）：rule 或 agent 两类都不低于 fixed，且至少一类 ≥10 pp → **PASS**。
- 必须说明的偏差：
  1. 现有 3 banks × 3 seeds（导师要 10 场景 × 3 seeds）；
  2. 无 LLM key，multilabel 作为 learned 代理；
  3. 规则阈值未做 dev/test 划分；
  4. oracle 是**诊断正确性上界**，不是 SR 上界（healthy 0.844 < fixed 0.911）。
- 下一步：10 banks 最小验证（80 或 240 runs）、R2 混合比例扫描（60 runs）、dev/test 划分。

## 6. 数据与可复现性

为控制仓库体积，**只保留元数据、逐次 `record.json`、`effective_config.json`、图表和小型 `summary.json`**；策略权重 `.npz`、任务 payload（`tasks_*`）与原始大规模结果目录未纳入。它们由固定种子确定性生成，重跑同一配置即可复现。完整清单与再生成命令见 [`horizon/results/README.md`](horizon/results/README.md)。

## 7. 引用与致谢

本仓库是研究原型，尚未附论文。第三方基线代码（如 LiLo-VLA、openpi、mimicgen）未复制进本仓库，只保留引用与笔记。
