# Breaking the Horizon — Subgoal-Attributed Agentic Real2Sim2Real

一个基于 MuJoCo 的 **real2sim2real 学习闭环**：用部署反馈归因失败子目标，再由选择器决定下一轮做哪种学习操作（校准孪生体 / 补数据 / DAgger / 停止）。本仓库是论文实验代码、配置、协议文档与当前结果的快照。

> 平台定位：同一 MuJoCo 引擎下的“真机代理”（部署配置代理），**不是**物理机器人，也不含跨模拟器验证。所有“真机”表述均为此代理。

---

## 1. 核心问题

给定一个与真实部署存在偏差的孪生体，系统能否**只根据部署反馈**判断该修什么，并在同一预算下比固定配方更好？

- **选择器**：`fixed` / `rule` / `oracle` / `random` / `round_robin` / `bandit`（后三个是对照，用于堵住“挑软柿子”的质疑）
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
├── v7_论文主张与验证路线.md        # 主张与验证路线
├── 汇报_2026-11-17/                # 组会材料、图、表、数据
└── horizon/                        # 代码（Python 包 + 脚本 + 配置 + 测试）
    ├── src/horizon/                # 任务图、数据生成、策略、监控归因、RSI、选择器
    ├── scripts/                    # 运行与报表脚本
    ├── configs/                    # 实验配置（YAML）
    ├── tests/                      # pytest
    └── results/                    # 结果快照（见 results/README.md）
```

`horizon/README.md` 中有更细的模块—论文方案对应表；本文件中的相对链接（`../*.md`）在仓库根目录可正常解析。

## 3. 当前状态（2026-11-18 13:00）

已完成：`results/four_causes_20261118`（rule 列已用当前代码重跑并合并）

| 项 | 值 |
|---|---|
| 网格 | 4 原因 × 2 场景实例（bank）× 6 选择器 = **48 runs** |
| 协议预算 | 初始化 80k / 每轮 40k / 8000 步训练 / 24 反馈 episode / 3 轮 |
| 进度 | **48 / 48**（`completion.json` = true） |
| rule 首步对 | **6 / 8**（几何 2/2、覆盖 2/2、健康 2/2；恢复 0/2） |

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

> 读数注意：每格只有 1–2 个样本（95% CI 约 ±6–8 pp），这里只用于方向筛查；论文主表需要 5 实例 × 3 种子（`configs/minimal_validation_4causes_paper.yaml`，216 runs）。

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

按 bank 分组留一（同一 bank 的多个 selector 看到同一 profile，旧脚本把它们当独立样本，88.9% 被放大），并补 4 类 × 20 个独立 feedback profile 后：**176 个独立 profile / 148 组**，几何 100%、覆盖 95%、健康 92%、恢复 100%，**mixed 0%**（10 个混合样本全被判成覆盖）。

→ 纯原因可分；混合原因不可靠。论文主张应是“单原因探针 + 混合压力测试 + 弃权”，而不是“分类器能分四类”。详见审计 §13。

**多标签修正**：把单标签换成 3 个二分类后，mixed 从 0% 变成“任一成分命中 100%、两成分全中 50%”（n=12，bank-grouped LOO；exact-set 89%）。论文接口应写成多标签后验，而不是单原因分类。

## 6. 数据与可复现性

为控制仓库体积，**只保留元数据、逐次 `record.json`、`effective_config.json`、图表和小型 `summary.json`**；策略权重 `.npz`、任务 payload（`tasks_*`）与原始大规模结果目录未纳入。它们由固定种子确定性生成，重跑同一配置即可复现。完整清单与再生成命令见 [`horizon/results/README.md`](horizon/results/README.md)。

## 7. 引用与致谢

本仓库是研究原型，尚未附论文。第三方基线代码（如 LiLo-VLA、openpi、mimicgen）未复制进本仓库，只保留引用与笔记。
