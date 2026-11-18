# 论文方案：Breaking the Horizon

> **原始研究方案，2026-10-26 路线说明。** 保留本文“部署反馈→孪生体/数据→策略更新”的 agentic learning 主框架，具体实施以 [Agentic Learning Harness 路线](Agentic_Learning_Harness_路线方案.md) 为准。H2 中“归因分配数据优于标准 DAgger”的机制未获支持，相关负结果继续保留；v5–v7 作为修复与协议工程记录。当前性能证据仍限于同一 MuJoCo 代理、pick_place 多阶段任务；本文真机数字占位符、扩展技能和强主假设不是已完成结果。

**Working title：**

> **Breaking the Horizon: Subgoal-Attributed Agentic Real2Sim2Real for Long-Horizon Manipulation**

备选标题：

- From Data Generation to Learning: A Self-Improving Agentic Real2Sim2Real System for Long-Horizon Manipulation
- Agentic Learning: Closing the Loop Between Policy Failure and Synthetic Data

---

## 0. Abstract（草稿，数字留空）

Long-horizon manipulation remains bottlenecked by the cost of generating demonstrations that cover compositional transitions and recovery behaviors. We present **X**, an agentic real2sim2real system that compiles a single RGB image and a natural-language instruction into a subgoal graph, generates feasible compositional demonstrations, monitors policy execution at subgoal resolution, attributes failures to their source, and regenerates targeted data. Across **N** long-horizon tasks (**H** stages) and **S** real scenes, X improves full-task real-world success from **A%** to **B%** over **K** self-improvement rounds, while using only **R** real-world trajectories per task. We show that open-loop data generation degrades exponentially with horizon length, whereas subgoal-attributed closed-loop generation flattens this curve. We further show that simulation-based evaluation ranks methods consistently with real-world evaluation (**Spearman ρ = **), enabling scalable iteration without proportional real-robot time.

---

## 1. Problem Statement

### 1.1 现实问题

- 长程操作（5–10 阶段）需要覆盖：子目标转移、物体状态变化、失败恢复、长时序信用分配。
- 真机采集长程数据的成本随阶段数近似线性甚至超线性增长。
- 仿真可以规模化，但现有 agentic real2sim2real 系统是**开环**的：数据在策略训练前生成完，策略失败不会改变数据分布。

### 1.2 研究缺口

现有工作分别解决了：

- **ARSTAG**：单图 + 指令 → 任务级场景与可行演示（短程、开环）。
- **EmbodiRSI**：策略失败 → 定向数据 → 更新（单场景、短程）。
- **Agentic Real2Sim**：真实 episode → 可仿真 twin（一次性转换、无任务合成）。

但没有工作解决：

> **在长程、组合式任务上，如何把“策略失败”转化为“下一轮该生成什么数据”的信用分配问题。**

### 1.3 核心问题

> **Can robot learning itself be an agentic system?**

即：系统能否观察策略在长程任务中的失败，定位到具体子目标与失败类型，并自主决定下一步的数据生成/场景修复/课程调整动作，从而在有限真机预算下持续提升？

---

## 2. Hypotheses

| 编号 | 假设 | 可证伪方式 |
|---|---|---|
| **H1** | 开环数据生成的成功率随阶段数指数衰减：\(S(H)=S_0 e^{-\alpha H}\) | 拟合 H=2→10 的开环曲线，检验 \(\alpha>0\) 且拟合优度高 |
| **H2** | 子目标级归因 + 定向再生成能显著降低衰减率 \(\alpha\) | 匹配数据预算下比较闭环 vs 开环的 \(\alpha\) |
| **H3** | 恢复分支数据主要提升“阶段转移”和“扰动后恢复”，而非名义执行 | 分层分析：名义 vs 扰动；转移 vs 阶段内 |
| **H4** | 经过校准的仿真评估与真机评估排序一致（ρ>0.7） | 8–10 个方法 × 10 任务，Spearman 相关 |
| **H5** | 单位任务成本随累计任务数下降（资产/技能/课程复用） | 记录 C_task vs N_task，检验下降趋势 |

**主假设**：H2。论文的主图由 H2 支撑。

---

## 3. Related Work & Positioning

| 工作 | 单位 | 闭环 | 时间跨度 | 失败信号 | 核心主张 | 与我们的差异 |
|---|---|---|---|---|---|---|
| ARSTAG | task | 开环 | 短 | 数据生成失败 | 数据生成可以 agentic | 我们闭环到策略失败，且做长程 |
| EmbodiRSI | deployment | 闭环 | 短 | 策略失败 | RSI 提升数据效率 | 我们做任务族 + 子目标级归因 |
| Agentic Real2Sim | episode | 开环 | 短 | 转换失败 | episode twin | 我们做任务合成 + 长程组合 |
| RoboGen / MimicGen / RoboTwin 2.0 | task | 开环 | 短 | 无 | 规模化数据生成 | 我们强调失败驱动与信用分配 |
| SenseNova-RoboRSI | harness | 半闭环 | 短 | 推理失败 | harness 级 RSI | 我们做权重级 + 环境级 RSI |
| Astra / π0.5 | — | — | — | — | 强空间/执行先验 | 我们把它当组件，不做基座竞争 |

**定位一句话：**

> ARSTAG 让 agent 学会“造数据”；我们让 agent 学会“判断该造什么数据”，并在长程任务上验证。

---

## 4. Contributions

1. **Task Graph Compiler**：从单图 + 指令编译出带前置条件、效果、成功谓词的子目标 DAG，并做可行性检查与自动修复。
2. **Compositional Data Generator**：子目标级专家演示 → 轨迹拼接 → 全轨迹验证 → 任务一致随机化 → 子目标边界恢复分支。
3. **Subgoal-Attributed RSI Controller**：执行监控 → 子目标级失败归因 → 选择改进动作（再生成/加恢复/调课程/修场景/校准验证器）→ 策略更新。
4. **Scaling Evidence**：给出成功率随阶段数、数据量、RSI 轮数的曲线，证明开环指数衰减、闭环拉平。
5. **Open Harness**：任务族、谓词、配置、评测协议、视频，形成可复现的 agentic learning benchmark。

---

## 5. Method

### 5.1 Task Graph Compiler

**输入**：单张 RGB（可选多视角）、自然语言指令、机器人模型、标定参数。

**步骤：**

1. **Task grounding**：VLM 联合图像与指令，构建任务相关场景图 \(G=(V,E)\)；只保留任务相关物体与支撑面。
2. **Subgoal decomposition**：LLM 将指令分解为有序子目标 \(g_1,\dots,g_H\)，每个子目标绑定技能模板与参数。
3. **Typed graph**：每个节点包含：
   - skill type（pick/place/push/open/close/insert/pour/wipe）
   - object arguments
   - preconditions（如 `grasped(obj)`, `clear(region)`）
   - effects（如 `in(obj, box)`, `closed(lid)`）
   - success predicate（可执行）
4. **Feasibility check**：IK 可达、碰撞、抓取成功率、放置稳定性、双臂协调。
5. **Repair loop**：不可行 → 调整参数/顺序/支撑面/技能；最多 N 次重试；保留失败原因。
6. **Output**：`TaskGraph` JSON，含每个子目标的谓词与验证方式。

**关键设计**：任务图不是自由文本计划，而是**可执行、可验证、可修复**的类型化结构。

### 5.2 Skill Library

原子技能（先刚体桌面）：

| 技能 | 参数 | 成功谓词 | 恢复原语 |
|---|---|---|---|
| pick | object, grasp pose | `grasped(obj)` | re-grasp, push-to-edge |
| place | object, target region | `on(obj, support)` | re-place, re-approach |
| push | object, direction | `moved(obj, Δ)` | re-push |
| open | articulated object | `open(obj, θ)` | pull, wiggle |
| close | articulated object | `closed(obj)` | press, re-align |
| insert | part, receptacle | `inserted(part, base)` | re-align, rotate |
| pour | container, target | `poured(obj, target)` | re-orient |
| wipe | tool, surface | `clean(surface)` | re-cover |

每个技能包含：

- 状态专家（state-based expert，仿真特权状态）
- 参数化接口（供任务图调用）
- 成功谓词
- 恢复原语（供数据生成与执行监控使用）

**复用逻辑**：技能库是跨任务摊销的核心；新任务优先组合已有技能，缺技能才扩展。

### 5.3 Compositional Data Generator

**目标**：生成覆盖“名义执行 + 阶段转移 + 扰动恢复”的长程演示。

流程：

1. **Per-subgoal demos**：对每个子目标，在采样的初始状态下调用状态专家，生成成功轨迹。
2. **Stitching**：在子目标结束状态拼接下一子目标；检查物体状态、夹爪状态、碰撞、时序。
3. **Full-trajectory validation**：整条轨迹在仿真中重放，验证所有谓词；失败则回退到最近可行子目标。
4. **Task-consistent randomization**：物体位姿、尺寸、颜色、遮挡、干扰物、光照、相机噪声；保持任务依赖关系（如“先放面包再放火腿”）。
5. **Recovery branches**：在每个子目标边界注入扰动（滑落、抓空、遮挡、错物体、碰撞），从实际到达状态生成恢复演示。
6. **Curation**：保留成功轨迹；记录失败类型；去重；写入 provenance。

**数据 schema（每条 episode）：**

```json
{
  "episode_id": "...",
  "task_id": "...",
  "subgoal_id": 3,
  "scene_id": "...",
  "graph_version": "...",
  "skill_version": "...",
  "sim_params_hash": "...",
  "policy_version": "...",
  "generator_version": "...",
  "success": true,
  "failure_type": null,
  "attribution": null,
  "frames": "...",
  "proprio": "...",
  "actions": "...",
  "randomization": {...},
  "provenance": "expert|policy|recovery|human"
}
```

### 5.4 Execution Monitor & Failure Attribution

**监控信号：**

- 子目标谓词是否满足
- 阶段超时
- 物体状态与期望的偏差
- 碰撞/力异常
- 夹爪状态
- 视觉变化（物体丢失、遮挡）

**失败分类：**

| 类型 | 例子 | 归属 |
|---|---|---|
| 场景/物理 | 物体悬空、支撑错、摩擦异常 | 重建/物理校准 |
| 抓取 | 抓空、滑落、夹持不稳 | 技能/数据 |
| 运动 | 不可达、碰撞、路径失败 | 任务图/运动规划 |
| 转移 | 上一阶段结束状态不满足下一阶段前置条件 | 组合/数据 |
| 指令 | 指令歧义、目标物体错 | 任务图重编译 |
| 验证器 | 假成功/假失败 | 重校准/人工审计 |
| 硬件 | 延迟、标定漂移 | 系统辨识 |

**归因输出**：`(subgoal_index, failure_type, cause, confidence)`。

**归因实现（由简到繁）：**

1. 规则 + 谓词 + 特权状态（MVP）
2. VLM 看关键帧与状态日志（补充语义）
3. 人工标注 100 条失败作为校准集
4. 训练一个轻量归因分类器（后期）

**关键指标**：归因准确率 vs 人工标签；混淆矩阵。

### 5.5 Subgoal-Attributed RSI Controller

**状态**：

\[
S_k=(\pi_k,\ \text{Sim}_k,\ \text{TaskGraph}_k,\ D_k,\ F_k,\ V_k)
\]

- \(F_k\)：子目标级失败分布
- \(V_k\)：验证器

**动作空间**：

| 动作 | 触发条件 |
|---|---|
| 再生成该子目标数据 | 技能/抓取失败集中 |
| 增加恢复分支 | 阶段转移失败 |
| 调整课程（增加阶段/难度） | 子目标成功率高 |
| 修复场景/物理 | 场景/物理失败 |
| 重编译任务图 | 指令/可行性失败 |
| 真机校准 | sim-real 偏差大 |
| 重校准验证器 | 假成功/假失败 |
| 更新策略 | 数据足够 |

**选择策略**：

- MVP：规则 + 优先级（按失败频率 × 修复成本）
- V1：bandit / 轻量 RL，学习“哪个动作最降低失败率/成本”
- 目标函数：

\[
\max \mathbb{E}[\text{real success}] - \lambda \cdot \text{real cost} - \mu \cdot \text{compute}
\]

**停止条件**：

- 连续 K 轮无显著提升
- 真机预算耗尽
- 验证器置信度低
- 失败类型收敛

### 5.6 Verifier & Calibration

**仿真侧**：状态谓词 + 特权状态 + 碰撞/接触检查。

**真机侧**：VLM judge + 关键帧 + 人工抽检（10%）。

**校准**：

- 用真机标签调整谓词阈值
- 报告 sim-real 成功率排序相关性
- 校准前/后对比

**这是论文可信度的关键**：没有校准，仿真提升不可信。

### 5.7 Policy

**架构 A（推荐）**：分层

- **System 2**：VLM/LLM 读取任务图与执行状态，选择下一子目标，触发恢复/重规划。
- **System 1**：π0.5 类 VLA，子目标条件化，输出动作 chunk。

**架构 B（对照）**：扁平 VLA，整条指令直接条件化。

**训练**：

1. BC on 成功演示
2. DAgger on 失败状态 + 纠错轨迹
3. 可选：仿真 RL（PPO/SAC）用稠密子目标 reward
4. 真机少量残差微调（10–20 条/任务）

**动作空间**：末端 delta pose + gripper；10–20 Hz；head + wrist cameras。

### 5.8 系统图

```text
Image + Instruction
        │
        ▼
Task Graph Compiler ──► TaskGraph (DAG + predicates)
        │
        ▼
Skill Library ──► State-based experts + recovery primitives
        │
        ▼
Compositional Data Generator ──► demos + recovery branches
        │
        ▼
Policy (System 2 subgoal selector + System 1 VLA)
        │
        ▼
Execution Monitor ──► subgoal-level failure attribution
        │
        ▼
RSI Controller ──► regenerate / repair / curriculum / calibrate / update
        │
        └──────────────► next round
```

---

## 6. Experimental Protocol

### 6.1 Research Questions

| RQ | 问题 | 对应假设 |
|---|---|---|
| RQ1 | 子目标归因 + 定向再生成是否比开环/均匀再生成更数据高效？ | H2 |
| RQ2 | 组合式数据 + 恢复分支是否提升长程全任务成功率？ | H3 |
| RQ3 | 成功率如何随阶段数、数据量、RSI 轮数缩放？ | H1/H2 |
| RQ4 | 仿真评估能否作为真机评估的代理？校准是否提升相关性？ | H4 |
| RQ5 | 哪些失败类型最常见，哪类修复算子性价比最高？ | — |
| RQ6 | 是否泛化到未见物体/指令/场景？是否遗忘旧任务？ | — |

### 6.2 Task Suite

先做 5 个任务族，每个 5–10 阶段；规模化后扩展到 10 个。

| 任务 | 阶段数 | 子目标示例 | 关键谓词 | 扰动 |
|---|---|---|---|---|
| 做三明治 | 6 | 面包→火腿→芝士→面包→对齐→装盘 | `on`, `stacked`, `aligned` | 滑落、顺序错、遮挡 |
| 装箱 | 6 | 物品1→物品2→物品3→合盖→扣锁→推到区 | `in`, `closed`, `at` | 掉落、盖子卡住、干扰物 |
| 装配玩具 | 6 | 底座→齿轮→轴→盖→拧螺丝→直立 | `inserted`, `fastened`, `upright` | 错位、掉落、错零件 |
| 分类回收 | 6–8 | 瓶→桶A，罐→桶B，纸→桶C，重复 | `in`, `class_match` | 投错、溢出、遮挡 |
| 摆盘 | 6–8 | 盘→杯→餐具→餐巾→调整 | `on`, `upright`, `spacing` | 倾倒、滑动、干扰物 |
| 备餐托盘 | 6 | 杯→点心→餐巾→摆放→端起 | `on`, `stable`, `arranged` | 掉落、碰撞、遮挡 |

**任务选择原则：**

- 阶段之间有明确状态依赖
- 每阶段有可执行谓词
- 包含至少 2 个“易失败”转移
- 可用现有刚体技能库覆盖

### 6.3 Scenes, Objects, Robot

- **场景**：3 个（厨房台面、打包台、装配台）
- **物体**：30+ 实例，8 类，3 尺寸，5 颜色，含干扰物
- **机器人**：双臂桌面平台（延续 ARSTAG 设置），2 相机（head + wrist），10–20 Hz
- **标定**：手眼标定、相机内参、桌面平面
- **安全**：速度限制、工作空间边界、急停

### 6.4 Baselines

| 基线 | 说明 | 数据预算 |
|---|---|---|
| ARSTAG open-loop | 同生成器，无 RSI | 匹配 |
| Flat BC | 无任务图，整条演示 | 匹配 |
| Real-only BC | 纯真机演示 | 50/100/200 per task |
| π0.5 zero-shot | 不微调 | 0 |
| π0.5 real fine-tune | 真机微调 | 50/100/200 |
| EmbodiRSI | 单任务 RSI | 匹配 |
| Agentic Real2Sim + BC | episode twin 数据 | 匹配 |
| Uniform regeneration | 无归因，均匀再生成 | 匹配 |
| Oracle attribution | 人工标注失败 | 上界 |
| Ours (full) | 完整系统 | 匹配 |

**关键对照**：`Uniform regeneration` 与 `Oracle attribution`。前者证明“不是多生成数据就行”，后者给出归因的上界。

### 6.5 Metrics

**Primary:**

- **Full-task real success**：所有子目标按顺序满足，且在时间限制内完成。

**Secondary:**

| 指标 | 定义 |
|---|---|
| Subgoal success | 单阶段谓词满足率 |
| Recovery success | 注入扰动后完成当前及后续阶段的比例 |
| Data efficiency | 达到 80% 渐近成功率所需真机轨迹数 |
| Sim-real ρ | 方法×任务的成功率 Spearman 相关 |
| Human interventions | 遥操作接管 + 手动 reset + prompt 修复次数 |
| Cost per task | GPU + VLM + 真机时间 + 人工时间 |
| Generalization | 未见物体/指令/场景的成功率 |
| Retention | 学新任务后旧任务成功率下降 |
| Attribution accuracy | 自动归因 vs 人工标签 |

### 6.6 Matched-Budget Protocol

所有方法在相同预算下比较：

- **仿真演示**：200 / 400 / 800 条每任务
- **真机轨迹**：10 / 20 / 50 / 200 条每任务
- **训练步数**：相同
- **策略架构**：相同
- **评测 trials**：每任务每方法 20–30 次
- **随机种子**：训练 3 seeds；评测固定初始分布

同时报告 **data scaling curves**，避免“只是数据多”。

### 6.7 Training Protocol

| 项目 | 设置 |
|---|---|
| Policy | π0.5 类 VLA；LoRA / 全量微调 |
| Action chunk | 50 |
| Batch size | 64 |
| LR | 1e-4（LoRA 可 2e-4） |
| BC steps | 50k |
| DAgger steps | 20k |
| RL（可选） | PPO/SAC，10k |
| Cameras | head + wrist |
| Proprio | joint + gripper |
| Seeds | 3 |

**训练阶段：**

1. 纯 BC（仿真数据）
2. BC + DAgger（失败状态纠错）
3. 可选 RL 微调（仿真稠密 reward）
4. 真机残差微调（10–20 条/任务）

### 6.8 Real-Robot Evaluation Protocol

- **Held-out**：场景布局、光照、物体实例、干扰物均与训练不同。
- **Unseen instructions**：同义改写、新物体组合。
- **Trials**：每任务每方法 20–30 次；顺序随机化。
- **Blind evaluator**：评测者不知道方法。
- **Perturbation suite**：物体位移、遮挡、干扰物、滑落。
- **Recording**：全程录像；失败帧标注。
- **Pre-registration**：主指标、样本量、成功判定在实验前固定。

### 6.9 Ablations

| 消融 | 目的 |
|---|---|
| No task graph（flat） | 任务图是否必要 |
| No feasibility check | 可行性筛选的贡献 |
| No composition（per-stage independent） | 组合式生成的贡献 |
| No attribution（uniform） | 信用分配是否关键 |
| No recovery branches | 恢复数据的贡献 |
| No RSI（open-loop） | 闭环的贡献 |
| No verifier calibration | 校准的贡献 |
| Randomization level（low/high） | 多样性与保真的权衡 |
| Hierarchical vs flat policy | 系统2/系统1组织方式 |
| Model routing（large vs small VLM） | 成本与效果的权衡 |

### 6.10 Scaling Experiments

**三个自变量：**

- 阶段数 \(H \in \{2,4,6,8,10\}\)
- 每子目标演示数 \(N \in \{50,100,200,400\}\)
- RSI 轮数 \(K \in \{0,1,2,3\}\)

**分析：**

- 对开环拟合 \(S(H)=S_0 e^{-\alpha H}\)，报告 \(\alpha\) 与 \(R^2\)。
- 对闭环拟合同样形式，比较 \(\alpha\)。
- 报告 \(S\) vs 累计数据量、\(S\) vs 成本。
- 报告 \(C_{task}\) vs 累计任务数（摊销曲线）。

**主图**：成功率 vs 阶段数，开环 vs 闭环。

### 6.11 Sim-Real Correlation

- 选 8–10 个方法（含强弱基线）。
- 每个方法在 10 个任务上做仿真与真机评测。
- 计算 Spearman \(\rho\)。
- 校准前 vs 校准后对比。
- 若 \(\rho<0.5\)，论文主张改为“仿真不能作为代理，必须真机在环”。

### 6.12 Failure Analysis

- 随机抽取 100 条失败，人工标注真实失败类型与子目标。
- 混淆矩阵：自动归因 vs 人工。
- 每种失败类型对应的修复算子成功率。
- 报告“归因准确率”和“修复命中率”。

### 6.13 Statistical Analysis

- 成功率：Wilson 置信区间。
- 方法差异：bootstrap 置信区间 / 配对检验。
- 跨任务：混合效应模型（task/scene 为随机效应）。
- 多重比较：Bonferroni 或 Holm 校正。
- 报告 effect size，不只报 p 值。

### 6.14 Infra & Cost Accounting

| 模块 | 方案 |
|---|---|
| 仿真 | MuJoCo（接触）+ Isaac Lab（并行/渲染），二选一为主 |
| 编排 | Ray / 容器化 job queue |
| 数据 | RLDS / LeRobot；metadata DB；S3/MinIO |
| 训练 | 8–32 GPU；异步 actor-learner |
| 真机 | 1–2 台；自动 reset；安全包络；评测调度 |
| 日志 | episode / subgoal / failure / cost 四张表 |

**成本表：**

- 每 1k rollout 的 GPU-hours
- 每任务 VLM 调用次数与费用
- 每任务真机小时
- 每任务人工介入分钟
- 总 $/task

---

## 7. Expected Results & Figures

| 图 | 内容 | 作用 |
|---|---|---|
| Fig.1 | 系统总览 | 让读者 30 秒理解 |
| Fig.2 | 任务族与阶段示例 | 展示长程复杂度 |
| Fig.3 | 成功率 vs 阶段数（开环 vs 闭环） | **主图** |
| Fig.4 | 数据效率曲线（真机轨迹数） | 证明省数据 |
| Fig.5 | RSI 每轮边际增益 | 证明递归有效 |
| Fig.6 | Sim-real 相关性散点 | 证明仿真可信 |
| Fig.7 | 归因混淆矩阵 | 证明信用分配 |
| Fig.8 | 成本摊销曲线 | 证明可 scaling |
| Fig.9 | 消融柱状图 | 证明每个模块必要 |
| Fig.10 | 定性恢复示例 | 视频/帧序列 |

**期望结论（待验证）：**

- 开环在 H>4 后显著衰减；闭环把衰减率降低 X%。
- 匹配预算下，归因再生成比均匀再生成高 X 个点。
- 真机数据从 200 条/任务降到 10–20 条/任务。
- 校准后 sim-real ρ 从 <0.5 提升到 >0.7。

---

## 8. Threats to Validity

| 威胁 | 说明 | 缓解 |
|---|---|---|
| 任务族偏差 | 只有桌面刚体 | 明确 scope；后续扩展形变/关节 |
| 单机器人 | 结论可能本体特定 | 至少 1 个额外本体或仿真迁移 |
| 样本量 | 真机 trials 有限 | Wilson CI；预注册；多场景 |
| 人工标注偏差 | 失败标签主观 | 双人标注；一致性系数 |
| 超参调优 | 方法间不公平 | 统一预算；共享超参搜索 |
| 数据泄漏 | 训练/评测场景重叠 | 严格 held-out；按场景/物体划分 |
| 仿真相关性样本小 | 方法×任务数量有限 | 增加方法/任务；报告置信区间 |

---

## 9. Milestones

| 阶段 | 目标 | 交付物 | Go/No-Go |
|---|---|---|---|
| **M0** | 任务族、谓词、日志、开环 baseline | 任务定义 + ARSTAG 复现 + 成本基线 | 开环曲线可复现 |
| **M1** | MVP 闭环 | 1 机器人、1 场景、3 个 5 阶段任务、2 轮 RSI | 闭环在同预算下优于开环 |
| **M2** | 任务族 + scaling | 5–10 任务、H/N/K 曲线、消融 | 主图成立；ρ>0.6 |
| **M3** | 完整真机评测 | 3 场景、盲评、统计、论文初稿 | ρ>0.7；数据效率达标 |
| **M4** | 开源与投稿 | 代码、任务、谓词、配置、视频、benchmark | 可复现 |

**No-Go 条件：**

- 闭环在同预算下不优于开环 → 转“验证器/校准”或“任务族 scaling”。
- 归因准确率 < 60% → 先解决归因，再谈 RSI。
- sim-real ρ < 0.5 → 必须真机在环，论文主张调整。

---

## 10. Risks & Mitigations

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 长程误差复利 | 高 | 高 | 恢复分支、子目标重规划、课程 |
| 归因不准 | 中 | 高 | 规则 + VLM + 人工校准；oracle 上界 |
| 数据生成成本高 | 中 | 中 | 技能复用、缓存、模型路由 |
| 仿真-真机脱节 | 中 | 高 | 物理校准、真机 held-out、验证器校准 |
| 被抢先 | 中 | 中 | 聚焦长程 + 归因 + 匹配预算 |
| 工程复杂度 | 高 | 中 | 模块化、先 MVP、先 rigid |
| 真机时间不足 | 中 | 高 | 仿真代理 + 优先采样 |

---

## 11. Open-Source Release Plan

- **代码**：任务图编译、技能库、组合生成、监控、RSI 控制器
- **任务族**：任务定义、阶段谓词、随机化配置
- **评测协议**：真机/仿真评测脚本、统计脚本
- **视频**：长程执行、恢复、失败案例
- **Leaderboard**：可选，先做静态结果表
- **License**：宽松开源，便于引用与复现

**发布策略**：论文接收前放 project page + 视频；接收后放完整代码与配置。

---

## 12. Venue Strategy

|  venue | 适配点 |
|---|---|
| **CoRL** | 系统 + 真机 + 长程，最匹配 |
| **RSS** | 系统 + 真机 + 新问题 |
| **NeurIPS / ICLR** | scaling 曲线 + 方法 generality |
| **ICRA** | 系统集成与真机 demo |
| **Workshop（先）** | 早期反馈，抢时间 |

**建议**：主攻 CoRL/RSS；若 scaling 证据强，转 NeurIPS。

---

## Appendix A：谓词示例

```python
def on(obj, support, tol=0.02):
    return abs(obj.z - (support.z + support.height)) < tol \
       and xy_inside(obj, support)

def in(obj, container):
    return point_in_volume(obj.center, container.interior) \
       and obj.stable

def grasped(obj):
    return gripper.contact(obj) and gripper.force > threshold

def inserted(part, base, tol=0.005):
    return pose_distance(part.pose, base.target_pose) < tol

def closed(lid):
    return lid.joint_angle > 0.95 * lid.max_angle
```

---

## Appendix B：日志 schema

| 表 | 字段 |
|---|---|
| episode | id, task, scene, policy_version, sim_hash, success, cost |
| subgoal | episode_id, index, skill, predicate, success, timeout |
| failure | episode_id, subgoal_index, type, cause, confidence |
| action | round, operator, reason, cost, outcome |

---

## Appendix C：Agent tool schema

- `segment(image, prompt)`
- `reconstruct(mask)`
- `plan_grasp(object)`
- `plan_motion(target)`
- `execute(program)`
- `verify(predicate, state)`
- `attribute(episode)`

所有工具返回结构化 JSON；agent 只能通过工具改变状态。

---

## Appendix D：默认超参数

| 参数 | 值 |
|---|---|
| 每子目标演示数 | 100 |
| RSI 轮数 | 3 |
| 真机轨迹/任务 | 10–20 |
| 失败采样比例 | 70% failure states / 30% nominal |
| 随机化强度 | 中 |
| 评测 trials | 30 |
| 训练 seeds | 3 |

---

## 一句话总结（给合作者）

> 我们不是再做一条 real2sim pipeline，而是把 ARSTAG 的“agentic 数据生成”推进到“agentic 学习”：用长程任务逼出子目标级信用分配，用闭环再生成打破成功率随任务长度衰减的曲线，并用匹配预算、真机盲评、sim-real 相关性把这件事证明到 CoRL/RSS 的标准。
