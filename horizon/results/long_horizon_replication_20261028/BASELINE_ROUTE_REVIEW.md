# 三种子复验后的路线决策备忘录

2026-10-28。本文在三种子复验运行期间依据原始规划、已完成旧实验及本地源码编写，**没有读取本批运行中的分数，没有新增实验**。下列选择条件用于完整结果通过产物核验之后；不是根据中间成绩选择胜出的路线，也不是已经证明的性能结论。

**建议顺序：先完成标准 β-DAgger 的完整长程复验；以观测通道一致性作为下一项单变量检查；有覆盖缺口证据才做阶段平衡；工具确有不同适用条件后再检验 agent 决策。** 不必把三条路线同时加入一个系统。

**1. 应保留的原始问题与应放弃的预设**

[v1 的实验轴](C:/Users/WYM/Desktop/real2sim/v1_group_meeting.md:133) 与 [v2 的主要指标及匹配预算](C:/Users/WYM/Desktop/real2sim/v2_paper_plan.md:367) 要求的是：部署反馈改变数据/孪生体，实际训练产生新策略，检验完整任务成功率随 H、数据量和轮次的变化，以及新任务的成本。任务图、归因、恢复是实现手段。原方案“子目标归因必然优于强闭环学习”的假设不能作为结论前提。

[v3](C:/Users/WYM/Desktop/real2sim/v3_实验进展.md:91) 已记录三个必须保留的事实：标准 β 混合 DAgger 是强基线；集中失败片段会挤占交接覆盖；oracle 归因并未自动带来领先。后续 60% 归因快照起点 DAgger 也未超过标准 DAgger，见 [基线记录](C:/Users/WYM/Desktop/real2sim/baselines/BASELINES.md:68)。这支持继续研究策略访问状态的学习，而不支持恢复“失败越集中采样越好”的默认设定。

原有 Markov 专家、噪声容差修正、谓词去抖、Torch 训练/NumPy rollout、来源哈希和成本账目继续保留。不要为了回到 v1 而回滚工程修正，也不要把吞吐优化作为算法成功率贡献。当前 pack/sort/stack 都以 pick_place 组合为主；全 H 训练曲线、短 H 向长 H 的分布外外推、多技能组合和真机迁移是不同问题。

**2. 哪些属于真实复用，哪些只是参考**

直接检查了 HORIZON 的依赖和导入；它未导入下面参考仓库的策略/训练包。[库存说明](C:/Users/WYM/Desktop/real2sim/baselines/BASELINES.md:1) 也明确这些 checkout 是参考材料。没有 runtime import 不能单独证明不存在任何相似代码；可以支持的准确说法是“本地独立实现／设计参考”，不能称“已经跑通官方方法”。

| 来源及本地证据 | 当前实际使用 | 不能据此声称 |
|---|---|---|
| HORIZON v1–v4 的 [strategies.py](C:/Users/WYM/Desktop/real2sim/horizon/src/horizon/rsi/strategies.py)、[model.py](C:/Users/WYM/Desktop/real2sim/horizon/src/horizon/policy/model.py)、[harness/experiment.py](C:/Users/WYM/Desktop/real2sim/horizon/src/horizon/harness/experiment.py) | 新 harness 真实调用旧专家、β-DAgger、BC 和几何估计器；这里是已有代码复用 | runner 的版本链本身是一种新学习算法 |
| ARSTAG [本地论文文本](C:/Users/WYM/Desktop/real2sim/paper_summaries/01_ARSTAG_Agentic_Real2Sim2Real_summary.txt)，[generator.py](C:/Users/WYM/Desktop/real2sim/horizon/src/horizon/datagen/generator.py) | 按论文配方重写状态专家、任务一致随机化、恢复演示和成功过滤；DAgger 是标准算法的本地实现 | 官方 ARSTAG 代码复现、原 Isaac Sim/机器人/视觉策略成绩的直接对比；本地“未公开代码”记录也不代表本日网络核验 |
| Agentic RSR `db1ddf5`：[MJCF fixture](C:/Users/WYM/Desktop/real2sim/baselines/agentic_rsr/tests/fixtures/stage3/code_as_policy_tabletop.xml)、[导入器](C:/Users/WYM/Desktop/real2sim/horizon/src/horizon/real2sim/mjcf_import.py)、[接入测试](C:/Users/WYM/Desktop/real2sim/horizon/tests/test_core.py:147) | 确实读入上游合成测试资产，再转换成自有 SceneSpec／任务／专家；导入器借鉴剥离机器人约束的接口思路 | 已运行自然视频重建、保留原机器人动力学、复现其程序策略；Real2Gym 也不能仅因支持 MJCF 就算完整接入 |
| MimicGen `72bd767`：[data_generator.py](C:/Users/WYM/Desktop/real2sim/baselines/mimicgen/mimicgen/datagen/data_generator.py:82)、[waypoint.py](C:/Users/WYM/Desktop/real2sim/baselines/mimicgen/mimicgen/datagen/waypoint.py:251) | 子任务边界、对象相对变换和对接执行的设计参考；本项目实际用状态专家生成 | 已使用其人类源演示变换生成器；“阶段平衡”是 MimicGen 官方算法 |
| LiLo-VLA `64aa29d`：[pipeline.py](C:/Users/WYM/Desktop/real2sim/baselines/LiLo-VLA/lilo_vla/core/pipeline.py:592)；CycleVLA `f3d0d7c`：[具体评测脚本](C:/Users/WYM/Desktop/real2sim/baselines/cyclevla_code/experiments/robot/libero-plus/run_libero_plus_eval_openpi_cyclevla.py:777) | 谓词验证、补回前置技能、切换时清空动作队列、恢复尝试计费的参考；未移植其 VLA/规划器 | 直接继承它们的长程能力。所引 CycleVLA 脚本调用 `restore_robot_only`，不能当作部署时可免费执行的物理恢复 |
| LeRobot `5517a32`：[sampler.py](C:/Users/WYM/Desktop/real2sim/baselines/lerobot/src/lerobot/datasets/sampler.py:26)、[ACT](C:/Users/WYM/Desktop/real2sim/baselines/lerobot/src/lerobot/policies/act/modeling_act.py:96) | episode 边界、精确恢复、动作 padding 和队列 reset 的参考，ACT/DP 仍是后续 backbone 候选 | 已训练 LeRobot/ACT，或 `EpisodeAwareSampler` 实现了 episode／stage 等权；其采样单位仍是帧 |
| SenseNova-RoboRSI `f7a349f`：[evaluation.md](C:/Users/WYM/Desktop/real2sim/baselines/SenseNova-RoboRSI/docs/evaluation.md)、[agents.md](C:/Users/WYM/Desktop/real2sim/baselines/SenseNova-RoboRSI/docs/agents.md) | 原生评分、代理版本、可比条件、运行轨迹的合约参考；HORIZON 实际账目是本地实现 | 本项目已经运行其 agent、复现其 benchmark 成绩或证明 improver 自身变强 |

EmbodiRSI 的 CEC/ADC 在这里是按任务类型近似实现，未复现论文的空间网格采集与全部训练设计；不能以“EmbodiRSI 式”的一个种子成绩宣称战胜官方方法。以上 commit 来自当前本地 checkout；详细限制见 [已有源码评审](C:/Users/WYM/Desktop/real2sim/horizon/docs/long_horizon_baseline_design.md)。

**3. 下一步必须面对哪些强而公平的对照**

| 要检验的主张 | 最小必要对照 | 公平性要求 |
|---|---|---|
| 访问策略状态是否改善长程学习 | 标准 β-DAgger vs 同恢复比例的名义数据扩增 | 共享 D0/π0、全 H 分布、累计保留标签、网络及更新步数；名义数据用嵌套预算；额外 rollout 与丢弃标签另计 |
| 新学习目标是否有价值 | 同一冻结数据上的标准 BC vs 单一新目标，再做各自策略采集的闭环比较 | 先隔离优化目标，再允许访问分布变化；不能让候选独享额外专家数据、起点或训练步数 |
| 校准后仍需权重学习 | 偏差下 S11 vs S10；校准是否帮助学习看 S11 vs S01 | 同种子相同初始数据/策略，配对任务、相同反馈机会，解释实际资源差异 |
| 归因／agent 选择是否有价值 | 最佳固定配方、透明规则 controller、agent；随机选择仅补充 | 同工具、可见反馈、总预算与候选试验成本；增加归因机制时保留同算子的均匀分配 |
| 长程结构是否有价值 | 若提出该主张，增加同观测和预算的顺序感知 flat baseline | 不能用缺失任务顺序信息的弱 flat 代替已有强对照 |

本批配置只有健康 nominal/DAgger 和偏差 S01/S10/S11，**没有偏差 S00 的最终配对格**，见 [bias 配置](C:/Users/WYM/Desktop/real2sim/horizon/configs/harness_long_horizon_replication_bias.yaml)。因此本批可以检验上述条件比较，不能直接计算完整 2×2 交互效应；不可用不同任务 bank 的 initial/curve 分数补成 S00。若要主张校准与学习的交互，需另立完整配对设计。

**4. 三条候选路线的选择门槛**

| 候选 | 何时优先 | 最小辨别实验与停止条件 |
|---|---|---|
| 观测通道一致性 | 存在可定位的训练／部署输入差异，且尚未做单变量性能检验。这个工程条件当前已满足；它本身不证明差异决定成功率 | 用同时记录 ideal 与 actor-observed grip 的同批轨迹、相同样本索引和专家标签，比较 legacy／observed 两种特征训练；冻结网络、β、预算、校准器和评测噪声。先核验特征重建，再做三种子完整任务对照。若无稳定收益，保留明确合约与负结果，不称“修复性能瓶颈” |
| 阶段平衡学习 | 完整复验后，健康 twin 的训练数据在后期 stage／交接附近覆盖不足，且这些位置也是可重复的失败集中处；仅总 SR 下降或总样本占比不均不足以立项 | 先在共享冻结数据上比较均匀 BC 与一个预定的温和权重方案。组内按 stage 平衡，同时保持原 family×H 总权重，设名义底线与上限；优化步数不变。空 stage 无法靠重权重补齐。若长 H 没改善、损害名义／其他任务族或有效样本只集中于少数 episode，则保留负结果，停止扩大权重搜索 |
| Agent 学习决策 | 至少两种操作在不同预定义条件下各有稳定用途，且可观测反馈能区分这些需求；最佳固定流程尚不能在预算内覆盖全部情形 | 让固定、规则和 agent 从同一工具集合选择“继续 DAgger／校准／覆盖候选／停止”，在未见任务分布比较完整 SR—预算曲线。若 agent 不优于最强固定/规则方案，且没有在事先规定的性能容忍范围内节省资源，则没有独立决策贡献 |

观测通道依据是明确的：[env.observe](C:/Users/WYM/Desktop/real2sim/horizon/src/horizon/sim/env.py:184) 给 grip position 加感知抖动；旧日志 `grip` 保存理想编码器，而部署控制器使用 `obs`。当前 [dataset.py](C:/Users/WYM/Desktop/real2sim/horizon/src/horizon/policy/dataset.py:45) 已有显式 legacy／observed 选项，[rollout.py](C:/Users/WYM/Desktop/real2sim/horizon/src/horizon/datagen/rollout.py:252) 保存 actor 当时收到的观测。**本次复验固定 legacy，不能中途切换**。旧日志没有该观测时不可重新抽噪声冒充原观测。该消融也不改变 `k_true` 专家条件标签、几何信念或校准输入，因此不能声称一次解决了所有训练／部署差异。

阶段诊断要报告每格的独立 episode 数、标签数、阶段到达率和失败位置；终态 prefix 不等于历史上到达过的阶段，300 条 cap 后的稀疏样本相邻也不等于时间相邻。现有 [SampleBlock](C:/Users/WYM/Desktop/real2sim/horizon/src/horizon/policy/dataset.py:23) 有 k/stage，但不足以还原边界窗口；边界加权需另存原始时间索引。现有加权 BC 使用 minibatch 内自归一化，不能等同严格分层采样。先验证固定权重，再考虑自适应选择；不要同时改采集起点、表示和网络。

另一个候选原因已有独立零结果：[旧 MixedController 的 90 次诊断](C:/Users/WYM/Desktop/real2sim/horizon/results/mixed_termination_20261028/README.md) 未观察到连续 8 行 learner 提前 done 而 oracle 仍 active 的停止，全部轨迹已[独立复核](C:/Users/WYM/Desktop/real2sim/horizon/results/mixed_termination_20261028/independent_verification.json)。因此不能把这次未观察到的机制当作主要瓶颈。该诊断仅一个训练种子的三个历史 checkpoint，不证明所有分布下不存在风险，也不估计修复收益。

**5. 复验结束后的决策和论文边界**

完整核验后先看三种子的配对差值与任务族×H 曲线，而不只看 pooled SR。若标准 DAgger 对名义扩增的优势不稳定，结论就是需要保留强名义对照，先做单变量机制检查；不以新增任务／改种子筛选优势。若 S11 相对 S10 没有额外学习收益，当前设定不足以支持联合学习主张，应检查覆盖或目标域差异；不能用修复命中率替代全任务学习收益。

三个种子的 mean／SD 是有限复制证据，共享评测任务也不是额外独立训练复制。候选筛选所看的 bank 属开发数据；选定唯一候选后再冻结确认协议，预先指定主 H／预算轴、最小有意义差异和可接受的任务族退化。全部失败、零增益、拒绝候选及其实际成本进入结果；有可靠变化且拟合适用时才报告指数 α。

Agent 决策成立之后，再检验带历史实验记忆的 improver 是否比初始／冻结 improver 更节省新任务成本：各组 actor、工具与反馈预算相同，经验版本冻结，避免把更强初始 actor 误算成记忆价值。这才接近 v2 的 H5 和“改进器的改进”，见 [E2/E3 路线](C:/Users/WYM/Desktop/real2sim/Agentic_Learning_Harness_路线方案.md:127) 与 [Physical2RSI 的边界](C:/Users/WYM/Desktop/real2sim/baselines/Physical2RSI.md:16)。固定修复序列不构成这项证据。

当前证据仍是受控 MuJoCo 代理。自然重建误差、独立目标模拟器、视觉策略、真实机器人及配对 sim-real 排序相关性需后续分别补齐。保持原生 full-task SR，另外报告进度和恢复成本；不拿外部 benchmark 的 Score 或参考论文表格与本地 SR 直接排名。下一项实验证据应减少一个明确不确定性，而不是增加一层系统名称。
