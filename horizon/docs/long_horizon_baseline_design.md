# 长程学习机制：本地基线参考与最小实验设计

日期：2026-10-28。本文件只读核对本地参考源码和当前 HORIZON 实现，提出下一项可执行设计；未安装依赖、修改 baseline 或核心代码，也未产生新的成功率。本地 checkout 版本见第 2 节。建议先保留当前状态策略和已验证的 8GB GPU 训练路径，研究完整任务中的数据覆盖与恢复，再扩展视觉策略。

## 1. 先区分长度外推与完整长程学习

原始 v3/v4 主实验在 H=2/4/6/8/10 上训练和评测，标准 β 混合 DAgger 是最强长程学习基线；`dagger_attr` 将 60% 采集起点集中在归因失败处反而退步。这两项结论应保留，不能靠更换方法名称重新假设“失败越集中采样越好”。

近期 [E1 外推诊断](../results/harness_transfer_20261026/TRANSFER.md) 只在 pack H=2/4 训练，将冻结策略评测至 H=6/8/10，24 个系统、2,880 次 rollout 的完整任务成功率均为零。该结果不能与原始全 H 训练分数直接比较。当前 [features.py](../src/horizon/policy/features.py) 除局部对象特征外，还输入 `k/10`、`H/10`，并按附近对象数量填充固定数量的邻居槽位。因此外推同时改变阶段数、显式数值特征和邻居/拥挤状态分布。这里列的是可检验原因，不是对零成功的已完成因果解释。

优先问题应为：**在覆盖全 H 的标准 DAgger 上，是否仍有可观测的后期子目标/交接状态欠覆盖；仅修正该覆盖，是否改善完整长程任务，而不挤占名义行为？**

暂不同时改变几何修复、验证器容差、网络架构和采集策略。先在健康 twin 上识别学习机制，再把已确定的变化带回 Real2Sim2Real 的校准×学习对照。

## 2. 基线中实际存在什么、可以借鉴什么

| 本地来源 | 已读的具体实现 | 适合当前状态策略的借鉴 | 边界与许可 |
|---|---|---|---|
| ARSTAG 论文资料；本地尚无官方研究代码 | [论文抽取文本](../../paper_summaries/01_ARSTAG_Agentic_Real2Sim2Real_summary.txt) III-B；HORIZON 的 [generator.py](../src/horizon/datagen/generator.py) `OpenLoopConfig/open_loop_job` | 状态专家、任务一致随机化、名义＋扰动恢复演示、成功过滤；保存随机化和实际失败代价 | 当前是论文配方启发的实现，非官方复现。完整训练不是运行一个 LLM 即自动得到，专家和可行性仍有任务假设。 |
| MimicGen `72bd767` | [data_generator.py](../../baselines/mimicgen/mimicgen/datagen/data_generator.py) `randomize_subtask_boundaries/select_source_demo/generate`；[selection_strategy.py](../../baselines/mimicgen/mimicgen/datagen/selection_strategy.py)；[waypoint.py](../../baselines/mimicgen/mimicgen/datagen/waypoint.py) `merge/execute` | 保持子段相邻边界一致、对象相对几何、对接段实际执行检查。可将“边界也需要覆盖”转为本地采样诊断 | 其生成依靠人类源轨迹变换与插值；当前状态专家不是同一生成器。随机边界不是失败归因，也不是 DAgger。NVIDIA 非商业研究/评估许可证，见 [LICENSE](../../baselines/mimicgen/LICENSE)；本次只借鉴设计，不复制代码。 |
| LiLo-VLA `64aa29d` | [pipeline.py](../../baselines/LiLo-VLA/lilo_vla/core/pipeline.py) 的 BDDL 技能完成检查、`_should_skip_reset`、place 失败后的 pick recovery 与 recovery 统计 | 区分单纯重试、重新估计接近位姿、补回前置技能；恢复之后重置相应 predicate baseline；所有恢复步数计费 | HORIZON 已有子目标回退和 Markov 专家恢复，新增价值必须由严格消融证明。这里没有移植 OpenVLA/运动规划/视觉系统。MIT，见 [LICENSE](../../baselines/LiLo-VLA/LICENSE)。 |
| CycleVLA `f3d0d7c` | [run_libero_plus_eval_openpi_cyclevla.py](../../baselines/cyclevla_code/experiments/robot/libero-plus/run_libero_plus_eval_openpi_cyclevla.py) 的 transit/backtrack、重试上限、动作队列清空和 MBR 候选记录 | 子目标改变时清空旧动作计划；保留失败尝试和恢复成本；将“进度正确”与完整任务成功分开 | **这个具体脚本调用 `restore_robot_only` 回溯仿真机器人状态**。不可把这种恢复直接当成可在真机执行的闭环动作。评测部署不能免费倒带；训练中合法 snapshot reset 也应单独记账。MIT，见 [LICENSE](../../baselines/cyclevla_code/LICENSE)。 |
| LeRobot `5517a32` | [sampler.py](../../baselines/lerobot/src/lerobot/datasets/sampler.py) `EpisodeAwareSampler/compute_sampler_state`；[ACT 实现](../../baselines/lerobot/src/lerobot/policies/act/modeling_act.py) `reset/select_action/forward` | 明确 episode 边界、可复现 shuffle/resume；序列训练屏蔽 `action_is_pad`；切换任务时清空动作队列 | EpisodeAwareSampler 保持每帧采样，**不是每 episode 或每子目标等权采样**。ACT chunk/temporal ensemble 是另一个学习/控制选择，不是本次覆盖机制的直接证明。Apache-2.0，见 [LICENSE](../../baselines/lerobot/LICENSE)。 |
| Agentic RSR `db1ddf5` | [randomization.py](../../baselines/agentic_rsr/src/sim/randomization.py) `EpisodeRandomizer/RandomizationConfig`；[SIM 文档](../../baselines/agentic_rsr/src/sim/README.md) | 分开保存物理场景变化、感知标定误差、控制误差；名义/随机化验收使用不同实例；失败经验与版本绑定 | 其程序策略修订不等价于本地 BC 权重训练。当前完整重建还需要模型/Blender/机器人依赖，不作为本次轻量优化前置条件。项目代码 Apache-2.0，权重/第三方许可另计。 |

上述均为本地源码核对，不是最新论文全面检索或官方方法端到端成绩复现。尤其不能把本地历史清单中“CycleVLA 会回退”泛化为所有回退都可物理执行。

## 3. 当前可以直接复用的实现

- [strategies.py](../src/horizon/rsi/strategies.py) `plan_dagger_beta` 已有 β=0.5^round、每个 episode 最多 300 条专家标注、新鲜完整任务起点与扰动采集。它应作为候选共同底座，不用崩溃过的 β=0 朴素版本做主要对手。
- [dataset.py](../src/horizon/policy/dataset.py) `SampleBlock` 已保留 episode key、H、provenance、条件子目标 k 和拓扑 stage；但没有逐样本原始时间索引和完整边界窗口标识。若做边界加权，需从真实时间序列构造窗口并保存，不能在稀疏抽样后把相邻样本误当连续时间。
- [model.py](../src/horizon/policy/model.py) `train_bc` 已接受样本权重 W、训练 Torch MLP、导出 NumPy 推理。可以先在同一份冻结数据上检验覆盖目标，不必先加入新 Transformer/VLA。
- [controllers.py](../src/horizon/policy/controllers.py) 已有 verifier 驱动的回退、去抖和混合专家采集。由 System 2 选择的子目标身份可用于训练诊断；如果使用专家阶段/真值标签，应明确仅用于仿真监督或离线分析，不让路由器读取不可观测真值。
- [learning_harness.md](learning_harness.md) 对应实现已保存 task/twin/data/policy 版本、实际训练记录、配对开发评测和成本账目。下一轮沿用这些基础设施，而非另造缺少来源的单次训练脚本。

## 4. 第一优先级：保留完整分布的覆盖机制

最小候选为 **全 H 标准 DAgger + 有上限的子目标覆盖加权**，先保持采集起点、β、扰动、控制器、特征和网络不变。其作用是重分配学习注意力，不是假装产生了更多独立数据；这是本项目待验证的设计，不冠以 MimicGen 或 LeRobot 官方算法名。

在训练集内统计 `(H, stage)` 或预先固定的粗阶段分组。令 N 为总样本数，G 为非空组数，n_g 为组内样本数，可比较统一权重与：

`w_i = (1 - λ) + λ * N / (G * n_g)`。

λ 与权重上限在开发前固定，建议第一轮只取一个保守候选，例如 λ=0.25、最大权重 4，再归一化；它们是开发设置，不是已证实的最优值。`train_bc` 当前的随机 minibatch 内自归一化权重不是严格的分层采样，应如实记录实现，不宣称两个估计器逐步相同。空组不能用权重补齐。

边界覆盖作为后续独立变量：根据条件子目标切换前后固定控制步窗口建立标签；同样保留名义权重底线。不要一次同时启用“阶段平衡＋边界加权＋失败起点重采样”，否则无法知道收益来源。

必须先报告覆盖，而不只报训练 loss：各 H/stage 的独立 episode 数、有效标签数、任务中到达该阶段的比例、交接窗口标签占比、恢复标签占比、专家/策略实际执行动作比例。较晚 stage 的高 loss 可能只是样本稀缺；完整任务 0% 也可能已经完成大部分阶段，须分开看。

若晚期阶段根本没有被访问，先检验 β 混合是否能送达、专家能否完成和训练 H 分布是否覆盖。只有在这之后才考虑加入少量训练 snapshot starts；必须保留完整任务采集对照及大部分新鲜起点，报告快照恢复成本，并与非归因的均匀 snapshot 选择比较。现有 60% `dagger_attr` 的负结果意味着不能直接恢复那条路线。

## 5. 分两层做最小实验，强对照不削弱

### 5.1 数据固定的机制筛选

目的：区分长度覆盖、表示外推和优化目标，而非先跑巨大矩阵。

1. 从全 H={2,4,6,8,10} 的统一任务分布收集共享 D0，保留标准恢复比例与相同 label budget；用固定标准 β-DAgger 流程获得一份有阶段记录的数据池。所有对照使用相同字节的数据、训练步数、batch 和随机种子。
2. 对比统一样本训练与单一候选覆盖权重。先用三个训练种子、pack 一族做开发筛选；pack 结果不外推为三族或跨技能结论。
3. 在同一未用于训练的任务 bank 上评估所有 H，分别报告名义/扰动完整 SR、H≥8、stage completion/first failure、恢复成功率与实际动作成本。
4. 单独用已有 H2/4 checkpoint 做长度外推参考，不与全 H 训练称为同任务训练分布下的公平优越性比较。若研究去掉 H/k 输入，另开相同数据下的表示消融，不能与权重候选同时变化。

旧数据可能已经用于开发选择，因此这一步是开发筛选；不打开保留确认库。不从本轮分数反复选择 λ 再把同批结果当确认。

### 5.2 真正的闭环学习比较

固定候选后，运行下面四种核心系统；每种使用自己的当前策略产生后续 DAgger 状态，初始 D0 和 π0 共享：

| 系统 | 作用 |
|---|---|
| ARSTAG 配方开环，按相同累计数据预算取嵌套子集 | 检查收益是否只是增加数据 |
| 标准 β 混合 DAgger | 最强必要基线；相同完整任务起点、训练/反馈预算 |
| DAgger + 固定覆盖候选 | 检查增加的机制是否超过标准闭环 |
| 无归因的固定/均匀覆盖对照 | 只有在引入反馈自适应覆盖时新增，用于隔离“agent 决策”与简单分层的价值 |

若候选只是固定权重，不必为了名称虚构第四种不同算法。得到稳定机制后再让 learning controller 根据覆盖诊断在“继续标准 DAgger / 增加边界覆盖 / 维持现状”之间选择；固定混合、随机选择、相同预算规则 controller 是其强对照。若 agent 只执行固定序列，应报告固定学习流程，不声称已证明 agentic 决策优势或递归改进器变强。

正式结果恢复原 v1–v3 轴：全任务 SR—H、SR—累计标签数、每轮学习增益、恢复/交接失败；可沿用 80k + 3×40k 的历史预算作可比设计，但运行时以当前冻结配置和资源为准。新协议的结果不能与历史数值拼接，且必须记额外仿真步数、discarded labels 和训练计算成本。指数 α 只在有足够成功变化并检查拟合适用性时报告；全零曲线不拟合出有解释力的 α。

通过上述学习机制筛选后，再回到 twin 校准×policy learning 的 2×2，检查新机制的收益是否在自然/代理几何差异下保留。不能同时修 geometry estimator 又改学习覆盖，然后把总提升全部归给 agent。

## 6. 恢复与动作分块的后续顺序

执行层恢复须是实际可执行动作：重新观察、重新抓取、清理前置条件或重新放置；部署评测不允许无代价恢复世界快照。训练中的状态恢复与部署中的物理恢复分开报告。

若原始单步 MLP 的主要问题是抖动或部分可观测状态，再单独验证短历史/小动作块。LeRobot ACT 的 chunk 默认值不能直接搬入当前 20Hz 系统；长开环块可能跨越抓取、放置与子目标切换并降低恢复速度。首先比较单步与短 chunk，切换子目标/恢复时清空队列，padding 不参与监督，并始终报告实际重新观测频率。ACT/DP 视觉训练是后续 backbone 扩展，不是当前长度机制的必要条件。

## 7. 8GB GPU 上的优化与证据边界

已有 [运行环境记录](../results/harness_runtime_20261026/README.md) 证明当前状态 BC 可在本机 GPU 训练，rollout 仍为 CPU/NumPy。新增覆盖权重是 O(N) 标量元数据，通常远小于图像模型成本；实际峰值显存和吞吐仍需测量，不能仅凭参数规模宣布加速。

值得优先测量的开销是 worker rollout/特征提取、数组拼接、训练索引传输、checkpoint 写入和评测。保持同一算法先比较缓存/批量化；随机数从 NumPy 改为设备端生成会改变训练序列，必须记录为新的实现版本，不声称与历史逐步一致。当前 NumPy rollout 与 Torch 前向的一致性检查应保留。

性能优化只能改善耗时，不单独计作算法成功率贡献。相反，采样权重、恢复规则、输入特征、动作块长度都改变学习或执行机制，应各自有配对对照。所有参考代码继续保持只读；若以后实际复用，记录原 commit、文件、许可与修改范围。

**当前建议的最短路线：全 H 的标准 DAgger → 共享数据上的覆盖诊断/单一加权消融 → 自己访问状态的闭环比较 → 再接 twin 校准与 agent 决策。** 这保留原始长程学习问题，也避免再次把固定策略的修复调度作为全部贡献。
