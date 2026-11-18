# RoboDojo 可选附录：证据状态与最小验证协议

日期：2026-10-26。本文是外部 benchmark 的可行性核对和待执行协议，**没有运行 RoboDojo，没有提交榜单，也没有得到新的 sim-real 相关性结果**。主实验继续采用自建、阶段数 H 可控的任务族；本附录不阻塞该实验。

## 1. 本次核验范围

已检查本地 `papers/`、`paper_summaries/`、`baselines/BASELINES.md` 和 SenseNova-RoboRSI 文档。本地论文目录未发现 RoboDojo 原文。本地材料包含上游项目 README 的副本及历史人工汇总，不能代替当前官方发布状态。

只读请求官方 GitHub 仓库元数据与项目网站时均返回 TLS/认证错误；升级网络访问的审批被中断，未取得页面内容。本次没有保存假冒官方快照，亦未从替代来源补成“已核实”。后续联网可用时再固定官方版本、文档和任务清单。

| 项目 | 当前可确认的内容 | 尚未确认的内容 |
|---|---|---|
| 任务数量、机器人、引擎 | 本地 [BASELINES.md](../../baselines/BASELINES.md) 第 25 行历史记录为“42 bimanual tasks，Isaac Sim，MIT” | 当前官方版本是否仍为 42；任务 ID、具体机器人、各任务阶段数、许可适用范围均未独立核验 |
| Agent track / sim 榜 | 本地 SenseNova README 引用 RoboDojo simulation leaderboard，并分别报告 score 和 success rate | 当前 track 规则、是否支持公开自助运行/提交、提交频率、账户要求和远程评测服务未核实；不假设排行榜存在就有公开 API |
| 本地 SenseNova 集成 | [README](../../baselines/SenseNova-RoboRSI/README.md) 的 BenchZoo 表将 RoboDojo 列为 **Coming soon** | 不能把其他 benchmark 的 HTTP adapter 当成已经可用的 RoboDojo adapter |
| 观测与动作 | 尚未获取 RoboDojo 官方合约 | 摄像头数量、RGB/depth/状态权限、动作维数、关节/末端控制、频率、双臂同步、可调用工具、reset 与终止接口均待核实 |
| 平台与依赖 | 历史清单写 Isaac Sim；本地 SenseNova 通用 benchmark images 面向 Linux | RoboDojo 固定的 Isaac Sim/Isaac Lab 版本、OS、CUDA、显存、资源下载量、CPU/远程模式均未核实 |
| Long-Horizon 分项 | 用户建议将其作为附录维度，合理但目前仍属待核实方案 | 是否有官方同名分项、成员 task IDs、汇总权重和成功规则未核实 |
| sim-real ρ | 需要同一候选系统在 sim 与 real 上的成对结果，见第 2 节 | 官方是否提供公开逐系统配对 real 数据、是否公布 ρ 及其计算口径未核实 |

待核对的官方入口：[项目网站](https://robodojo-benchmark.com/)、[排行榜](https://robodojo-benchmark.com/leaderboard)、[GitHub 仓库](https://github.com/RoboDojo-Benchmark/RoboDojo)。这些链接是后续核验入口，不代表本次已成功读取。

## 2. Score、完整任务 SR、长程分项与 ρ 必须分开

本地 [SenseNova README 的 RoboDojo 表](../../baselines/SenseNova-RoboRSI/README.md) 写明其 **average score 为 60.08，success rate 为 54.17%**，并注明结果正在提交、官方结果可能不同。这足以确认该报告中两列不是同一个指标；不能把 60.08 写成 60.08% 完整任务成功率，也不能把该第三方自报分数当作本项目或当前官方成绩。

- **Score**：沿用对应官方版本的 native scorer 和聚合方式。是否包含阶段完成度、部分奖励或加权项目，本次没有核实；不得凭数值大小猜测定义。
- **完整任务 SR**：对每次 episode 的官方 full-task success 布尔值求平均，同时报告成功次数和总次数。若官方只返回 score 而无 full-success 定义，不能自行用 score 阈值制造“官方 SR”。
- **Long-Horizon 分项**：若官方提供固定任务列表与规则，按该列表单列 Score 和 SR；若只是我们按任务结构自选子集，则明确写“自定义长程子集”，在跑分前冻结 task IDs 和选择依据。不得把任务数 42 当作阶段数 H=42。
- **成功率随 H 的曲线**：固定任务集合里的不同任务同时改变技能、物体、难度和时限，无法替代自建任务中控制其他因素、只改变 H 的实验。即使官方有长程分项，也不据此直接拟合“长度导致的衰减”并与自建 α 混合解释。
- **sim-real 排序相关性**：设候选系统 i 的仿真成功率为 s_i、真实部署成功率为 r_i，可对成对向量计算 Spearman ρ。候选必须有明确相同版本和对应任务分布、指标与预算；缺任一 real 结果的系统不能补零或用第三方别的策略成绩替代。只有 sim 榜单无法计算 sim-real ρ。

若真实端仍是另一模拟器，只能写 **cross-simulator correlation**；如果是同 MuJoCo 的另一配置，则写内部部署代理相关性。均不能改名成真实机器人相关性。少量候选的 ρ 会不稳定，应同时给逐系统原始分数、配对排序错误、预测误差，并按任务/训练种子结构报告不确定性。校准使用的任务与用于验证排序的任务分离。

## 3. Windows / 8GB 与现有夹爪的实际适配边界

当前机器的项目记录为 Windows、RTX 5060 Laptop、约 8GB 显存。**本次没有安装或启动 RoboDojo，不能断言其可在本机低成本完整运行，也不能仅凭一般 Isaac Sim 要求断言绝对不可运行。** 先取得该 benchmark 的固定版本要求；若不满足，再考虑已有 Linux/GPU 机器或正式可用的远程服务。远程服务的存在和费用也需要单独核验，不假设 SenseNova 对其他环境的 HTTP 支持能自动提供 RoboDojo。

HORIZON 当前是 4 自由度浮动夹爪、20Hz 的增量控制，其 pack/sort/stack 策略及状态向量不能直接迁移到双臂 benchmark。需要一个明确的 adapter：

1. `reset(task_id, seed)` 保留原生任务与初始分布；观测只暴露该 track 允许的信息。
2. 映射目标物、子目标与允许的动作工具；保留 native controller 的单位、坐标系、双臂约束和频率。不能仅 padding 动作维数就当移植成功。
3. `step`、超时、重试、恢复、动作步数由原生环境约束；恢复动作计入预算。
4. native scorer 独立记录 score/full-success；学习 harness 的 judge 不覆盖官方评分。
5. 为该机器人使用独立且固定的低层 skill/视觉策略。可迁移的首要对象是反馈组织、学习操作选择和经验版本管理，**不是当前夹爪 checkpoint 的无条件零样本迁移**。

附录第一步只做一个固定任务的 reset—观测—动作—反馈—评分冒烟及成本测量。通过后才决定是否有资源做长程子集，最后才考虑全任务集合。更换机器人/技能模型带来的收益和 harness 改进分别对照。

## 4. 最小附录协议（尚未执行）

目的：检验主实验中得到的 harness 设计能否接入外部原生任务并产生可重复收益；不是以刷榜决定主方法。

1. **冻结合约**：记录官方 commit/任务 manifest/资产版本/运行环境/track、允许观测与工具、步数和重试上限、score/SR 定义。先确认自助运行是否开放。
2. **冻结子集**：在性能比较前选择可运行、包含多阶段的少量任务。若没有官方 Long-Horizon 子集，以任务结构列出自定义子集。报告所有失败，不按方法结果挑任务。
3. **共享 actor 与资源**：以同一个低层机器人策略对比静态 harness、固定反馈流程、主实验选定的反馈驱动 harness。若比较权重学习，明确新增训练数据/步数与 checkpoint；冻结 actor 的比较单独报告为 harness 改进，不冒充 policy learning。
4. **匹配预算**：同任务种子、原生动作/时限/重试预算和允许反馈；记录模型调用、实际 rollout、人工/专家标注、训练成本。候选试运行不能隐藏在预算外。
5. **保留独立确认**：开发任务用于选方案，另留任务/种子进行最终比较。一次冒烟只证明接口可用，不能报成功率提升。
6. **分别报告**：每个任务及聚合的 native Score、完整任务 SR、动作和反馈成本；如果有合法官方长程子集，单列其结果。附录成绩不与自建 H 曲线合并拟合。
7. **ρ 是独立扩展**：只有获得合法可用的 paired real 数据，且与候选系统版本和指标匹配，才追加 sim-real 排序分析；没有则明确“不测”，不从 sim 榜推算。榜单提交不属于此协议的自动执行步骤。

## 5. 可立即复用的本地实现设计

不需要等 RoboDojo 运行起来才借鉴以下设计，且本次没有复制代码：

| 来源及许可 | 已核对的实现/文档 | 可用于本项目的具体设计 |
|---|---|---|
| SenseNova-RoboRSI，Apache-2.0；保留 [LICENSE](../../baselines/SenseNova-RoboRSI/LICENSE) | [evaluation.md](../../baselines/SenseNova-RoboRSI/docs/evaluation.md)、`roborsi/evaluation/agent_artifact.py` | 实验比较前核验任务、动作预算、actor/config 与运行依赖；声明的版本与观察到的 runtime 分开存。避免把不同条件的分数直接相减。 |
| 同仓库，Apache-2.0 | [agents.md](../../baselines/SenseNova-RoboRSI/docs/agents.md)、`roborsi/agents/package.py`、`tests/integration_tests/robots/libero/test_roborsi_batch_resume.py` | 将 agent 的 code/resources/config 形成显式版本；新 run ID 恢复兼容的已完成 case，保留父结果链接；同一经验可以导出训练监督，再以新 checkpoint 回到相同评测 runtime。 |

这些是可靠 harness 的基础设施，不自动构成算法创新。SenseNova 的 RoboDojo 集成在当前本地 README 中尚未公开为可用项，故这里只借鉴其已存在的通用版本/比较设计，不声称已经复现其 RoboDojo 成绩。

## 6. 决策

保留 RoboDojo 为可选外部附录。当前主实验继续验证“部署反馈 → twin/数据操作 → 真实权重更新 → 独立任务评测”的 agentic learning 闭环，以及阶段数可控的长程性能。取得官方合约和可用运行资源后再进行一个任务的适配冒烟；缺少 paired real 数据时，不将 sim-real ρ 纳入已完成交付。
