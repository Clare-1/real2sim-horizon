# 三种子长程复验：已完成

2026-10-28启动，2026-10-31完成。15/15系统、45次CUDA训练、全部独立产物核验及联合重校验通过。204项软件回归通过，PNG/SVG已检查。

- [中文进展与研究决策](../../../实验记录_长程复验_2026-10-31.md)：适合工作汇报，包含结果、代码优化、成本、负结果与下一步。
- [完整数值和曲线](RESULTS.md)、[机器可读结果及哈希](comparison.json)、[PNG](long_horizon_replication.png)、[SVG](long_horizon_replication.svg)。
- [原始结果独立复算](INDEPENDENT_RESULT_AUDIT.md)、[成本/覆盖独立审计](COST_COVERAGE_AUDIT.md)。
- [冻结计划](PLAN.md)、[复现命令](REPRODUCIBILITY.md)、[基线来源与路线](BASELINE_ROUTE_REVIEW.md)。
- [同任务特权专家检查](expert_feasibility/README.md)：600次，名义300/300、扰动299/300，不并入主实验成本。

主要结果：偏差下校准＋DAgger的H≥8完整SR为75.56±4.59% / 71.94±6.03%（名义/扰动）；相对校准冻结策略提高40.00 / 44.17个百分点，三个种子均为正。健康DAgger相对恢复数据基线的长程均值较高，但seed2退步，且sort收益伴随pack/stack权衡。

统计为三个训练seed的均值±样本SD，不是置信区间。仍是同MuJoCo开发代理，未检验真机迁移、agent决策优势或改进器迁移；没有S00最终格，不估计完整因子交互。

**计时解释：** 原始worker_seconds是各rollout循环经过时间之和，部分跨越10月28日至31日的间隔，不能视为CPU时间或速度基准。具体见[运行时说明](RUNTIME_NOTE.md)。主实验实际23,670次rollout、450,910,150物理步；评测复用避免3,600次重复rollout，不估算节省的墙钟时间。
