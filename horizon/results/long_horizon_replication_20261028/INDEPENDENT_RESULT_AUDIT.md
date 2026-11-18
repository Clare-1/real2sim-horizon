# 三种子结果的独立科学复核

审计时间：2026-10-31T06:23:28.920634+00:00。

本审计直接读取15个 `runs/...json` 的逐episode成功标记、轮次belief/策略/成本记录及实际训练NPZ的stage数组，先独立重算，再核对联合报告。未调用报告的计算函数，未训练、未执行新rollout、未修改源代码或原始结果。

结论：一次几何校准后继续学习的增益在三个训练种子均重复出现；健康DAgger的总体优势不稳定，主要体现为长sort的改善，并伴随pack与名义stack的退步。本批支持代理环境中的机制复现，尚不支持agent决策优势或RSI收益。

## 统计与配对核验

- SR = 完整任务success为true的episode数 / 对应episode数；all-H每seed、每模式300个任务，H≥8为120个，单family×H为20个。
- 每个表格中的seed0/1/2分别是三次训练结果；均值与样本SD均先逐seed计算，SD使用n−1分母，不是置信区间。
- 每模式的300个最终任务身份（family/H/task_seed/rollout_seed/task_sha256）在所有方法和seed间完全一致；不是900个不同任务。
- 健康两组每seed的初始data/policy哈希相同；偏差三组每seed的初始data/policy哈希相同。
- 独立核对联合JSON的840条final_metrics（包括终态前缀/满足数/最大同时满足数）、504条paired_contrasts（包括胜/平/负）、280条final_summary和168条paired_summary，全部一致；浮点容差1e−12。
- 这次科学审计不替代已完成的全量工件验证；主要新增的是原始统计复算与结论边界。

## 逐seed最终SR

单位：%。N=nominal，P=perturbed；seed列按0 / 1 / 2顺序。

| 方法 | 模式 | 范围 | seed 0 / 1 / 2 | 均值 ± SD |
|---|---|---|---|---|
| healthy_nominal | N | all-H | 78.33 / 81.67 / 85.67 | 81.89 ± 3.67 |
| healthy_nominal | N | H≥8 | 66.67 / 70.00 / 75.00 | 70.56 ± 4.19 |
| healthy_nominal | P | all-H | 73.00 / 76.67 / 79.33 | 76.33 ± 3.18 |
| healthy_nominal | P | H≥8 | 59.17 / 65.00 / 65.83 | 63.33 ± 3.63 |
| healthy_dagger | N | all-H | 83.67 / 82.33 / 76.00 | 80.67 ± 4.10 |
| healthy_dagger | N | H≥8 | 79.17 / 77.50 / 71.67 | 76.11 ± 3.94 |
| healthy_dagger | P | all-H | 81.00 / 81.00 / 70.00 | 77.33 ± 6.35 |
| healthy_dagger | P | H≥8 | 76.67 / 78.33 / 62.50 | 72.50 ± 8.70 |
| bias_S01 | N | all-H | 30.33 / 30.00 / 29.67 | 30.00 ± 0.33 |
| bias_S01 | N | H≥8 | 28.33 / 25.83 / 25.00 | 26.39 ± 1.73 |
| bias_S01 | P | all-H | 28.67 / 27.33 / 26.67 | 27.56 ± 1.02 |
| bias_S01 | P | H≥8 | 25.00 / 20.83 / 20.00 | 21.94 ± 2.68 |
| bias_S10 | N | all-H | 44.33 / 48.33 / 51.00 | 47.89 ± 3.36 |
| bias_S10 | N | H≥8 | 36.67 / 31.67 / 38.33 | 35.56 ± 3.47 |
| bias_S10 | P | all-H | 37.67 / 45.00 / 42.33 | 41.67 ± 3.71 |
| bias_S10 | P | H≥8 | 25.00 / 30.83 / 27.50 | 27.78 ± 2.93 |
| bias_S11 | N | all-H | 84.67 / 79.00 / 78.00 | 80.56 ± 3.60 |
| bias_S11 | N | H≥8 | 80.00 / 75.83 / 70.83 | 75.56 ± 4.59 |
| bias_S11 | P | all-H | 83.00 / 77.67 / 75.33 | 78.67 ± 3.93 |
| bias_S11 | P | H≥8 | 75.83 / 75.00 / 65.00 | 71.94 ± 6.03 |

## 逐seed配对差

单位：百分点。相同seed、相同任务，左减右。

| 对照 | 模式 | 范围 | seed 0 / 1 / 2 | 均值 ± SD |
|---|---|---|---|---|
| healthy_dagger − healthy_nominal | N | all-H | 5.33 / 0.67 / -9.67 | -1.22 ± 7.68 |
| healthy_dagger − healthy_nominal | N | H≥8 | 12.50 / 7.50 / -3.33 | 5.56 ± 8.09 |
| healthy_dagger − healthy_nominal | P | all-H | 8.00 / 4.33 / -9.33 | 1.00 ± 9.13 |
| healthy_dagger − healthy_nominal | P | H≥8 | 17.50 / 13.33 / -3.33 | 9.17 ± 11.02 |
| bias_S11 − bias_S10 | N | all-H | 40.33 / 30.67 / 27.00 | 32.67 ± 6.89 |
| bias_S11 − bias_S10 | N | H≥8 | 43.33 / 44.17 / 32.50 | 40.00 ± 6.51 |
| bias_S11 − bias_S10 | P | all-H | 45.33 / 32.67 / 33.00 | 37.00 ± 7.22 |
| bias_S11 − bias_S10 | P | H≥8 | 50.83 / 44.17 / 37.50 | 44.17 ± 6.67 |
| bias_S11 − bias_S01 | N | all-H | 54.33 / 49.00 / 48.33 | 50.56 ± 3.29 |
| bias_S11 − bias_S01 | N | H≥8 | 51.67 / 50.00 / 45.83 | 49.17 ± 3.00 |
| bias_S11 − bias_S01 | P | all-H | 54.33 / 50.33 / 48.67 | 51.11 ± 2.91 |
| bias_S11 − bias_S01 | P | H≥8 | 50.83 / 54.17 / 45.00 | 50.00 ± 4.64 |

## H=10的任务差异

单元格为seed0 / 1 / 2的SR，括号内为三seed均值。

| 方法 | family | nominal | perturbed |
|---|---|---|---|
| healthy_nominal | pack | 85.00 / 75.00 / 80.00 (80.00) | 80.00 / 75.00 / 70.00 (75.00) |
| healthy_nominal | sort | 15.00 / 15.00 / 10.00 (13.33) | 20.00 / 20.00 / 15.00 (18.33) |
| healthy_nominal | stack | 90.00 / 100.00 / 100.00 (96.67) | 65.00 / 80.00 / 80.00 (75.00) |
| healthy_dagger | pack | 85.00 / 65.00 / 60.00 (70.00) | 80.00 / 75.00 / 45.00 (66.67) |
| healthy_dagger | sort | 55.00 / 70.00 / 70.00 (65.00) | 50.00 / 75.00 / 55.00 (60.00) |
| healthy_dagger | stack | 75.00 / 70.00 / 85.00 (76.67) | 85.00 / 65.00 / 80.00 (76.67) |
| bias_S01 | pack | 0.00 / 0.00 / 0.00 (0.00) | 0.00 / 0.00 / 0.00 (0.00) |
| bias_S01 | sort | 70.00 / 55.00 / 75.00 (66.67) | 70.00 / 60.00 / 45.00 (58.33) |
| bias_S01 | stack | 5.00 / 0.00 / 5.00 (3.33) | 5.00 / 0.00 / 5.00 (3.33) |
| bias_S10 | pack | 55.00 / 30.00 / 35.00 (40.00) | 40.00 / 35.00 / 20.00 (31.67) |
| bias_S10 | sort | 0.00 / 20.00 / 25.00 (15.00) | 0.00 / 15.00 / 5.00 (6.67) |
| bias_S10 | stack | 35.00 / 35.00 / 30.00 (33.33) | 15.00 / 15.00 / 20.00 (16.67) |
| bias_S11 | pack | 75.00 / 70.00 / 70.00 (71.67) | 75.00 / 70.00 / 65.00 (70.00) |
| bias_S11 | sort | 65.00 / 60.00 / 55.00 (60.00) | 60.00 / 55.00 / 40.00 (51.67) |
| bias_S11 | stack | 75.00 / 85.00 / 80.00 (80.00) | 75.00 / 85.00 / 70.00 (76.67) |

健康DAgger对H10 sort的提升在三个seed均存在：nominal为+40/+55/+60 pp，perturbed为+30/+55/+40 pp；相对地，H10 pack为0/−10/−20和0/0/−25 pp，nominal stack为−15/−30/−15 pp。不能用总平均掩盖这个稳定的任务交换。

偏差S11相对S01的巨大总体收益主要来自受支撑高度影响的pack/stack恢复；H10 sort均值反而下降6.67 pp（两模式均如此）。sort没有同样的支撑高度故障，因此此差异不能直接解释为几何校准损害sort，也不能说明所有任务都受益。共享策略和训练状态分布的影响需要专门对照。

## 首轮校准与冻结核验

配置中的真实plate顶面与tray顶面均为20 mm；初始belief均为6 mm。下面n是相关轨迹中的观测帧数，不能当成独立样本量。

| seed | plate估计/mm (n) | tray估计/mm (n) | 对20 mm的误差/mm |
|---|---|---|---|
| 0 | 19.662 (1597) | 20.467 (4119) | plate -0.338; tray +0.467 |
| 1 | 20.535 (1602) | 19.925 (1650) | plate +0.535; tray -0.075 |
| 2 | 19.893 (3337) | 19.919 (1874) | plate -0.107; tray -0.081 |

三个seed均满足：S10/S11首轮反馈观测字节哈希相同、首轮校准belief相同、calibration_attempted=[true,false,false]；第2/3轮没有新校准报告或校准成本；全部后续recon_after及final_recon与首轮相同。S10策略保持初始哈希，S01重建保持初始错误belief。没有依据校准质量筛选seed，也没有事后选择校准轮次。

所有支撑估计距真值≤0.535 mm，因而这三个seed中不存在明显失真的首轮校准。该数字只描述本次固定20 mm支撑和当前观测通道，不能外推到未见材质、厚度或真实相机。

## 实际数据覆盖和采集成本

下表直接读取D0及三轮新增NPZ的stage数组，列出H10最后阶段（stage=9）的累计标签。

| 方法 | family | stage9标签：seed0 / 1 / 2 |
|---|---|---|
| healthy_nominal | pack | 2459 / 2101 / 3219 |
| healthy_nominal | sort | 1832 / 2393 / 2128 |
| healthy_nominal | stack | 1610 / 1577 / 2331 |
| healthy_dagger | pack | 1951 / 1462 / 1968 |
| healthy_dagger | sort | 1697 / 2339 / 2041 |
| healthy_dagger | stack | 1810 / 1338 / 1894 |
| bias_S11 | pack | 1819 / 1428 / 2004 |
| bias_S11 | sort | 1634 / 2309 / 2202 |
| bias_S11 | stack | 1908 / 1854 / 1840 |

数据不支持“末阶段零覆盖”的解释；普通数据也含较多末阶段样本。健康DAgger的H10 sort末阶段标签数在各seed并不更多，却明显提高SR，提示状态分布/恢复经验比简单末阶段计数更值得研究；这仍是关联证据，不能据此确定因果机制。

健康两组追加保留标签及优化步相同，但实际追加采集物理步为DAgger 18,980,150、名义恢复 11,420,200，前者为后者的1.662倍。不能把匹配标签预算写成匹配计算成本或样本效率胜出。S10只有80k标签和8k优化步，S11为200k和32k，S11−S10是学习机制对照，不是等计算效率对照。

## 证据强度与下一项单变量实验

最强的现有结论是：在人为注入的支撑高度偏差下，固定错误twin中的学习无法恢复pack/stack；一次合理校准加后续学习在三seed中都恢复了大部分完整任务成功率，H≥8的增益也一致。健康DAgger长程均值有所提高，但seed2退步且family交换明确，不能宣称普遍优于名义恢复数据。

原始v3/v4路线已经检验过失败起点采集：[原始基线记录](C:/Users/WYM/Desktop/real2sim/baselines/BASELINES.md:68)中的 `dagger_attr` 将60%的采集episode从自动归因失败子目标的开头快照启动，结果未超过标准β混合DAgger。因此不能忽略已有负结果，把集中失败快照作为自然胜出的下一步。本批末阶段已有充分覆盖，也没有新增证据证明缺少可恢复失败状态才是主要瓶颈。

建议首先隔离 **observed与legacy训练输入通道**：精确重放冻结轨迹，核验任务、动作、标签、样本身份与选择均一致，再由同一批已记录观测分别构造控制器实际收到的gripper输入和历史ideal-encoder输入。固定训练种子、初始化、优化预算、网络、twin及评测任务，只改变这一输入通道；不让两个学习策略重新采集不同轨迹。重放中的动作/标签/样本匹配核验通过之前，不把性能差归因于输入通道。完整报告三seed、全部H与H10各family结果，保留40M确认库。

随后才考虑 **固定比例的恢复专家数据与标准DAgger数据混合**，针对本批重复出现的pack/stack与sort交换做独立对照。比例在结果产生前固定，预算、family×H和stage覆盖、观测模式及其他学习设置保持一致，避免同时引入阶段重加权或失败起点改变。检验能否保留sort收益并缓解pack/stack退步，分别报告完整SR、末阶段覆盖及物理采集成本；该混合比例不因当前单元格分数而自适应选择，也不预设会胜出。

快照恢复作为延后的候选方案保留：只有新增诊断明确显示有价值的可恢复状态缺口，且常规采集确实难以覆盖时，才值得重新设计。届时仍须对照完整任务起点和非归因的均匀有效快照，并计入候选池、无效恢复及失败采集成本。当前数据没有要求立即恢复60%失败起点路线，也没有证明targeting、RSI、学习型agent或递归自改进有效。

本批只有三个训练seed及一个共享开发任务库；每个H10/family单元格20个任务。没有S00完整因子组、独立真实机器人、独立物理引擎或未见分布确认，所有校准与学习均在同MuJoCo代理世界。误差条为描述性SD，不做显著性、SOTA或真机迁移结论。

## 审计输入哈希

联合报告 comparison.json SHA256：`6b5b817a8ab99b0df2d52684c2372cad787f42bd641708701c65fa9c1ae4e268`。

| 原始运行（相对本目录） | SHA256 |
|---|---|
| fullH_nominal/runs/healthy/S01_s0.json | `a7d49b79d347283bc1e2c5980df6b3b0c1e06b329c3888eb7a907591d5db22e0` |
| fullH_nominal/runs/healthy/S01_s1.json | `b8209ffd8a2d0e3a02a55b8b4667e24452abbac6eb50ef1f9119b3f8698d7250` |
| fullH_nominal/runs/healthy/S01_s2.json | `2531f1e97f756e2dbf551a8aeb3ce360575f9b9f9aa5285cab3c3525afc5af21` |
| fullH_dagger/runs/healthy/S01_s0.json | `2c344714ef8c08d2812df0c53688900822776041e95de567ab4e77c846eacbcb` |
| fullH_dagger/runs/healthy/S01_s1.json | `85aa861bf2cbbfdbaabe6c51ab6a7ad7898c2f88bcc15fe9f22a64d3b620179e` |
| fullH_dagger/runs/healthy/S01_s2.json | `e818c25c7c22a08acd15f625743e84cfc579fd6068f8e70e38c4140506c6178d` |
| fullH_bias/runs/height_bias/S01_s0.json | `a77c5aa7af2699197c9a7b4a3739e9955b40993aba20c437f2e86edca4866095` |
| fullH_bias/runs/height_bias/S01_s1.json | `d7832eb34c7d4fc5b4a1b60aaac06a8f01c856053ddb0859016f7d1c8690f0e0` |
| fullH_bias/runs/height_bias/S01_s2.json | `402a35a89ab3322d8e28ba8ed06ddff7f7beefc6e3f37dd60e93e796af34e2f8` |
| fullH_bias/runs/height_bias/S10_s0.json | `728eb35550b9cbcce7aafcc9af221b5903c11f0bcbbbd2b356b38ad6c2a9389b` |
| fullH_bias/runs/height_bias/S10_s1.json | `fc84a5c53e597b6215e9ad6288a60aba3cc3684af3c7ad9e505bf01c5c7da1bb` |
| fullH_bias/runs/height_bias/S10_s2.json | `f1938971a491cc11f2ef06e27a49b412b606706eb6283abca61bdee8bcd74d80` |
| fullH_bias/runs/height_bias/S11_s0.json | `1a8c30817bb39dcbd59dfd1386a2480b969f7afcc47f6e9cf0861ffc0cddb605` |
| fullH_bias/runs/height_bias/S11_s1.json | `89cb817649efbd21525aaebc3959abca4c53fd30f98986b05769affbd7812ef0` |
| fullH_bias/runs/height_bias/S11_s2.json | `665be40c013d3d368f203387ae97cb6a2bb618f74088a215e05e717bcbc6a38b` |

重算只需Python标准库；stage计数需NumPy。计算方式为先按family/H筛选每个run的final[mode].episodes，再对success求平均；配对差按上述完整任务身份匹配；跨seed标准差使用statistics.stdev。NPZ按metadata中block顺序读取block{i}_stage，累计stage==H−1。
