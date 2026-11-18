# 复现与实现核验

本批计划在主实验结果产生之前冻结，见 [PLAN.md](PLAN.md)。运行的源码、配置分别保存在 `fullH_dagger/source`、`fullH_nominal/source`、`fullH_bias/source`。不能在这些目录里修改代码后继续同一个实验；不同实现或配置须使用新输出目录。

## Windows 运行环境

工作目录为 `C:\Users\WYM\Desktop\real2sim\horizon`。本批使用已验证的 bundled Python 与项目 `.venv` 依赖，MuJoCo 3.16.0、Torch 2.14.1+cu130、RTX 5060 Laptop 8GB。CPU rollout 采用 NumPy 推理，BC learner 使用 CUDA。三个 runner 的 worker 数为 4 / 4 / 12；并发训练时间不是受控的速度比较。

运行从10月28日跨至10月31日，恢复核对时原进程继续执行。跨日间隔与耗时解释见 [运行时说明](RUNTIME_NOTE.md)；不把累计墙钟时长当作连续计算时间。

下面以健康 DAgger 为例。健康恢复专家数据组替换 `dagger` 为 `nominal`（4 workers），偏差组替换为 `bias`（12 workers）。已有目录只允许同配置、同源码、同 workers 的恢复；完整目录会核验并跳过完成项。

```powershell
$outDir = 'results\long_horizon_replication_20261028\fullH_dagger'
$pythonExe = 'C:\Users\WYM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH="$outDir\source\src;.venv\Lib\site-packages"
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
& $pythonExe -B "$outDir\source\scripts\run_harness.py" --config "$outDir\source\configs\harness_long_horizon_replication_dagger.yaml" --out $outDir --workers 4 --project-root .
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed' }
& $pythonExe -B "$outDir\source\scripts\validate_harness_artifacts.py" $outDir
if ($LASTEXITCODE -ne 0) { throw 'Artifact validation failed' }
& $pythonExe -B "$outDir\source\scripts\report_harness.py" $outDir --no-plots
if ($LASTEXITCODE -ne 0) { throw 'Report failed' }
```

三个目录均完成、独立核验通过之后，才能生成联合报告：

```powershell
$env:PYTHONPATH='src;.venv\Lib\site-packages'
& $pythonExe -B results/long_horizon_replication_20261028/report_source/report_long_horizon_replication.py results/long_horizon_replication_20261028
if ($LASTEXITCODE -ne 0) { throw 'Combined report failed' }
```

联合报告的四个脚本冻结在 `report_source`，自身也记录在最终 `comparison.json` 的输入哈希中。独立验证器必须与原核验时的版本相同；不能仅依赖一份陈旧的通过标志。产物包含数据、策略、任务、反馈观测、训练统计、阶段覆盖、实际物理步数和配对结果；没有读取保留确认库。

## 实现验证与额外诊断

- `smoke_uncached`、`smoke_cached`、`smoke_observed` 是微预算真实仿真集成检查，均独立核验通过。前两者初始/最终策略、各轮数据、逐 episode 成绩和 belief 一致，复用节省 8 次实际 rollout。这里只验证实现，不评价方法性能。
- `training_observation_mode: legacy` 在所有主实验中显式固定。新 `observed` 选项保存控制器实际收到的 gripper observation；没有额外观测随机数调用，不改变既有几何反馈通道。
- [新旧默认路径对照](legacy_compatibility/compatibility.json) 在固定 pack/H2 任务上检查 p=0/.3、新旧各40控制步：12个历史日志数组与5个训练数组逐字节一致，传感器读取次数及环境/扰动/控制器 RNG 终态一致；p=.3 的 slip 事件也相同。[可复现脚本](legacy_compatibility/check_legacy_compatibility.py) 拒绝覆盖既有结果。额外成本4次 rollout、160控制步、4,600物理步，不是性能实验或穷尽等价性证明。
- [混合控制器诊断](../mixed_termination_20261028/README.md) 使用旧冻结策略和独立 88M bank，90 次 rollout 未发生严格定义的连续错误提前终止。这个有限诊断不支持当前修改控制器，也不证明所有分布下都安全。
- [本批任务库专家检查](expert_feasibility/README.md) 先冻结计划，再用原特权状态专家执行与主实验哈希一致的300个任务、两种扰动条件，共600次；名义300/300、扰动299/300成功，成本7,762,750物理步，独立核验通过。它是另计成本的可执行性诊断，不是观测匹配的学习基线。唯一失败完整保留。
- `source_before` 保留修改前实现。历史单种子结果仍在 `../long_horizon_dev_20261028`，不与本批合并为额外训练种子。

完整回归 **204/204 通过**，耗时 26.85 秒，见 [tests_final.log](tests_final.log)；其中报告专项 30 项覆盖真实 PNG/SVG 导出。主实验性能结果只在完整矩阵通过核验后汇总，不以正在运行时的日志预览选择策略或轮次。
