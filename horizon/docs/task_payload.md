# 显式场景任务进入 rollout worker

日期：2026-10-26。此接口是 [Agentic Learning Harness 路线](../../Agentic_Learning_Harness_路线方案.md) 的第一项工程基础。它让现有专家、学习策略和 β 混合采集执行给定的场景与任务图，不再强制从模板采样。它没有实现新的 planner，也没有证明自然重建迁移或学习性能提高。

## 接口

- `horizon.tasks.serialization.task_to_payload(task) -> dict`：导出独立的 JSON 可序列化对象。
- `task_from_payload(payload, expected_family=..., expected_H=..., expected_seed=...) -> TaskInstance`：检查 schema、身份与图结构，重算可行性报告，不自动修复或替换输入场景。
- `Job(..., task_payload=payload)`：worker 使用显式任务；不传时保持原有 `make_task(family, H, seed)` 路径。

payload schema 版本为 1，包含 `family/H/seed/instruction/scene/graph/belief/belief_graph/attempts/log`。物理场景与机器人信念分别保存；它是模拟器任务输入，不能直接作为 agent 可见反馈。`TaskInstance.controller_view()` 提供机器人视角，但完整日志的评分真值隔离仍属于后续 harness 工作。

显式 payload 的 family、H、seed 必须与 Job 对应。无效输入由 worker 返回 `JobResult.error`，不静默回退到模板。图结构无效时拒绝；几何/符号可行性问题保存在 `task.report`，调用方决定是否允许执行诊断。

现有 `world/recon/grounding/physics` 变换继续在显式 task 基础上按原顺序执行。若同时传入相应覆盖参数，它们仍按旧 worker 语义更新场景或信念；要直接执行 payload 中保存的完整系统状态，就不要再传这些覆盖参数。该接口当前没有扩展 SceneSpec 的形状、关节物体或技能类型。

## 从已有 MJCF 接入

下面的示例从 `horizon/` 目录运行，需要当前环境能够导入 MuJoCo 和 NumPy。上游 fixture 是手工合成的测试文件，只用于验证接口。

```python
from pathlib import Path

from horizon.datagen.parallel import Job, Runner
from horizon.real2sim.mjcf_import import import_mjcf
from horizon.taskgraph.compiler import TemplateCompiler
from horizon.taskgraph.feasibility import check
from horizon.tasks.factory import TaskInstance
from horizon.tasks.serialization import task_to_payload


def main():
    source = Path("../baselines/agentic_rsr/tests/fixtures/stage3/code_as_policy_tabletop.xml")
    scene = import_mjcf(source)
    instruction = (
        "Sort the objects into the bins matching their colours, "
        "in this order: blue_block, green_block."
    )
    graph = TemplateCompiler().compile(scene, instruction)
    report = check(scene, graph)
    if not report.ok:
        raise ValueError(f"Imported task is infeasible: {report}")
    task = TaskInstance(graph.family, graph.H, 0, scene, graph, instruction, report)
    job = Job(
        task.family, task.H, task.seed, seed=0,
        task_payload=task_to_payload(task), keep_log=True,
    )
    with Runner(workers=1) as runner:
        result = runner.map([job])[0]
    if result.error:
        raise RuntimeError(result.error)
    print({"success": result.success, "transitions": result.n_valid})


if __name__ == "__main__":  # Windows spawn 必须保护入口
    main()
```

该示例执行现有专家。学习策略使用原有 `Runner(policy={"params": params})` 与 `controller="hier"`；DAgger 使用原有 `controller="mixed"`、`learner`、`beta` 和专家标注参数。让 `run_method/run_simreal` 在每次采集时选用外部 task manifest，是下一步整合工作，当前接口本身没有修改这些入口。

## 验证覆盖与限制

[test_task_payload.py](../tests/test_task_payload.py) 覆盖 JSON 往返、独立 belief 与 belief_graph、元数据不匹配、无效图和非有限数值、明确不调用模板的专家 rollout、Windows spawn 与串行结果一致，以及外部场景上的 world/recon 变换。

测试使用可识别名称的外部 JSON 场景，避免“payload 被忽略但模板任务仍成功”造成假通过。成功的专家执行不等于学习策略成功，也不等于完整 Real2Sim2Real。

实际完整回归：**88 passed in 8.51s**，包括新增 15 项测试。串行与 Windows spawn 的外部专家任务均成功，动作数组逐元素相同；本轮没有训练或新性能实验。

本机 `.venv-cpu/Scripts/python.exe` 启动被 Windows 应用控制拦截。验证使用已可正常运行的 Codex Python 3.12.14，复用项目现有 CPU 依赖，没有改变系统策略或 DLL；原系统 Temp/pytest 目录有 ACL 问题，改用工作区测试临时目录。实际命令从 `horizon/` 执行：

```powershell
$env:PYTHONPATH='src;.venv-cpu\Lib\site-packages'
& 'C:\Users\WYM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q -p no:cacheprovider --basetemp=.task-payload-full-test-tmp-20261026 tests
```

这是当前机器的已验证测试环境，不代表 Torch/CUDA 或完整训练路径已在该启动方式下验证。复跑可另选新的工作区 `--basetemp`，避免复用其他测试会话的临时目录。
