# 三条线部署完整性审计（2026-11-22）

**结论：没有全部部署完整。** 三条线都到了“可开工的代码 + 权重”程度，但只有真机的 **sim/mock 路径**端到端验证；LLM 缺真实调用验证，LIBERO 仍是骨架。

## 总表

| 线 | 已验证（附证据） | 仍缺 | 判定 |
|---|---|---|---|
| **LLM** | prompt 无泄漏、parser 拒绝非法动作、无 key 硬失败、`client=` 注入、mock pilot 4 次决策 0 调用（`tests/test_llm_planner.py`，4 passed） | 真实 Anthropic 调用（无 key）、retry/backoff、成本上限、**LLM + 经验**变体、模型名有效性 | **pilot-ready，未生产验证** |
| **真机** | 接口、JSONL 回放、部署日志、校准 helper（`tests/test_real_interfaces.py`，4 passed）；**task payload → TaskInstance → HierarchicalController → TaskFeaturizer → NumpyMLP** 端到端 mock：`policy_dim=55`，7 步，success 1/1 | 机器人驱动（reset/step/close）、AprilTag/位姿源 + 相机→基座标定、夹爪动作映射、安全限位、真实成功判定（Verifier）、与归因兼容的完整 EpisodeLog | **sim/mock 完整；硬件适配层缺失（外部）** |
| **LIBERO** | HDF5 读取（合成数据测试）、布局先验、MockLiberoEnv、ImageBC fit/save/load/act、train/eval mock（`tests/test_libero_scaffold.py`，4 passed） | 官方 LIBERO 环境适配（`--env libero` 目前显式抛错）、LIBERO-PRO 扰动、DP/ACT（现在只是小 CNN）、**训练权重**、新布局演示生成（MimicGen/脚本专家） | **骨架，不能直接跑实验三** |

## LLM：为什么只能算 pilot-ready

已完成：

```bash
PYTHONPATH=src python scripts/llm_pilot.py --results results/four_causes_paper8_20261120 --limit 4 --mock
# 4 decisions, usage 全 0，无 API 调用
```

缺口：
1. **没有真实 API 调用**：`ANTHROPIC_API_KEY` 未配置；模型名 `claude-sonnet-4-20250514` 未在真实端点上校验。
2. 没有 retry/backoff、超时重试、成本上限（advisor 要求 API ≤1000 美元）。
3. 没有 **LLM + 经验**变体（导师表 2 的一行）；目前只有无经验 LLM。
4. 没有把对话/usage 落成正式结果文件（pilot 只有 JSON）。

最小验收：有 key 后跑 8 个 profile，确认 usage/成本/动作合法；再跑 `configs/minimal_validation_llm.yaml` 的 10×1。

## 真机：sim/mock 端到端已通，硬件仍是外部

已完成（仓库内可复现）：

```bash
cd horizon
PYTHONPATH=src python scripts/run_real_deploy.py --robot mock \
  --task ../weights/examples/geometry_task.json \
  --policy ../weights/mujoco_4causes/geometry_pi0.npz \
  --episodes 1 --max-steps 8 --out /tmp/real_task.jsonl
# task=pack_H4_s81002127 objects=4 policy_dim=55
# episodes=1 steps=7 success=1/1
```

这证明：任务 payload 能还原 `TaskInstance`，`HierarchicalController` 能选子目标，`TaskFeaturizer` 能生成 55 维特征，`NumpyMLP` 能出动作。

缺口（都需要机器人/相机）：
1. 实现 `Robot.reset/step/close`（UR/Franka/other SDK）与安全限位。
2. 实现 `PoseSource.read()`：AprilTag/位姿估计 + **相机→机器人基座**外参标定。
3. 夹爪动作映射（增量 xyz/yaw/开合 → 厂商命令）。
4. 真实成功判定：目前 mock 用最后一帧 `success=True`；真机要在 perceived state 上跑 `Verifier`。
5. 日志只存了 `grip_pos/obj_pos/success`，还没有 `EpisodeLog` 的完整字段（`events/held_est/action`），归因与 `geometry_repair` 需要补齐。

## LIBERO：还是骨架

已完成：

```bash
PYTHONPATH=src python scripts/train_libero.py --dataset <hdf5> --out weights/libero/bc.pt
PYTHONPATH=src python scripts/eval_libero.py --policy weights/libero/bc.pt --env mock --episodes 3
```

缺口：
1. `eval_libero.py --env libero` 目前显式 `raise SystemExit`，没有接官方 `OffScreenRenderEnv` / `LIBERO-PRO`。
2. 没有 DP/ACT；`ImageBC` 只是可跑的小 CNN，不能替代导师要求的 DP。
3. **没有权重**：训练需要官方 LIBERO 数据（磁盘/checkout 阻塞）；`weights/libero/` 为空。
4. 没有新布局演示生成（实验三需要专家/MimicGen），`twin.layout_prior` 只更新采样先验，不生成数据。

## 现在可以交接什么

- **LLM**：代码、pilot、配置、测试；接 key 即可先跑 8-profile pilot。
- **真机**：sim/mock 端到端 + 状态权重 + 示例 payload；接驱动/位姿源即可上真机。
- **LIBERO**：数据读取/训练/评测骨架 + 合成测试；接 LIBERO 数据/环境后先训 ImageBC，DP 需另接。

**不要把“有文件”当成“部署完整”**：上表第三列才是每条线开工前必须补的适配层。
