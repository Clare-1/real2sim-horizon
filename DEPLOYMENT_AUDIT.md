# 三条线部署完整性审计（2026-11-22 第二版）

**结论：代码侧已闭环；外部只剩 key（LLM）、硬件驱动/位姿源（真机）、LIBERO 包/数据/DP 权重。**
上一版把“代码适配层”和“外部资源”混在一起；这一版逐项区分。

## 总表

| 线 | 代码侧已完成（有测试/ smoke） | 外部仍缺 | 判定 |
|---|---|---|---|
| **LLM** | prompt 无泄漏、parser、无 key 硬失败、`client=` 注入、**retry/backoff**、**max_calls/max_usd**、**memory（LLM+经验）**、usage 含 usd；7 tests | `ANTHROPIC_API_KEY`、真实模型名校验 | **code-side complete；key pending** |
| **真机** | PoseSource/Robot、mock/JSONL、部署日志、twin 校准、**Verifier 成功判定**、**EpisodeLog 构建**、**task→TaskInstance→HierarchicalController→TaskFeaturizer→NumpyMLP**（dim=55，7 步）；6 tests + smoke | 机器人驱动、AprilTag/位姿源 + 相机→基座外参、夹爪映射、安全限位；EpisodeLog 的 phase/valid 需人工标注 | **code-side complete；hardware pending** |
| **LIBERO** | HDF5 读取、布局先验、MockEnv、**官方 env 适配器**、**DPAdapter**、ImageBC、train/eval（`--bddl`）；6 tests | `libero` 包 + 数据 + BDDL、**训练权重**、新布局演示生成（MimicGen/专家） | **code-side complete；env/data/weights pending** |

## 已实测的命令

```bash
# LLM（无 key；client 注入 + 重试/上限）
PYTHONPATH=src python scripts/llm_pilot.py --results results/four_causes_paper8_20261120 --limit 3 --mock

# 真机（任务 payload + 真实控制器 + 权重 + Verifier 成功判定 + EpisodeLog）
PYTHONPATH=src python scripts/run_real_deploy.py --robot mock \
  --task weights/examples/geometry_task.json \
  --policy weights/mujoco_4causes/geometry_pi0.npz \
  --episodes 1 --max-steps 8 --out /tmp/real_task.jsonl --episode-out /tmp/real_episode.npz
# task=pack_H4_s81002127 objects=4 policy_dim=55; steps=7 success=1/1

# LIBERO（mock；官方 env 需 --bddl + libero 包）
PYTHONPATH=src python scripts/train_libero.py --dataset <hdf5> --out weights/libero/bc.pt
PYTHONPATH=src python scripts/eval_libero.py --policy weights/libero/bc.pt --env mock --episodes 3
```

## LLM：还差什么

- 真实 key 下的 8-profile pilot（模型名、价格、超时需在真实端点确认）。
- API 成本上限已实现（`max_usd`/`max_calls`），但价格表是占位值，接 key 后按实际模型改 `price_in_per_mtok/price_out_per_mtok`。
- “LLM + 经验”已实现为 `memory` 字段 + `llm_memory` selector；经验内容必须由开发场景产生，不能含真值原因。

## 真机：还差什么

- 实现 `Robot.reset/step/close`（厂商 SDK）与安全限位。
- 实现 `PoseSource.read()`：AprilTag/位姿估计 + 相机→基座外参；输出字段与 `PoseSample` 对齐。
- 夹爪动作映射（增量 xyz/yaw/开合 → 厂商命令）。
- `EpisodeLog` 的 `phase/valid` 目前是占位；若要用 `attribute_auto` 做细粒度归因，需要人工/自动标注。

## LIBERO：还差什么

- 安装官方 `libero` 包并准备 BDDL/数据；`LiberoEnvAdapter` 的相机 key 可能随版本不同，必要时覆盖 `_image`。
- 用 `train_libero.py` 训 ImageBC（小 CNN）或把 DP/ACT predictor 包进 `DPAdapter`。
- **没有随仓库提供 LIBERO 权重**；必须在 LIBERO 数据上训练。
- 实验三还需要“新布局演示生成”（专家或 MimicGen），当前 `twin.layout_prior` 只更新采样先验，不生成数据。

## 交接判定

- **LLM**：代码可交；给 key 即可跑 pilot。
- **真机**：代码可交；接驱动/位姿源即可上真机；sim/mock 已证明策略链路。
- **LIBERO**：代码可交；接包/数据/权重后先跑 ImageBC，DP 需另接。
