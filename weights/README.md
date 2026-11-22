# 权重清单（handoff）

本目录是给外部三条线（LLM / 真机 / LIBERO）直接开工用的权重。校验和见 `SHA256SUMS.txt`。

## MuJoCo 四类原因（状态策略，非图像）

| 文件 | 来源 | 用途 |
|---|---|---|
| `mujoco_4causes/{geometry,coverage,recovery,healthy}_pi0.npz` | `results/four_causes_paper8_20261120/shared/<scenario>/s0/policy_0.npz` | 初始策略 π0 |
| `mujoco_4causes/{scenario}_fixed_final.npz` | `.../runs/<scenario>/bank0/fixed/s0/policy_3.npz` | fixed 配方第 3 轮（校准 + DAgger） |

加载：

```python
from horizon.rsi.loop import load_params
from horizon.policy.model import NumpyMLP
policy = NumpyMLP(load_params("weights/mujoco_4causes/geometry_pi0.npz"))
action = policy.act(features)  # features = TaskFeaturizer.hier(perceived_state, k)
```

输入是 `policy/features.py` 的 `TaskFeaturizer.hier(state, k)`：物体位姿、夹爪位姿/开合、目标槽位、近邻物体；**不是图像**。真机需要用 AprilTag/位姿估计得到同样的 perceived state。

## 诊断 / 价值模型

| 文件 | 内容 | 用途 |
|---|---|---|
| `models/multilabel_model_heldout_paper.json` | 3 个二分类 logistic（geometry/coverage/recovery），阈值 0.3，训练排除 banks 0–2 | `MultiLabelSelector`；不要在这些 bank 上重训 |
| `models/value_model.json` | 40 条反事实样本的 ridge，LOO R²=0.138 | **负结果**，仅作基线，不要当主结果 |

## 没有包含的权重

- **LLM**：不需要权重，需要 `ANTHROPIC_API_KEY`。
- **LIBERO**：没有训练好的图像策略权重。`horizon/libero/` 提供数据/训练/评测骨架，权重必须在 LIBERO 数据上训练（磁盘/checkout 是外部阻塞）。
- **真机**：没有真实硬件标定文件；`horizon/real/` 提供 PoseSource/Robot 接口与 mock/CSV 回放。
