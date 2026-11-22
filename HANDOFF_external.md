# 外部三条线 handoff（LLM / 真机 / LIBERO）

> **完整性核验**：逐项证据与缺口见 [`DEPLOYMENT_AUDIT.md`](DEPLOYMENT_AUDIT.md)。真机的 sim/mock 路径已端到端验证；LLM 待 key；LIBERO 仍是骨架。

**一句话**：代码路径、接口、mock、测试和 MuJoCo 权重都已就绪；LLM 只差 API key，真机只差驱动/位姿源，LIBERO 只差数据/环境。

## 1. 目录与权重

| 路径 | 内容 |
|---|---|
| `weights/mujoco_4causes/` | 4 场景 ×（`pi0`、`fixed_final`）状态策略，16 MB；见 `weights/README.md` |
| `weights/models/` | `multilabel_model_heldout_paper.json`（阈值 0.3）、`value_model.json`（负结果） |
| `weights/SHA256SUMS.txt` | 全部权重校验和 |
| `src/horizon/real/` | PoseSource/Robot 协议、mock/CSV 回放、部署日志、twin 校准 |
| `src/horizon/libero/` | HDF5 数据读取、布局先验、mock 环境、ImageBC 基线 |
| `scripts/llm_pilot.py` | LLM selector 的无 rollout pilot（`--mock` 无需 key） |
| `scripts/run_real_deploy.py` | 真机部署循环（mock/CSV） |
| `scripts/train_libero.py` / `eval_libero.py` | LIBERO 训练/评测入口 |
| `configs/minimal_validation_llm.yaml` | fixed/rule/llm/oracle 的最小验证配置 |
| `tests/test_llm_planner.py` / `test_real_interfaces.py` / `test_libero_scaffold.py` | 对应测试 |

## 2. LLM（只差 key）

```bash
export ANTHROPIC_API_KEY=...
cd horizon
PYTHONPATH=src python scripts/llm_pilot.py --results results/four_causes_paper8_20261120 --limit 8
# 无 key 的管线自测：
PYTHONPATH=src python scripts/llm_pilot.py --results ... --limit 8 --mock
```

- `LLMPlannerSelector`：每轮一次调用、temperature 0、token 记账；**没有 key 会硬失败**，不会静默退回规则。
- `client=` 可注入传输（pilot/测试用）；生产用 urllib 调 Anthropic。
- 全量：`configs/minimal_validation_llm.yaml`（2 causes × 10 banks × 1 seed × [fixed, rule, llm, oracle]）。
- 成本：每轮 1 次调用；先跑 `--limit 8` 看 token，再决定正式场景数。

## 3. 真机（只差驱动/位姿源）

硬件假设：单臂 + 平行夹爪 + 1–2 相机 + AprilTag/位姿估计（状态策略，**不是图像策略**）。

```python
from horizon.real import PoseSample, run_episodes
from horizon.policy.model import NumpyMLP
from horizon.rsi.loop import load_params
from horizon.policy.features import TaskFeaturizer

mlp = NumpyMLP(load_params("weights/mujoco_4causes/geometry_fixed_final.npz"))
feat = TaskFeaturizer(scene, graph, verifier)          # 由任务 payload 构建
policy_fn = lambda s: mlp.act(feat.hier(s.to_world_state(), k))
run_episodes(robot, policy_fn, episodes=20, max_steps=400, log_path="logs/run.jsonl")
```

- 需要你实现：`Robot.reset/step/close`（驱动）、`PoseSource.read()`（tag/位姿）、安全限位。
- mock 自测：`PYTHONPATH=src python scripts/run_real_deploy.py --robot mock --episodes 1`
- CSV 回放：`--robot csv --csv logs/episode.jsonl`
- twin 校准：`horizon.real.calibrate.median_support/update_recon_bias`；完整鲁棒估计器在 `horizon.rsi.simreal.geometry_repair`。
- 注意：状态策略在 MuJoCo pack H4 上训练；真机任务必须复用同一 `scene`/`graph` payload，否则特征不匹配。

## 4. LIBERO（只差数据/环境）

```bash
# 训练（需要官方 LIBERO HDF5；权重不随仓库分发）
PYTHONPATH=src python scripts/train_libero.py --dataset /path/libero_10.hdf5 --out weights/libero/bc.pt
# 无 LIBERO 环境的 smoke
PYTHONPATH=src python scripts/eval_libero.py --policy weights/libero/bc.pt --env mock --episodes 3
```

- `dataset.py` 默认键：`data/demo_i/obs/agentview_rgb`、`obs/ee_states`、`actions`，可用 CLI 覆盖。
- `policy.py` 是小的 ImageBC（CNN）；要换 Diffusion Policy/ACT，只要保持 `fit/save/load/act` 接口。
- `twin.py`：从首帧物体 xy 估计布局先验，写入 scene payload；不读特权布局参数。
- `eval_libero.py --env libero` 目前显式要求官方 `libero` 包，并留了一个清晰的接入点。
- 权重：没有随仓库提供；必须在 LIBERO 数据上训练（磁盘/checkout 是外部阻塞）。

## 5. 测试与状态

```bash
cd horizon
PYTHONPATH=src python -m pytest -q tests/test_llm_planner.py tests/test_real_interfaces.py tests/test_libero_scaffold.py
# 12 passed, 1 skipped (h5py missing on this machine)
```

- LLM：代码/记账/无 key 硬失败/注入传输全部就绪；**只差 key**。
- 真机：接口/mock/日志/校准就绪；**只差驱动与位姿源**。
- LIBERO：数据/训练/评测/布局先验骨架就绪；**只差数据与环境，权重需训练**。
