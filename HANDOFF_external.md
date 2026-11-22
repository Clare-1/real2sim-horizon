# 外部三条线 handoff（LLM / 真机 / LIBERO）

> **完整性核验**：逐项证据与缺口见 [`DEPLOYMENT_AUDIT.md`](DEPLOYMENT_AUDIT.md)。真机的 sim/mock 路径已端到端验证；LLM 已用真实端点跑通（20 次调用）；LIBERO 仍是骨架。

**一句话**：代码路径、接口、mock、测试和 MuJoCo 权重都已就绪；LLM 已接通真实端点（只差论文级网格），真机只差驱动/位姿源，LIBERO 只差数据/环境。

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

## 2. LLM（真实端点已跑通；下一步是论文级网格）

**已验证的端点**：`https://minara.cloud/v1`，模型 `openai/gpt-6.2-astra`（模型名必须带厂商前缀，否则 404 `model_not_found`）。实测约 **$18/Mtok**，20 次调用共 $0.039。

```bash
# 已验证的真实调用（凭据在仓库外，chmod 600）
cd horizon
PYTHONPATH=src python scripts/llm_pilot.py --provider openai --env-file ~/.config/horizon/llm.env --probe
PYTHONPATH=src python scripts/llm_pilot.py --provider openai --env-file ~/.config/horizon/llm.env \
    --results results/four_causes_paper8_20261120 --per-cause 3 --max-usd 0.5
PYTHONPATH=src python scripts/report_llm_pilot.py     # LLM vs 脚本选择器同表

# 方式 A：Anthropic 原生（若换回官方端点）
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=src python scripts/llm_pilot.py --results results/four_causes_paper8_20261120 --limit 8

# 无 key 的管线自测：
PYTHONPATH=src python scripts/llm_pilot.py --results ... --limit 8 --mock
```

实测结果（`results/llm_vs_rule_20261122.md`）：20 次调用，首步 **12/20**；几何 5/5、覆盖 4/5、健康 3/5、**恢复 0/5**（与规则一致——观测通道里没有恢复信号）。

- `LLMPlannerSelector`：每轮一次调用、temperature 0、token 记账；**没有 key 会硬失败**，不会静默退回规则。
- `provider="openai"` 走 `/chat/completions`，`provider="anthropic"` 走 `/v1/messages`；`client=` 可注入传输。
- key 只放在仓库外（`~/.config/horizon/llm.env`，chmod 600），**不要提交**。
- 全量：`configs/minimal_validation_llm.yaml`（2 causes × 10 banks × 1 seed × [fixed, rule, llm, oracle]）。
- 成本：每轮 1 次调用；先跑 `--limit 8` 看 token，再决定正式场景数。

## 3. 真机：松灵 AgileX Piper（代码已到位，只差实测标定）

适配器已实现，不需要你写驱动：

| 文件 | 内容 |
|---|---|
| `horizon/src/horizon/real/piper.py` | `PiperArm`（piper_sdk 封装，单位换算）+ `PiperRobot`（与仿真一致的动作空间） |
| `horizon/src/horizon/real/apriltag.py` | `AprilTagPoseSource`（tag → 物体位姿）+ `StaticPoseSource`（固定布局） |
| `weights/examples/piper_tag_map.json` | `{tag_id: object_index}` 示例 |
| `weights/examples/piper_cam_calib.json` | 相机内参 + 相机→基座外参示例 |
| `horizon/tests/test_piper_adapter.py` | 6 个测试（单位换算、步长上限、夹爪方向、无 SDK 报错） |

上机步骤（真机那侧执行）：

```bash
# 1) CAN（Piper 固定 1 Mbps）
sudo ip link set up can0 type can bitrate 1000000
candump can0            # 能看到反馈帧才继续

# 2) SDK
pip install piper_sdk

# 3) 先只读位姿、不动作（安全）
python - <<'PY'
from horizon.real.piper import PiperArm
arm = PiperArm().connect(); arm.enable()
print(arm.end_pose())          # (x,y,z) m, (rx,ry,rz) rad
PY

# 4) 标定 tag 与相机外参后跑一次部署
PYTHONPATH=src python scripts/run_real_deploy.py \
    --robot piper --can can0 \
    --tag-map weights/examples/piper_tag_map.json \
    --cam-calib weights/examples/piper_cam_calib.json \
    --task weights/examples/geometry_task.json \
    --policy weights/mujoco_4causes/geometry_pi0.npz \
    --episodes 1 --max-steps 200 --out logs/piper_run.jsonl
```

关键约定（与仿真完全一致，改一处就会不匹配）：

- 动作 `[dx, dy, dz, dyaw, grip] ∈ [-1,1]`；单步最大 `max_step=15 mm`、`max_yaw_step=0.15 rad`；
- 夹爪 `grip=+1` 闭合、`-1` 全开（`gripper_open_m=70 mm`）；
- 位置单位 0.001 mm、角度单位 0.001°（`PiperArm` 内部换算，上层只用 SI）；
- **必须实测**：`home_xyz`、`home_rpy_deg`（Piper 竖直抓取常用 `(0,60,0)°`）、`workspace_lo/hi`、相机外参、tag 尺寸。

安全：`PiperRobot.reset()` 会先 `enable()` 再回 home；第一次上电请把手放在急停上，并把 `speed_pct` 降到 10。

通用接口（如果换其它臂，只实现这两个协议）：

```python
class Robot(Protocol):   # horizon/real/interfaces.py
    def reset(self) -> PoseSample: ...
    def step(self, action: np.ndarray) -> PoseSample: ...
    def close(self) -> None: ...
class PoseSource(Protocol):
    def read(self) -> PoseSample: ...
```

- mock 自测：`PYTHONPATH=src python scripts/run_real_deploy.py --robot mock --episodes 1`
- CSV 回放：`--robot csv --csv logs/episode.jsonl`
- twin 校准：`horizon.real.calibrate.median_support/update_recon_bias`；鲁棒估计器在 `horizon.rsi.simreal.geometry_repair`。
- 注意：状态策略在 MuJoCo pack H4 上训练；真机任务必须复用同一 `scene`/`graph` payload，否则特征维度不匹配。

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

- LLM：**真实端点已跑通**（20 次调用 $0.039，首步 12/20）；下一步是论文级网格。
- 真机：接口/mock/日志/校准就绪；**只差驱动与位姿源**。
- LIBERO：数据/训练/评测/布局先验骨架就绪；**只差数据与环境，权重需训练**。
