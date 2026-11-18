# 恢复信号探针（v2 指标）

`legacy` = 当前批次实际写入的值；v2 = 从轨迹事件实测的值。

| scenario | policy | SR | armed eps | events | **P(fail\|armed)** | P(armed\|fail) | legacy rate/triggers |
|---|---|---:|---:|---:|---:|---:|---:|
| geometry | `/home/aurora/real2sim/horizon/results/four_causes_20261118/runs/geometry/bank0/fixed/s0/policy_3.npz` | 0.167 | 24 | 10 | **0.833** | 1.000 | 0.000 / 0 |
| coverage | `/home/aurora/real2sim/horizon/results/four_causes_20261118/runs/coverage/bank0/fixed/s0/policy_3.npz` | 0.542 | 24 | 14 | **0.458** | 1.000 | 0.000 / 0 |
| recovery | `/home/aurora/real2sim/horizon/results/four_causes_20261118/runs/recovery/bank0/fixed/s0/policy_3.npz` | 0.750 | 24 | 82 | **0.250** | 1.000 | 0.000 / 0 |

失败类型（armed 且失败）：

- geometry: {'placement': 9, 'stall': 8, 'transition': 2, 'grasp': 1}
- coverage: {'placement': 3, 'transition': 6, 'grasp': 2}
- recovery: {'placement': 3, 'transition': 2, 'grasp': 1}
