# 恢复信号探针（v2 指标）

`legacy` = 当前批次实际写入的值；v2 = 从轨迹事件实测的值。

| scenario | policy | SR | armed eps | events | **P(fail\|armed)** | P(armed\|fail) | legacy rate/triggers |
|---|---|---:|---:|---:|---:|---:|---:|
| geometry | `results/four_causes_20261118/runs/geometry/bank0/fixed/s0/policy_3.npz` | 0.167 | 24 | 10 | **0.833** | 1.000 | 0.000 / 0 |
| coverage | `results/four_causes_20261118/runs/coverage/bank0/fixed/s0/policy_3.npz` | 0.542 | 24 | 14 | **0.458** | 1.000 | 0.000 / 0 |
| recovery | `results/four_causes_20261118/runs/recovery/bank0/fixed/s0/policy_3.npz` | 0.708 | 24 | 104 | **0.292** | 1.000 | 0.000 / 0 |
| healthy | `results/four_causes_20261118/runs/healthy/bank0/fixed/s0/policy_3.npz` | 0.875 | 24 | 18 | **0.125** | 1.000 | 0.000 / 0 |

失败类型（armed 且失败）：

- geometry: {'placement': 9, 'stall': 8, 'transition': 2, 'grasp': 1}
- coverage: {'placement': 3, 'transition': 6, 'grasp': 2}
- recovery: {'placement': 3, 'transition': 3, 'grasp': 1}
- healthy: {'transition': 2, 'placement': 1}
