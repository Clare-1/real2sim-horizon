# 首步正确率：LLM vs 脚本选择器（同一 4 原因网格）

48 格筛查批次（`results/four_causes_20261118`，每格 2 个 bank） + LLM pilot（20 次调用，$0.0388）

| 选择器 | geo | cov | rec | hlth | 合计 |
|---|---:|---:|---:|---:|---:|
| bandit | 2/2 | 0/2 | 0/2 | 0/2 | 2/8 |
| fixed | 2/2 | 0/2 | 0/2 | 0/2 | 2/8 |
| oracle | 2/2 | 2/2 | 2/2 | 2/2 | 8/8 |
| random | 0/2 | 0/2 | 0/2 | 0/2 | 0/8 |
| round_robin | 2/2 | 0/2 | 0/2 | 0/2 | 2/8 |
| rule | 2/2 | 2/2 | 0/2 | 2/2 | 6/8 |
| **LLM** | 5/5 | 4/5 | 0/5 | 3/5 | 12/20 |

## 最终 SR（脚本选择器；LLM 只有首步决策，没有在线 rollout）

| 选择器 | geo | cov | rec | hlth | 平均 |
|---|---:|---:|---:|---:|---:|
| bandit | 0.925 | 1.000 | 0.850 | 1.000 | 0.944 |
| fixed | 0.775 | 0.675 | 0.850 | 0.975 | 0.819 |
| oracle | 0.925 | 0.950 | 0.875 | 0.875 | 0.906 |
| random | 0.450 | 0.925 | 1.000 | 0.950 | 0.831 |
| round_robin | 0.875 | 0.825 | 0.650 | 0.850 | 0.800 |
| rule | 0.900 | 0.875 | 1.000 | 0.875 | 0.912 |

## LLM 逐条决策

- geometry b0 → `calibrate_geometry` (OK, expected `calibrate_geometry`)
- geometry b1 → `calibrate_geometry` (OK, expected `calibrate_geometry`)
- coverage b0 → `regenerate_targeted` (OK, expected `regenerate_targeted`)
- coverage b1 → `regenerate_targeted` (OK, expected `regenerate_targeted`)
- recovery b0 → `regenerate_targeted` (MISS, expected `add_recovery_data`)
- recovery b1 → `dagger_round` (MISS, expected `add_recovery_data`)
- healthy b0 → `calibrate_geometry` (MISS, expected `stop`)
- healthy b1 → `stop` (OK, expected `stop`)
- geometry b0 → `calibrate_geometry` (OK, expected `calibrate_geometry`)
- geometry b1 → `calibrate_geometry` (OK, expected `calibrate_geometry`)
- geometry b2 → `calibrate_geometry` (OK, expected `calibrate_geometry`)
- coverage b0 → `regenerate_targeted` (OK, expected `regenerate_targeted`)
- coverage b1 → `regenerate_targeted` (OK, expected `regenerate_targeted`)
- coverage b2 → `calibrate_geometry` (MISS, expected `regenerate_targeted`)
- recovery b0 → `regenerate_targeted` (MISS, expected `add_recovery_data`)
- recovery b1 → `dagger_round` (MISS, expected `add_recovery_data`)
- recovery b2 → `calibrate_geometry` (MISS, expected `add_recovery_data`)
- healthy b0 → `stop` (OK, expected `stop`)
- healthy b1 → `stop` (OK, expected `stop`)
- healthy b2 → `calibrate_geometry` (MISS, expected `stop`)
