# 四类原因 × 选择器：进度与结果

- 数据源：`results/four_causes_20261118`（per-run record.json (batch in progress)）
- 已完成：**31** / 48　→ 仍在运行
- 场景实例（bank）：[0, 1]；原因：['geometry_bias', 'coverage_deficit', 'recovery_deficit']；选择器：['fixed', 'rule', 'random', 'round_robin', 'bandit', 'oracle']

## 1. 最终 SR（按原因；括号内为该格样本数）

| 选择器 | geometry_bias | coverage_deficit | recovery_deficit | 平均 |
|---|---:|---:|---:|---:|
| fixed | 0.775 (2) | 0.675 (2) | 0.850 (2) | 0.767 |
| rule | 0.875 (2) | 0.925 (2) | 0.950 (1) | 0.910 |
| random | 0.450 (2) | 0.925 (2) | 1.000 (1) | 0.750 |
| round_robin | 0.875 (2) | 0.825 (2) | 0.650 (1) | 0.810 |
| bandit | 0.925 (2) | 1.000 (2) | 0.850 (1) | 0.940 |
| oracle | 0.925 (2) | 0.950 (2) | 0.900 (1) | 0.930 |

## 2. 首步选对率（按原因）

| 选择器 | geometry_bias | coverage_deficit | recovery_deficit | 合计 |
|---|---:|---:|---:|---:|
| fixed | 2/2 | 0/2 | 0/2 | 2/6 |
| rule | 2/2 | 2/2 | 0/1 | 4/5 |
| random | 0/2 | 0/2 | 0/1 | 0/5 |
| round_robin | 2/2 | 0/2 | 0/1 | 2/5 |
| bandit | 2/2 | 0/2 | 0/1 | 2/5 |
| oracle | 2/2 | 2/2 | 1/1 | 5/5 |

## 3. 首步操作分布（行=真实原因，列=第一步选的操作）

### fixed

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit | 100% |  |  |  |  |  |  |
| recovery_deficit | 100% |  |  |  |  |  |  |

### rule

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit |  |  | 100% |  |  |  |  |
| recovery_deficit |  |  |  |  |  | 100% |  |

### random

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias |  |  |  |  |  | 100% |  |
| coverage_deficit |  |  |  |  |  | 100% |  |
| recovery_deficit |  |  |  |  |  | 100% |  |

### round_robin

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit | 100% |  |  |  |  |  |  |
| recovery_deficit | 100% |  |  |  |  |  |  |

### bandit

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit | 100% |  |  |  |  |  |  |
| recovery_deficit | 100% |  |  |  |  |  |  |

### oracle

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit |  |  | 100% |  |  |  |  |
| recovery_deficit |  |  |  | 100% |  |  |  |

## 4. 相对 fixed 的配对差（同一 seed × 场景实例；95% bootstrap CI）

| 原因 | 选择器 | 配对数 | 平均差 (pp) | 95% CI |
|---|---|---:|---:|---|
| geometry_bias | rule | 2 | +10.0 | [+10.0, +10.0] |
| geometry_bias | random | 2 | -32.5 | [-40.0, -25.0] |
| geometry_bias | round_robin | 2 | +10.0 | [+10.0, +10.0] |
| geometry_bias | bandit | 2 | +15.0 | [+10.0, +20.0] |
| geometry_bias | oracle | 2 | +15.0 | [+15.0, +15.0] |
| coverage_deficit | rule | 2 | +25.0 | [+25.0, +25.0] |
| coverage_deficit | random | 2 | +25.0 | [+25.0, +25.0] |
| coverage_deficit | round_robin | 2 | +15.0 | [+10.0, +20.0] |
| coverage_deficit | bandit | 2 | +32.5 | [+30.0, +35.0] |
| coverage_deficit | oracle | 2 | +27.5 | [+25.0, +30.0] |

## 5. 成本（每格均值）

| 选择器 | 部署 episodes | 新增标签 | 校准次数 |
|---|---:|---:|---:|
| fixed | 72 | 80000 | 1.0 |
| rule | 72 | 104000 | 0.4 |
| random | 72 | 120000 | 0.0 |
| round_robin | 72 | 80000 | 1.0 |
| bandit | 72 | 80000 | 1.0 |
| oracle | 72 | 96000 | 0.4 |
