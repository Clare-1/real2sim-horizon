# 四类原因 × 选择器：进度与结果

- 数据源：`results/four_causes_paper8_20261120`（summary.json）
- 已完成：**288** / 288　→ **已完成**
- 场景实例（bank）：[0, 1, 2]；原因：['geometry_bias', 'coverage_deficit', 'recovery_deficit', 'healthy']；选择器：['fixed', 'rule', 'random', 'round_robin', 'bandit', 'oracle', 'linucb', 'multilabel']

## 1. 最终 SR（按原因；括号内为该格样本数）

| 选择器 | geometry_bias | coverage_deficit | recovery_deficit | healthy | 平均 |
|---|---:|---:|---:|---:|---:|
| fixed | 0.772 (9) | 0.600 (9) | 0.833 (9) | 0.911 (9) | 0.779 |
| rule | 0.900 (9) | 0.928 (9) | 0.900 (9) | 0.928 (9) | 0.914 |
| random | 0.300 (9) | 0.939 (9) | 0.867 (9) | 0.972 (9) | 0.769 |
| round_robin | 0.894 (9) | 0.872 (9) | 0.822 (9) | 0.939 (9) | 0.882 |
| bandit | 0.961 (9) | 0.983 (9) | 0.817 (9) | 0.978 (9) | 0.935 |
| oracle | 0.939 (9) | 0.972 (9) | 0.939 (9) | 0.844 (9) | 0.924 |
| linucb | 0.939 (9) | 0.917 (9) | 0.356 (9) | 0.961 (9) | 0.793 |
| multilabel | 0.828 (9) | 0.928 (9) | 0.828 (9) | 0.883 (9) | 0.867 |

## 2. 首步选对率（按原因）

| 选择器 | geometry_bias | coverage_deficit | recovery_deficit | healthy | 合计 |
|---|---:|---:|---:|---:|---:|
| fixed | 9/9 | 0/9 | 0/9 | 0/9 | 9/36 |
| rule | 9/9 | 9/9 | 0/9 | 3/9 | 21/36 |
| random | 0/9 | 0/9 | 0/9 | 0/9 | 0/36 |
| round_robin | 9/9 | 0/9 | 0/9 | 0/9 | 9/36 |
| bandit | 9/9 | 0/9 | 0/9 | 0/9 | 9/36 |
| oracle | 9/9 | 9/9 | 9/9 | 9/9 | 36/36 |
| linucb | 9/9 | 0/9 | 0/9 | 0/9 | 9/36 |
| multilabel | 9/9 | 9/9 | 5/9 | 5/9 | 28/36 |

## 3. 首步操作分布（行=真实原因，列=第一步选的操作）

### fixed

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit | 100% |  |  |  |  |  |  |
| recovery_deficit | 100% |  |  |  |  |  |  |
| healthy | 100% |  |  |  |  |  |  |

### rule

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit |  |  | 100% |  |  |  |  |
| recovery_deficit |  |  | 33% |  |  | 67% |  |
| healthy |  |  | 44% |  |  | 22% | 33% |

### random

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias |  |  |  |  |  | 100% |  |
| coverage_deficit |  |  |  |  |  | 100% |  |
| recovery_deficit |  |  |  |  |  | 100% |  |
| healthy |  |  |  |  |  | 100% |  |

### round_robin

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit | 100% |  |  |  |  |  |  |
| recovery_deficit | 100% |  |  |  |  |  |  |
| healthy | 100% |  |  |  |  |  |  |

### bandit

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit | 100% |  |  |  |  |  |  |
| recovery_deficit | 100% |  |  |  |  |  |  |
| healthy | 100% |  |  |  |  |  |  |

### oracle

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit |  |  | 100% |  |  |  |  |
| recovery_deficit |  |  |  | 100% |  |  |  |
| healthy |  |  |  |  |  |  | 100% |

### linucb

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit | 100% |  |  |  |  |  |  |
| recovery_deficit | 100% |  |  |  |  |  |  |
| healthy | 100% |  |  |  |  |  |  |

### multilabel

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit |  |  | 100% |  |  |  |  |
| recovery_deficit | 11% |  |  | 56% |  | 33% |  |
| healthy |  |  |  | 22% |  | 22% | 56% |

## 4. 相对 fixed 的配对差（同一 seed × 场景实例；95% bootstrap CI）

| 原因 | 选择器 | 配对数 | 平均差 (pp) | 95% CI |
|---|---|---:|---:|---|
| geometry_bias | rule | 9 | +12.8 | [+6.1, +19.4] |
| geometry_bias | random | 9 | -47.2 | [-55.0, -40.0] |
| geometry_bias | round_robin | 9 | +12.2 | [+5.6, +18.9] |
| geometry_bias | bandit | 9 | +18.9 | [+12.2, +25.6] |
| geometry_bias | oracle | 9 | +16.7 | [+10.0, +23.3] |
| geometry_bias | linucb | 9 | +16.7 | [+10.0, +23.3] |
| geometry_bias | multilabel | 9 | +5.6 | [+1.1, +10.6] |
| coverage_deficit | rule | 9 | +32.8 | [+24.4, +41.1] |
| coverage_deficit | random | 9 | +33.9 | [+27.2, +41.1] |
| coverage_deficit | round_robin | 9 | +27.2 | [+15.0, +40.0] |
| coverage_deficit | bandit | 9 | +38.3 | [+31.1, +45.6] |
| coverage_deficit | oracle | 9 | +37.2 | [+30.0, +45.6] |
| coverage_deficit | linucb | 9 | +31.7 | [+23.3, +40.6] |
| coverage_deficit | multilabel | 9 | +32.8 | [+26.1, +40.0] |
| recovery_deficit | rule | 9 | +6.7 | [+2.2, +11.1] |
| recovery_deficit | random | 9 | +3.3 | [-3.3, +10.0] |
| recovery_deficit | round_robin | 9 | -1.1 | [-9.4, +5.6] |
| recovery_deficit | bandit | 9 | -1.7 | [-5.6, +2.2] |
| recovery_deficit | oracle | 9 | +10.6 | [+5.0, +16.1] |
| recovery_deficit | linucb | 9 | -47.8 | [-57.2, -38.3] |
| recovery_deficit | multilabel | 9 | -0.6 | [-5.6, +4.4] |
| healthy | rule | 9 | +1.7 | [-3.9, +7.2] |
| healthy | random | 9 | +6.1 | [+1.7, +10.6] |
| healthy | round_robin | 9 | +2.8 | [-6.1, +10.6] |
| healthy | bandit | 9 | +6.7 | [+2.8, +10.6] |
| healthy | oracle | 9 | -6.7 | [-11.1, -2.2] |
| healthy | linucb | 9 | +5.0 | [-0.6, +10.6] |
| healthy | multilabel | 9 | -2.8 | [-8.3, +3.3] |

## 5. 成本（每格均值）

| 选择器 | 部署 episodes | 新增标签 | 校准次数 |
|---|---:|---:|---:|
| fixed | 72 | 80000 | 1.0 |
| rule | 64 | 81111 | 0.2 |
| random | 71 | 114444 | 0.0 |
| round_robin | 72 | 77778 | 1.0 |
| bandit | 72 | 77778 | 1.0 |
| oracle | 60 | 74444 | 0.2 |
| linucb | 72 | 78889 | 1.0 |
| multilabel | 59 | 55556 | 0.3 |
