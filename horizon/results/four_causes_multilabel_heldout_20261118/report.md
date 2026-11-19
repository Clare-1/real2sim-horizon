# 四类原因 × 选择器：进度与结果

- 数据源：`results/four_causes_multilabel_heldout_20261118`（summary.json）
- 已完成：**8** / 8　→ **已完成**
- 场景实例（bank）：[0, 1]；原因：['geometry_bias', 'coverage_deficit', 'recovery_deficit', 'healthy']；选择器：['multilabel']

## 1. 最终 SR（按原因；括号内为该格样本数）

| 选择器 | geometry_bias | coverage_deficit | recovery_deficit | healthy | 平均 |
|---|---:|---:|---:|---:|---:|
| multilabel | 0.900 (2) | 0.925 (2) | 0.800 (2) | 0.875 (2) | 0.875 |

## 2. 首步选对率（按原因）

| 选择器 | geometry_bias | coverage_deficit | recovery_deficit | healthy | 合计 |
|---|---:|---:|---:|---:|---:|
| multilabel | 2/2 | 2/2 | 2/2 | 1/2 | 7/8 |

## 3. 首步操作分布（行=真实原因，列=第一步选的操作）

### multilabel

| 真实原因 | calibrate_geometry | regenerate_uniform | regenerate_targeted | add_recovery_data | randomize_physics | dagger_round | stop |
|---|---:|---:|---:|---:|---:|---:|---:|
| geometry_bias | 100% |  |  |  |  |  |  |
| coverage_deficit |  |  | 100% |  |  |  |  |
| recovery_deficit |  |  |  | 100% |  |  |  |
| healthy |  |  |  | 50% |  |  | 50% |

## 4. 相对 fixed 的配对差（同一 seed × 场景实例；95% bootstrap CI）

| 原因 | 选择器 | 配对数 | 平均差 (pp) | 95% CI |
|---|---|---:|---:|---|

## 5. 成本（每格均值）

| 选择器 | 部署 episodes | 新增标签 | 校准次数 |
|---|---:|---:|---:|
| multilabel | 60 | 55000 | 0.2 |
