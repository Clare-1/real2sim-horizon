# 图 R1　配方 × 原因（最终 SR）

| 配方 | geometry_bias | coverage_deficit | recovery_deficit | healthy | 平均 |
|---|---:|---:|---:|---:|---:|
| fixed | 0.772 | 0.600 | 0.833 | 0.911 | 0.779 |
| rule | 0.900 | 0.928 | 0.900 | 0.928 | 0.914 |
| random | 0.300 | 0.939 | 0.867 | 0.972 | 0.769 |
| round_robin | 0.894 | 0.872 | 0.822 | 0.939 | 0.882 |
| bandit | 0.961 | 0.983 | 0.817 | 0.978 | 0.935 |
| linucb | 0.939 | 0.917 | 0.356 | 0.961 | 0.793 |
| multilabel | 0.828 | 0.928 | 0.828 | 0.883 | 0.867 |
| oracle | 0.939 | 0.972 | 0.939 | 0.844 | 0.924 |

没有任何一行在所有列上都最好：fixed 在 geometry 为 0.77，在 coverage 只有 0.60；multilabel 的 coverage 高但 geometry/recovery/healthy 不是最高。
