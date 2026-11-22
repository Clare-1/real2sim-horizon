# 10-bank 最小验证：dev (0-2) vs held-out (3-9)

数据：`results/minimal_validation_advisor_20261121`，seed=0；dev=[0, 1, 2]，test=[3, 4, 5, 6, 7, 8, 9]

| 选择器 | 原因 | dev (0-2) | held-out (3-9) | 差 (test-dev) |
|---|---|---:|---:|---:|
| fixed | geometry_bias | 0.750 | 0.871 | +12.1 pp |
| fixed | coverage_deficit | 0.717 | 0.721 | +0.5 pp |
| rule | geometry_bias | 0.833 | 0.907 | +7.4 pp |
| rule | coverage_deficit | 0.917 | 0.929 | +1.2 pp |
| bandit | geometry_bias | 0.917 | 0.979 | +6.2 pp |
| bandit | coverage_deficit | 1.000 | 0.936 | -6.4 pp |
| oracle | geometry_bias | 0.917 | 0.986 | +6.9 pp |
| oracle | coverage_deficit | 0.967 | 0.993 | +2.6 pp |

## 逐 bank

| 选择器 | 原因 | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | geometry_bias | 0.80 | 0.75 | 0.70 | 0.85 | 0.80 | 0.80 | 0.95 | 0.90 | 0.90 | 0.90 |
| fixed | coverage_deficit | 0.70 | 0.65 | 0.80 | 0.65 | 0.80 | 0.65 | 0.80 | 0.65 | 0.70 | 0.80 |
| rule | geometry_bias | 0.95 | 0.85 | 0.70 | 0.85 | 0.95 | 1.00 | 0.75 | 0.95 | 0.90 | 0.95 |
| rule | coverage_deficit | 0.85 | 0.90 | 1.00 | 0.95 | 0.95 | 1.00 | 0.90 | 0.90 | 0.80 | 1.00 |
| bandit | geometry_bias | 1.00 | 0.85 | 0.90 | 1.00 | 1.00 | 1.00 | 0.95 | 0.95 | 0.95 | 1.00 |
| bandit | coverage_deficit | 1.00 | 1.00 | 1.00 | 0.85 | 0.95 | 0.90 | 1.00 | 0.90 | 0.95 | 1.00 |
| oracle | geometry_bias | 0.95 | 0.90 | 0.90 | 0.95 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| oracle | coverage_deficit | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 |

说明：rule 阈值是硬编码的，不是在这 3 个 dev banks 上调的；因此这里检验的是“换 bank 后排序是否保持”，不是完整的调参/测试分离。