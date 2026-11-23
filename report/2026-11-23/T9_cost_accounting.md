# T9 成本口径：标签 vs 部署，以及盈亏平衡点

每格 9 个实例（3 seeds x 3 banks）。`new_labels` 只算选择器自己产出的标签（初始 80k 不计）。

| 原因 | 臂 | 标签/格 | 部署/格 | 最终 SR |
|---|---|---:|---:|---:|
| geometry_bias | fixed | 80000 | 72.0 | 0.772 |
| geometry_bias | rule | 66667 | 72.0 | 0.900 |
| coverage_deficit | fixed | 80000 | 72.0 | 0.600 |
| coverage_deficit | rule | 97778 | 66.7 | 0.928 |
| recovery_deficit | fixed | 80000 | 72.0 | 0.833 |
| recovery_deficit | rule | 115556 | 72.0 | 0.900 |
| healthy | fixed | 80000 | 72.0 | 0.911 |
| healthy | rule | 44444 | 45.3 | 0.928 |

`L` = 一次专家部署标注折合多少条仿真标签。总成本 = 标签 + L x 部署。
盈亏平衡 `L*` 满足 `label_rule + L*·ep_rule = label_fixed + L*·ep_fixed`，即
`L* = (label_rule - label_fixed) / (ep_fixed - ep_rule)`。

| 原因 | 规则相对固定的标签差 | 部署差 | L*（条/次） | 读法 |
|---|---:|---:|---:|---|
| geometry_bias | -13333 | +0.0 | — | 规则同时更省标签且不多部署：任何 L 下都更便宜 |
| coverage_deficit | +17778 | -5.3 | 3333 | 专家每次超过 3333 条标签时，规则反而更贵 |
| recovery_deficit | +35556 | +0.0 | — | 规则多花标签、部署相同：任何 L 下都更贵（+6.7 pp 是买来的，不是省出来的） |
| healthy | -35556 | -26.7 | — | 规则两项都省（标签少 35.6k、部署少 26.7 次）：更好且更便宜 |

注意：恢复类上规则多花 35.6k 标签、部署次数相同，+6.7 pp 是用标签换来的，
不能写成"更省部署"。覆盖率类部署少 5.3 次但多花 17.8k 标签，
只有当一次专家部署贵过约 3.3k 条仿真标签时，规则在总成本上才占优。
