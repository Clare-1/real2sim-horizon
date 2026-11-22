# 规则阈值：只用 dev banks 调，held-out banks 报

- 数据：`results/minimal_validation_advisor_20261121`（rule 记录 60 条；dev banks (0, 1, 2) = 18，held-out = 42）
- 只回放**第一步决策**；后续轮次依赖前面买到的数据，不能离线复现。

| 阈值组 | dev score | dev 首步对 | dev 误停 | held-out score | held-out 首步对 | held-out 误停 |
|---|---:|---:|---:|---:|---:|---:|
| 默认 (tau_geo=4mm, conc=0.5, sr=0.85) | +1.000 | 18 | 0 | +0.952 | 40 | 0 |
| **dev 最优** {"healthy_sr": 0.8, "tau_concentration": 0.4, "tau_geo": 0.003} | +1.000 | 18 | 0 | +0.952 | 40 | 0 |

判定：dev 最优组在 held-out 上若不高于默认组，就保留默认阈值并在论文里说明阈值是在开发集上固定、未在测试集上调。

