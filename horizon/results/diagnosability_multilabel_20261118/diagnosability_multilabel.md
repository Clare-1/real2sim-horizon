# Multi-label diagnosability (bank-grouped leave-one-out)

profiles=184  groups=156  threshold=0.3

| threshold | geometry_bias | coverage_deficit | recovery_deficit | exact | any mix | all mix |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 95%/77% | 100%/93% | 90%/56% | 72% | 100% | 67% |
| 0.3 | 95%/85% | 100%/95% | 90%/68% | 82% | 100% | 72% |
| 0.4 | 93%/90% | 100%/95% | 87%/76% | 85% | 100% | 61% |
| 0.5 | 89%/92% | 100%/99% | 80%/80% | 86% | 100% | 39% |
| 0.6 | 87%/93% | 99%/99% | 73%/88% | 86% | 100% | 28% |

（每格为 recall / precision）

**selected threshold = 0.3**（按 exact-set，再按全混合命中）

| label | n | recall | precision |
|---|---:|---:|---:|
| geometry_bias | 75 | 95% | 85% |
| coverage_deficit | 75 | 100% | 95% |
| recovery_deficit | 30 | 90% | 68% |

exact-set accuracy (incl. healthy): 82%
mixtures n=18: any component hit 100%, all components hit 72%

