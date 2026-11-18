# Multi-label diagnosability (bank-grouped leave-one-out)

profiles=178  groups=150  threshold=0.3

| threshold | geometry_bias | coverage_deficit | recovery_deficit | exact | any mix | all mix |
|---|---:|---:|---:|---:|---:|---:|
| 0.2 | 97%/78% | 100%/92% | 100%/69% | 78% | 100% | 83% |
| 0.3 | 94%/86% | 100%/95% | 100%/80% | 86% | 100% | 67% |
| 0.4 | 92%/89% | 100%/95% | 100%/86% | 88% | 100% | 50% |
| 0.5 | 92%/90% | 100%/99% | 96%/85% | 89% | 100% | 50% |
| 0.6 | 86%/92% | 99%/99% | 83%/91% | 88% | 100% | 17% |

（每格为 recall / precision）

**selected threshold = 0.3**（按 exact-set，再按全混合命中）

| label | n | recall | precision |
|---|---:|---:|---:|
| geometry_bias | 71 | 94% | 86% |
| coverage_deficit | 71 | 100% | 95% |
| recovery_deficit | 24 | 100% | 80% |

exact-set accuracy (incl. healthy): 86%
mixtures n=12: any component hit 100%, all components hit 67%

