# LLM selector in the main table

LLM grid: 8 runs in `results/four_causes_llm_20261122` (4 causes x 2 banks x 1 seed).
Comparison: 288 runs in `results/four_causes_paper8_20261120` (3 seeds x 3 banks).
Both are final test SR; the LLM row is screening-level (2 banks, 1 seed).

| selector | geo | cov | rec | hlth | mean | first-step |
|---|---|---|---|---|---|---|
| bandit | 0.961 | 0.983 | 0.817 | 0.978 | 0.935 | 9/36 |
| llm | 0.925 | 0.875 | 1.000 | 0.925 | 0.931 | 5/8 |
| oracle | 0.939 | 0.972 | 0.939 | 0.844 | 0.924 | 36/36 |
| rule | 0.900 | 0.928 | 0.900 | 0.928 | 0.914 | 21/36 |
| multilabel | 0.828 | 0.928 | 0.828 | 0.883 | 0.867 | 28/36 |
| linucb | 0.939 | 0.917 | 0.356 | 0.961 | 0.793 | 9/36 |
| fixed | 0.772 | 0.600 | 0.833 | 0.911 | 0.779 | 9/36 |

## LLM cost

- runs: 8; calls: 23; spend: $0.0441 (mean $0.0019/call)
- actions per run: coverage/b0: regenerate_targeted→stop, coverage/b1: regenerate_targeted→stop, geometry/b0: calibrate_geometry→regenerate_targeted→stop, geometry/b1: calibrate_geometry→regenerate_targeted→regenerate_targeted→stop, healthy/b0: calibrate_geometry→regenerate_targeted→regenerate_targeted→stop, healthy/b1: stop, recovery/b0: regenerate_targeted→calibrate_geometry→regenerate_targeted→deploy_dagger, recovery/b1: dagger_round→regenerate_targeted→regenerate_targeted→regenerate_targeted

