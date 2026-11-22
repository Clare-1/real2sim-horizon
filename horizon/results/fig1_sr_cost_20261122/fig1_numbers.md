# 图 1（当前 paper8）　SR vs 累计部署

| cause | selector | 0 | 24 | 48 | 72 |
|---|---|---:|---:|---:|---:|
| geometry_bias | fixed | 0.328 | 0.694 | 0.756 | 0.772 |
| geometry_bias | rule | 0.328 | 0.694 | 0.922 | 0.900 |
| geometry_bias | bandit | 0.328 | 0.694 | 0.950 | 0.961 |
| geometry_bias | linucb | 0.328 | 0.694 | 0.950 | 0.939 |
| geometry_bias | multilabel | 0.328 | 0.694 | 0.811 | 0.828 |
| geometry_bias | oracle | 0.328 | 0.694 | 0.950 | 0.939 |
| coverage_deficit | fixed | 0.083 | 0.083 | 0.339 | 0.600 |
| coverage_deficit | rule | 0.083 | 0.911 | 0.883 | 0.928 |
| coverage_deficit | bandit | 0.083 | 0.083 | 0.906 | 0.983 |
| coverage_deficit | linucb | 0.083 | 0.083 | 0.083 | 0.917 |
| coverage_deficit | multilabel | 0.083 | 0.911 | 0.894 | 0.928 |
| coverage_deficit | oracle | 0.083 | 0.911 | 0.950 | 0.972 |
| recovery_deficit | fixed | 0.350 | 0.350 | 0.767 | 0.833 |
| recovery_deficit | rule | 0.350 | 0.589 | 0.778 | 0.900 |
| recovery_deficit | bandit | 0.350 | 0.350 | 0.428 | 0.817 |
| recovery_deficit | linucb | 0.350 | 0.350 | 0.428 | 0.356 |
| recovery_deficit | multilabel | 0.350 | 0.683 | 0.828 | 0.828 |
| recovery_deficit | oracle | 0.350 | 0.783 | 0.917 | 0.939 |
| healthy | fixed | 0.844 | 0.844 | 0.922 | 0.911 |
| healthy | rule | 0.844 | 0.911 | 0.917 | 0.928 |
| healthy | bandit | 0.844 | 0.844 | 0.933 | 0.978 |
| healthy | linucb | 0.844 | 0.844 | 0.933 | 0.961 |
| healthy | multilabel | 0.844 | 0.894 | 0.883 | 0.883 |
| healthy | oracle | 0.844 | 0.844 | 0.844 | 0.844 |
