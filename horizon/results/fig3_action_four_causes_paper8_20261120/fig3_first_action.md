# Fig. 3  first operation per cause (row = cause, column = chosen operation)

runs=288  causes=['geometry_bias', 'coverage_deficit', 'recovery_deficit', 'healthy']  selectors=['bandit', 'fixed', 'linucb', 'multilabel', 'oracle', 'random', 'round_robin', 'rule']

## all selectors pooled

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 88% |  |  |  |  | 12% |  | 72 |
| coverage_deficit | 50% |  | 38% |  |  | 12% |  | 72 |
| recovery_deficit | 51% |  | 4% | 19% |  | 25% |  | 72 |
| healthy | 50% |  | 6% | 3% |  | 18% | 24% | 72 |

## bandit

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 9 |
| coverage_deficit | 100% |  |  |  |  |  |  | 9 |
| recovery_deficit | 100% |  |  |  |  |  |  | 9 |
| healthy | 100% |  |  |  |  |  |  | 9 |

## fixed

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 9 |
| coverage_deficit | 100% |  |  |  |  |  |  | 9 |
| recovery_deficit | 100% |  |  |  |  |  |  | 9 |
| healthy | 100% |  |  |  |  |  |  | 9 |

## linucb

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 9 |
| coverage_deficit | 100% |  |  |  |  |  |  | 9 |
| recovery_deficit | 100% |  |  |  |  |  |  | 9 |
| healthy | 100% |  |  |  |  |  |  | 9 |

## multilabel

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 9 |
| coverage_deficit |  |  | 100% |  |  |  |  | 9 |
| recovery_deficit | 11% |  |  | 56% |  | 33% |  | 9 |
| healthy |  |  |  | 22% |  | 22% | 56% | 9 |

## oracle

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 9 |
| coverage_deficit |  |  | 100% |  |  |  |  | 9 |
| recovery_deficit |  |  |  | 100% |  |  |  | 9 |
| healthy |  |  |  |  |  |  | 100% | 9 |

## random

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias |  |  |  |  |  | 100% |  | 9 |
| coverage_deficit |  |  |  |  |  | 100% |  | 9 |
| recovery_deficit |  |  |  |  |  | 100% |  | 9 |
| healthy |  |  |  |  |  | 100% |  | 9 |

## round_robin

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 9 |
| coverage_deficit | 100% |  |  |  |  |  |  | 9 |
| recovery_deficit | 100% |  |  |  |  |  |  | 9 |
| healthy | 100% |  |  |  |  |  |  | 9 |

## rule

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 9 |
| coverage_deficit |  |  | 100% |  |  |  |  | 9 |
| recovery_deficit |  |  | 33% |  |  | 67% |  | 9 |
| healthy |  |  | 44% |  |  | 22% | 33% | 9 |
