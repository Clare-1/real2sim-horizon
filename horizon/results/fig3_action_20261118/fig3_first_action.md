# Fig. 3  first operation per cause (row = cause, column = chosen operation)

runs=38  causes=['geometry_bias', 'coverage_deficit', 'recovery_deficit', 'healthy']  selectors=['bandit', 'fixed', 'oracle', 'random', 'round_robin', 'rule']

## all selectors pooled

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 83% |  |  |  |  | 17% |  | 12 |
| coverage_deficit | 50% |  | 33% |  |  | 17% |  | 12 |
| recovery_deficit | 50% |  |  | 17% | 8% | 25% |  | 12 |
| healthy | 50% |  |  |  |  | 50% |  | 2 |

## bandit

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit | 100% |  |  |  |  |  |  | 2 |
| recovery_deficit | 100% |  |  |  |  |  |  | 2 |
| healthy |  |  |  |  |  |  |  | 0 |

## fixed

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit | 100% |  |  |  |  |  |  | 2 |
| recovery_deficit | 100% |  |  |  |  |  |  | 2 |
| healthy | 100% |  |  |  |  |  |  | 1 |

## oracle

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit |  |  | 100% |  |  |  |  | 2 |
| recovery_deficit |  |  |  | 100% |  |  |  | 2 |
| healthy |  |  |  |  |  |  |  | 0 |

## random

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias |  |  |  |  |  | 100% |  | 2 |
| coverage_deficit |  |  |  |  |  | 100% |  | 2 |
| recovery_deficit |  |  |  |  |  | 100% |  | 2 |
| healthy |  |  |  |  |  |  |  | 0 |

## round_robin

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit | 100% |  |  |  |  |  |  | 2 |
| recovery_deficit | 100% |  |  |  |  |  |  | 2 |
| healthy |  |  |  |  |  |  |  | 0 |

## rule

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit |  |  | 100% |  |  |  |  | 2 |
| recovery_deficit |  |  |  |  | 50% | 50% |  | 2 |
| healthy |  |  |  |  |  | 100% |  | 1 |
