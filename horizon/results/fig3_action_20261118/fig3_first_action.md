# Fig. 3  first operation per cause (row = cause, column = chosen operation)

runs=48  causes=['geometry_bias', 'coverage_deficit', 'recovery_deficit', 'healthy']  selectors=['bandit', 'fixed', 'oracle', 'random', 'round_robin', 'rule']

## all selectors pooled

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 83% |  |  |  |  | 17% |  | 12 |
| coverage_deficit | 50% |  | 33% |  |  | 17% |  | 12 |
| recovery_deficit | 50% |  |  | 17% |  | 33% |  | 12 |
| healthy | 50% |  |  |  |  | 17% | 33% | 12 |

## bandit

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit | 100% |  |  |  |  |  |  | 2 |
| recovery_deficit | 100% |  |  |  |  |  |  | 2 |
| healthy | 100% |  |  |  |  |  |  | 2 |

## fixed

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit | 100% |  |  |  |  |  |  | 2 |
| recovery_deficit | 100% |  |  |  |  |  |  | 2 |
| healthy | 100% |  |  |  |  |  |  | 2 |

## oracle

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit |  |  | 100% |  |  |  |  | 2 |
| recovery_deficit |  |  |  | 100% |  |  |  | 2 |
| healthy |  |  |  |  |  |  | 100% | 2 |

## random

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias |  |  |  |  |  | 100% |  | 2 |
| coverage_deficit |  |  |  |  |  | 100% |  | 2 |
| recovery_deficit |  |  |  |  |  | 100% |  | 2 |
| healthy |  |  |  |  |  | 100% |  | 2 |

## round_robin

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit | 100% |  |  |  |  |  |  | 2 |
| recovery_deficit | 100% |  |  |  |  |  |  | 2 |
| healthy | 100% |  |  |  |  |  |  | 2 |

## rule

| true cause | calibrate_geom | regenerate_uni | regenerate_tar | add_recovery_d | randomize_phys | dagger_round | stop | n |
|---|---|---|---|---|---|---|---|---|
| geometry_bias | 100% |  |  |  |  |  |  | 2 |
| coverage_deficit |  |  | 100% |  |  |  |  | 2 |
| recovery_deficit |  |  |  |  |  | 100% |  | 2 |
| healthy |  |  |  |  |  |  | 100% | 2 |
