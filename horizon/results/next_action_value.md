# 下一步动作价值（给定上一轮动作）

数值 = 该动作在下一轮带来的 ΔSR 均值（n = 出现的次数）。括号内为该原因下这个动作的边际均值，用于对比“条件”与“边际”的差。

## coverage_deficit

### 上一轮 = `add_recovery_data`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.089 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | — | 0 | +0.169 |
| regenerate_targeted | — | 0 | +0.594 |
| regenerate_uniform | — | 0 | -0.000 |
| stop | +0.000 | 1 | +0.000 |

### 上一轮 = `calibrate_geometry`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.089 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.256 | 9 | +0.169 |
| regenerate_targeted | +0.822 | 18 | +0.594 |
| regenerate_uniform | -0.000 | 9 | -0.000 |
| stop | — | 0 | +0.000 |

### 上一轮 = `dagger_round`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.089 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.181 | 16 | +0.169 |
| regenerate_targeted | +0.175 | 10 | +0.594 |
| regenerate_uniform | — | 0 | -0.000 |
| stop | +0.000 | 3 | +0.000 |

### 上一轮 = `regenerate_targeted`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.089 | 9 | +0.089 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | -0.018 | 28 | +0.169 |
| regenerate_targeted | +0.037 | 15 | +0.594 |
| regenerate_uniform | — | 0 | -0.000 |
| stop | +0.000 | 11 | +0.000 |

### 上一轮 = `regenerate_uniform`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.089 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | — | 0 | +0.169 |
| regenerate_targeted | +0.833 | 9 | +0.594 |
| regenerate_uniform | — | 0 | -0.000 |
| stop | — | 0 | +0.000 |

## geometry_bias

### 上一轮 = `calibrate_geometry`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.013 |
| calibrate_geometry | — | 0 | +0.367 |
| dagger_round | +0.062 | 16 | +0.021 |
| regenerate_targeted | +0.254 | 25 | +0.125 |
| regenerate_uniform | +0.250 | 22 | +0.180 |
| stop | — | 0 | +0.000 |

### 上一轮 = `dagger_round`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.013 |
| calibrate_geometry | — | 0 | +0.367 |
| dagger_round | +0.022 | 16 | +0.021 |
| regenerate_targeted | -0.111 | 9 | +0.125 |
| regenerate_uniform | — | 0 | +0.180 |
| stop | — | 0 | +0.000 |

### 上一轮 = `regenerate_targeted`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.013 | 8 | +0.013 |
| calibrate_geometry | — | 0 | +0.367 |
| dagger_round | -0.031 | 21 | +0.021 |
| regenerate_targeted | — | 0 | +0.125 |
| regenerate_uniform | — | 0 | +0.180 |
| stop | +0.000 | 5 | +0.000 |

### 上一轮 = `regenerate_uniform`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.013 |
| calibrate_geometry | — | 0 | +0.367 |
| dagger_round | -0.050 | 1 | +0.021 |
| regenerate_targeted | -0.012 | 8 | +0.125 |
| regenerate_uniform | -0.012 | 8 | +0.180 |
| stop | +0.000 | 5 | +0.000 |

## healthy

### 上一轮 = `add_recovery_data`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | -0.100 | 1 | +0.042 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | — | 0 | +0.047 |
| regenerate_targeted | — | 0 | +0.061 |
| regenerate_uniform | — | 0 | +0.089 |
| stop | +0.000 | 2 | +0.000 |

### 上一轮 = `calibrate_geometry`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.042 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.078 | 9 | +0.047 |
| regenerate_targeted | +0.089 | 18 | +0.061 |
| regenerate_uniform | +0.089 | 9 | +0.089 |
| stop | — | 0 | +0.000 |

### 上一轮 = `dagger_round`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.042 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.000 | 11 | +0.047 |
| regenerate_targeted | +0.014 | 7 | +0.061 |
| regenerate_uniform | — | 0 | +0.089 |
| stop | +0.000 | 6 | +0.000 |

### 上一轮 = `regenerate_targeted`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.044 | 9 | +0.042 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.006 | 17 | +0.047 |
| regenerate_targeted | — | 0 | +0.061 |
| regenerate_uniform | — | 0 | +0.089 |
| stop | +0.000 | 3 | +0.000 |

### 上一轮 = `regenerate_uniform`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.042 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | — | 0 | +0.047 |
| regenerate_targeted | +0.028 | 9 | +0.061 |
| regenerate_uniform | — | 0 | +0.089 |
| stop | — | 0 | +0.000 |

## mixed_cr

### 上一轮 = `add_recovery_data`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.275 | 2 | +0.417 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | — | 0 | +0.268 |
| regenerate_targeted | — | 0 | +0.325 |

### 上一轮 = `calibrate_geometry`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.417 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.200 | 2 | +0.268 |
| regenerate_targeted | +0.325 | 4 | +0.325 |

### 上一轮 = `dagger_round`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.417 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.238 | 4 | +0.268 |
| regenerate_targeted | +0.375 | 2 | +0.325 |

### 上一轮 = `regenerate_targeted`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.487 | 4 | +0.417 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.325 | 6 | +0.268 |
| regenerate_targeted | — | 0 | +0.325 |

## mixed_gc

### 上一轮 = `calibrate_geometry`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.025 |
| calibrate_geometry | — | 0 | +0.050 |
| dagger_round | +0.075 | 2 | +0.071 |
| regenerate_targeted | +0.850 | 8 | +0.575 |

### 上一轮 = `dagger_round`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.025 |
| calibrate_geometry | — | 0 | +0.050 |
| dagger_round | +0.525 | 2 | +0.071 |
| regenerate_targeted | +0.025 | 2 | +0.575 |

### 上一轮 = `regenerate_targeted`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.025 | 2 | +0.025 |
| calibrate_geometry | — | 0 | +0.050 |
| dagger_round | -0.075 | 6 | +0.071 |
| regenerate_targeted | +0.025 | 2 | +0.575 |

## mixed_gcr

### 上一轮 = `calibrate_geometry`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.400 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.200 | 2 | +0.183 |
| regenerate_targeted | +0.400 | 8 | +0.310 |

### 上一轮 = `dagger_round`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.400 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.375 | 2 | +0.183 |
| regenerate_targeted | -0.050 | 2 | +0.310 |

### 上一轮 = `regenerate_targeted`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.400 | 4 | +0.400 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.150 | 6 | +0.183 |
| regenerate_targeted | — | 0 | +0.310 |

## mixed_gr

### 上一轮 = `add_recovery_data`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.175 | 2 | +0.383 |
| calibrate_geometry | — | 0 | +0.025 |
| dagger_round | — | 0 | +0.246 |
| regenerate_targeted | — | 0 | +0.281 |

### 上一轮 = `calibrate_geometry`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.650 | 2 | +0.383 |
| calibrate_geometry | — | 0 | +0.025 |
| dagger_round | +0.500 | 2 | +0.246 |
| regenerate_targeted | +0.375 | 6 | +0.281 |

### 上一轮 = `dagger_round`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.383 |
| calibrate_geometry | — | 0 | +0.025 |
| dagger_round | +0.175 | 2 | +0.246 |
| regenerate_targeted | +0.000 | 2 | +0.281 |

### 上一轮 = `regenerate_targeted`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.325 | 2 | +0.383 |
| calibrate_geometry | — | 0 | +0.025 |
| dagger_round | +0.208 | 6 | +0.246 |
| regenerate_targeted | — | 0 | +0.281 |

## recovery_deficit

### 上一轮 = `add_recovery_data`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.078 | 18 | +0.277 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.175 | 2 | +0.240 |
| regenerate_targeted | — | 0 | +0.044 |
| regenerate_uniform | — | 0 | +0.078 |
| stop | +0.000 | 5 | +0.000 |

### 上一轮 = `calibrate_geometry`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.650 | 1 | +0.277 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.417 | 9 | +0.240 |
| regenerate_targeted | +0.078 | 18 | +0.044 |
| regenerate_uniform | +0.078 | 9 | +0.078 |
| stop | — | 0 | +0.000 |

### 上一轮 = `dagger_round`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.277 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.114 | 25 | +0.240 |
| regenerate_targeted | +0.072 | 9 | +0.044 |
| regenerate_uniform | — | 0 | +0.078 |
| stop | +0.000 | 6 | +0.000 |

### 上一轮 = `regenerate_targeted`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | +0.389 | 9 | +0.277 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | +0.274 | 21 | +0.240 |
| regenerate_targeted | — | 0 | +0.044 |
| regenerate_uniform | — | 0 | +0.078 |
| stop | — | 0 | +0.000 |

### 上一轮 = `regenerate_uniform`

| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |
|---|---:|---:|---:|
| add_recovery_data | — | 0 | +0.277 |
| calibrate_geometry | — | 0 | +0.000 |
| dagger_round | — | 0 | +0.240 |
| regenerate_targeted | -0.072 | 9 | +0.044 |
| regenerate_uniform | — | 0 | +0.078 |
| stop | — | 0 | +0.000 |

