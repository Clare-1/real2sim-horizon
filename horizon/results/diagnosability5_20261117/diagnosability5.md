# Five-class diagnosability (leave-one-out)

samples=243  classes=3  accuracy=88.9%

| true \ predicted | coverage_d | geometry_b | mixed_caus |
|---|---|---|---|
| coverage_deficit | 99 | 0 | 6 |
| geometry_bias | 0 | 108 | 0 |
| mixed_cause | 21 | 0 | 9 |

| cause | n | recall |
|---|---:|---:|
| coverage_deficit | 105 | 94% |
| geometry_bias | 108 | 100% |
| mixed_cause | 30 | 30% |
