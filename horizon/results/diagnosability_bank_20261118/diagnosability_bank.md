# Bank-grouped diagnosability (5 classes)

records=416  distinct_profiles=184  groups=150  evaluated=184  accuracy=87.5%

| true \ predicted | coverage_de | geometry_bi | healthy | mixed_cause | recovery_de |
|---|---|---|---|---|---|
| coverage_deficit | 55 | 1 | 0 | 3 | 0 |
| geometry_bias | 0 | 59 | 0 | 0 | 0 |
| healthy | 0 | 0 | 22 | 0 | 2 |
| mixed_cause | 16 | 1 | 0 | 1 | 0 |
| recovery_deficit | 0 | 0 | 0 | 0 | 24 |

| cause | groups | n | recall |
|---|---:|---:|---:|
| coverage_deficit | 59 | 59 | 93% [89%, 98%] |
| geometry_bias | 59 | 59 | 100% [100%, 100%] |
| healthy | 24 | 24 | 92% [85%, 100%] |
| mixed_cause | 18 | 18 | 6% [0%, 11%] |
| recovery_deficit | 24 | 24 | 100% [100%, 100%] |
