# Bank-grouped diagnosability (5 classes)

records=368  distinct_profiles=176  groups=148  evaluated=176  accuracy=91.5%

| true \ predicted | coverage_de | geometry_bi | healthy | mixed_cause | recovery_de |
|---|---|---|---|---|---|
| coverage_deficit | 56 | 1 | 0 | 2 | 0 |
| geometry_bias | 0 | 59 | 0 | 0 | 0 |
| healthy | 0 | 0 | 22 | 0 | 2 |
| mixed_cause | 10 | 0 | 0 | 0 | 0 |
| recovery_deficit | 0 | 0 | 0 | 0 | 24 |

| cause | groups | n | recall |
|---|---:|---:|---:|
| coverage_deficit | 59 | 59 | 95% [91%, 100%] |
| geometry_bias | 59 | 59 | 100% [100%, 100%] |
| healthy | 24 | 24 | 92% [85%, 100%] |
| mixed_cause | 10 | 10 | 0% [0%, 0%] |
| recovery_deficit | 24 | 24 | 100% [100%, 100%] |
