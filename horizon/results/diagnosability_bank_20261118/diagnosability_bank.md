# Bank-grouped diagnosability (5 classes)

records=288  distinct_profiles=96  groups=68  evaluated=96  accuracy=85.4%

| true \ predicted | coverage_de | geometry_bi | healthy | mixed_cause | recovery_de |
|---|---|---|---|---|---|
| coverage_deficit | 37 | 1 | 0 | 1 | 0 |
| geometry_bias | 0 | 39 | 0 | 0 | 0 |
| healthy | 0 | 2 | 0 | 0 | 2 |
| mixed_cause | 8 | 0 | 0 | 2 | 0 |
| recovery_deficit | 0 | 0 | 0 | 0 | 4 |

| cause | groups | n | recall |
|---|---:|---:|---:|
| coverage_deficit | 39 | 39 | 95% [90%, 100%] |
| geometry_bias | 39 | 39 | 100% [100%, 100%] |
| healthy | 4 | 4 | 0% [0%, 0%] |
| mixed_cause | 10 | 10 | 20% [0%, 40%] |
| recovery_deficit | 4 | 4 | 100% [100%, 100%] |
