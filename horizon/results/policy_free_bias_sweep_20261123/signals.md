# 策略无关诊断信号（nominal）

A2 = 专家用真值规划；A = 专家用 twin 规划、在真值里执行；B = 专家用 twin 规划、在 twin 里执行；
C = 策略在真值里执行；D = 策略在 twin 里执行（自检）。

| 原因 | A2 | A | B | C | D | twin_cost (A2−A) | knowledge_gap (A−C) | selfcheck_bias (D−C) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coverage_deficit | 1.00 | 1.00 | 1.00 | nan | nan | **+0.00** | **+nan** | **+nan** |
| geometry_bias | 1.00 | 0.33 | 1.00 | nan | nan | **+0.67** | **+nan** | **+nan** |
| healthy | 1.00 | 1.00 | 1.00 | nan | nan | **+0.00** | **+nan** | **+nan** |
| recovery_deficit | 1.00 | 1.00 | 1.00 | nan | nan | **+0.00** | **+nan** | **+nan** |
