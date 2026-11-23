# LLM vs rule on mixed causes

- 数据：`results/experiment_llm_mixed_20261123`（24 格，3 条件 × 2 banks × 1 seed）
- 对照：`results/experiment_llm_mixed_rulefix_20261123`（同一规则，`tau_labels_low` 200k）

## 最终 SR

| 臂 | mixed_gc | mixed_cr | mixed_gcr | 平均 | $/格（实计） | LLM 调用/格 |
|---|---:|---:|---:|---:|---:|---:|
| rule | 0.725 | 0.900 | 0.575 | **0.733** | 0.0000 | 0.0 |
| rule_fix | — | — | — | **nan** | 0.0000 | 0.0 |
| llm | 0.850 | 0.850 | 0.300 | **0.667** | 0.0145 | 3.0 |
| llm_memory | 0.925 | 0.350 | 0.300 | **0.525** | 0.0164 | 3.0 |
| oracle | 0.925 | 0.950 | 0.800 | **0.892** | 0.0000 | 0.0 |

## 配对差（对 `rule`，按 (condition, bank) 匹配）

| 臂 | ΔSR (pp) | 95% CI | n |
|---|---:|---|---:|
| rule_fix | +nan | [+nan,+nan] | 0 |
| llm | -6.7 | [-25.0,+10.0] | 6 |
| llm_memory | -20.8 | [-45.8,+8.3] | 6 |
| oracle | +15.8 | [+3.3,+28.3] | 6 |

## 读法

1. 若 `rule_fix` ≈ `rule`，说明标签门在混合条件上不是主要瓶颈，LLM 的差异不能用它解释。
2. 若 `llm` 仅在 2 类混合上占优、在 3 类混合上落后，结论是"LLM 的收益随原因数增加而反转"，
   不能写成"LLM 会处理规则没写到的原因"——本平台可修原因已被规则四分支覆盖。
3. 2 banks × 1 seed 只能做筛查：区间跨零时既不支持"更好"也不支持"等价"。

## 决策序列（C 校准 / T 定向 / U 均匀 / R 恢复 / D DAgger / - 停止）

| 臂 | 条件/bank | 决策 | final SR |
|---|---|---|---:|
| rule | mixed_cr/b0 | T → D → D | 1.000 |
| rule | mixed_cr/b1 | T → D → D | 0.800 |
| rule | mixed_gc/b0 | C → T → D | 0.950 |
| rule | mixed_gc/b1 | C → T → D | 0.500 |
| rule | mixed_gcr/b0 | C → T → D | 0.650 |
| rule | mixed_gcr/b1 | C → T → D | 0.500 |
| llm | mixed_cr/b0 | T → deploy_dagger → deploy_dagger | 0.800 |
| llm | mixed_cr/b1 | T → deploy_dagger → deploy_dagger | 0.900 |
| llm | mixed_gc/b0 | C → T → T | 0.950 |
| llm | mixed_gc/b1 | C → T → deploy_dagger | 0.750 |
| llm | mixed_gcr/b0 | C → T → T | 0.200 |
| llm | mixed_gcr/b1 | C → T → T | 0.400 |
| llm_memory | mixed_cr/b0 | T → T → T | 0.500 |
| llm_memory | mixed_cr/b1 | T → T → T | 0.200 |
| llm_memory | mixed_gc/b0 | C → T → T | 0.950 |
| llm_memory | mixed_gc/b1 | C → T → T | 0.900 |
| llm_memory | mixed_gcr/b0 | C → T → T | 0.200 |
| llm_memory | mixed_gcr/b1 | C → T → T | 0.400 |
| oracle | mixed_cr/b0 | T → R → R | 0.950 |
| oracle | mixed_cr/b1 | T → R → R | 0.950 |
| oracle | mixed_gc/b0 | C → T → T | 0.950 |
| oracle | mixed_gc/b1 | C → T → T | 0.900 |
| oracle | mixed_gcr/b0 | C → T → R | 0.800 |
| oracle | mixed_gcr/b1 | C → T → R | 0.800 |
