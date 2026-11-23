# RSS 2027 Stage 1 投稿（拓展摘要）

`rss2027_submission.tex` 是**唯一**的正文源文件（不再拆成分节文件），`rss2027_submission.pdf` 是已编译版本，`fig1_recipe_cause.pdf` 是开篇图（用表 1 的实测数字画的配方×原因热力图）。

```bash
# 需要 TeX Live / TinyTeX；本机验证用 TinyTeX
pdflatex rss2027_submission && bibtex rss2027_submission && pdflatex rss2027_submission && pdflatex rss2027_submission
```

## 当前状态

- **6 页**：前 5 页正文，第 6 页参考文献（符合 RSS 2027 Stage 1 的 5+1 上限）。
- 结构：Abstract / I Introduction / II Related Work / III Problem Formulation（式 1–2 + 两个界）/ IV Which Repairs Help?（表 1、图 1、表 2）/ V Method（C1 策略无关诊断、C2 预算内决策、C3 提交前核验、混合原因先修主因）/ VI Experiments（表 3、表 4、表 5）/ VII Discussion（含成本口径）/ VIII Conclusion。
- 18 条引用全部解析，无未定义引用、无 overfull box；正文数字均来自 `../horizon/results/` 下的批次，未填预测值。
- 图 1 是实测热力图（表 1 的 8×3 数字）。仍未画的：真机散点、诊断混淆矩阵、Piper/LIBERO 外部验证（正文 §VI "Not measured" 已写明这是范围而非结果）。

## 与实验目录的对应

| 论文位置 | 数据来源 |
|---|---|
| Fig. 1 / Table 1（8 配方 × 3 原因，72 格） | `horizon/results/experiment1_table1_3banks_20261123/` |
| Table 2（探针 A2/A/C/D） | `results/policy_free_signals_paper_20261123/signals.md` |
| Table 3 / Table 4（选择器 288 格、分原因配对差） | `results/four_causes_paper8_20261120/`、`paired_vs_rule.md` |
| Table 5（三类混合的决策链） | `results/experiment_llm_mixed_20261123/`、`llm_mixed_report.md` |
| §VI 混合原因与 LLM | `results/experiment_llm_mixed_20261123/`、`results/llm_mixed_report.md` |
| 校准门消融 | `results/verify_ablation_20261116/`、`results/verify_pred_20261116/`、`results/calibrator_ablation_20261112/TABLE8.md` |
| 顺序干预（§VI 末段，1 bank） | `results/experiment_conflict_order_20261123/`（跑完后回填） |