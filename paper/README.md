# RSS 2027 Stage 1 投稿（拓展摘要）

`rss2027_submission.tex` 是正文源文件（`\input` 四个分节文件），`rss2027_submission.pdf` 是已编译版本。

```bash
# 需要 TeX Live / TinyTeX；本机验证用 TinyTeX
pdflatex rss2027_submission && bibtex rss2027_submission && pdflatex rss2027_submission && pdflatex rss2027_submission
```

## 当前状态

- **6 页**：前 5 页正文，第 6 页参考文献（符合 RSS 2027 Stage 1 的 5+1 上限）。
- 19 条引用全部解析；正文数字均来自 `../horizon/results/` 下的批次，未填预测值。
- 留空处：Fig. 1（配方×原因热力图）、Fig. 2（真机散点 / 诊断混淆矩阵）、Table 4（Piper / LIBERO 外部验证）、以及顺序干预的配对图（`external_scope.tex` 末尾的空框）。

## 与实验目录的对应

| 论文位置 | 数据来源 |
|---|---|
| Table 1（8 配方 × 3 原因，72 格） | `horizon/results/experiment1_table1_3banks_20261123/` |
| Table 2（选择器 × 4 原因，288 格） | `horizon/results/four_causes_paper8_20261120/` |
| Table 3（效率与诊断） | 同上 + `paper8_report.md` |
| §VI 混合原因与 LLM | `results/experiment_llm_mixed_20261123/`、`results/llm_mixed_report.md` |
| 探针表（$A_2,A,C,D$） | `results/policy_free_signals_paper_20261123/signals.md` |
| 校准门消融 | `results/verify_ablation_20261116/`、`results/verify_pred_20261116/`、`results/calibrator_ablation_20261112/TABLE8.md` |
| 顺序干预（待填） | `results/experiment_conflict_order_20261123/`（跑完后回填 §VI 与图） |