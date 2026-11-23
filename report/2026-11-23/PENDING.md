# 在跑的批次 / 跑完后怎么并进来

更新时间：2026-11-23 15:40。本文件由人更新；数字是当时的进度。

| 批次 | 目标格数 | 现在 | 进论文哪里 | 完成后替换 |
|---|---:|---:|---|---|
| 顺序 / 冲突实验 `experiment_conflict_order_20261123` | 36 | **14** | §VI "Order"、图 B、命题 2 的因果版 | `tables/T7_order_partial.md`、`figures/figB_order.png`；正文那句 "intervention in progress" 可改成完整估计 |
| 表 A1 扰动评测 `experiment_a1_perturbed_20261123` | 24 | 0（排队） | 附录：排序是不是"干净评测"的产物 | 新增 `tables/TA1_perturbed_eval.md` |
| paper_llm_rows `paper_llm_rows_20261123` | 72 | 2 | 表 2 的 LLM 正式行（3 seeds × 3 banks × 2 臂） | `tables/T6` 里的 24 格筛查行升级为正式行 |

已完成但仍值得标注的两批：`experiment_llm_mixed_rulefix_20261123`（6/6，**门打开与不打开完全相同**）、`experiment_llm_mixed_20261123`（24/24）。

## 一键刷新

```bash
bash /home/aurora/real2sim/组会汇报_2026-11-23/refresh.sh
```

它会重跑：T1/T3/T4/T6/T7 的报告、7 张历史图、4 张组会图（A 混合各臂 / B 顺序 / C 配对热图 / D 核验闸门），并把结果覆盖到本文件夹。**只重算，不改口径。**

## 会自动跑吗

已挂 `watch_and_refresh.sh`：等三个批次全部出现 `completion.json` 后自动执行 `refresh.sh`，并在 `REFRESH_LOG.md` 追加一行时间戳和当时各批的格数。若只想手动跑，忽略它即可。

## 判断"跑完了"的判据

不数目录，读 `completion.json`：

```bash
for d in experiment_conflict_order_20261123 experiment_a1_perturbed_20261123 paper_llm_rows_20261123; do
  echo -n "$d: "; python3 -c "import json;d=json.load(open('results/$d/completion.json'));print(d['runs'],'/',d['expected_runs'],d['complete'])" 2>/dev/null || echo "未写"
done
```