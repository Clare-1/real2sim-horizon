# 汇报目录说明（2026-11-17）

组会用。先读 `组会汇报.md`，所有数字都能在 `tables/` 和 `data/` 里复核。

## 目录结构

```
汇报_2026-11-17/
├── 组会汇报.md            ← 主文档：结论 + 已完成 + 待做 + 下一步 + 需拍板
├── README.md              ← 本文件
├── 实验进展_2026-11-16.md  ← 全过程记录（17 节，含所有中间结果与自我更正）
├── 证据索引_2026-11-16.md  ← 主张 ↔ 数字 ↔ 文件，写论文时逐条引
├── 文献_LoopWAM_2026-11-16.md ← LoopWAM 精读（5 问逐条 + 定位句）
├── 合规复核_2026-11-16.md  ← 导师意见逐条 ✅/⚠️/❌/🔒
├── figures/               ← 5 张图（PNG，可直接贴 PPT）
├── tables/                ← 4 张表（Markdown）
└── data/                  ← 2 个 CSV（原始数字，可自己画图）
```

## figures/

| 文件 | 内容 | 一句话结论 |
|---|---|---|
| `fig1_sr_vs_cost.png` | SR 随累计部署次数 | 覆盖不足下固定配方在 0.6 走平，规则第 2 轮跳到 0.9+ |
| `fig3_confusion.png` | 二类诊断混淆矩阵 | 95.9%（27/3；0/43），3 个假阳性全在一侧 |
| `fig_R2.png` | 混杂度 vs 选择器优势 | **+22.0 → +3.4 pp** 单调下降 |
| `fig7_budget_sweep.png` | 部署预算 8/24/48 | **无单调趋势**；每格 n=2，不能进正文 |
| `long_horizon_replication_rebuilt.png` | 长程复验（重画） | 原图是坏的（平在 0.5）；真实值 0.90/0.78/0.83/0.82/0.71 |

## tables/

| 文件 | 内容 |
|---|---|
| `实验表格汇总_2026-11-12.md` | 一键生成的 8 张表（R1 / 2-3 / 8 / 混合 / 预算平分 / 核验×2 / RoboDojo） |
| `tab3_efficiency.md` | 效率表：AUC 与达到 oracle 90% 所需部署 |
| `fig7_numbers.md` | 图 7 的数字表 |
| `diagnosability5.md` | 多类可辨识性（当前 3 类，混合 recall 仅 30%） |

## data/

| 文件 | 内容 |
|---|---|
| `main_3seeds.csv` | 每（选择器 × 原因）的均值、CI、首步正确数、标签量 |
| `paired_deltas.csv` | 相对固定配方的配对差与 CI（3 组来源：3seeds / n50 / mixed） |

## 复核方式

所有表和图都由脚本从 `horizon/results/` 原始产物重新生成，不是手抄：

```bash
cd horizon
PYTHONPATH=src .venv-linux/bin/python scripts/make_all_tables.py        # 8 张表
PYTHONPATH=src .venv-linux/bin/python scripts/make_paper_figures.py     # 图 1、图 3、表 3
PYTHONPATH=src .venv-linux/bin/python scripts/make_proposition_figure.py # 图 R2
PYTHONPATH=src .venv-linux/bin/python scripts/make_fig7_budget.py       # 图 7
PYTHONPATH=src .venv-linux/bin/python scripts/run_diagnosability5.py    # 多类混淆矩阵
```

回归测试：`PYTHONPATH=src .venv-linux/bin/python -m pytest -q tests/`（214 passed / 1 既有 failed / 1 skipped）。
