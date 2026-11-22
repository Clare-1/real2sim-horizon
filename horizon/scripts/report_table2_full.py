"""Advisor Table 2, full version: pure causes (paper8) + mixed causes in one table.

The advisor's Table 2 has a ``混合`` column next to the five single causes. Our pure-cause batch
(paper8) and the mixed-cause grid were run separately, so this script merges them and collapses the
four mixtures into one column, printing how many conditions behind each cell.

    python scripts/report_table2_full.py \
        --pure results/four_causes_paper8_20261120 --mixed results/mixed_grid_20261118
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PURE = ("geometry_bias", "coverage_deficit", "recovery_deficit", "healthy")
PURE_LABEL = {"geometry_bias": "几何", "coverage_deficit": "覆盖",
              "recovery_deficit": "恢复", "healthy": "健康"}
ROWS = ("fixed", "random", "round_robin", "bandit", "rule", "linucb", "multilabel", "oracle")


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    args = sys.argv[1:]
    pure = Path(args[args.index("--pure") + 1]) if "--pure" in args else ROOT / "results/four_causes_paper8_20261120"
    mixed = Path(args[args.index("--mixed") + 1]) if "--mixed" in args else ROOT / "results/mixed_grid_20261118"
    pure = pure if pure.is_absolute() else ROOT / pure
    mixed = mixed if mixed.is_absolute() else ROOT / mixed
    p_recs, m_recs = load(pure), load(mixed)
    causes = sorted({r["cause"] for r in m_recs} - set(PURE))

    def mean(recs, sel, cause=None):
        v = [r["final_sr"] for r in recs if r["selector"] == sel
             and (cause is None or r["cause"] == cause)]
        return (sum(v) / len(v)) if v else None

    out = ["# 表 2（完整版）：纯原因 + 混合原因", "",
           f"- 纯原因：`{pure.relative_to(ROOT)}`（{len(p_recs)} 格）",
           f"- 混合原因：`{mixed.relative_to(ROOT)}`（{len(m_recs)} 格，条件 {', '.join(causes)}）",
           "- `混合` 列 = 上述混合条件的均值；`全部` = 该选择器所有可算条件的均值。", ""]
    cols = list(PURE) + ["mixed"]
    out += ["| 选择器 | " + " | ".join(PURE_LABEL[c] for c in PURE) + " | 混合 | 全部 |",
            "|---|" + "---:|" * (len(PURE) + 2)]
    for sel in ROWS:
        vals, cells = [], []
        for c in PURE:
            m = mean(p_recs, sel, c)
            cells.append(f"{m:.3f}" if m is not None else "-")
            if m is not None:
                vals.append(m)
        mm = mean(m_recs, sel)
        cells.append(f"{mm:.3f}" if mm is not None else "-")
        if mm is not None:
            vals.append(mm)
        out.append(f"| {sel} | " + " | ".join(cells) + " | " +
                   (f"**{sum(vals)/len(vals):.3f}**" if vals else "-") + " |")
    out += ["", "## 每格样本量", "", "| 条件 | " + " | ".join(ROWS) + " |",
            "|---|" + "---:|" * len(ROWS)]
    for c in cols:
        src = p_recs if c in PURE else m_recs
        cells = [str(len([r for r in src if r["selector"] == s and (c == "mixed" or r["cause"] == c)]))
                 for s in ROWS]
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out += ["", "## 混合条件的逐条 SR", "", "| 条件 | " + " | ".join(ROWS) + " |",
            "|---|" + "---:|" * len(ROWS)]
    for c in causes:
        cells = [f"{mean(m_recs, s, c):.3f}" if mean(m_recs, s, c) is not None else "-" for s in ROWS]
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    dest = ROOT / "results/table2_full.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
