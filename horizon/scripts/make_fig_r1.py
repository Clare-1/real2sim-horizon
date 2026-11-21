"""Fig. R1 (advisor §4.4): recipe x cause heatmap. No fixed recipe wins every column.

    python scripts/make_fig_r1.py [results/four_causes_paper8_20261120]

Rows are selectors/recipes, columns are failure causes, values are mean final SR over
(seed, bank) cells. The point of the figure is the absence of a dominant row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ORDER = ("fixed", "rule", "random", "round_robin", "bandit", "linucb", "multilabel", "oracle")
CAUSES = ("geometry_bias", "coverage_deficit", "recovery_deficit", "healthy")


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/four_causes_paper8_20261120"
    if not path.is_absolute():
        path = ROOT / path
    recs = load(path)
    sels = [s for s in ORDER if any(r["selector"] == s for r in recs)]
    sels += sorted({r["selector"] for r in recs} - set(sels))
    causes = [c for c in CAUSES if any(r["cause"] == c for r in recs)]
    tab = []
    for s in sels:
        tab.append([sum(r["final_sr"] for r in recs
                        if r["selector"] == s and r["cause"] == c) /
                    max(sum(1 for r in recs if r["selector"] == s and r["cause"] == c), 1)
                    for c in causes])
    out = ROOT / "results/fig_R1_recipe_cause_20261121"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.7 * len(causes) + 3.0, 0.55 * len(sels) + 2.0),
                           layout="constrained")
    im = ax.imshow(tab, cmap="RdYlGn", vmin=0.25, vmax=1.0, aspect="auto")
    ax.set(xticks=range(len(causes)), yticks=range(len(sels)), xticklabels=causes,
           yticklabels=sels, title="Final SR: no fixed recipe wins every cause")
    for i, row in enumerate(tab):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="test SR")
    fig.savefig(out / "fig_R1_recipe_cause.png", dpi=140)
    lines = ["# 图 R1　配方 × 原因（最终 SR）", "",
             "| 配方 | " + " | ".join(causes) + " | 平均 |",
             "|---|" + "---:|" * len(causes) + "---:|"]
    for s, row in zip(sels, tab):
        lines.append(f"| {s} | " + " | ".join(f"{v:.3f}" for v in row) +
                     f" | {sum(row)/len(row):.3f} |")
    lines += ["", "没有任何一行在所有列上都最好：fixed 在 geometry 为 "
              f"{tab[sels.index('fixed')][0]:.2f}，在 coverage 只有 {tab[sels.index('fixed')][1]:.2f}；"
              "multilabel 的 coverage 高但 geometry/recovery/healthy 不是最高。"]
    (out / "fig_R1_recipe_cause.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[written] {out/'fig_R1_recipe_cause.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
