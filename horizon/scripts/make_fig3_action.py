"""Fig. 3 as the advisor specifies it: rows = true cause, columns = the FIRST operation chosen.

This is different from a cause-vs-predicted-cause confusion matrix: it answers "given this cause,
which operation did the selector try first", which is what the paper needs.

Default: one heatmap with 4 rows (the four gradable causes) and one column per admissible
operation, pooled over selectors. ``--per-selector`` additionally writes one panel per selector.

Only the current four-cause batch is used. The voided ``five_causes_20261117`` run and the
mixed-cause experiment are deliberately excluded so the rows are exactly the causes the paper
claims.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["results/four_causes_20261118"]
CAUSES = ("geometry_bias", "coverage_deficit", "recovery_deficit", "healthy")
ACTIONS = ("calibrate_geometry", "regenerate_uniform", "regenerate_targeted",
           "add_recovery_data", "randomize_physics", "dagger_round", "stop")


def load(dirs):
    recs = []
    for d in dirs:
        p = ROOT / d
        if not p.exists():
            continue
        f = p / "summary.json"
        if f.exists():
            recs += json.loads(f.read_text(encoding="utf-8"))["records"]
        else:
            recs += [json.loads(x.read_text(encoding="utf-8"))
                     for x in sorted(p.glob("runs/*/*/*/*/record.json"))]
    return [r for r in recs if r.get("first_action")]


def matrix(recs, causes, selectors=None):
    rows = []
    for c in causes:
        cnt = Counter(r["first_action"] for r in recs
                      if r["cause"] == c and (selectors is None or r["selector"] in selectors))
        n = sum(cnt.values()) or 1
        rows.append([cnt.get(a, 0) / n for a in ACTIONS])
    return rows


def write_table(title: str, causes, rows, n_by_cause) -> list[str]:
    lines = [f"## {title}", "",
             "| true cause | " + " | ".join(a[:14] for a in ACTIONS) + " | n |",
             "|---|" + "---|" * (len(ACTIONS) + 1)]
    for c, row in zip(causes, rows):
        lines.append(f"| {c} | " + " | ".join(f"{v:.0%}" if v else "" for v in row) +
                     f" | {n_by_cause.get(c, 0)} |")
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-selector", action="store_true")
    ap.add_argument("--results", default="results/four_causes_20261118",
                    help="result directory to read (default: the screening batch)")
    args = ap.parse_args()

    recs = load([args.results])
    causes = [c for c in CAUSES if any(r["cause"] == c for r in recs)]
    causes += sorted({r["cause"] for r in recs} - set(causes))
    sels = sorted({r["selector"] for r in recs})
    name = Path(args.results).name
    out_name = "fig3_action_20261118" if name == "four_causes_20261118" else f"fig3_action_{name}"
    out = ROOT / "results" / out_name
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Fig. 3  first operation per cause (row = cause, column = chosen operation)",
             "", f"runs={len(recs)}  causes={causes}  selectors={sels}", ""]

    rows = matrix(recs, causes)
    n_all = {c: sum(1 for r in recs if r["cause"] == c) for c in causes}
    lines += write_table("all selectors pooled", causes, rows, n_all)
    fig, ax = plt.subplots(figsize=(8.4, 0.9 * len(causes) + 2.0), layout="constrained")
    im = ax.imshow(rows, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(len(ACTIONS)), yticks=range(len(causes)),
           xticklabels=[a.replace("_", "\n") for a in ACTIONS], yticklabels=causes,
           title=f"first operation by true cause  (n={len(recs)} runs)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(out / "fig3_first_action.png", dpi=120)
    plt.close(fig)

    if args.per_selector and sels:
        fig, axes = plt.subplots(1, len(sels), figsize=(4.4 * len(sels), 0.9 * len(causes) + 2.0),
                                 layout="constrained", squeeze=False)
        for col, sel in enumerate(sels):
            srows = matrix(recs, causes, {sel})
            n_sel = {c: sum(1 for r in recs if r["cause"] == c and r["selector"] == sel)
                     for c in causes}
            lines += write_table(sel, causes, srows, n_sel)
            ax = axes[0][col]
            im = ax.imshow(srows, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set(xticks=range(len(ACTIONS)), yticks=range(len(causes)),
                   xticklabels=[a.replace("_", "\n") for a in ACTIONS], yticklabels=causes,
                   title=sel)
            plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
            fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle("Fig. 3  first operation chosen, by true failure cause and selector")
        fig.savefig(out / "fig3_first_action_by_selector.png", dpi=120)
        plt.close(fig)
    else:
        for sel in sels:
            n_sel = {c: sum(1 for r in recs if r["cause"] == c and r["selector"] == sel)
                     for c in causes}
            lines += write_table(sel, causes, matrix(recs, causes, {sel}), n_sel)

    (out / "fig3_first_action.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)[:2400])
    return 0


if __name__ == "__main__":
    sys.exit(main())

