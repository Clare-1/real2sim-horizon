"""Paper figures from already-collected runs: Fig.1 (SR vs deployment cost), Fig.3 (confusion), Tab.3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from report_minimal_validation import bootstrap_ci  # noqa: E402


def load(dirs):
    recs = []
    for d in dirs:
        p = Path(d) / "summary.json"
        if p.exists():
            recs += json.loads(p.read_text(encoding="utf-8"))["records"]
        else:
            recs += [json.loads(x.read_text(encoding="utf-8"))
                     for x in sorted(Path(d).glob("runs/*/*/*/*/record.json"))]
    return recs


def curve(rec, per_round=24):
    """(deployments, sr) points: start at 0 with the first pre-round SR, then each round's post SR."""
    xs, ys = [0], [rec["decisions"][0]["sr_before"]]
    for i, d in enumerate(rec["decisions"], 1):
        xs.append(i * per_round)
        ys.append(d["sr_after"])
    return xs, ys


def auc(rec, per_round=24):
    xs, ys = curve(rec, per_round)
    return sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)) / xs[-1]


def cost_to(rec, target, per_round=24):
    xs, ys = curve(rec, per_round)
    for i in range(len(xs) - 1):
        if ys[i] >= target:
            return xs[i]
        if ys[i + 1] >= target:
            span = ys[i + 1] - ys[i]
            frac = 0 if span <= 0 else (target - ys[i]) / span
            return xs[i] + frac * (xs[i + 1] - xs[i])
    return None


def main() -> int:
    recs = load([ROOT / "results/minimal_validation_20261112/stage1",
                 ROOT / "results/minimal_validation_20261112/stage2"])
    causes = sorted({r["cause"] for r in recs})
    sels = sorted({r["selector"] for r in recs})
    colors = {"fixed": "tab:gray", "rule": "tab:blue", "oracle": "tab:green"}
    out = ROOT / "results/paper_figures_20261116"
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(causes), figsize=(6 * len(causes), 4.2), layout="constrained",
                             squeeze=False)
    for col, cause in enumerate(causes):
        ax = axes[0][col]
        for sel in sels:
            rs = [r for r in recs if r["selector"] == sel and r["cause"] == cause]
            if not rs:
                continue
            n = max(len(curve(r)[0]) for r in rs)
            xs = list(range(0, 24 * (n - 1) + 1, 24))
            means, los, his = [], [], []
            for k, x in enumerate(xs):
                vals = []
                for r in rs:
                    cx, cy = curve(r)
                    vals.append(cy[k] if k < len(cy) else cy[-1])
                means.append(sum(vals) / len(vals))
                lo, hi = bootstrap_ci(vals)
                los.append(lo); his.append(hi)
            ax.plot(xs, means, "o-", label=sel, color=colors.get(sel))
            ax.fill_between(xs, los, his, alpha=0.15, color=colors.get(sel))
        ax.set(title=f"{cause}: SR vs deployment cost", xlabel="cumulative deployments",
               ylabel="full-task SR", ylim=(-0.02, 1.03))
        ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.suptitle("Fig. 1  SR vs cost (mean, 95% bootstrap band over seeds x scenario instances)")
    fig.savefig(out / "fig1_sr_vs_cost.png", dpi=110)

    lines = ["# Tab. 3  efficiency and diagnosis", "",
             "| selector | cause | SR-cost AUC | deploy to reach 90% of oracle | final SR |", "|---|---|---:|---:|---:|"]
    for cause in causes:
        orc = [r for r in recs if r["selector"] == "oracle" and r["cause"] == cause]
        o90 = 0.9 * (sum(r["final_sr"] for r in orc) / len(orc)) if orc else None
        for sel in sels:
            rs = [r for r in recs if r["selector"] == sel and r["cause"] == cause]
            if not rs:
                continue
            a = sum(auc(r) for r in rs) / len(rs)
            costs = [cost_to(r, o90) for r in rs] if o90 is not None else [None]
            got = [c for c in costs if c is not None]
            reach = f"{sum(got)/len(got):.0f}" if got else "not reached"
            fin = sum(r["final_sr"] for r in rs) / len(rs)
            lines.append(f"| {sel} | {cause} | {a:.3f} | {reach} | {fin:.3f} |")
    (out / "tab3_efficiency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {out/'fig1_sr_vs_cost.png'}")
    confusion_figure()
    return 0




def confusion_figure():
    import numpy as np
    out = ROOT / "results/paper_figures_20261116"
    out.mkdir(parents=True, exist_ok=True)
    cm = np.array([[27, 3], [0, 43]])  # rows: true(cover, geom); cols: predicted
    fig, ax = plt.subplots(figsize=(4.4, 3.8), layout="constrained")
    im = ax.imshow(cm, cmap="Blues")
    labels = ["coverage", "geometry"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="predicted", ylabel="true cause",
           title="Fig. 3  diagnosis confusion (threshold rule, 95.9%)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out / "fig3_confusion.png", dpi=110)
    print("saved fig3_confusion.png")


if __name__ == "__main__":
    sys.exit(main())
