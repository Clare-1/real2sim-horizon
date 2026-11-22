"""Fig. 1 (current protocol): test SR vs cumulative deployment episodes.

Built offline from each record's per-round ``sr_before/sr_after``; no new rollouts. One panel per
cause; selectors that are not present are skipped.

    python scripts/make_fig1_cost.py [results/four_causes_paper8_20261120]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from report_minimal_validation import bootstrap_ci  # noqa: E402

CAUSES = ("geometry_bias", "coverage_deficit", "recovery_deficit", "healthy")
SELS = ("fixed", "rule", "bandit", "linucb", "multilabel", "oracle")


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def points(rec, step: float):
    ds = rec.get("decisions") or []
    if not ds:
        return []
    pts = [(0.0, float(ds[0]["sr_before"]))]
    for i, d in enumerate(ds):
        pts.append((step * (i + 1), float(d["sr_after"])))
    return pts


def at(rec, x: float, step: float) -> float:
    pts = points(rec, step)
    y = pts[0][1]
    for px, py in pts:
        if px <= x + 1e-9:
            y = py
    return y


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/four_causes_paper8_20261120"
    if not path.is_absolute():
        path = ROOT / path
    recs = load(path)
    cfg = {}
    exp = path / "experiment.json"
    if exp.exists():
        cfg = json.loads(exp.read_text(encoding="utf-8"))["config"]
    step = float(cfg.get("feedback_episodes", 24))
    xs = [step * i for i in range(int(cfg.get("rounds", 3)) + 1)]
    causes = [c for c in CAUSES if any(r["cause"] == c for r in recs)]
    sels = [s for s in SELS if any(r["selector"] == s for r in recs)]

    out = ROOT / "results/fig1_sr_cost_20261122"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(causes), figsize=(4.2 * len(causes), 3.8),
                             layout="constrained", squeeze=False)
    lines = ["# 图 1（当前 paper8）　SR vs 累计部署", "",
             "| cause | selector | " + " | ".join(f"{int(x)}" for x in xs) + " |",
             "|---|---|" + "---:|" * len(xs)]
    for i, cause in enumerate(causes):
        ax = axes[0][i]
        for sel in sels:
            rs = [r for r in recs if r["cause"] == cause and r["selector"] == sel]
            if not rs:
                continue
            means, los, his = [], [], []
            for x in xs:
                vals = [at(r, x, step) for r in rs]
                lo, hi = bootstrap_ci(vals)
                means.append(float(np.mean(vals)))
                los.append(lo)
                his.append(hi)
            ax.plot(xs, means, "o-", label=sel, lw=1.5, ms=3)
            ax.fill_between(xs, los, his, alpha=0.12)
            lines.append(f"| {cause} | {sel} | " +
                         " | ".join(f"{m:.3f}" for m in means) + " |")
        ax.set(title=cause, xlabel="cumulative deployment episodes", ylabel="test SR",
               ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=6)
    fig.suptitle("Fig. 1  test SR vs cumulative deployment (paper8, 9 cells/point)")
    fig.savefig(out / "fig1_sr_cost.png", dpi=140)
    (out / "fig1_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:10]))
    print(f"[written] {out/'fig1_sr_cost.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
