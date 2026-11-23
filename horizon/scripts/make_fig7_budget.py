"""Fig. 7: final SR vs deployment budget per round (8 / 24 / 48 feedback episodes)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from report_minimal_validation import bootstrap_ci  # noqa: E402
BANKS = 2  # compare the same two scenario instances across budgets


def load_glob(path):
    p = Path(path) / "summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["records"]
    return [json.loads(x.read_text(encoding="utf-8"))
            for x in sorted(Path(path).glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fb8", default=None)
    ap.add_argument("--fb24", default=None)
    ap.add_argument("--fb48", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.fb8 or args.fb48:
        for name in ("fb8", "fb24", "fb48"):
            if getattr(args, name) is None:
                raise SystemExit(f"--{name} is required when using explicit sources")
        sources = {8: load_glob(ROOT / args.fb8), 24: load_glob(ROOT / args.fb24),
                   48: load_glob(ROOT / args.fb48)}
        out = ROOT / (args.out or "results/fig7_budget_20261123")
        return _plot(sources, out)
    sources = {
        8: load_glob(ROOT / "results/budget_sweep_20261116/fb8") if (ROOT / "results/budget_sweep_20261116/fb8").exists() else [],
        24: [r for r in load_glob(ROOT / "results/minimal_validation_20261112/stage1")
             if r.get("bank_index", 0) < BANKS],
        48: load_glob(ROOT / "results/budget_sweep_20261116/fb48") if (ROOT / "results/budget_sweep_20261116/fb48").exists() else [],
    }
    return _plot(sources, ROOT / "results/paper_figures_20261116")


def _plot(sources, out) -> int:
    causes = sorted({r["cause"] for v in sources.values() for r in v})
    sels = sorted({r["selector"] for v in sources.values() for r in v})
    colors = {"fixed": "tab:gray", "rule": "tab:blue", "bandit": "tab:orange", "oracle": "tab:green"}
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, max(len(causes), 1), figsize=(6 * max(len(causes), 1), 4.2),
                             layout="constrained", squeeze=False)
    rows = ["# Fig. 7  deployment-budget sweep", "",
            "| cause | selector | 8/round | 24/round | 48/round |", "|---|---|---:|---:|---:|"]
    for col, cause in enumerate(causes):
        ax = axes[0][col]
        for sel in sels:
            xs, ms, los, his = [], [], [], []
            for fb in (8, 24, 48):
                vals = [r["final_sr"] for r in sources[fb] if r["cause"] == cause and r["selector"] == sel]
                if not vals:
                    continue
                xs.append(fb); ms.append(sum(vals) / len(vals))
                lo, hi = bootstrap_ci(vals) if len(vals) > 1 else (float("nan"), float("nan"))
                los.append(lo); his.append(hi)
            if not xs:
                continue
            ax.plot(xs, ms, "o-", label=sel, color=colors.get(sel))
            if all(v == v for v in los):
                ax.fill_between(xs, los, his, alpha=0.15, color=colors.get(sel))
        ax.set(title=f"{cause}: final SR vs deployment budget", xlabel="feedback episodes per round",
               ylabel="final SR", xscale="log", xticks=[8, 24, 48], ylim=(-0.02, 1.03))
        ax.grid(alpha=0.25); ax.legend(frameon=False)
    for cause in causes:
        for sel in sels:
            cells = []
            for fb in (8, 24, 48):
                vals = [r["final_sr"] for r in sources[fb] if r["cause"] == cause and r["selector"] == sel]
                cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "-")
            rows.append(f"| {cause} | {sel} | " + " | ".join(cells) + " |")
    fig.savefig(out / "fig7_budget_sweep.png", dpi=110)
    (out / "fig7_numbers.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
