"""Fig. R2 (advisor §4.7): fixed-recipe regret minus selector regret as the dominant cause varies.

Uses per-cause means from an existing run (default: the 10-bank minimal validation). For a test
mixture with geometry fraction alpha, the expected regret of a selector is
    alpha * regret(selector, geometry) + (1-alpha) * regret(selector, coverage)
where regret = oracle mean SR - selector mean SR. No new rollouts are needed.

    python scripts/make_fig_r2.py [results/minimal_validation_advisor_20261121]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAUSES = ("geometry_bias", "coverage_deficit")


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/minimal_validation_advisor_20261121"
    if not path.is_absolute():
        path = ROOT / path
    recs = load(path)
    sels = [s for s in ("fixed", "rule", "bandit") if any(r["selector"] == s for r in recs)]
    means = {}
    for s in sels + ["oracle"]:
        for c in CAUSES:
            v = [r["final_sr"] for r in recs if r["selector"] == s and r["cause"] == c]
            means[(s, c)] = sum(v) / len(v) if v else float("nan")
    oracle = {c: means[("oracle", c)] for c in CAUSES}
    regret = {(s, c): oracle[c] - means[(s, c)] for s in sels for c in CAUSES}

    alphas = [i / 20 for i in range(21)]
    curve = {s: [a * regret[(s, "geometry_bias")] + (1 - a) * regret[(s, "coverage_deficit")]
                 for a in alphas] for s in sels}
    gap = {s: [a * regret[("fixed", "geometry_bias")] + (1 - a) * regret[("fixed", "coverage_deficit")]
               - curve[s][i] for i, a in enumerate(alphas)] for s in sels if s != "fixed"}

    out = ROOT / "results/fig_R2_regret_20261122"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for s in sels:
        axes[0].plot(alphas, [100 * v for v in curve[s]], "o-", label=s)
    axes[0].set(xlabel="geometry fraction alpha (0 = pure coverage, 1 = pure geometry)",
                ylabel="expected regret vs oracle (pp)", title="expected regret under a mixture")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    for s, v in gap.items():
        axes[1].plot(alphas, [100 * x for x in v], "o-", label=f"fixed - {s}")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(xlabel="geometry fraction alpha", ylabel="fixed regret - selector regret (pp)",
                title="advantage of selection over the fixed recipe")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(out / "fig_R2_regret.png", dpi=140)

    lines = ["# 图 R2　主导原因占比 vs 遗憾", "",
             f"数据：`{path.relative_to(ROOT)}`", "",
             "| selector | geometry regret (pp) | coverage regret (pp) |", "|---|---:|---:|"]
    for s in sels:
        lines.append(f"| {s} | {100*regret[(s,'geometry_bias')]:.1f} | "
                     f"{100*regret[(s,'coverage_deficit')]:.1f} |")
    lines += ["", "| alpha | fixed-rule gap (pp) | fixed-bandit gap (pp) |", "|---|---:|---:|"]
    for i, a in enumerate(alphas):
        lines.append(f"| {a:.2f} | {100*gap['rule'][i]:+.1f} | {100*gap['bandit'][i]:+.1f} |")
    lines += ["", "说明：这是从每类原因的单轮均值算出的期望遗憾，不是新 rollout；"
              "单轮命题是定性的，多轮/bandit 的界需要单独推导。"]
    (out / "fig_R2_regret.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"[written] {out/'fig_R2_regret.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
