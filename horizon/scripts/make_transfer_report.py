"""Tables and figures for the H5 transfer experiment (results/<name>/transfer_s<seed>.json).

* success on the new family vs. its own data budget, by number of previously learned families;
* C_task: smallest budget reaching the target success (log-linear interpolation), by prior set.

Palette: categorical slots 1-3 of the reference palette in fixed order (0, 1, 2 prior families).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from horizon.eval.metrics import wilson  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOT = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}
MARK = {0: "o", 1: "s", 2: "^"}
NAME = {0: "no prior family", 1: "1 prior family", 2: "2 prior families"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def c_task(budgets: list[int], succ: list[float], target: float) -> float:
    """Smallest budget reaching ``target`` (log-linear interpolation); inf if never reached."""
    for i, (b, s) in enumerate(zip(budgets, succ)):
        if s >= target:
            if i == 0:
                return float(b)
            b0, s0 = budgets[i - 1], succ[i - 1]
            f = (target - s0) / max(1e-9, s - s0)
            if b0 == 0:
                return float(b0 + f * (b - b0))
            return float(2 ** (np.log2(b0) + f * (np.log2(b) - np.log2(b0))))
    return float("inf")


def fmt_c(c: float) -> str:
    return "—" if not np.isfinite(c) else ("0" if c == 0 else f"{c / 1000:.1f}k")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", default="results/transfer", nargs="?")
    args = ap.parse_args()
    res = Path(args.results)
    files = sorted(res.glob("transfer_s*.json"))
    if not files:
        raise SystemExit(f"no results in {res}")
    runs = [json.loads(p.read_text()) for p in files]
    cfg = runs[0]["config"]
    target_s = float(cfg.get("target", 0.5))
    budgets = [int(b) for b in cfg["budgets"]]
    fams = list(cfg.get("families", ["pack", "sort", "stack"]))
    cell = defaultdict(lambda: [0, 0])  # (target, prior tuple, budget) -> [k, n] pooled over seeds
    for r in runs:
        for row in r["rows"]:
            c = cell[(row["target"], tuple(row["prior"]), row["budget"])]
            c[0] += row["k"]
            c[1] += row["n"]
    figs = res / "figures"
    figs.mkdir(exist_ok=True)
    md = [f"# H5: new-task cost vs. number of previously learned task families ({res.name})", "",
          f"Seeds: {len(runs)}. Prior families contribute {int(cfg['prior_budget']) // 1000}k transitions each; "
          f"the new family gets b transitions. Test: the new family's held-out set ({len(cfg.get('H_values', [2, 4, 6, 8, 10]))} H "
          f"values × {cfg.get('test', {}).get('n_per', 20)} per seed). C_task = budget at which success reaches "
          f"{target_s:.0%} (log-linear interpolation; — = not reached within {budgets[-1] // 1000}k).", "",
          "## Success on the new family", "",
          "| new family | prior families | " + " | ".join(f"b={b // 1000}k" for b in budgets) + " | C_task |",
          "|---|---|" + "---|" * len(budgets) + "---|"]
    ctab = {}
    for t in fams:
        priors = sorted({p for (tt, p, _) in cell if tt == t}, key=lambda p: (len(p), p))
        for p in priors:
            s = [cell[(t, p, b)][0] / max(1, cell[(t, p, b)][1]) for b in budgets]
            ctab[(t, p)] = c_task(budgets, s, target_s)
            md.append(f"| {t} | {', '.join(p) or '—'} | " + " | ".join(f"{x:.2f}" for x in s) + f" | {fmt_c(ctab[(t, p)])} |")
    # aggregated by number of prior families
    md += ["", "## Pooled over new families (1 prior: both choices pooled)", "",
           "| prior families | " + " | ".join(f"b={b // 1000}k" for b in budgets) + " | mean C_task (reached) | reached |",
           "|---|" + "---|" * len(budgets) + "---|---|"]
    agg = {}
    for npr in (0, 1, 2):
        ks = [sum(v[0] for (t, p, bb), v in cell.items() if len(p) == npr and bb == b) for b in budgets]
        ns = [sum(v[1] for (t, p, bb), v in cell.items() if len(p) == npr and bb == b) for b in budgets]
        agg[npr] = (ks, ns)
        cs = [c for (t, p), c in ctab.items() if len(p) == npr]
        fin = [c for c in cs if np.isfinite(c)]
        md.append(f"| {npr} | " + " | ".join(f"{k / max(1, n):.2f} [{wilson(k, n)[1]:.2f}, {wilson(k, n)[2]:.2f}]"
                                              for k, n in zip(ks, ns))
                  + f" | {fmt_c(float(np.mean(fin))) if fin else '—'} | {len(fin)}/{len(cs)} |")
    # figure: small multiples by new family + pooled panel
    xs = np.arange(len(budgets))
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(3.0 * (len(fams) + 1), 3.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, t in zip(axes, fams + ["all"]):
        style(ax)
        for npr in (0, 1, 2):
            if t == "all":
                ks, ns = agg[npr]
                ys = [k / max(1, n) for k, n in zip(ks, ns)]
            else:
                ps = [p for (tt, p, _) in cell if tt == t and len(p) == npr]
                ps = sorted(set(ps))
                if not ps:
                    continue
                ys = [sum(cell[(t, p, b)][0] for p in ps) / max(1, sum(cell[(t, p, b)][1] for p in ps)) for b in budgets]
            ax.plot(xs, ys, color=SLOT[npr], lw=2, marker=MARK[npr], ms=5, mec=SURFACE, mew=1.5, label=NAME[npr])
        ax.axhline(target_s, color=AXIS, lw=1, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels(["0" if b == 0 else f"{b // 1000}k" for b in budgets])
        ax.set_title(t if t != "all" else "pooled", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("new-family transitions", color=INK2, fontsize=8)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Success on the new family", color=INK2)
    leg = axes[-1].legend(frameon=False, fontsize=7, loc="upper left")
    for tx in leg.get_texts():
        tx.set_color(INK2)
    fig.tight_layout()
    fig.savefig(figs / "transfer.png", facecolor=SURFACE)
    plt.close(fig)
    md += ["", "![transfer](figures/transfer.png)", ""]
    (res / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {res / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
