"""Export a developmental repair comparison as PNG/PDF with seed-level points."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from make_cost_report import load_runs

ORDER = ["routed", "gated", "cascade", "cascade_no_replay", "perfect"]
COLORS = {"routed": "#2a78d6", "gated": "#8b64bc", "cascade": "#eb6834",
          "cascade_no_replay": "#eda100", "perfect": "#1baf7a"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    runs = load_runs(args.folder)
    methods = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    for offset, key, label, hatch in ((-.18, "real_nominal", "Nominal", ""),
                                      (.18, "real_pert0.15", "Perturbed", "//")):
        for i, method in enumerate(methods):
            values = [r["final"][key]["overall"] for _, r in sorted(runs[method].items())]
            color = COLORS.get(method, "#777777")
            ax.bar(i + offset, np.mean(values), .34, color=color, edgecolor="white", hatch=hatch,
                   alpha=.8, label=label if i == 0 else None)
            jitter = np.linspace(-.045, .045, len(values)) if len(values) > 1 else [0]
            ax.scatter(i + offset + np.asarray(jitter), values, s=22, c="#171717", zorder=4)
    ax.set_xticks(range(len(methods)), [m.replace("cascade_no_replay", "cascade\nno replay") for m in methods],
                  rotation=15, ha="right")
    ax.set_title("Success and individual training seeds")
    ax.legend(frameon=False)
    ax = axes[1]
    for method in methods:
        rs = list(runs[method].values())
        Hs = sorted(rs[0]["final"]["real_nominal"]["by_H"], key=int)
        curves = np.array([[r["final"]["real_nominal"]["by_H"][H]["k"] /
                            r["final"]["real_nominal"]["by_H"][H]["n"] for H in Hs] for r in rs])
        color = COLORS.get(method, "#777777")
        for curve in curves:
            ax.plot(list(map(int, Hs)), curve, color=color, alpha=.2, linewidth=.8)
        ax.plot(list(map(int, Hs)), curves.mean(0), color=color, marker="o", markersize=4,
                label=method, linewidth=1.8, linestyle="--" if "no_replay" in method else "-")
    ax.set_title("Nominal success by horizon")
    ax.set_xlabel("Subgoals (H)")
    ax.set_xticks(list(map(int, Hs)))
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Task success")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    seeds = sorted({s for rs in runs.values() for s in rs})
    first = next(iter(next(iter(runs.values())).values()))
    context = "fixed policies" if first.get("protocol") == "fixed_policy_probe" else "retrained policies"
    fig.suptitle(f"MuJoCo proxy · {context} · {len(seeds)} training seeds · developmental experiment", fontsize=11)
    for extension in ("png", "pdf"):
        path = args.folder / ("repair_comparison." + extension)
        fig.savefig(path, dpi=200)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
