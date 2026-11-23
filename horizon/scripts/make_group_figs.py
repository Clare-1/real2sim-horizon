"""Three views the paper tables cannot show at a glance, for a group meeting.

    python scripts/make_group_figs.py [out_dir]

(A) mixed-cause arms: rule / rule with the label gate opened / API planner / planner with memory /
    cause-informed reference, per mixture and on average.
(B) order intervention: the same two operations in both orders, plus the same-operation-twice
    controls, from whatever cells have finished. Bars are labelled with the number of banks.
(C) paired difference from the rule within each condition, as a signed heatmap, so the signs that
    cancel in a mean column stay visible.

All three read `record.json` files; nothing is typed in by hand, so re-running after the remaining
cells finish updates every number.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(pattern: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ROOT.glob(pattern))]


def fig_mixed(out: Path) -> None:
    arms = [
        ("rule", "results/experiment_llm_mixed_20261123/runs/*/*/rule/s0/record.json"),
        ("rule, gate open", "results/experiment_llm_mixed_rulefix_20261123/runs/*/*/rule/s0/record.json"),
        ("API planner", "results/experiment_llm_mixed_20261123/runs/*/*/llm/s0/record.json"),
        ("planner + memory", "results/experiment_llm_mixed_20261123/runs/*/*/llm_memory/s0/record.json"),
        ("cause-informed", "results/experiment_llm_mixed_20261123/runs/*/*/oracle/s0/record.json"),
    ]
    conditions = ["mixed_gc", "mixed_cr", "mixed_gcr"]
    vals, labels = [], []
    for name, pat in arms:
        recs = load(pat)
        per = [np.mean([r["final_sr"] for r in recs if r["cause"] == c]) if any(
            r["cause"] == c for r in recs) else np.nan for c in conditions]
        vals.append(per + [np.nanmean(per)])
        labels.append(name)
    vals = np.array(vals, dtype=float)

    fig, ax = plt.subplots(figsize=(5.0, 2.7), dpi=300)
    x = np.arange(len(conditions) + 1)
    w = 0.15
    for i, name in enumerate(labels):
        ax.bar(x + (i - 2) * w, vals[i], w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["geo+cov", "cov+rec", "geo+cov+rec", "mean"], fontsize=7.5)
    ax.set_ylabel("final SR", fontsize=7.5)
    ax.set_ylim(0, 1.22)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.0, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    for i, name in enumerate(labels):
        for j, v in enumerate(vals[i]):
            if np.isfinite(v):
                ax.text(x[j] + (i - 2) * w, v + 0.02, f"{v:.2f}", ha="center", fontsize=4.6)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "figA_mixed_arms.png")
    plt.close(fig)


def fig_order(out: Path) -> None:
    recs = load("results/experiment_conflict_order_20261123/runs/*/*/*/s0/record.json")
    if not recs:
        return
    order = ["seq:calibrate_geometry+calibrate_geometry",
             "seq:calibrate_geometry+regenerate_targeted",
             "seq:regenerate_targeted+calibrate_geometry",
             "seq:regenerate_targeted+regenerate_targeted",
             "seq:calibrate_geometry+regenerate_uniform",
             "seq:regenerate_uniform+calibrate_geometry",
             "seq:calibrate_geometry+dagger_round",
             "seq:dagger_round+calibrate_geometry",
             "seq:dagger_round+regenerate_targeted"]
    short = {"seq:calibrate_geometry+calibrate_geometry": "C→C",
             "seq:calibrate_geometry+regenerate_targeted": "C→T",
             "seq:regenerate_targeted+calibrate_geometry": "T→C",
             "seq:regenerate_targeted+regenerate_targeted": "T→T",
             "seq:calibrate_geometry+regenerate_uniform": "C→U",
             "seq:regenerate_uniform+calibrate_geometry": "U→C",
             "seq:calibrate_geometry+dagger_round": "C→D",
             "seq:dagger_round+calibrate_geometry": "D→C",
             "seq:dagger_round+regenerate_targeted": "D→T"}
    cause = "geometry" if any(r["scenario"] == "geometry" for r in recs) else recs[0]["scenario"]
    recs = [r for r in recs if r["scenario"] == cause]
    # Compare only banks where every sequence has a record. Mixing n=1 and n=2 bars would compare
    # different task sets, which is exactly the confound the intervention is meant to remove.
    by_seq: dict[str, dict[int, float]] = defaultdict(dict)
    for r in recs:
        by_seq[r["selector"]][int(r["bank_index"])] = float(r["final_sr"])
    common = set.intersection(*[set(d) for d in by_seq.values()]) if by_seq else set()
    by = {s: [d[b] for b in sorted(common)] for s, d in by_seq.items() if common and s in by_seq}
    names, means, ns = [], [], []
    for s in order:
        if s in by:
            names.append(short[s])
            means.append(float(np.mean(by[s])))
            ns.append(len(by[s]))
    fig, ax = plt.subplots(figsize=(4.2, 2.3), dpi=300)
    x = np.arange(len(names))
    colors = ["#888888" if "→" not in n else ("#1f77b4" if n.startswith("C") else "#d62728")
              for n in names]
    ax.bar(x, means, 0.62, color=colors)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + 0.015, f"{m:.2f}", ha="center", fontsize=6.5)
        ax.text(i, 0.02, f"n={n}", ha="center", fontsize=5.2, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("final SR", fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    ax.set_title(f"order intervention, {cause}, banks in common (blue: calibration first, "
                 f"red: data first)", fontsize=6.6)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "figB_order.png")
    plt.close(fig)


def fig_paired(out: Path) -> None:
    # All seeds: the paper table averages three seeds x three banks, so a seed-0-only slice would
    # disagree with it.
    recs = load("results/four_causes_paper8_20261120/runs/*/*/*/s*/record.json")
    if not recs:
        return
    causes = ["geometry_bias", "coverage_deficit", "recovery_deficit", "healthy"]
    sels = ["fixed", "random", "round_robin", "bandit", "linucb", "multilabel", "oracle"]
    cells = defaultdict(dict)
    for r in recs:
        cells[(r["cause"], r["selector"])][(int(r["seed"]), int(r["bank_index"]))] = float(r["final_sr"])
    M = np.full((len(sels), len(causes)), np.nan)
    for i, s in enumerate(sels):
        for j, c in enumerate(causes):
            base = cells.get((c, "rule"), {})
            other = cells.get((c, s), {})
            keys = set(base) & set(other)
            if keys:
                M[i, j] = 100 * np.mean([other[k] - base[k] for k in keys])
    fig, ax = plt.subplots(figsize=(3.9, 2.3), dpi=300)
    im = ax.imshow(M, cmap="RdBu", vmin=-40, vmax=40, aspect="auto")
    ax.set_xticks(range(len(causes)))
    ax.set_xticklabels(["geometry", "coverage", "recovery", "healthy"], fontsize=7)
    ax.set_yticks(range(len(sels)))
    ax.set_yticklabels(sels, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.0f}", ha="center", va="center", fontsize=6.2,
                        color="black" if abs(M[i, j]) < 25 else "white")
    ax.set_title("paired ΔSR vs rule (pp)", fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "figC_paired_vs_rule.png")
    plt.close(fig)


def fig_verify(out: Path) -> None:
    """Left: does the candidate repair reduce the held-out prediction error?
    Right: what each acceptance rule costs in deployment success (geometry only, three seeds).
    """
    pred = ROOT / "results/verify_pred_20261116/summary.json"
    abl = ROOT / "results/verify_ablation_20261116/summary.json"
    if not pred.exists() or not abl.exists():
        return
    pred_recs = json.loads(pred.read_text())["records"]
    checks = [(c, r["scenario"], int(r["rejected"])) for r in pred_recs
              if r["arm"] == "verify_pred" for c in r["checks"]]
    ab = json.loads(abl.read_text())["records"]
    geo = {arm: [r for r in ab if r["arm"] == arm and r["scenario"] == "geometry"]
           for arm in ("no_verify", "verify")}

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3), dpi=300,
                             gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    # Geometry runs start from a biased twin; coverage runs start from a correct one, so a
    # candidate there is a false alarm waiting to be caught. Colour by scenario.
    for scen, colour in (("geometry", "#1f77b4"), ("coverage", "#7f7f7f")):
        pts = [c for c, s, _ in checks if s == scen]
        if pts:
            ax.scatter([p["err_before_mm"] for p in pts], [p["err_after_mm"] for p in pts],
                       s=14, color=colour, zorder=3, label=f"{scen} (n={len(pts)})")
    bad = [(c, rej) for c, s, rej in checks if c["err_after_mm"] > c["err_before_mm"] + 2.0]
    for c, rej in bad:
        ax.scatter([c["err_before_mm"]], [c["err_after_mm"]], s=90, facecolors="none",
                   edgecolors="#d62728", lw=1.2, zorder=4)
        ax.annotate("rejected" if rej else "accepted", (c["err_before_mm"], c["err_after_mm"]),
                    textcoords="offset points", xytext=(6, 2), fontsize=5.6, color="#d62728")
    xs = [c["err_before_mm"] for c, _, _ in checks]
    ys = [c["err_after_mm"] for c, _, _ in checks]
    lim = max(xs + ys) * 1.1
    ax.plot([0, lim], [0, lim], lw=0.8, ls="--", color="grey")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("error before repair (mm)", fontsize=7)
    ax.set_ylabel("error after repair (mm)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title(f"held-out height error, {len(checks)} fixture checks", fontsize=7)
    ax.legend(fontsize=5.4, frameon=False, loc="lower right")
    ax.grid(lw=0.3, alpha=0.4)
    ax.text(0.03, 0.95, "below line = repair helped", transform=ax.transAxes, fontsize=5.8,
            va="top", color="grey")

    ax = axes[1]
    arms = [("no gate", "no_verify", "#888888"), ("policy gate", "verify", "#d62728"),
            ("prediction gate", "verify_pred", "#1f77b4")]
    vals, names, rej = [], [], []
    for label, key, color in arms:
        if key == "verify_pred":
            recs = [r for r in json.loads(pred.read_text())["records"]
                    if r["arm"] == "verify_pred" and r["scenario"] == "geometry"]
        else:
            recs = geo[key]
        vals.append(float(np.mean([r["sr_nominal"] for r in recs])))
        names.append(label)
        rej.append(sum(int(r.get("rejected", 0) or 0) for r in recs))
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.55, color=[c for _, _, c in arms])
    for i, (v, k) in enumerate(zip(vals, rej)):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=6.5)
        ax.text(i, 0.03, f"rej {k}/9", ha="center", fontsize=5.6, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("deployment SR (geometry)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title("acceptance rule, three seeds", fontsize=7)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(out / "figD_verification.png")
    plt.close(fig)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/group_figs"
    out.mkdir(parents=True, exist_ok=True)
    fig_mixed(out)
    fig_order(out)
    fig_paired(out)
    fig_verify(out)
    print(f"[written] {out}")
    for p in sorted(out.glob("*.png")):
        print("  ", p.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())