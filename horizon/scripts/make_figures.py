"""Paper figures + table views from results/<name>/*_s<seed>.json.

Fig. 3 (main): full-task success vs number of stages H at the matched budget (closed-loop final
budget; open loop read off its nested curve at the same size), with binomial-MLE fits
S(H) = S0 exp(-alpha H). Fig. 4: success vs supervised-transition budget (open-loop nested curve
vs closed-loop rounds). Fig. 9: decay rate alpha per method with a parametric bootstrap CI.
Fig. 7: automatic-vs-oracle attribution confusion. Counts are pooled over seeds; every figure
has a markdown table twin.

Palette: slots 1-4 of the reference categorical palette in fixed order; other methods are drawn
in muted gray with distinct marker shapes so the open- vs closed-loop story leads.
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

from horizon.eval.metrics import fit_horizon_decay, wilson  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# categorical slots 1-4 in the palette's fixed order (validated on the adjacent pairlist used by line charts)
SLOT = {"open_loop": "#2a78d6", "ours": "#eb6834", "oracle": "#1baf7a", "uniform": "#eda100",
        "dagger_beta": "#e87ba4"}
# de-emphasised methods share the muted ink and are told apart by marker shape
MARKER = {"embodirsi": "s", "dagger": "^", "ours_no_recovery": "D", "flat_open_loop": "v", "flat_ours": "P",
          "flat_uniform": "X", "hybrid": "h"}
MAIN = ["open_loop", "dagger_beta", "uniform", "ours", "oracle", "embodirsi"]  # single-seed diagnostics stay in the tables
# architecture comparison: hierarchical (task-graph conditioned) vs flat (order-aware attention) policies
ARCH = {"open_loop": "#2a78d6", "flat_open_loop": "#2a78d6", "uniform": "#eda100", "flat_uniform": "#eda100",
        "ours": "#eb6834", "flat_ours": "#eb6834"}
LABEL = {
    "open_loop": "Open loop (ARSTAG recipe)", "ours": "Ours: subgoal-attributed RSI", "oracle": "Oracle attribution",
    "uniform": "Uniform regeneration", "dagger": "DAgger, naive (beta=0)", "embodirsi": "EmbodiRSI-style (CEC+ADC)",
    "dagger_beta": "DAgger (beta-mixed)", "flat_uniform": "No task graph + uniform",
    "dagger_attr": "DAgger, attribution-guided", "open_loop_norec": "Open loop w/o recovery demos",
    "ours_no_recovery": "Ours w/o recovery ops", "flat_open_loop": "No task graph, open loop", "flat_ours": "No task graph + ours",
    "hybrid": "Hybrid: half attributed, half uniform",
}
ORDER = ["open_loop", "ours", "oracle", "uniform", "hybrid", "dagger_beta", "dagger_attr", "dagger", "embodirsi",
         "ours_no_recovery", "open_loop_norec", "flat_open_loop", "flat_ours", "flat_uniform"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK2, width=1)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def cond_label(c: str) -> str:
    return "nominal" if c == "nominal" else f"perturbed, p={c.replace('pert', '')}"


def color(m):
    return SLOT.get(m, MUTED)


def label(m):
    return LABEL.get(m, m)


# ---------------------------------------------------------------------- loading
def load(results: Path) -> dict:
    runs = defaultdict(list)
    for p in sorted(results.glob("*_s*.json")):
        if p.name.endswith("_train.json"):
            continue
        r = json.loads(p.read_text())
        runs[r["method"]].append(r)
    return runs


def curve_points(run: dict) -> list[dict]:
    """[(n, {cond: by_H counts})] for every evaluated budget of one run (falls back to final eval)."""
    pts = run["train"].get("curve") or []
    if pts:
        return [{"n": c["n"], "round": c["round"], "by_H": {cond: v["by_H"] for cond, v in c["test"].items()}} for c in pts]
    return [{"n": run["train"]["n"], "round": 0, "by_H": {cond: v["by_H"] for cond, v in run["eval"].items()}}]


def counts_at(runs: list, cond: str, n: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-H successes/trials pooled over seeds at budget n (None = each run's final point)."""
    k, t = defaultdict(int), defaultdict(int)
    for r in runs:
        pts = curve_points(r)
        pt = pts[-1] if n is None else min(pts, key=lambda p: abs(p["n"] - n))
        if n is not None and abs(pt["n"] - n) > 0.02 * n:
            continue
        for h, v in pt["by_H"][cond].items():
            k[int(h)] += v["k"]
            t[int(h)] += v["n"]
    Hs = np.array(sorted(k))
    return Hs, np.array([k[h] for h in Hs]), np.array([t[h] for h in Hs])


def boot_alpha(Hs, k, n, reps: int = 400, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    p = k / np.maximum(n, 1)
    a = [fit_horizon_decay(Hs, rng.binomial(n, p), n)["alpha"] for _ in range(reps)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def conds(runs: dict) -> list[str]:
    any_run = next(iter(runs.values()))[0]
    return list(any_run["eval"].keys())


def matched_budget(runs: dict) -> int | None:
    closed = [r["train"]["n"] for m, rs in runs.items() if m != "open_loop" and not m.startswith("flat_open") for r in rs]
    return int(np.median(closed)) if closed else None


def alpha_table(runs: dict, budget: int | None) -> list[str]:
    """Decay rate alpha and long-horizon success of every method at the matched budget (both conditions)."""
    cs = conds(runs)
    rows = ["| Method | seeds | " + " | ".join(f"α {cond_label(c)} [95% CI]" for c in cs) + " |",
            "|---|---|" + "---|" * len(cs)]
    for m in [x for x in ORDER if x in runs]:
        cells = []
        for c in cs:
            Hs, k, n = counts_at(runs[m], c, budget if m.startswith("open_loop") or m.startswith("flat_open") else None)
            if len(Hs) < 2:
                cells.append("-")
                continue
            fit = fit_horizon_decay(Hs, k, n)
            lo, hi = boot_alpha(Hs, k, n)
            cells.append(f"{fit['alpha']:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append(f"| {label(m)} | {len(runs[m])} | " + " | ".join(cells) + " |")
    return rows


# ---------------------------------------------------------------------- figures
def fig_horizon(runs: dict, cond: str, budget: int | None, out: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    rows = []
    ms = [m for m in ORDER if m in runs and m in MAIN]
    ms = [m for m in ms if m not in SLOT] + [m for m in ms if m in SLOT]
    hh = np.linspace(1.5, 10.5, 100)
    Hcols = None
    dodge = {m: (i - (len(ms) - 1) / 2) * 0.09 for i, m in enumerate(ms)}
    for m in ms:
        Hs, k, n = counts_at(runs[m], cond, budget if m.startswith("open_loop") or m.startswith("flat_open") else None)
        if len(Hs) < 2:
            continue
        Hcols = Hs
        fit = fit_horizon_decay(Hs, k, n)
        emph = m in SLOT
        col = color(m)
        p = np.array([wilson(int(a), int(b)) for a, b in zip(k, n)])
        ax.plot(hh, fit["S0"] * np.exp(-fit["alpha"] * hh), color=col, lw=2 if emph else 1.2, alpha=0.95 if emph else 0.6,
                solid_capstyle="round", zorder=3 if emph else 2)
        xs = Hs + dodge[m]
        ax.errorbar(xs, p[:, 0], yerr=[p[:, 0] - p[:, 1], p[:, 2] - p[:, 0]], fmt="none", ecolor=col, elinewidth=1,
                    alpha=0.7 if emph else 0.35, zorder=3)
        ax.scatter(xs, p[:, 0], s=40 if emph else 22, color=col, edgecolor=SURFACE, linewidth=2, zorder=4,
                   marker=MARKER.get(m, "o"),
                   label=f"{label(m)}  (α={fit['alpha']:.3f})")
        lo, hi = boot_alpha(Hs, k, n)
        rows.append(f"| {label(m)} | " + " | ".join(f"{a}/{b}" for a, b in zip(k, n))
                    + f" | {k.sum() / n.sum():.3f} | {fit['alpha']:.3f} [{lo:.3f}, {hi:.3f}] | {fit['per_stage']:.3f} |")
    ax.set_xlabel("Number of stages H", color=INK2)
    ax.set_ylabel("Full-task success", color=INK2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(1.2, 10.8)
    bt = f" at {budget // 1000}k transitions" if budget else ""
    ax.set_title(f"Success vs. horizon{bt} ({cond_label(cond)})", color=INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    if Hcols is None:
        return []
    head = "| Method | " + " | ".join(f"H={h}" for h in Hcols) + " | overall | α [95% CI] | per-stage |"
    return [head, "|" + "---|" * (len(Hcols) + 4), *rows]


def fig_budget(runs: dict, out: Path, methods: list | None = None, palette: dict | None = None,
               title: str = "Data efficiency") -> list[str]:
    cs = conds(runs)
    methods = methods or [m for m in ORDER if m in MAIN]
    fig, axes = plt.subplots(1, len(cs), figsize=(5.2 * len(cs), 3.9), dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    rows = ["| Method | condition | budget (k) | success | seeds |", "|---|---|---|---|---|"]
    for ax, cond in zip(axes[0], cs):
        style(ax)
        for m in [x for x in methods if x in runs]:
            by_n = defaultdict(list)
            for r in runs[m]:
                for p in curve_points(r):
                    k = sum(v["k"] for v in p["by_H"][cond].values())
                    n = sum(v["n"] for v in p["by_H"][cond].values())
                    by_n[round(p["n"] / 1000) * 1000].append((k, n))
            if not by_n:
                continue
            xs = np.array(sorted(by_n))
            ys = np.array([sum(a for a, _ in by_n[x]) / sum(b for _, b in by_n[x]) for x in xs])
            sd = np.array([np.std([a / b for a, b in by_n[x]]) if len(by_n[x]) > 1 else 0.0 for x in xs])
            col = (palette or {}).get(m) or color(m)
            emph = m in SLOT or (palette is not None and m in palette)
            flat = m.startswith("flat_")
            ax.plot(xs / 1000, ys, color=col, lw=2 if emph else 1.2, marker=MARKER.get(m, "o"), ms=5 if emph else 4.5, mec=SURFACE,
                    mew=1.5, label=label(m), alpha=(0.55 if flat else 1.0) if emph else 0.7, zorder=3 if emph else 2,
                    linestyle=(0, (1.5, 1.5)) if (palette is not None and flat) else "-")
            if (sd > 0).any():
                ax.fill_between(xs / 1000, ys - sd, ys + sd, color=col, alpha=0.10 if emph else 0.05, lw=0)
            for x, y in zip(xs, ys):
                rows.append(f"| {label(m)} | {cond} | {x // 1000} | {y:.3f} | {len(by_n[x])} |")
        ax.set_xlabel("Supervised transitions (thousands)", color=INK2)
        ax.set_ylabel("Full-task success (all H)", color=INK2)
        ax.set_title(f"{title} ({cond_label(cond)})", color=INK, fontsize=11, loc="left")
    leg = axes[0][0].legend(frameon=False, fontsize=7.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return rows


def pooled_at(runs: list, cond: str, n: int) -> tuple[int, int, int]:
    """(successes, trials, seeds) pooled over seeds at budget n (within 2%)."""
    k = t = s = 0
    for r in runs:
        for p in curve_points(r):
            if abs(p["n"] - n) <= 0.02 * n:
                k += sum(v["k"] for v in p["by_H"][cond].values())
                t += sum(v["n"] for v in p["by_H"][cond].values())
                s += 1
                break
    return k, t, s


def diff_table(runs: dict, base: str = "open_loop", methods: list | None = None) -> list[str]:
    """Method minus ``base`` success at each shared budget, 95% CI (two-proportion normal approx.)."""
    if base not in runs:
        return []
    methods = methods or [x for x in ORDER if x != base and not x.startswith("flat_")]
    budgets = sorted({round(p["n"] / 1000) * 1000 for m, rs in runs.items() if m != base
                      for r in rs for p in curve_points(r)})
    rows = [f"| Method | condition | budget (k) | method | {label(base)} | Δ (pp) | 95% CI (pp) | seeds |",
            "|---|---|---|---|---|---|---|---|"]
    for m in [x for x in methods if x in runs]:
        for cond in conds(runs):
            for b in budgets:
                k1, n1, s1 = pooled_at(runs[m], cond, b)
                k0, n0, s0 = pooled_at(runs[base], cond, b)
                if n1 == 0 or n0 == 0:
                    continue
                p1, p0 = k1 / n1, k0 / n0
                se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
                d = 100 * (p1 - p0)
                rows.append(f"| {label(m)} | {cond_label(cond)} | {b // 1000} | {p1:.3f} | {p0:.3f} | {d:+.1f} | "
                            f"[{d - 196 * se:+.1f}, {d + 196 * se:+.1f}] | {min(s1, s0)} |")
    return rows


def long_horizon_table(runs: dict, budget: int | None, h_min: int = 8) -> list[str]:
    """Success restricted to H >= h_min at the matched budget (the regime the paper's claim is about)."""
    rows = [f"| Method | condition | success at H>={h_min} | 95% CI | episodes |", "|---|---|---|---|---|"]
    for m in [x for x in ORDER if x in runs]:
        for cond in conds(runs):
            want = budget if (m.startswith("open_loop") or m.startswith("flat_open")) else None
            Hs, k, n = counts_at(runs[m], cond, want)
            sel = Hs >= h_min
            if not sel.any():
                continue
            kk, nn = int(k[sel].sum()), int(n[sel].sum())
            p, lo, hi = wilson(kk, nn)
            rows.append(f"| {label(m)} | {cond_label(cond)} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {nn} |")
    return rows


def cost_table(runs: dict) -> list[str]:
    """Compute cost per method (H5): simulation spent on data, policy rollouts for RSI evaluation, wall time."""
    rows = ["| Method | training transitions | sim steps for data (M) | sim steps per transition | "
            "RSI evaluation episodes | wall time (min) | seeds |", "|---|---|---|---|---|---|---|"]
    for m in [x for x in ORDER if x in runs]:
        steps, trans, evals, walls = [], [], [], []
        for r in runs[m]:
            hist = r["train"]["history"]
            steps.append(sum(h.get("sim_steps", 0) for h in hist))
            trans.append(r["train"]["n"])
            evals.append(sum(h.get("eval_n", 0) for h in hist))
            walls.append(r.get("wall", 0) / 60)
        if not trans:
            continue
        rows.append(f"| {label(m)} | {np.mean(trans) / 1000:.0f}k | {np.mean(steps) / 1e6:.1f} | "
                    f"{np.mean(steps) / max(1, np.mean(trans)):.0f} | {np.mean(evals):.0f} | {np.mean(walls):.1f} | {len(trans)} |")
    rows.append("")
    rows.append("Open-loop rows count the whole nested pool (up to 320k); sim steps include the 25 physics "
                "substeps of every control step.")
    return rows


def failure_table(runs: dict, budget: int | None) -> list[str]:
    """Remaining failures by (oracle) type at the matched budget, pooled over seeds."""
    from horizon.monitor.attribution import FAILURE_TYPES

    rows = ["| Method | condition | failures / episodes | " + " | ".join(FAILURE_TYPES) + " |",
            "|---|---|---|" + "---|" * len(FAILURE_TYPES)]
    for m in [x for x in ORDER if x in runs]:
        for cond in conds(runs):
            cnt: dict = defaultdict(int)
            n_ep = n_fail = 0
            for r in runs[m]:
                pts = r["train"].get("curve") or []
                pt = None
                if pts:
                    want = budget if (m.startswith("open_loop") or m.startswith("flat_open")) else None
                    pt = pts[-1] if want is None else min(pts, key=lambda p: abs(p["n"] - want))
                    if want is not None and abs(pt["n"] - want) > 0.02 * want:
                        continue
                    summ = pt["test"][cond]
                else:
                    summ = r["eval"][cond]
                for t, c in summ.get("failure_types_oracle", {}).items():
                    cnt[t] += c
                    n_fail += c
                n_ep += sum(v["n"] for v in summ["by_H"].values())
            if n_ep:
                rows.append(f"| {label(m)} | {cond_label(cond)} | {n_fail}/{n_ep} | "
                            + " | ".join(str(cnt.get(t, 0)) for t in FAILURE_TYPES) + " |")
    return rows


def fig_confusion(runs: dict, out: Path) -> list[str]:
    from horizon.monitor.attribution import FAILURE_TYPES

    src = runs.get("ours") or runs.get("oracle")
    if not src:
        return []
    mat = np.zeros((len(FAILURE_TYPES), len(FAILURE_TYPES)))
    for r in src:
        for h in r["train"]["history"][1:]:
            for i, a in enumerate(FAILURE_TYPES):
                for j, b in enumerate(FAILURE_TYPES):
                    mat[i, j] += h["attribution"]["matrix"][a][b]
    keep = [i for i in range(len(FAILURE_TYPES)) if mat[i].sum() + mat[:, i].sum() > 0]
    if not keep:
        return []
    m = mat[np.ix_(keep, keep)]
    names = [FAILURE_TYPES[i] for i in keep]
    norm = m / np.maximum(m.sum(1, keepdims=True), 1)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if m[i, j] > 0:
                ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.5, color=INK2)
    ax.set_xlabel("Automatic attribution", color=INK2)
    ax.set_ylabel("Oracle (ground truth)", color=INK2)
    acc = np.trace(m) / max(1, m.sum())
    ax.set_title(f"Failure-type attribution during RSI (acc. {acc:.2f})", color=INK, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    head = "| oracle \\ auto | " + " | ".join(names) + " |"
    return [head, "|" + "---|" * (len(names) + 1)] + \
        [f"| {names[i]} | " + " | ".join(str(int(v)) for v in m[i]) + " |" for i in range(len(names))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--budget", type=int, default=None, help="matched budget for the horizon figure")
    args = ap.parse_args()
    res = Path(args.results)
    runs = load(res)
    if not runs:
        raise SystemExit(f"no results in {res}")
    figs = res / "figures"
    figs.mkdir(exist_ok=True)
    budget = args.budget or matched_budget(runs)
    md = [f"# Results: {res.name}", "",
          "Seeds per method: " + ", ".join(f"{m}={len(runs[m])}" for m in ORDER if m in runs) + ".",
          f"Matched budget for the horizon figure: {budget} supervised transitions. Counts pooled over seeds;"
          " α is the binomial-MLE decay rate of S(H)=S0·exp(-αH) with a parametric-bootstrap 95% CI.", ""]
    for cond in conds(runs):
        md += [f"## Success vs. horizon at matched budget ({cond})", "", f"![horizon](figures/horizon_{cond}.png)", ""]
        md += fig_horizon(runs, cond, budget, figs / f"horizon_{cond}.png") + [""]
    md += ["## Data efficiency: success vs. budget", "", "![budget](figures/budget.png)", ""]
    md += fig_budget(runs, figs / "budget.png") + [""]
    if any(m.startswith("flat_") for m in runs):
        md += ["## Architecture: task-graph conditioned (solid) vs. flat attention policy (dotted)", "",
               "![arch](figures/architecture.png)", ""]
        md += fig_budget(runs, figs / "architecture.png", methods=list(ARCH), palette=ARCH,
                         title="Hierarchical vs. flat") + [""]
    if "open_loop_norec" in runs:
        md += ["## H3: open loop with vs. without recovery demonstrations", "", "![h3](figures/recovery_ablation.png)", ""]
        md += fig_budget(runs, figs / "recovery_ablation.png", methods=["open_loop", "open_loop_norec"],
                         palette={"open_loop": "#2a78d6", "open_loop_norec": "#eb6834"},
                         title="Recovery demos") + [""]
        md += ["", "Paired difference (with − without recovery demos):", ""]
        md += diff_table(runs, base="open_loop_norec", methods=["open_loop"]) + [""]
    if "dagger_attr" in runs and "dagger_beta" in runs:
        md += ["## Attribution-guided vs. standard DAgger", ""]
        md += diff_table(runs, base="dagger_beta", methods=["dagger_attr"]) + [""]
    if "hybrid" in runs:
        md += ["## Hybrid allocation vs. its two halves", "",
               "Δ = hybrid − attribution-only (ours) and hybrid − uniform, same starting data and test tasks.", ""]
        md += diff_table(runs, base="ours", methods=["hybrid"]) + [""]
        md += diff_table(runs, base="uniform", methods=["hybrid"]) + [""]
    rows = diff_table(runs)
    if rows:
        md += ["## Paired difference to open loop at equal budget", "",
               "Same starting data, same test tasks; Δ = method − open loop in percentage points.", ""] + rows + [""]
    rows = diff_table(runs, base="flat_open_loop", methods=["flat_ours", "flat_uniform"])
    if len(rows) > 2:
        md += ["## Flat policy: paired difference to the flat open loop", ""] + rows + [""]
    md += ["## Decay rate α of every method at the matched budget", ""] + alpha_table(runs, budget) + [""]
    rows = long_horizon_table(runs, budget)
    if len(rows) > 2:
        md += [f"## Long horizons only (H >= 8) at {budget // 1000 if budget else '?'}k", ""] + rows + [""]
    md += ["## Compute cost (data-generation side; H5 itself is tested in results/transfer)", ""] + cost_table(runs) + [""]
    rows = failure_table(runs, budget)
    if len(rows) > 2:
        md += [f"## Remaining failures by type at {budget // 1000 if budget else '?'}k (oracle labels)", ""] + rows + [""]
    rows = fig_confusion(runs, figs / "attribution_confusion.png")
    if rows:
        md += ["## Attribution confusion during RSI rounds", "", "![conf](figures/attribution_confusion.png)", ""] + rows + [""]
    (res / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {res / 'RESULTS.md'} and {len(list(figs.glob('*.png')))} figures")


if __name__ == "__main__":
    main()
