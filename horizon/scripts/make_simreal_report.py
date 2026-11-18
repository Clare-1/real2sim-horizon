"""Tables and figures for the sim-real proxy experiments (results/<name>/*_s<seed>.json).

* final real success per routing variant (Wilson CI, pooled over seeds) and resources used;
* real success after each round, which repair each round chose, which injected errors were still
  present at that point (privileged, report only) and the resulting repair precision;
* H4: does success in the robot's own simulator predict real success? Spearman over the 15
  (family, H) cells, pooled separately for runs whose final twin is calibrated or not.

Works for the single-error run (results/simreal: wrong support heights, lenient judge) and the
three-error run (results/simreal3: + strict judge, swapped bins). Palette: categorical slots 1-5 of
the reference palette in fixed order.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from horizon.eval.metrics import wilson  # noqa: E402
from horizon.real2sim.recon_error import RealWorld, ReconModel  # noqa: E402
from horizon.tasks.factory import make_task  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ORDER = ["sim_only", "routed", "gated", "perfect", "repair_all", "random_route"]
# five categorical slots in fixed order; the gated variant is "ours" too: same hue, other marker, dashed
COLOR = {"sim_only": "#2a78d6", "routed": "#eb6834", "gated": "#eb6834", "perfect": "#1baf7a",
         "repair_all": "#eda100", "random_route": "#e87ba4"}
MARK = {"sim_only": "o", "routed": "s", "gated": "P", "perfect": "^", "repair_all": "D", "random_route": "v"}
DASH = {"gated": (0, (4, 2))}
LABEL = {"sim_only": "Sim-only DAgger (no real feedback)", "routed": "Ours: attribution-routed repair (1 per round)",
         "gated": "Ours: evidence-gated repairs (no budget)",
         "perfect": "Perfect twin (upper bound)", "repair_all": "Every repair every round",
         "random_route": "Random repair (incl. decoy)"}
SHORT = {"routed": "Ours: routed repair", "gated": "Ours: evidence-gated", "repair_all": "Every repair, every round",
         "random_route": "Random repair", "sim_only": "Sim-only DAgger", "perfect": "Perfect twin"}
HARMFUL_JUDGE = 0.7


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def load(d: Path) -> dict:
    runs = defaultdict(list)
    for p in sorted(d.glob("*_s*.json")):
        r = json.loads(p.read_text())
        runs[r["config"]["name"]].append(r)
    return runs


def pooled(rs: list, key: str) -> tuple[int, int]:
    k = sum(v["k"] for r in rs for v in r["final"][key]["by_H"].values())
    n = sum(v["n"] for r in rs for v in r["final"][key]["by_H"].values())
    return k, n


def cells(r: dict, key: str) -> dict:
    return {c: v["k"] / max(1, v["n"]) for c, v in r["final"][key]["by_family_H"].items()}


def _support_tops(recon: ReconModel, world: RealWorld) -> dict[str, float]:
    out = {}
    for fam in ("pack", "stack"):
        sc = recon.apply(world.apply(make_task(fam, 3, 0).scene))
        out.update({f.name: f.top_z for f in sc.fixtures if f.kind == "plate"})
    return out


def final_errors(r: dict) -> list[str]:
    """Injected errors still present in a run's final beliefs (privileged; report only)."""
    world = RealWorld.from_dict(r["config"].get("world"))
    b = r.get("final_belief")
    if b is None:  # single-error runs: reconstruct from the last round + the repairs it chose
        last = r["history"][-1]
        recon = ReconModel.from_dict(last.get("recon", last.get("belief", {}).get("recon", {})))
        if "geometry_report" in last:
            for name, v in last["geometry_report"].items():
                if abs(v["estimated"] - v["believed"]) > 0.003 and v["believed"] > 1e-4:
                    recon.corrections[name] = recon.corrections.get(name, 1.0) * v["estimated"] / v["believed"]
        judge = last.get("verifier_report", {}).get("chosen", last["verifier_scale"]) or last["verifier_scale"]
        grounding = {}
    else:
        recon, judge, grounding = ReconModel.from_dict(b["recon"]), b["judge_scale"], b["grounding"].get("confuse", {})
    if r["config"]["route"] == "perfect":
        recon = ReconModel()
    true = _support_tops(ReconModel(), world)
    believed = _support_tops(recon, world)
    out = []
    if any(abs(true[k] - believed.get(k, true[k])) > 0.003 for k in true):
        out.append("geometry")
    if judge < HARMFUL_JUDGE:
        out.append("verifier")
    if grounding:
        out.append("graph")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", default="results/simreal", nargs="?")
    args = ap.parse_args()
    res = Path(args.results)
    runs = load(res)
    if not runs:
        raise SystemExit(f"no results in {res}")
    figs = res / "figures"
    figs.mkdir(exist_ok=True)
    ms = [m for m in ORDER if m in runs]
    any_cfg = runs[ms[0]][0]["config"]
    errs = ["support heights reconstructed 70% too low"]
    errs.append(f"judge tolerance ×{any_cfg.get('judge_scale', 2.5)}" + (" (strict)" if any_cfg.get("judge_scale", 2.5) < 1 else
                                                                         " (lenient; harmless here)"))
    if any_cfg.get("grounding", {}).get("confuse"):
        errs.append("compiler grounding error " + ", ".join(f"{a}→{b}" for a, b in any_cfg["grounding"]["confuse"].items()))
    md = [f"# Sim-real proxy: {res.name}", "",
          "Seeds: " + ", ".join(f"{m}={len(runs[m])}" for m in ms) + ". The robot trains only in its own simulator; "
          "the real (proxy) world differs by: " + "; ".join(errs) + ".", "",
          "## Final success", "",
          "| Method | real (nominal) | 95% CI | real (perturbed) | sim (nominal) | real episodes | audits | repairs used | errors left (runs) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in ms:
        k, n = pooled(runs[m], "real_nominal")
        kp, np_ = pooled(runs[m], "real_pert0.15")
        ks, ns = pooled(runs[m], "sim_nominal")
        p, lo, hi = wilson(k, n)
        reps = Counter(x for r in runs[m] for x in r["repairs"])
        left = Counter(e for r in runs[m] for e in final_errors(r))
        md.append(f"| {LABEL[m]} | {p:.3f} | [{lo:.3f}, {hi:.3f}] | {kp / max(1, np_):.3f} | {ks / max(1, ns):.3f} | "
                  f"{np.mean([r['real_episodes_used'] for r in runs[m]]):.0f} | {np.mean([r['audits_used'] for r in runs[m]]):.0f} | "
                  + (", ".join(f"{a}×{b}" for a, b in sorted(reps.items())) or "-") + " | "
                  + (", ".join(f"{a} ({b})" for a, b in sorted(left.items())) or "none") + " |")
    md += ["", "Per seed (real nominal): " + "; ".join(
        f"{SHORT[m]} " + "/".join(f"{r['final']['real_nominal']['overall']:.2f}"
                                                         for r in sorted(runs[m], key=lambda r: r['config']['seed'])) for m in ms) + "."]
    # per-family final real success
    md += ["", "## Final real success by family (nominal)", "", "| Method | pack | sort | stack |", "|---|---|---|---|"]
    for m in ms:
        fam = defaultdict(lambda: [0, 0])
        for r in runs[m]:
            for c, v in r["final"]["real_nominal"]["by_family_H"].items():
                fam[c.split("_")[0]][0] += v["k"]
                fam[c.split("_")[0]][1] += v["n"]
        md.append(f"| {LABEL[m]} | " + " | ".join(f"{fam[f][0] / max(1, fam[f][1]):.2f}" for f in ("pack", "sort", "stack")) + " |")
    # learning curves
    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    md += ["", "## Real success of the policy entering each round (nominal, small test set) and final", "",
           "| Method | round 1 | round 2 | round 3 | final |", "|---|---|---|---|---|"]
    for m in ms:
        R = min(len(r["history"]) for r in runs[m])
        ys = [float(np.mean([r["history"][i]["real_test"] for r in runs[m]])) for i in range(R)]
        k, n = pooled(runs[m], "real_nominal")
        ys.append(k / max(1, n))
        xs = list(range(1, R + 2))
        ax.plot(xs, ys, color=COLOR[m], lw=2, marker=MARK[m], ms=6, mec=SURFACE, mew=1.5, label=LABEL[m],
                linestyle=DASH.get(m, "-"))
        md.append(f"| {LABEL[m]} | " + " | ".join(f"{y:.3f}" for y in ys) + " |")
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(["round 1", "round 2", "round 3", "final"])
    ax.set_ylabel("Real success (all H)", color=INK2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Deployment success across self-improvement rounds", color=INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(figs / "rounds.png", facecolor=SURFACE)
    plt.close(fig)
    md += ["", "![rounds](figures/rounds.png)", ""]
    # routing decisions and their precision
    md += ["## Routing decisions", "",
           "Per round: the repair chosen, and (in brackets, privileged) the injected errors still present when it was chosen.",
           "", "| Method | seed | round 1 | round 2 | round 3 |", "|---|---|---|---|---|"]
    prec = {}
    for m in ms:
        if m in ("sim_only", "perfect"):
            continue
        hit = tot = 0
        for r in sorted(runs[m], key=lambda r: r["config"]["seed"]):
            cells_ = []
            for h in r["history"]:
                ch = h.get("chosen", [])
                act = h.get("active")
                if act is not None:
                    hit += sum(c in act for c in ch)
                    tot += len(ch)
                cells_.append((",".join(ch) or "none") + (f" [{','.join(act) or '-'}]" if act is not None else ""))
            md.append(f"| {LABEL[m]} | {r['config']['seed']} | " + " | ".join(cells_) + " |")
        if tot:
            prec[m] = (hit, tot)
    if prec:
        md += ["", "Repair precision (repairs that targeted an error still present / repairs run): " +
               "; ".join(f"{LABEL[m]} {h}/{t}" for m, (h, t) in prec.items()) + "."]
    md += ["", "Evidence used by the routed variant (geometry share from the twin test; judge / plan shares from audits):", "",
           "| seed | round | flagged | geometry | judge | plan | twin gap pack / sort / stack | audits: judge / plan / false success |",
           "|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs.get("routed", []), key=lambda r: r["config"]["seed"]):
        for h in r["history"]:
            tw = h.get("twin", {})
            gaps = " / ".join(f"{tw[f]['gap']:+.2f}" if f in tw else "-" for f in ("pack", "sort", "stack"))
            a = h.get("audit", {})
            sh = h.get("shares", {})
            md.append(f"| {r['config']['seed']} | {h['round']} | {','.join(h.get('flagged', [])) or '-'} | "
                      f"{sh.get('geometry', 0):.3f} | {sh.get('verifier', 0):.3f} | {sh.get('graph', 0):.3f} | {gaps} | "
                      f"{a.get('judge', '-')} / {a.get('plan', '-')} / {a.get('false_success', '-')} |")
    # H4: own-simulator success vs. real success
    md += ["", "## H4: does success in the robot's own simulator predict real success?", "",
           "Spearman ρ between final sim and real success over the 15 (family, H) cells of each run (mean over runs), "
           "the same pooled over runs, and the mean sim − real gap. Bottom rows: all runs grouped by whether the final "
           "twin is calibrated (no injected error left).", "",
           "| Runs | ρ (mean over runs) | pooled ρ | mean sim − real | runs |", "|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, len(ms), figsize=(2.3 * len(ms), 2.9), dpi=200, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    groups: dict[str, list] = {m: runs[m] for m in ms}
    groups["calibrated twin (all methods)"] = [r for m in ms for r in runs[m] if not final_errors(r)]
    groups["uncalibrated twin (all methods)"] = [r for m in ms for r in runs[m] if final_errors(r)]
    for g, rs in groups.items():
        if not rs:
            continue
        rhos, xs_all, ys_all = [], [], []
        for r in rs:
            s_, re_ = cells(r, "sim_nominal"), cells(r, "real_nominal")
            keys = sorted(set(s_) & set(re_))
            x = np.array([s_[c] for c in keys])
            y = np.array([re_[c] for c in keys])
            if np.std(x) > 0 and np.std(y) > 0:
                rhos.append(spearmanr(x, y).statistic)
            xs_all += list(x)
            ys_all += list(y)
        pr = spearmanr(xs_all, ys_all).statistic if np.std(xs_all) > 0 and np.std(ys_all) > 0 else float("nan")
        def rho_txt(v: float) -> str:  # undefined when one side is constant (e.g. real success 0 everywhere)
            return "n/a" if not np.isfinite(v) else f"{v:.2f}"

        md.append(f"| {LABEL.get(g, g)} | {rho_txt(float(np.mean(rhos)) if rhos else float('nan'))} | {rho_txt(pr)} | "
                  f"{np.mean(np.array(xs_all) - np.array(ys_all)):+.3f} | {len(rs)} |")
        if g in COLOR:
            ax = axes[ms.index(g)]
            style(ax)
            ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
            ax.scatter(xs_all, ys_all, s=16, color=COLOR[g], marker=MARK[g], edgecolor=SURFACE, linewidth=0.6,
                       alpha=0.9, zorder=3)
            ax.set_title(f"{SHORT[g]}\npooled ρ = {rho_txt(pr)}", color=INK, fontsize=8, loc="left")
            ax.set_xlim(-0.04, 1.04)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xlabel("sim success", color=INK2, fontsize=8)
    axes[0].set_ylabel("real (proxy) success", color=INK2, fontsize=8)
    fig.suptitle("Sim vs. real success per (family, H) cell, pooled over seeds", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figs / "sim_vs_real.png", facecolor=SURFACE)
    plt.close(fig)
    md += ["", "![h4](figures/sim_vs_real.png)", ""]
    # judge behaviour on the final test
    md += ["## Judge verdicts on the final real test (nominal)", "",
           "| Method | judged done | false success | timeouts |", "|---|---|---|---|"]
    for m in ms:
        eps = [e for r in runs[m] for e in r["final"]["real_nominal"]["episodes"]]
        jd = [e.get("judged_done", e["steps"] < e["H"] * 180 + 60) for e in eps]
        fs = sum(1 for e, j in zip(eps, jd) if j and not e["success"])
        md.append(f"| {LABEL[m]} | {sum(jd) / max(1, len(eps)):.3f} | {fs / max(1, len(eps)):.3f} | "
                  f"{(len(eps) - sum(jd)) / max(1, len(eps)):.3f} |")
    (res / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {res / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
