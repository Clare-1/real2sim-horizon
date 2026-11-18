"""H4 in the sim-real proxy: does a twin rank complete systems (policy + its beliefs) like reality?

Every final system of a sim-real run (``<method>_s<seed>_policy.npz`` + its final beliefs) is
evaluated inside two twins of the real (proxy) world: the robot's initial reconstruction
(uncalibrated) and the reconstruction after the routed variant's repairs (calibrated, estimated from
real data only). Systems keep their own beliefs (believed geometry, judge, plan) in every
evaluation, exactly as on the real system; only the physics is replaced by the twin. Real success is
the run's final real test (same test seeds). Spearman ρ over systems between twin and real success;
the paper's H4 asks for ρ > 0.7 with a calibrated twin.

Usage: python scripts/h4_rank.py results/simreal3 [--n-per 20] [--workers N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from horizon.datagen.parallel import Job, Runner  # noqa: E402
from horizon.datagen.rollout import PerturbationConfig  # noqa: E402
from horizon.eval.evaluate import TEST_SEED_BASE, summarize  # noqa: E402
from horizon.monitor.verifier import VerifierConfig  # noqa: E402
from horizon.rsi.loop import load_params  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"sim_only": "#2a78d6", "routed": "#eb6834", "gated": "#eb6834", "perfect": "#1baf7a",
         "repair_all": "#eda100", "random_route": "#e87ba4"}
MARK = {"sim_only": "o", "routed": "s", "gated": "P", "perfect": "^", "repair_all": "D", "random_route": "v"}
LABEL = {"sim_only": "Sim-only DAgger", "routed": "Ours: routed repair", "gated": "Ours: evidence-gated",
         "perfect": "Perfect twin", "repair_all": "Every repair every round", "random_route": "Random repair"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def jobs_for(run: dict, physics: dict, n_per: int) -> list[Job]:
    cfg, b = run["config"], run["final_belief"]
    recon = b["recon"]
    g = b["grounding"] if b["grounding"].get("confuse") else None
    out = []
    for fi, fam in enumerate(("pack", "sort", "stack")):
        for H in (2, 4, 6, 8, 10):
            for i in range(n_per):
                ts = TEST_SEED_BASE + fi * 100_000 + H * 1000 + i
                out.append(Job(family=fam, H=H, task_seed=ts, seed=ts + 17, controller=cfg.get("policy", "hier"),
                               world=dict(cfg["world"]), recon=recon, real=True, physics=physics, grounding=g,
                               verifier_cfg=VerifierConfig(tol_scale=b["judge_scale"]), pert=PerturbationConfig(),
                               want_samples=False, keep_log=True, provenance="twin_eval"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--n-per", type=int, default=20)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    res = Path(args.results)
    runs = [json.loads(p.read_text()) for p in sorted(res.glob("*_s[0-9].json"))]
    runs = [r for r in runs if (res / f"{r['config']['name']}_s{r['config']['seed']}_policy.npz").exists()]
    if not runs:
        raise SystemExit("no runs with saved policies")
    routed = sorted([r for r in runs if r["config"]["name"] == "routed"], key=lambda r: r["config"]["seed"])
    twins = {"uncalibrated": runs[0]["config"]["bias"],
             "calibrated": routed[0]["final_belief"]["recon"] if routed else None}
    out_path = res / "h4_rank.json"
    table = json.loads(out_path.read_text()) if out_path.exists() else {}
    for r in runs:
        key = f"{r['config']['name']}_s{r['config']['seed']}"
        row = table.setdefault(key, {"method": r["config"]["name"], "seed": r["config"]["seed"],
                                     "real": r["final"]["real_nominal"]["overall"],
                                     "real_cells": {c: v["k"] / v["n"] for c, v in r["final"]["real_nominal"]["by_family_H"].items()}})
        todo = [t for t in twins if twins[t] is not None and t not in row]
        if not todo:
            continue
        params = load_params(res / f"{key}_policy.npz")
        with Runner(args.workers, policy={"params": params}) as runner:
            for t in todo:
                s = summarize(runner.map(jobs_for(r, twins[t], args.n_per)), (2, 4, 6, 8, 10))
                row[t] = s["overall"]
                row[t + "_cells"] = {c: v["k"] / v["n"] for c, v in s["by_family_H"].items()}
                print(f"{key}: real {row['real']:.3f}  {t} twin {row[t]:.3f}", flush=True)
        out_path.write_text(json.dumps(table, indent=1))
    # summary
    md = ["## H4: does a twin rank complete systems like the real world?", "",
          f"{len(table)} final systems (policy + its own beliefs), each evaluated on the real (proxy) world and inside two "
          "twins; only the physics differs. Uncalibrated twin = the robot's initial reconstruction; calibrated twin = the "
          "reconstruction after the routed variant's repairs (seed 0), estimated from real data only.", "",
          "| twin | Spearman ρ over systems | mean twin − real | ρ over (system, cell) pairs |", "|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, t in zip(axes, ("uncalibrated", "calibrated")):
        style(ax)
        rows = [v for v in table.values() if t in v]
        if len(rows) < 3:
            continue
        x = np.array([v[t] for v in rows])
        y = np.array([v["real"] for v in rows])
        rho = spearmanr(x, y).statistic
        xc = [v[t + "_cells"][c] for v in rows for c in v["real_cells"] if c in v[t + "_cells"]]
        yc = [v["real_cells"][c] for v in rows for c in v["real_cells"] if c in v[t + "_cells"]]
        md.append(f"| {t} | {rho:.2f} | {np.mean(x - y):+.3f} | {spearmanr(xc, yc).statistic:.2f} |")
        for m in COLOR:
            pts = [(v[t], v["real"]) for v in rows if v["method"] == m]
            if pts:
                ax.scatter(*zip(*pts), s=42, color=COLOR[m], marker=MARK[m], edgecolor=SURFACE, linewidth=1.2,
                           label=LABEL[m], zorder=3)
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(f"Success in the {t} twin", color=INK2)
        ax.set_title(f"{t.capitalize()} twin: ρ = {rho:.2f}", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("Success on the real (proxy) world", color=INK2)
    leg = axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    for tx in leg.get_texts():
        tx.set_color(INK2)
    fig.tight_layout()
    (res / "figures").mkdir(exist_ok=True)
    fig.savefig(res / "figures" / "h4_rank.png", facecolor=SURFACE)
    plt.close(fig)
    md += ["", "![h4 rank](figures/h4_rank.png)", ""]
    (res / "H4.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
