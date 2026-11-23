"""Fig. 2 / Table 11 (current encoding): test SR vs horizon H.

Reads ``h_extrapolation.json`` written by ``check_h_extrapolation.py`` (list of
``{H, in_training, sr_nominal, sr_perturbed}``), plots SR vs H for the nominal
and perturbed evaluation modes, and marks which H values were trained on.

    python scripts/make_fig2_h.py [results/h_extrapolation_20261123 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> dict:
    f = path / "h_extrapolation.json"
    if not f.exists():
        raise SystemExit(f"missing {f}")
    rows = json.loads(f.read_text(encoding="utf-8"))
    return {"path": path, "rows": rows}


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]] or [ROOT / "results/h_extrapolation_20261123"]
    runs = []
    for p in paths:
        runs.append(load(p if p.is_absolute() else ROOT / p))

    out = ROOT / "results/fig2_sr_vs_h_20261123"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")
    lines = ["# 图 2 / 表 11：H 外推（当前编码）", ""]
    for ax, mode, key in ((axes[0], "nominal", "sr_nominal"), (axes[1], "perturbed", "sr_perturbed")):
        for run in runs:
            rows = run["rows"]
            hs = [r["H"] for r in rows]
            ys = [r[key] for r in rows]
            label = run["path"].name
            ax.plot(hs, ys, "o-", label=label, lw=1.6, ms=4)
            train = [r["H"] for r in rows if r["in_training"]]
            unseen = [r["H"] for r in rows if not r["in_training"]]
            if train:
                ax.plot([r["H"] for r in rows if r["in_training"]],
                        [r[key] for r in rows if r["in_training"]], "o", ms=8,
                        mfc="none", mec="tab:green", mew=1.5)
            if unseen:
                ax.plot([r["H"] for r in rows if not r["in_training"]],
                        [r[key] for r in rows if not r["in_training"]], "s", ms=8,
                        mfc="none", mec="tab:red", mew=1.5)
        ax.set(title=f"{mode}", xlabel="horizon H (subgoals per task)", ylabel="test SR",
               xticks=sorted({r["H"] for run in runs for r in run["rows"]}), ylim=(-0.02, 1.02))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Fig. 2  test SR vs horizon; green circles = trained H, red squares = unseen H")
    fig.savefig(out / "fig2_sr_vs_h.png", dpi=140)

    lines += ["| run | mode | " + " | ".join(f"H={h}" for h in sorted({r['H'] for run in runs for r in run['rows']})) + " |",
              "|---|---|" + "---:|" * len(sorted({r["H"] for run in runs for r in run["rows"]}))]
    for run in runs:
        for mode, key in (("nominal", "sr_nominal"), ("perturbed", "sr_perturbed")):
            by_h = {r["H"]: r[key] for r in run["rows"]}
            hs = sorted(by_h)
            lines.append(f"| {run['path'].name} | {mode} | " + " | ".join(
                (f"**{by_h[h]:.3f}**" if not next(r for r in run['rows'] if r['H'] == h)["in_training"]
                 else f"{by_h[h]:.3f}") for h in hs) + " |")
    lines += ["", "粗体 = 未参与训练的 H（外推点）。"]
    (out / "fig2_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[written] {out/'fig2_sr_vs_h.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
