"""Advisor §四.7 / 图 R2: regret gap between a fixed recipe and a selector as the dominant
cause varies.

Everything is measured, not assumed: the per-cause success rates come from the minimal-validation
runs, epsilon comes from the diagnosability check, and the mixing curve is a reweighting of
single-cause cells (stated as an assumption: causes do not interact inside a scenario).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load(dirs):
    recs = []
    for d in dirs:
        p = Path(d) / "summary.json"
        if p.exists():
            recs += json.loads(p.read_text(encoding="utf-8"))["records"]
        else:
            recs += [json.loads(x.read_text(encoding="utf-8"))
                     for x in sorted(Path(d).glob("runs/*/*/*/*/record.json"))]
    return recs


def main() -> int:
    recs = load([ROOT / "results/minimal_validation_20261112/stage1",
                 ROOT / "results/minimal_validation_20261112/stage2"])
    causes = sorted({r["cause"] for r in recs})
    sr = {}
    for sel in ("fixed", "rule", "oracle"):
        for c in causes:
            rs = [r["final_sr"] for r in recs if r["selector"] == sel and r["cause"] == c]
            if rs:
                sr[(sel, c)] = sum(rs) / len(rs)
    print("measured per-cause SR:")
    for c in causes:
        print(f"  {c:16s} fixed={sr[('fixed', c)]:.3f} rule={sr[('rule', c)]:.3f} "
              f"oracle={sr[('oracle', c)]:.3f}")

    geo = "geometry_bias" if "geometry_bias" in causes else causes[1]
    cov = [c for c in causes if c != geo][0]
    r_fix = {c: sr[("oracle", c)] - sr[("fixed", c)] for c in causes}
    r_rule = {c: sr[("oracle", c)] - sr[("rule", c)] for c in causes}
    print("regret vs oracle: fixed=(geo %.3f, cov %.3f) rule=(geo %.3f, cov %.3f)"
          % (r_fix[geo], r_fix[cov], r_rule[geo], r_rule[cov]))

    ps = [i / 20 for i in range(21)]
    gap_fix = [p * r_fix[geo] + (1 - p) * r_fix[cov] for p in ps]
    gap_rule = [p * r_rule[geo] + (1 - p) * r_rule[cov] for p in ps]
    diff = [100 * (a - b) for a, b in zip(gap_fix, gap_rule)]

    eps = 0.0
    diag = ROOT / "results/diagnosability_20261116/diagnosability.md"
    if diag.exists():
        for line in diag.read_text(encoding="utf-8").splitlines():
            if line.startswith("accuracy"):
                eps = 1.0 - float(line.split("=")[1].strip().rstrip("%")) / 100.0
                break
    delta = min(sr[("oracle", c)] - sr[("fixed", c)] for c in causes)
    print(f"epsilon (misdiagnosis)={eps:.3f}  delta(min per-cause gain)={100*delta:.1f} pp")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    ax[0].plot(ps, diff, "o-", label="measured: R_fixed − R_rule")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(xlabel="P(dominant cause = geometry bias)", ylabel="regret gap (pp)",
              title="Fig. R2: selector advantage vs cause mix")
    ax[0].grid(alpha=0.25)
    ax[0].legend(frameon=False)
    ax[1].bar(["Δ (min per-cause gain)", "ε (misdiagnosis)", "R_fixed(geo)", "R_rule(geo)"],
              [100 * delta, eps, 100 * r_fix[geo], 100 * r_rule[geo]],
              color=["tab:blue", "tab:orange", "tab:red", "tab:green"])
    ax[1].set(ylabel="value (pp or probability)", title="proposition inputs, all measured")
    ax[1].grid(alpha=0.25, axis="y")
    out = ROOT / "results/proposition_20261116"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_R2.png", dpi=110)
    (out / "numbers.json").write_text(json.dumps({
        "per_cause_sr": {f"{k[0]}|{k[1]}": v for k, v in sr.items()},
        "regret_fixed": r_fix, "regret_rule": r_rule, "epsilon": eps, "delta": delta,
        "gap_at_p0_pp": diff[0], "gap_at_p1_pp": diff[-1]}, indent=2), encoding="utf-8")
    print(f"gap at p=0: {diff[0]:+.1f} pp   at p=1: {diff[-1]:+.1f} pp")
    print(f"saved {out/'fig_R2.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
