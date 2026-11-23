"""Paired selector differences with bootstrap CIs, matched on (seed, bank).

A paper table that compares absolute SR invites the objection that the differences are within
noise.  This script resamples *matched cells* (same seed and bank, so the task instances are the
same) and reports the paired difference and its 95% interval.  Cells that are present for one
selector but not the other are dropped rather than imputed.

    python scripts/paired_diffs.py [results_dir] [--baseline rule] [--others a,b,c]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def boot_ci(vals: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/four_causes_paper8_20261120")
    ap.add_argument("--baseline", default="rule")
    ap.add_argument("--others", default="")
    args = ap.parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path
    recs = load(path)
    if not recs:
        print(f"no records in {path}")
        return 1

    cells: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    for r in recs:
        key = (r["cause"], r["selector"])
        cells[key][(int(r["seed"]), int(r["bank_index"]))] = float(r["final_sr"])

    causes = sorted({c for c, _ in cells})
    selectors = sorted({s for _, s in cells})
    others = [s for s in (args.others.split(",") if args.others else selectors)
              if s and s != args.baseline and any(k[1] == s for k in cells)]

    lines = [f"# 配对差（baseline = `{args.baseline}`，按 (seed, bank) 匹配）", "",
             "数值为 ΔSR（pp），括号内是 cells 上的 bootstrap 95% 区间；n = 匹配的格数。", ""]
    header = "| 原因 | " + " | ".join(f"`{s}`" for s in others) + " |"
    lines += [header, "|---|" + "---|" * len(others)]
    pooled: dict[str, list[float]] = defaultdict(list)
    for cause in causes:
        base = cells.get((cause, args.baseline), {})
        row = [f"| {cause} |"]
        for sel in others:
            cur = cells.get((cause, sel), {})
            keys = sorted(set(base) & set(cur))
            diffs = [cur[k] - base[k] for k in keys]
            pooled[sel] += diffs
            if not diffs:
                row.append("— |")
                continue
            m, lo, hi = boot_ci(diffs)
            star = "*" if (lo > 0 or hi < 0) else ""
            row.append(f"{100 * m:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}]{star} (n={len(diffs)}) |")
        lines.append(" ".join(row))

    lines += ["", "**跨原因汇总（把每个原因的配对差再合在一起）**", "",
              "| 选择器 | 平均 ΔSR (pp) | 95% CI | n |", "|---|---:|---|---:|"]
    for sel in others:
        m, lo, hi = boot_ci(pooled[sel])
        lines.append(f"| `{sel}` | {100 * m:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {len(pooled[sel])} |")

    text = "\n".join(lines)
    print(text)
    out = path / f"paired_vs_{args.baseline}.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
