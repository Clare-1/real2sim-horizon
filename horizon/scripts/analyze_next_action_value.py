"""Next-action value conditioned on what was already done.

The marginal ΔSR of an operation (``analyze_operation_conflicts.py``) answers "is this operation
good?", which is not the question a selector faces: a selector decides *given the history*.  This
script computes, from every recorded run, the mean ΔSR of round k+1 as a function of
(cause, action at round k, action at round k+1), i.e. the empirical next-action value table.

No rollouts and no training: it only reads ``record.json`` decision logs.

    python scripts/analyze_next_action_value.py [results_dir ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIRS = ["results/four_causes_paper8_20261120", "results/mixed_grid_20261118"]


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    dirs = [Path(a) for a in sys.argv[1:]] or [ROOT / d for d in DEFAULT_DIRS]
    recs: list[dict] = []
    for d in dirs:
        if not (d / "summary.json").exists() and not list(d.glob("runs/*/*/*/*/record.json")):
            continue
        recs += load(d)
    if not recs:
        print("no records")
        return 1

    # cells[(cause, prev_action, action)] = list of ΔSR
    cells = defaultdict(list)
    pairs = defaultdict(list)   # (cause, action) marginal, for reference
    for r in recs:
        dec = sorted(r.get("decisions", []), key=lambda d: d["round"])
        for i, d in enumerate(dec):
            delta = d["sr_after"] - d["sr_before"]
            pairs[(r["cause"], d["action"])].append(delta)
            if i:
                cells[(r["cause"], dec[i - 1]["action"], d["action"])].append(delta)

    causes = sorted({c for c, _ in pairs})
    lines = ["# 下一步动作价值（给定上一轮动作）", "",
             "数值 = 该动作在下一轮带来的 ΔSR 均值（n = 出现的次数）。"
             "括号内为该原因下这个动作的边际均值，用于对比“条件”与“边际”的差。", ""]
    for cause in causes:
        lines += [f"## {cause}", ""]
        prevs = sorted({p for c, p, _ in cells if c == cause})
        acts = sorted({a for c, _, a in cells if c == cause} | {a for c, a in pairs if c == cause})
        for prev in prevs:
            lines += [f"### 上一轮 = `{prev}`", "",
                      "| 下一轮动作 | ΔSR（条件） | n | ΔSR（边际） |", "|---|---:|---:|---:|"]
            for a in acts:
                v = cells.get((cause, prev, a), [])
                m = pairs.get((cause, a), [])
                if not v and not m:
                    continue
                cond = f"{sum(v) / len(v):+.3f}" if v else "—"
                marg = f"{sum(m) / len(m):+.3f}" if m else "—"
                lines.append(f"| {a} | {cond} | {len(v)} | {marg} |")
            lines.append("")
    text = "\n".join(lines)
    print(text)
    out = ROOT / "results/next_action_value.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
