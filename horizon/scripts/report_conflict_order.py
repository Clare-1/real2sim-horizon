"""Conflict / ordering report: does the ORDER of two operations matter?

Reads the ``seq:<a>+<b>`` cells and reports, per cause:
  final SR, first-step SR, and the non-commutativity gap between a+ b and b+a.

The point is not "which sequence wins" but whether operations are additive,
super-additive (the first one enables the second) or destructive (the second loses value
without the first).  Pre-registered sequences only; no selection is fitted here.

    python scripts/report_conflict_order.py [results/experiment_conflict_order_20261123]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SHORT = {
    "calibrate_geometry": "C",
    "regenerate_targeted": "T",
    "regenerate_uniform": "U",
    "add_recovery_data": "R",
    "dagger_round": "D",
    "randomize_physics": "P",
    "deploy_dagger": "X",
    "mixed_train": "M",
    "stop": "S",
}


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def label(selector: str) -> str:
    if not selector.startswith("seq:"):
        return selector
    return "→".join(SHORT.get(a, a) for a in selector.split(":", 1)[1].split("+") if a)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "results/experiment_conflict_order_20261123"
    if not path.is_absolute():
        path = ROOT / path
    recs = load(path)
    if not recs:
        print(f"no records under {path}")
        return 1

    per = defaultdict(list)
    for r in recs:
        per[(r["cause"], label(r["selector"]))].append(r)

    causes = sorted({c for c, _ in per})
    lines = ["# 冲突 / 顺序实验（预注册序列）", "",
             "SR = 最终 SR（跨 bank 均值）；首步 = 第一步动作；Δ首轮 = 第一轮 SR 变化。", ""]
    for cause in causes:
        rows = sorted(k[1] for k in per if k[0] == cause)
        lines += [f"## {cause}", "",
                  "| 序列 | final SR | 首轮 ΔSR | 首步动作 | n |",
                  "|---|---:|---:|---|---:|"]
        for name in rows:
            group = per[(cause, name)]
            first = group[0]["decisions"][0] if group[0]["decisions"] else {}
            d1 = (first.get("sr_after", 0.0) - first.get("sr_before", 0.0)) if first else float("nan")
            lines.append(f"| {name} | {mean([g['final_sr'] for g in group]):.3f} | {d1:+.3f} | "
                         f"{first.get('action', '-')} | {len(group)} |")
        # non-commutativity: any a+b vs b+a pair present in this cause
        pairs = defaultdict(dict)
        for name in rows:
            if "→" not in name:
                continue
            a, b = name.split("→", 1)
            pairs[(a, b)] = name
        seen = set()
        gaps = []
        for (a, b), forward in pairs.items():
            reverse = pairs.get((b, a))
            if reverse and (b, a) not in seen and (a, b) != (b, a):
                seen.add((a, b))
                sf = mean([g["final_sr"] for g in per[(cause, forward)]])
                sr = mean([g["final_sr"] for g in per[(cause, reverse)]])
                gaps.append((a, b, sf, sr, sf - sr))
        if gaps:
            lines += ["", "| 顺序对 | 正序 SR | 反序 SR | 差（非交换性） |", "|---|---:|---:|---:|"]
            for a, b, sf, sr, g in gaps:
                lines.append(f"| {a}+{b} | {sf:.3f} | {sr:.3f} | {g:+.3f} |")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    (path / "conflict_order.md").write_text(text + "\n", encoding="utf-8")
    print(f"[written] {path / 'conflict_order.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
