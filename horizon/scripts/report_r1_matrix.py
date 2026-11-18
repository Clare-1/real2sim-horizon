"""Summarise the R1 matrix: operation x cause, with paired deltas and 95% bootstrap CI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_minimal_validation import bootstrap_ci  # noqa: E402

OPS = ("nothing", "calibrate_geometry", "regenerate_uniform", "regenerate_targeted", "all")


def main(root: str) -> int:
    recs = json.loads((Path(root) / "summary.json").read_text(encoding="utf-8"))["records"]
    causes = sorted({r["cause"] for r in recs})
    print(f"cells={len(recs)}  causes={causes}\n")
    print("| 操作 | " + " | ".join(f"{c} ΔSR (pp)" for c in causes) + " |")
    print("|---|" + "---|" * len(causes))
    for op in OPS:
        cells = []
        for cause in causes:
            d = [r["delta"] for r in recs if r["op"] == op and r["cause"] == cause]
            if not d:
                cells.append("-")
                continue
            lo, hi = bootstrap_ci(d)
            cells.append(f"{100*sum(d)/len(d):+.1f} [{100*lo:+.1f},{100*hi:+.1f}] n={len(d)}")
        print(f"| {op} | " + " | ".join(cells) + " |")

    print("\n| 原因 | 种子 | 基线 SR | nothing | calibrate | uniform | targeted | all |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for cause in causes:
        base = {r["seed"]: r["sr_before"] for r in recs if r["cause"] == cause}
        for seed in sorted(base):
            row = {r["op"]: r["sr_after"] for r in recs if r["cause"] == cause and r["seed"] == seed}
            print(f"| {cause} | {seed} | {base[seed]:.3f} | " +
                  " | ".join(f"{row.get(op, float('nan')):.3f}" for op in OPS) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
