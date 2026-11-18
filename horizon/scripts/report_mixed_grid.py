"""Mixed-cause grid report: how much of the oracle gap does each selector recover?

Classification accuracy is the wrong metric for mixtures: a selector can pick a component action
or a two-step sequence and still help. This script scores final SR relative to the fixed recipe and
the oracle:

    gap_recovery = (SR_selector - SR_fixed) / (SR_oracle - SR_fixed)

    python scripts/report_mixed_grid.py [results/mixed_grid_20261118]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = {
    "mixed_gc": {"calibrate_geometry", "regenerate_targeted"},
    "mixed_gr": {"calibrate_geometry", "add_recovery_data"},
    "mixed_cr": {"regenerate_targeted", "add_recovery_data"},
    "mixed_gcr": {"calibrate_geometry", "regenerate_targeted", "add_recovery_data"},
    "mixed_cause": {"calibrate_geometry", "regenerate_targeted"},
}


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/mixed_grid_20261118"
    if not path.is_absolute():
        path = ROOT / path
    recs = load(path)
    conds = [c for c in COMPONENTS if any(r["cause"] == c for r in recs)]
    conds += sorted({r["cause"] for r in recs} - set(conds))
    sels = [s for s in ("fixed", "rule", "random", "round_robin", "bandit", "oracle")
            if any(r["selector"] == s for r in recs)]
    sels += sorted({r["selector"] for r in recs} - set(sels))
    expected = None
    exp_file = path / "experiment.json"
    if exp_file.exists():
        cfg = json.loads(exp_file.read_text(encoding="utf-8"))["config"]
        expected = (len(cfg["scenarios"]) * len(cfg["seeds"]) *
                    int(cfg.get("scenario_banks", 1)) * len(cfg["selectors"]))

    out = ["# 混合原因网格：选择器能恢复多少 oracle gap", "",
           f"- 数据源：`{path.relative_to(ROOT)}`",
           f"- 已完成：**{len(recs)}**" + (f" / {expected}" if expected else ""), ""]
    out += ["## 1. 最终 SR", "",
            "| 条件 | " + " | ".join(sels) + " |",
            "|---|" + "---:|" * len(sels)]
    means = {}
    for c in conds:
        cells = []
        for s in sels:
            v = [r["final_sr"] for r in recs if r["cause"] == c and r["selector"] == s]
            means[(c, s)] = (sum(v) / len(v)) if v else None
            cells.append(f"{means[(c, s)]:.3f}" if v else "-")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("")

    out += ["## 2. gap recovery（相对 fixed；1.0 = 追平 oracle）", "",
            "| 条件 | fixed | oracle | " + " | ".join(s for s in sels if s not in ("fixed", "oracle")) + " |",
            "|---|---:|---:|" + "---:|" * len([s for s in sels if s not in ("fixed", "oracle")])]
    for c in conds:
        f, o = means.get((c, "fixed")), means.get((c, "oracle"))
        cells = []
        for s in sels:
            if s in ("fixed", "oracle"):
                continue
            m = means.get((c, s))
            cells.append("-" if (f is None or o is None or m is None or abs(o - f) < 1e-9)
                         else f"{100*(m - f)/(o - f):+.0f}%")
        out.append(f"| {c} | " + (f"{f:.3f}" if f is not None else "-") + " | " +
                   (f"{o:.3f}" if o is not None else "-") + " | " + " | ".join(cells) + " |")
    out.append("")

    out += ["## 3. 首步是否落在任一正确操作（混合不要求唯一答案）", "",
            "| 条件 | " + " | ".join(sels) + " |",
            "|---|" + "---:|" * len(sels)]
    for c in conds:
        allowed = COMPONENTS.get(c, set())
        cells = []
        for s in sels:
            rs = [r for r in recs if r["cause"] == c and r["selector"] == s]
            if not rs:
                cells.append("-")
                continue
            k = sum(1 for r in rs if r.get("first_action") in allowed)
            cells.append(f"{k}/{len(rs)}")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("")

    out += ["## 4. 成本（每格均值）", "", "| 选择器 | 标签 | 部署 |", "|---|---:|---:|"]
    for s in sels:
        rs = [r for r in recs if r["selector"] == s]
        if rs:
            out.append(f"| {s} | {sum(r['new_labels'] for r in rs)/len(rs):.0f} | "
                       f"{sum(r['deploy_episodes'] for r in rs)/len(rs):.0f} |")
    out.append("")
    text = "\n".join(out)
    dest = path / "mixed_report.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
