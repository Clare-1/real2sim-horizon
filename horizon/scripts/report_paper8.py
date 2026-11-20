"""Paper-batch report: selector x cause with 95% bootstrap CIs over (seed, bank) cells.

    python scripts/report_paper8.py [results/four_causes_paper8_20261120]

Unlike report_four_causes.py this treats each (selector, cause) cell as a mean over independent
(seed, bank) instances and reports a CI, which is what a paper table needs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from report_minimal_validation import bootstrap_ci  # noqa: E402


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/four_causes_paper8_20261120"
    if not path.is_absolute():
        path = ROOT / path
    recs = load(path)
    expected = None
    exp = path / "experiment.json"
    if exp.exists():
        cfg = json.loads(exp.read_text(encoding="utf-8"))["config"]
        expected = (len(cfg["scenarios"]) * len(cfg["seeds"]) *
                    int(cfg.get("scenario_banks", 1)) * len(cfg["selectors"]))
    causes = [c for c in ("geometry_bias", "coverage_deficit", "recovery_deficit", "healthy")
              if any(r["cause"] == c for r in recs)]
    sels = [s for s in ("fixed", "rule", "random", "round_robin", "bandit",
                        "linucb", "multilabel", "learned_value", "oracle")
            if any(r["selector"] == s for r in recs)]
    sels += sorted({r["selector"] for r in recs} - set(sels))

    out = ["# 论文批次：选择器 × 原因（95% bootstrap CI）", "",
           f"- 数据源：`{path.relative_to(ROOT)}`",
           f"- 已完成：**{len(recs)}**" + (f" / {expected}" if expected else ""), ""]
    out += ["| 选择器 | " + " | ".join(causes) + " | 平均 | 首步对 |",
            "|---|" + "---:|" * len(causes) + "---:|---:|"]
    for s in sels:
        cells, vals = [], []
        ok, tot = 0, 0
        for c in causes:
            rs = [r for r in recs if r["selector"] == s and r["cause"] == c]
            if not rs:
                cells.append("-")
                continue
            v = [r["final_sr"] for r in rs]
            lo, hi = bootstrap_ci(v)
            mean = sum(v) / len(v)
            cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            vals += v
            ok += sum(1 for r in rs if r.get("first_step_correct"))
            tot += len(rs)
        avg = f"{sum(vals)/len(vals):.3f}" if vals else "-"
        out.append(f"| {s} | " + " | ".join(cells) + f" | {avg} | {ok}/{tot} |")
    out.append("")

    n_cells = {c: len({(r.get("seed"), r.get("bank_index", 0)) for r in recs if r["cause"] == c})
               for c in causes}
    out += ["每个 (selector, cause) 格由 " + ", ".join(f"{c}: {n_cells[c]}" for c in causes) +
            " 个 (seed, bank) 实例组成；单实例不足 2 个时 CI 无定义（显示 nan）。", ""]
    text = "\n".join(out)
    dest = path / "paper8_report.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
