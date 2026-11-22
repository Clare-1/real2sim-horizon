"""Dev/test bank split for the 10-bank minimal validation (seed 0).

Banks 0-2 are the ones used by the paper8 batch, banks 3-9 are new. The rule thresholds are
hard-coded (not tuned per bank), so this is a held-out-instance check, not a tuning experiment:
if the ranking survives on banks 3-9, the rule's advantage is not a bank-0-2 artefact.

    python scripts/split_banks_report.py [results/minimal_validation_advisor_20261121] [seed]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEV = {0, 1, 2}
TEST = set(range(3, 10))


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/minimal_validation_advisor_20261121"
    if not path.is_absolute():
        path = ROOT / path
    seed_arg = sys.argv[2] if len(sys.argv) > 2 else "0"
    recs = load(path)
    if seed_arg != "all":
        recs = [r for r in recs if r.get("seed") == int(seed_arg)]
    label = "all seeds" if seed_arg == "all" else f"seed={seed_arg}"
    sels = [s for s in ("fixed", "rule", "bandit", "oracle")
            if any(r["selector"] == s for r in recs)]
    causes = [c for c in ("geometry_bias", "coverage_deficit")
              if any(r["cause"] == c for r in recs)]
    banks = sorted({r["bank_index"] for r in recs})

    def mean(sel, cause, pool):
        v = [r["final_sr"] for r in recs
             if r["selector"] == sel and r["cause"] == cause and r["bank_index"] in pool]
        return sum(v) / len(v) if v else float("nan")

    out = ["# 10-bank 最小验证：dev (0-2) vs held-out (3-9)", "",
           f"数据：`{path.relative_to(ROOT)}`，{label}；dev={sorted(DEV)}，test={sorted(TEST)}",
           "", "| 选择器 | 原因 | dev (0-2) | held-out (3-9) | 差 (test-dev) |",
           "|---|---|---:|---:|---:|"]
    for s in sels:
        for c in causes:
            d, t = mean(s, c, DEV), mean(s, c, TEST)
            out.append(f"| {s} | {c} | {d:.3f} | {t:.3f} | {100*(t-d):+.1f} pp |")
    out += ["", "## 逐 bank", "",
            "| 选择器 | 原因 | " + " | ".join(f"b{b}" for b in banks) + " |",
            "|---|---|" + "---:|" * len(banks)]
    for s in sels:
        for c in causes:
            cells = []
            for b in banks:
                v = [r["final_sr"] for r in recs
                     if r["selector"] == s and r["cause"] == c and r["bank_index"] == b]
                cells.append(f"{v[0]:.2f}" if v else "-")
            out.append(f"| {s} | {c} | " + " | ".join(cells) + " |")
    out += ["", "说明：rule 阈值是硬编码的，不是在这 3 个 dev banks 上调的；"
            "因此这里检验的是“换 bank 后排序是否保持”，不是完整的调参/测试分离。"]
    text = "\n".join(out)
    dest = path / "bank_split_report.md"
    dest.write_text(text, encoding="utf-8")
    print(text[:1800])
    print(f"\n[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
