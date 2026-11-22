"""Put the LLM pilot next to the scripted selectors on the same 4-cause grid.

The 48-run screening batch already records ``first_step_correct`` per run, so the only new
number is the LLM row. This script reads both and prints one table; it never re-runs anything.

    python scripts/report_llm_pilot.py
    python scripts/report_llm_pilot.py --runs results/four_causes_20261118 \
        --llm results/llm_pilot_20261122.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CAUSES = ("geometry_bias", "coverage_deficit", "recovery_deficit", "healthy")
SHORT = {"geometry_bias": "geo", "coverage_deficit": "cov",
         "recovery_deficit": "rec", "healthy": "hlth"}


def load_records(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/four_causes_20261118")
    ap.add_argument("--llm", nargs="+",
                    default=["results/llm_pilot_20261122.json",
                             "results/llm_pilot_paper8_20261122.json"])
    args = ap.parse_args()

    runs = load_records(ROOT / args.runs)
    pilots = [json.loads((ROOT / p).read_text(encoding="utf-8")) for p in args.llm]

    first = defaultdict(lambda: defaultdict(list))
    sr = defaultdict(lambda: defaultdict(list))
    for r in runs:
        first[r["selector"]][r["cause"]].append(int(bool(r.get("first_step_correct"))))
        sr[r["selector"]][r["cause"]].append(float(r["final_sr"]))

    llm_first = defaultdict(list)
    llm_calls = 0
    llm_usd = 0.0
    llm_rows = []
    for p in pilots:
        llm_calls += int(p.get("n_calls", len(p["decisions"])))
        llm_usd += float(p.get("spent_usd", 0.0))
        for row in p["decisions"]:
            llm_first[row["cause"]].append(int(bool(row["correct"])))
            llm_rows.append(row)

    lines = ["# 首步正确率：LLM vs 脚本选择器（同一 4 原因网格）", "",
             "48 格筛查批次（`results/four_causes_20261118`，每格 2 个 bank）"
             f" + LLM pilot（{llm_calls} 次调用，${llm_usd:.4f}）", "",
             "| 选择器 | " + " | ".join(SHORT[c] for c in CAUSES) + " | 合计 |",
             "|---|" + "---:|" * (len(CAUSES) + 1)]

    for sel in sorted(first):
        cells = []
        tot, ok = 0, 0
        for c in CAUSES:
            v = first[sel].get(c, [])
            cells.append(f"{sum(v)}/{len(v)}" if v else "-")
            tot += len(v)
            ok += sum(v)
        lines.append(f"| {sel} | " + " | ".join(cells) + f" | {ok}/{tot} |")

    cells = []
    tot, ok = 0, 0
    for c in CAUSES:
        v = llm_first.get(c, [])
        cells.append(f"{sum(v)}/{len(v)}" if v else "-")
        tot += len(v)
        ok += sum(v)
    lines.append(f"| **LLM** | " + " | ".join(cells) + f" | {ok}/{tot} |")

    lines += ["", "## 最终 SR（脚本选择器；LLM 只有首步决策，没有在线 rollout）", "",
              "| 选择器 | " + " | ".join(SHORT[c] for c in CAUSES) + " | 平均 |",
              "|---|" + "---:|" * (len(CAUSES) + 1)]
    for sel in sorted(sr):
        cells = []
        vals = []
        for c in CAUSES:
            v = sr[sel].get(c, [])
            cells.append(f"{sum(v) / len(v):.3f}" if v else "-")
            vals += v
        mean = f"{sum(vals) / len(vals):.3f}" if vals else "-"
        lines.append(f"| {sel} | " + " | ".join(cells) + f" | {mean} |")

    lines += ["", "## LLM 逐条决策", ""]
    for row in llm_rows:
        flag = "OK" if row["correct"] else "MISS"
        lines.append(f"- {row['scenario']} b{row['bank_index']} → `{row['action']}` "
                     f"({flag}, expected `{row.get('expected')}`)")

    text = "\n".join(lines) + "\n"
    dest = ROOT / "results" / "llm_vs_rule_20261122.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"[written] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
