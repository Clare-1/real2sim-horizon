"""Merge the LLM selector into the paper table.

The LLM grid is small (4 causes x 2 banks, 1 seed) because every call costs money, so it is
reported next to the 288-run paper8 grid instead of being pooled with it. Numbers are therefore
screening-level; the point is that the LLM row now has a *final SR*, not only first-step choices.

    python scripts/report_llm_grid.py [results/four_causes_llm_20261122]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER8 = ROOT / "results" / "four_causes_paper8_20261120"
CAUSES = ("geometry_bias", "coverage_deficit", "recovery_deficit", "healthy")
SHORT = {"geometry_bias": "geo", "coverage_deficit": "cov",
         "recovery_deficit": "rec", "healthy": "hlth"}
COMPARE = ("fixed", "rule", "bandit", "multilabel", "linucb", "oracle")


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def main() -> int:
    llm_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "four_causes_llm_20261122"
    if not llm_dir.is_absolute():
        llm_dir = ROOT / llm_dir
    llm = load(llm_dir)
    base = load(PAPER8) if PAPER8.exists() else []

    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in base + llm:
        agg[(r["selector"], r["cause"])].append(r)

    lines = ["# LLM selector in the main table", "",
             f"LLM grid: {len(llm)} runs in `{llm_dir.relative_to(ROOT)}` "
             f"(4 causes x 2 banks x 1 seed).",
             f"Comparison: {len(base)} runs in `{PAPER8.relative_to(ROOT)}` (3 seeds x 3 banks).",
             "Both are final test SR; the LLM row is screening-level (2 banks, 1 seed).", ""]

    head = "| selector | " + " | ".join(SHORT[c] for c in CAUSES) + " | mean | first-step |"
    lines += [head, "|" + "---|" * (len(CAUSES) + 3)]
    rows = []
    for sel in list(COMPARE) + ["llm"]:
        vals, hits, total = [], 0, 0
        for c in CAUSES:
            rs = agg.get((sel, c), [])
            if not rs:
                vals.append(None)
                continue
            vals.append(sum(r["final_sr"] for r in rs) / len(rs))
            hits += sum(1 for r in rs if r.get("first_step_correct"))
            total += len(rs)
        if all(v is None for v in vals):
            continue
        mean = [v for v in vals if v is not None]
        mean = sum(mean) / len(mean) if mean else None
        rows.append((sel, vals, mean, hits, total))
    rows.sort(key=lambda x: (-(x[2] or 0)))
    for sel, vals, mean, hits, total in rows:
        cells = ["—" if v is None else f"{v:.3f}" for v in vals]
        lines.append(f"| {sel} | " + " | ".join(cells) +
                     f" | {'—' if mean is None else f'{mean:.3f}'} | {hits}/{total} |")

    llm_calls = sum(r.get("llm_calls", 0) for r in llm)
    llm_usd = sum(r.get("llm_usd", 0.0) for r in llm)
    lines += ["", "## LLM cost", "",
              f"- runs: {len(llm)}; calls: {llm_calls}; spend: ${llm_usd:.4f} "
              f"(mean ${llm_usd / max(llm_calls, 1):.4f}/call)",
              f"- actions per run: " +
              ", ".join(f"{r['scenario']}/b{r['bank_index']}: " +
                        "→".join(d["action"] for d in r["decisions"])
                        for r in sorted(llm, key=lambda x: (x["scenario"], x["bank_index"]))), ""]

    text = "\n".join(lines)
    dest = ROOT / "results" / "llm_in_main_table.md"
    dest.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
