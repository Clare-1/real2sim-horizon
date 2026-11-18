"""Progress-safe report for the four-cause selector experiment.

Unlike ``report_minimal_validation.py`` this works while the batch is still running: when
``summary.json`` is absent it reads the per-run ``record.json`` files that already exist, so the
same script is the checkpoint viewer and the final reporter.

    python scripts/report_four_causes.py [results/four_causes_20261118]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from report_minimal_validation import bootstrap_ci, paired_deltas  # noqa: E402

ACTIONS = ("calibrate_geometry", "regenerate_uniform", "regenerate_targeted",
           "add_recovery_data", "randomize_physics", "dagger_round", "stop")


def load(path: Path):
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"], "summary.json"
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]
    return recs, "per-run record.json (batch in progress)"


def expected_runs(path: Path) -> int | None:
    f = path / "experiment.json"
    if not f.exists():
        return None
    cfg = json.loads(f.read_text(encoding="utf-8"))["config"]
    return (len(cfg.get("scenarios", [])) * len(cfg.get("seeds", [])) *
            int(cfg.get("scenario_banks", 1)) * len(cfg.get("selectors", [])))


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else ROOT / "results/four_causes_20261118"
    if not path.is_absolute():
        path = ROOT / path
    recs, source = load(path)
    exp = expected_runs(path)
    complete = (path / "completion.json").exists()

    causes = [c for c in ("geometry_bias", "coverage_deficit", "recovery_deficit", "healthy")
              if any(r["cause"] == c for r in recs)]
    causes += sorted({r["cause"] for r in recs} - set(causes))
    sels = [s for s in ("fixed", "rule", "random", "round_robin", "bandit", "oracle")
            if any(r["selector"] == s for r in recs)]
    sels += sorted({r["selector"] for r in recs} - set(sels))
    banks = sorted({r.get("bank_index", 0) for r in recs})

    out = ["# 四类原因 × 选择器：进度与结果", "",
           f"- 数据源：`{path.relative_to(ROOT)}`（{source}）",
           f"- 已完成：**{len(recs)}**" + (f" / {exp}" if exp else "") +
           ("　→ **已完成**" if complete else "　→ 仍在运行"),
           f"- 场景实例（bank）：{banks}；原因：{causes}；选择器：{sels}", ""]

    out += ["## 1. 最终 SR（按原因；括号内为该格样本数）", "",
            "| 选择器 | " + " | ".join(causes) + " | 平均 |",
            "|---|" + "---:|" * len(causes) + "---:|"]
    for sel in sels:
        cells, vals = [], []
        for c in causes:
            v = [r["final_sr"] for r in recs if r["selector"] == sel and r["cause"] == c]
            cells.append(f"{sum(v)/len(v):.3f} ({len(v)})" if v else "-")
            vals += v
        out.append(f"| {sel} | " + " | ".join(cells) +
                   (f" | {sum(vals)/len(vals):.3f} |" if vals else " | - |"))
    out.append("")

    out += ["## 2. 首步选对率（按原因）", "",
            "| 选择器 | " + " | ".join(causes) + " | 合计 |",
            "|---|" + "---:|" * len(causes) + "---:|"]
    for sel in sels:
        cells, ok, tot = [], 0, 0
        for c in causes:
            rs = [r for r in recs if r["selector"] == sel and r["cause"] == c]
            k = sum(1 for r in rs if r.get("first_step_correct"))
            cells.append(f"{k}/{len(rs)}" if rs else "-")
            ok += k
            tot += len(rs)
        out.append(f"| {sel} | " + " | ".join(cells) + f" | {ok}/{tot} |")
    out.append("")

    out += ["## 3. 首步操作分布（行=真实原因，列=第一步选的操作）", ""]
    for sel in sels:
        out += [f"### {sel}", "",
                "| 真实原因 | " + " | ".join(a for a in ACTIONS) + " |",
                "|---|" + "---:|" * len(ACTIONS)]
        for c in causes:
            cnt = Counter(r.get("first_action") for r in recs
                          if r["selector"] == sel and r["cause"] == c)
            n = sum(cnt.values()) or 1
            out.append(f"| {c} | " + " | ".join(
                f"{cnt[a]/n:.0%}" if cnt.get(a) else "" for a in ACTIONS) + " |")
        out.append("")

    out += ["## 4. 相对 fixed 的配对差（同一 seed × 场景实例；95% bootstrap CI）", "",
            "| 原因 | 选择器 | 配对数 | 平均差 (pp) | 95% CI |",
            "|---|---|---:|---:|---|"]
    for c in causes:
        for sel in sels:
            if sel == "fixed":
                continue
            d = paired_deltas(recs, sel, c)
            if len(d) < 2:
                continue
            lo, hi = bootstrap_ci(d)
            out.append(f"| {c} | {sel} | {len(d)} | {100*sum(d)/len(d):+.1f} | "
                       f"[{100*lo:+.1f}, {100*hi:+.1f}] |")
    out.append("")

    out += ["## 5. 成本（每格均值）", "",
            "| 选择器 | 部署 episodes | 新增标签 | 校准次数 |",
            "|---|---:|---:|---:|"]
    for sel in sels:
        rs = [r for r in recs if r["selector"] == sel]
        if not rs:
            continue
        out.append(f"| {sel} | {sum(r['deploy_episodes'] for r in rs)/len(rs):.0f} | "
                   f"{sum(r['new_labels'] for r in rs)/len(rs):.0f} | "
                   f"{sum(r.get('calibrations', 0) for r in rs)/len(rs):.1f} |")
    out.append("")

    text = "\n".join(out)
    dest = path / ("report.md" if complete else "partial_report.md")
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
