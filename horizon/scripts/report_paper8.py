"""Paper-batch report: selector x cause with 95% bootstrap CIs over (seed, bank) cells.

    python scripts/report_paper8.py [results/four_causes_paper8_20261120]

Unlike report_four_causes.py this treats each (selector, cause) cell as a mean over independent
(seed, bank) instances and reports a CI, which is what a paper table needs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

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
    cfg = {}
    if exp.exists():
        cfg = json.loads(exp.read_text(encoding="utf-8"))["config"]
        expected = (len(cfg["scenarios"]) * len(cfg["seeds"]) *
                    int(cfg.get("scenario_banks", 1)) * len(cfg["selectors"]))
    step = float(cfg.get("feedback_episodes", 24))
    horizon = step * int(cfg.get("rounds", 3))
    # Preferred order first, then anything else present (mixed-cause runs use mixed_gc / mixed_cr /
    # mixed_gcr, which the old hard-coded list dropped, producing empty columns and nan AUC).
    preferred = ("geometry_bias", "coverage_deficit", "recovery_deficit", "healthy",
                 "mixed_gc", "mixed_gr", "mixed_cr", "mixed_gcr")
    present = {r["cause"] for r in recs}
    causes = [c for c in preferred if c in present]
    causes += sorted(present - set(causes))
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

    means = {}
    for s in sels + ["oracle"]:
        for c in causes:
            v = [r["final_sr"] for r in recs if r["selector"] == s and r["cause"] == c]
            if v:
                means[(s, c)] = sum(v) / len(v)

    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    def points(rec):
        ds = rec.get("decisions") or []
        if not ds:
            return []
        pts = [(0.0, float(ds[0]["sr_before"]))]
        for i, d in enumerate(ds):
            pts.append((step * (i + 1), float(d["sr_after"])))
        return pts

    def area(rec):
        pts = points(rec)
        if not pts:
            return None
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        if x[-1] < horizon:
            x.append(horizon)
            y.append(y[-1])
        return float(trap(y, x) / horizon)

    def first_reach(rec, thr):
        for x, y in points(rec):
            if y >= thr:
                return x
        return None

    has_health = "healthy" in causes
    head = ("| 选择器 | SR-成本 AUC | oracle-90% 部署（达到/总数） | regret (pp) | 首步对 |"
            + (" 健康标签/格 | 健康部署/格 |" if has_health else ""))
    out += ["## 表 3（当前 paper8）：效率与诊断", "", head,
            "|---|" + "---:|" * (4 + (2 if has_health else 0))]
    for s in sels:
        aucs, reaches = [], []
        regrets = []
        n_ok = sum(1 for r in recs if r["selector"] == s and r.get("first_step_correct"))
        n_all = sum(1 for r in recs if r["selector"] == s)
        for c in causes:
            thr = 0.9 * means.get(("oracle", c), 0.0)
            for r in [r for r in recs if r["selector"] == s and r["cause"] == c]:
                a = area(r)
                if a is not None:
                    aucs.append(a)
                d = first_reach(r, thr)
                if d is not None:
                    reaches.append(d)
            if ("oracle", c) in means and (s, c) in means:
                regrets.append(means[("oracle", c)] - means[(s, c)])
        health = [r for r in recs if r["selector"] == s and r["cause"] == "healthy"]
        h_lab = sum(r["new_labels"] for r in health) / len(health) if health else 0.0
        h_dep = sum(r["deploy_episodes"] for r in health) / len(health) if health else 0.0
        n_cause = sum(1 for r in recs if r["selector"] == s)
        row = (f"| {s} | {np.mean(aucs):.3f} | "
               f"{np.mean(reaches):.1f} ({len(reaches)}/{n_cause}) | "
               f"{100*np.mean(regrets):+.1f} | {n_ok}/{n_all} |")
        if has_health:
            row += f" {h_lab:.0f} | {h_dep:.0f} |"
        out.append(row)
    out += ["", "AUC 由每轮 `sr_before/sr_after` 折线在 0–horizon 上积分；"
            "oracle-90% 是首次达到 0.9×oracle 均值时的累计部署；健康花费按标签/部署 episodes 报。", ""]

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
