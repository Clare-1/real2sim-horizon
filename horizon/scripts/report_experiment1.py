"""Advisor Table 1 layout: the eight fixed pipelines against each failure condition.

Rows are pipelines (open loop / fixed twin / domain randomisation / deployment DAgger /
sim+deployment mix / calibrate once / repeated calibration / true params), columns are the failure
conditions, and the last columns give the cost and the paired difference against the fixed-twin
DAgger baseline (advisor method 2).

    python scripts/report_experiment1.py [results/experiment1_table1_20261122]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# pipeline -> (label, uses feedback, uses twin, calibrates twin)  -- the advisor's first columns
META = {
    "open_loop": ("1 开环生成 200k", "✗", "✓", "✗"),
    "always_dagger": ("2 固定 twin + DAgger", "✓", "✓", "✗"),
    "domain_randomize": ("3 领域随机化 + DAgger", "✓", "✓", "✗"),
    "deploy_dagger": ("4 部署环境 DAgger", "✓", "✗", "—"),
    "mixed_train": ("5 仿真 + 部署混合", "✓", "✓", "✗"),
    "fixed": ("6 首轮校准后冻结", "✓", "✓", "✓"),
    "calibrate_alternate": ("7 重复校准 + DAgger", "✓", "✓", "✓"),
    "true_params": ("8 真实参数当 twin", "✓", "✓", "真值"),
}
ORDER = list(META)
COND_LABEL = {"geometry_bias": "高度偏差", "coverage_deficit": "覆盖不足",
              "recovery_deficit": "恢复不足", "healthy": "健康"}


def load(path: Path) -> list[dict]:
    f = path / "summary.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["records"]
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(path.glob("runs/*/*/*/*/record.json"))]


def boot(vals, n=2000, seed=0):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    means = rng.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/experiment1_table1_20261122"
    if not path.is_absolute():
        path = ROOT / path
    recs = load(path)
    conds = [c for c in COND_LABEL if any(r["cause"] == c for r in recs)]
    rows = [s for s in ORDER if any(r["selector"] == s for r in recs)]
    rows += sorted({r["selector"] for r in recs} - set(rows))
    base = "always_dagger"

    # Pair by (cause, seed, bank): the matched design is what makes a paired test meaningful.
    # The previous version paired *sorted* SR lists, which matched different task instances.
    def val(sel, cond):
        return {(int(r["seed"]), int(r["bank_index"])): float(r["final_sr"])
                for r in recs if r["selector"] == sel and r["cause"] == cond}

    def cell(sel, cond):
        return [r["final_sr"] for r in recs if r["selector"] == sel and r["cause"] == cond]

    out = [f"# 表 1（实验一）：8 条固定流程 × 失败原因", "",
           f"- 数据源：`{path.relative_to(ROOT)}`，**{len(recs)}** 格", ""]
    out += ["| 组别 | 方法 | 用反馈 | 用 twin | 校准 twin | " +
            " | ".join(COND_LABEL[c] for c in conds) + " | 平均 | 标签/格 | 部署/格 |",
            "|---|---|---|---|---|" + "---:|" * len(conds) + "---:|---:|---:|"]
    for sel in rows:
        label, fb, twin, cal = META.get(sel, (sel, "?", "?", "?"))
        cells, allv = [], []
        for c in conds:
            v = cell(sel, c)
            cells.append(f"{np.mean(v):.3f}" if v else "-")
            allv += v
        lab = [r["new_labels"] for r in recs if r["selector"] == sel]
        dep = [r["deploy_episodes"] for r in recs if r["selector"] == sel]
        out.append(f"| | {label} | {fb} | {twin} | {cal} | " + " | ".join(cells) +
                   f" | {'**' + format(np.mean(allv), '.3f') + '**' if allv else '-'} | "
                   f"{np.mean(lab) if lab else 0:.0f} | {np.mean(dep) if dep else 0:.0f} |")

    out += ["", "## 与方法 2（固定 twin + DAgger）的配对差", "",
            "按 (seed, bank) 匹配同一批任务实例；每格为该原因下的 ΔSR（pp）与 95% 区间，"
            "最后一列把各原因的配对差合并（跨原因汇总，需与分原因一起读）。", "",
            "| 方法 | " + " | ".join(COND_LABEL[c] for c in conds) + " | 汇总 |",
            "|---|" + "---|" * (len(conds) + 1)]
    for sel in rows:
        if sel == base:
            continue
        label = META.get(sel, (sel,))[0]
        row, pooled = [], []
        for c in conds:
            a, b = val(sel, c), val(base, c)
            keys = sorted(set(a) & set(b))
            diffs = [a[k] - b[k] for k in keys]
            pooled += diffs
            if not diffs:
                row.append("—")
                continue
            lo, hi = boot(diffs)
            star = "*" if (lo > 0 or hi < 0) else ""
            row.append(f"{100*np.mean(diffs):+.1f} [{100*lo:+.1f},{100*hi:+.1f}]{star} (n={len(diffs)})")
        if pooled:
            lo, hi = boot(pooled)
            star = "*" if (lo > 0 or hi < 0) else ""
            row.append(f"**{100*np.mean(pooled):+.1f}** [{100*lo:+.1f},{100*hi:+.1f}]{star} (n={len(pooled)})")
        else:
            row.append("—")
        out.append(f"| {label} | " + " | ".join(row) + " |")

    out += ["", "方法 8（真实参数当 twin）是上界参考，不是可部署方法；`*` 表示区间不含 0。", ""]
    text = "\n".join(out) + "\n"
    dest = path / "table1_report.md"
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
