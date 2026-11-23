"""Tables 9 and 10 of experiment 5.

Table 9  bias-response curve: reconstruction bias 0.20 / 0.46 / 0.70 x fixed / rule / oracle.
Table 10 agent ablation:    llm / llm_memory / llm_minimal / llm_cheap at the 4-cause grid,
                            with LLM calls and spend per cell.

    python scripts/report_experiment5.py [--bias DIR] [--agent DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLS = {"geometry_bias": "几何", "coverage_deficit": "覆盖",
        "recovery_deficit": "恢复", "healthy": "健康"}
ARM_LABEL = {"llm": "LLM（全工具，无经验）", "llm_memory": "LLM + 经验",
             "llm_minimal": "LLM 去专门工具", "llm_cheap": "LLM 便宜模型"}
BIAS_LABEL = {"bias_020": "0.20（弱）", "bias_046": "0.46（校准）", "bias_070": "0.70（强）"}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
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
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def table9(recs, out_lines):
    if not recs:
        out_lines += ["## 表 9：偏差档响应", "", "_未跑_", ""]
        return
    scen = [s for s in BIAS_LABEL if any(r.get("scenario") == s for r in recs)]
    scen += sorted({r.get("scenario") for r in recs} - set(scen) - {None})
    sels = sorted({r["selector"] for r in recs})
    out_lines += ["## 表 9：偏差档响应（高度偏差 0.20 / 0.46 / 0.70）", "",
                  "| 偏差档 | " + " | ".join(sels) + " | 首步对 | 标签/格 |",
                  "|---|" + "---:|" * len(sels) + "---:|---:|"]
    for s in scen:
        cells, firsts = [], []
        for sel in sels:
            v = [r["final_sr"] for r in recs if r.get("scenario") == s and r["selector"] == sel]
            cells.append(f"{np.mean(v):.3f}" if v else "-")
            firsts += [bool(r.get("first_step_correct")) for r in recs
                       if r.get("scenario") == s and r["selector"] == sel]
        lab = [r["new_labels"] for r in recs if r.get("scenario") == s]
        out_lines.append(f"| {BIAS_LABEL.get(s, s)} | " + " | ".join(cells) +
                         f" | {sum(firsts)}/{len(firsts)} | {np.mean(lab) if lab else 0:.0f} |")
    out_lines += ["", "偏差越大，越依赖“校准 twin”类动作；oracle 是诊断上界。", ""]


def table10(recs, out_lines):
    if not recs:
        out_lines += ["## 表 10：LLM agent 消融", "", "_未跑_", ""]
        return
    causes = [c for c in COLS if any(r["cause"] == c for r in recs)]
    arms = [a for a in ARM_LABEL if any(r["selector"] == a for r in recs)]
    arms += sorted({r["selector"] for r in recs} - set(arms))
    out_lines += ["## 表 10：LLM agent 消融", "",
                  "| 臂 | " + " | ".join(COLS[c] for c in causes) +
                  " | 平均 | 首步对 | 调用/格 | 花费/格 (USD) | 动作数/格 |",
                  "|---|" + "---:|" * len(causes) + "---:|---:|---:|---:|---:|"]
    for a in arms:
        cells, allv, firsts, calls, usd, acts = [], [], [], [], [], []
        for c in causes:
            rs = [r for r in recs if r["selector"] == a and r["cause"] == c]
            v = [r["final_sr"] for r in rs]
            cells.append(f"{np.mean(v):.3f}" if v else "-")
            allv += v
            firsts += [bool(r.get("first_step_correct")) for r in rs]
            calls += [int(r.get("llm_calls", 0)) for r in rs]
            usd += [float(r.get("llm_usd", 0.0)) for r in rs]
            acts += [len(r.get("decisions") or []) for r in rs]
        lo, hi = boot(allv)
        mean = f"**{np.mean(allv):.3f}**" if allv else "-"
        out_lines.append(
            f"| {ARM_LABEL.get(a, a)} | " + " | ".join(cells) + f" | {mean} | "
            f"{sum(firsts)}/{len(firsts)} | {np.mean(calls) if calls else 0:.1f} | "
            f"{np.mean(usd) if usd else 0:.4f} | {np.mean(acts) if acts else 0:.1f} |")
    out_lines += ["", "花费取自各 run 记录的 `llm_usd`（按配置中的单价估算）；",
                  "`llm_cheap` 在该代理上由服务端计费为 $0，估算列仍按单价折算，两者需同时报告。", ""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", default="results/experiment5_bias_levels_20261123")
    ap.add_argument("--agent", default="results/experiment5_agent_ablation_20261123")
    args = ap.parse_args()
    bias = load((ROOT / args.bias) if not Path(args.bias).is_absolute() else Path(args.bias))
    agent = load((ROOT / args.agent) if not Path(args.agent).is_absolute() else Path(args.agent))
    lines = ["# 实验五：表 9（偏差档）与表 10（LLM agent 消融）", ""]
    table9(bias, lines)
    table10(agent, lines)
    dest = ROOT / "results/experiment5_report.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[written] {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
