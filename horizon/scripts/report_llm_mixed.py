"""LLM vs rule on mixed causes, with the rule's label gate both ways.

Three batches are read together because they answer one question: on a failure that has more
than one component, does an API-backed planner beat the hand-written rule, and does the answer
survive giving the rule the one documented fix (`tau_labels_low` 20k -> 200k)?

    python scripts/report_llm_mixed.py

Absolute SR and paired differences are reported separately. The paired column resamples matched
(condition, bank) cells and drops cells that only one arm has, so a missing run is never imputed.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BATCHES = {
    "rule": ("results/experiment_llm_mixed_20261123", "rule"),
    "rule_fix": ("results/experiment_llm_mixed_rulefix_20261123", "rule"),
    "llm": ("results/experiment_llm_mixed_20261123", "llm"),
    "llm_memory": ("results/experiment_llm_mixed_20261123", "llm_memory"),
    "oracle": ("results/experiment_llm_mixed_20261123", "oracle"),
}
CAUSES = ["mixed_gc", "mixed_cr", "mixed_gcr"]


def load(dirname: str, selector: str) -> list[dict]:
    d = ROOT / dirname
    out = []
    for p in sorted(d.glob("runs/*/*/*/*/record.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("selector") == selector:
            out.append(r)
    return out


SHORT = {"calibrate_geometry": "C", "regenerate_targeted": "T", "regenerate_uniform": "U",
         "add_recovery_data": "R", "dagger_round": "D", "stop": "-", "randomize_physics": "P"}


def boot_ci(vals: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n, a.size), replace=True).mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    tables: dict[str, dict[tuple[str, int], float]] = {}
    meta: dict[str, dict[str, float]] = {}
    for arm, (dirname, selector) in BATCHES.items():
        recs = load(dirname, selector)
        tables[arm] = {(r["cause"], int(r["bank_index"])): float(r["final_sr"]) for r in recs}
        usd = [r.get("llm_usd", 0.0) or 0.0 for r in recs]
        # `llm_usage` is a list of per-call dicts; the served-side charge is per call.
        prov = [sum(float(u.get("provider_cost", 0.0) or 0.0) for u in (r.get("llm_usage") or []))
                for r in recs]
        calls = [int(r.get("llm_calls", 0) or 0) for r in recs]
        meta[arm] = {
            "n": len(recs),
            "calls": float(np.mean(calls)) if calls else 0.0,
            "usd": float(np.mean(usd)) if usd else 0.0,
            "provider": float(np.mean(prov)) if prov else 0.0,
        }

    out = ["# LLM vs rule on mixed causes", "",
           "- 数据：`results/experiment_llm_mixed_20261123`（24 格，3 条件 × 2 banks × 1 seed）",
           "- 对照：`results/experiment_llm_mixed_rulefix_20261123`（同一规则，`tau_labels_low` 200k）",
           "", "## 最终 SR", "",
           "| 臂 | " + " | ".join(CAUSES) + " | 平均 | $/格（实计） | LLM 调用/格 |",
           "|---|" + "---:|" * (len(CAUSES) + 3)]
    for arm in BATCHES:
        cells = tables[arm]
        row = [f"{cells[c, b]:.3f}" for c, b in [(c, b) for c in CAUSES
                                                for b in sorted({bb for cc, bb in cells if cc == c})]]
        per = [np.mean([v for (cc, _), v in cells.items() if cc == c]) for c in CAUSES]
        allv = list(cells.values())
        fmt = [f"{v:.3f}" if np.isfinite(v) else "—" for v in per]
        out.append(f"| {arm} | " + " | ".join(fmt) +
                   f" | **{np.mean(allv):.3f}** | {meta[arm]['provider']:.4f} | {meta[arm]['calls']:.1f} |")
    out += ["", "## 配对差（对 `rule`，按 (condition, bank) 匹配）", "",
            "| 臂 | ΔSR (pp) | 95% CI | n |", "|---|---:|---|---:|"]
    base = tables["rule"]
    for arm in BATCHES:
        if arm == "rule":
            continue
        d = [100 * (v - base[k]) for k, v in tables[arm].items() if k in base]
        m, lo, hi = boot_ci(d)
        out.append(f"| {arm} | {m:+.1f} | [{lo:+.1f},{hi:+.1f}] | {len(d)} |")
    out += ["", "## 读法", "",
            "1. 若 `rule_fix` ≈ `rule`，说明标签门在混合条件上不是主要瓶颈，LLM 的差异不能用它解释。",
            "2. 若 `llm` 仅在 2 类混合上占优、在 3 类混合上落后，结论是\"LLM 的收益随原因数增加而反转\"，",
            "   不能写成\"LLM 会处理规则没写到的原因\"——本平台可修原因已被规则四分支覆盖。",
            "3. 2 banks × 1 seed 只能做筛查：区间跨零时既不支持\"更好\"也不支持\"等价\"。"]

    # The SR column says who won; this section says why. C=calibrate, T=targeted regeneration,
    # U=uniform regeneration, R=recovery data, D=DAgger, -=stop.
    out += ["", "## 决策序列（C 校准 / T 定向 / U 均匀 / R 恢复 / D DAgger / - 停止）", "",
            "| 臂 | 条件/bank | 决策 | final SR |", "|---|---|---|---:|"]
    for arm, (dirname, selector) in BATCHES.items():
        for r in sorted(load(dirname, selector), key=lambda r: (r["cause"], r["bank_index"])):
            seq = " → ".join(SHORT.get(d.get("action"), str(d.get("action"))) for d in r.get("decisions") or [])
            out.append(f"| {arm} | {r['cause']}/b{r['bank_index']} | {seq} | {float(r['final_sr']):.3f} |")
    text = "\n".join(out) + "\n"
    (ROOT / "results/llm_mixed_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())