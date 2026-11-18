"""Summarise a selector-driven minimal-validation run into the advisor's table."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def bootstrap_ci(values, reps=2000, seed=0):
    """Percentile bootstrap 95% CI of the mean over (seed x scenario-instance) cells."""
    import random
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def paired_deltas(recs, sel, cause, baseline="fixed"):
    """Pair on (seed, bank): the same scenario instance and training seed, different selector."""
    key = lambda r: (r.get("seed"), r.get("bank_index", 0))
    base = {key(r): r["final_sr"] for r in recs if r["selector"] == baseline and r["cause"] == cause}
    out = []
    for r in recs:
        if r["selector"] == sel and r["cause"] == cause and key(r) in base:
            out.append(r["final_sr"] - base[key(r)])
    return out


def main(root: str) -> int:
    data = json.loads((Path(root) / "summary.json").read_text(encoding="utf-8"))
    recs = data["records"]
    causes = sorted({r["cause"] for r in recs})
    sels = sorted({r["selector"] for r in recs})
    print(f"runs={len(recs)}  causes={causes}  selectors={sels}\n")

    print("| 选择器 | 第一步选对（按原因） | 最终 SR（按原因） | 部署次数 | 新增标签 |")
    print("|---|---|---|---:|---:|")
    for sel in sels:
        cells, srs, dep, lab = [], [], [], []
        for cause in causes:
            rs = [r for r in recs if r["selector"] == sel and r["cause"] == cause]
            if not rs:
                continue
            ok = sum(1 for r in rs if r["first_step_correct"])
            cells.append(f"{cause}: {ok}/{len(rs)}")
            srs.append(f"{cause}: {sum(r['final_sr'] for r in rs)/len(rs):.3f}")
            dep += [r["deploy_episodes"] for r in rs]
            lab += [r["new_labels"] for r in rs]
        print(f"| {sel} | {'; '.join(cells)} | {'; '.join(srs)} | {sum(dep)/len(dep):.0f} | {sum(lab)/len(lab):.0f} |")

    banks = sorted({r.get("bank_index", 0) for r in recs})
    print("\n--- 相对固定配方的配对差（按 seed × 场景实例配对，95% bootstrap CI）---")
    print("| 原因 | 选择器 | 配对数 | 平均差 (pp) | 95% CI |")
    print("|---|---|---:|---:|---|")
    for cause in causes:
        for sel in sels:
            if sel == "fixed":
                continue
            d = paired_deltas(recs, sel, cause)
            if len(d) < 2:
                continue
            lo, hi = bootstrap_ci(d)
            print(f"| {cause} | {sel} | {len(d)} | {100*sum(d)/len(d):+.1f} | "
                  f"[{100*lo:+.1f}, {100*hi:+.1f}] |")

    if len(banks) > 1:
        print(f"\n--- 场景实例（bank）数 = {len(banks)}；按 bank 拆分的最终 SR ---")
        header = "| 选择器 | 原因 | " + " | ".join(f"bank{b}" for b in banks) + " |"
        print(header)
        print("|---|---|" + "---:|" * len(banks))
        for sel in sels:
            for cause in causes:
                by_bank = {}
                for r in recs:
                    if r["selector"] == sel and r["cause"] == cause:
                        by_bank.setdefault(r.get("bank_index", 0), []).append(r["final_sr"])
                if not by_bank:
                    continue
                vals = [f"{sum(by_bank[b])/len(by_bank[b]):.2f}" if b in by_bank else "-" for b in banks]
                print(f"| {sel} | {cause} | " + " | ".join(vals) + " |")

    print("\n--- 每轮决策 ---")
    for r in recs:
        chain = " -> ".join(f"{d['action']}" for d in r["decisions"])
        print(f"{r['scenario']:9s} {r['selector']:7s} s{r['seed']}: {chain}   "
              f"first_ok={r['first_step_correct']}  final={r['final_sr']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
