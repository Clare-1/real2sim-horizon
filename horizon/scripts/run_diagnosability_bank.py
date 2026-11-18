"""Bank-grouped diagnosability: the honest version of run_diagnosability5.py.

Why this exists
---------------
Feedback jobs are seeded independently of the selector, so for one (scenario, bank, round) every
selector sees the *same* profile. The old script counted each selector as a sample, which put exact
duplicates in both train and test of a leave-one-out split. On the 2026-11-18 data 272 records
collapse to ~74 distinct profiles (3x to 12x duplicates), so its 88.9% is not a generalisation
number.

This script:
  * deduplicates by (cause, source, bank, profile);
  * holds out whole banks (leave-one-bank-out), never individual duplicate rows;
  * reports how many distinct profiles/groups exist, so the sample size is visible;
  * gives per-class recall with a group-level bootstrap CI.

    python scripts/run_diagnosability_bank.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_diagnosability5 import TYPES, fit_softmax  # noqa: E402

SOURCES = [
    "results/minimal_validation_20261112/stage1",
    "results/minimal_validation_20261112/stage2",
    "results/minimal_validation_n50_20261116",
    "results/minimal_validation_mixed_20261116",
    "results/four_causes_20261118",
]


def load() -> list[dict]:
    out = []
    for d in SOURCES:
        p = ROOT / d
        f = p / "summary.json"
        if f.exists():
            recs = json.loads(f.read_text(encoding="utf-8"))["records"]
        else:
            recs = [json.loads(x.read_text(encoding="utf-8"))
                    for x in sorted(p.glob("runs/*/*/*/*/record.json"))]
        for r in recs:
            r["_source"] = d
            if r.get("decisions") and r["decisions"][0].get("profile"):
                out.append(r)
    # Independent profiles collected by scripts/collect_profiles.py (one per cause x bank).
    prof_path = ROOT / "results/diagnosis_profiles_20261118/profiles.jsonl"
    if prof_path.exists():
        for line in prof_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out.append({"cause": row["cause"], "_source": "profiles",
                        "bank_index": row["bank_index"],
                        "decisions": [{"profile": row["profile"]}]})
    return out


def features(rec: dict) -> list[float]:
    prof = rec["decisions"][0]["profile"]
    per = prof.get("per_subgoal") or {}
    ft = prof.get("failure_types") or {}
    total = sum(ft.values()) or 1
    return [1000.0 * float(prof.get("residual") or 0.0), float(prof.get("concentration") or 0.0),
            float(len(per)), float(sum(per.values())),
            float(prof.get("sr") or 0.0)] + [ft.get(t, 0) / total for t in TYPES]


def main() -> int:
    recs = load()
    units, seen = [], set()
    for r in recs:
        key = (r["cause"], r["_source"], r.get("bank_index", 0),
               json.dumps(r["decisions"][0]["profile"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        units.append(r)

    def group(u):
        return (u["cause"], u["_source"], u.get("bank_index", 0))

    groups = sorted({group(u) for u in units})
    causes = sorted({u["cause"] for u in units})
    idx = {c: i for i, c in enumerate(causes)}
    groups_per_cause = Counter(u["cause"] for u in units)
    print(f"records={len(recs)}  distinct profiles={len(units)}  groups={len(groups)}")
    print("distinct profiles per cause:", dict(groups_per_cause))
    print("classes:", causes)

    X = np.array([features(u) for u in units], float)
    y = np.array([idx[u["cause"]] for u in units])
    pred = np.full(len(y), -1)
    for g in groups:
        hold = np.array([group(u) == g for u in units])
        cause = g[0]
        # Hold out a bank only when the cause has another bank to learn from; otherwise the class
        # would disappear from training and the number would be meaningless.
        if groups_per_cause[cause] < 2:
            continue
        train = ~hold
        if len(set(y[train])) < len(causes):
            continue
        clf = fit_softmax(X[train], y[train], len(causes))
        pred[hold] = clf(X[hold])

    mask = pred >= 0
    acc = float((pred[mask] == y[mask]).mean()) if mask.any() else float("nan")
    lines = ["# Bank-grouped diagnosability (5 classes)", "",
             f"records={len(recs)}  distinct_profiles={len(units)}  groups={len(groups)}  "
             f"evaluated={int(mask.sum())}  accuracy={acc:.1%}", "",
             "| true \\ predicted | " + " | ".join(c[:11] for c in causes) + " |",
             "|---|" + "---|" * len(causes)]
    for i, c in enumerate(causes):
        lines.append(f"| {c} | " + " | ".join(
            str(int(((y == i) & (pred == j)).sum())) for j in range(len(causes))) + " |")

    lines += ["", "| cause | groups | n | recall |", "|---|---:|---:|---:|"]
    rng = np.random.default_rng(0)
    for i, c in enumerate(causes):
        sel = (y == i) & mask
        n, ng = int(sel.sum()), groups_per_cause[c]
        rec = float((pred[sel] == i).mean()) if n else float("nan")
        # group-level bootstrap over this class' groups
        gs = sorted({group(u) for u, m in zip(units, sel) if m})
        vals = []
        for _ in range(2000):
            pick = [gs[k] for k in rng.integers(0, len(gs), len(gs))] if gs else []
            idxs = [k for k, u in enumerate(units) if group(u) in pick and sel[k]]
            if idxs:
                vals.append(float((pred[idxs] == i).mean()))
        ci = f"[{np.percentile(vals, 2.5):.0%}, {np.percentile(vals, 97.5):.0%}]" if vals else "-"
        lines.append(f"| {c} | {ng} | {n} | {rec:.0%} {ci} |")
    text = "\n".join(lines)
    out = ROOT / "results/diagnosability_bank_20261118"
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosability_bank.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"\n[written] {out/'diagnosability_bank.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
