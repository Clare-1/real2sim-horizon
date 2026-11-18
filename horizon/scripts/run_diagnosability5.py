"""Five-to-six-class diagnosability: can the failure cause be recovered from feedback alone?

Extends run_diagnosability.py from binary to multi-class. Features are exactly what a selector is
allowed to see: calibration residual, failure concentration, how many subgoals failed, how many
failures, and the attributed failure-type shares. Nothing about the injected cause enters.

Leave-one-out multinomial logistic regression, plus the confusion matrix. If dynamics cannot be
separated, that is the answer and the paper must say so.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

SOURCES = [
    "results/minimal_validation_20261112/stage1",
    "results/minimal_validation_20261112/stage2",
    "results/minimal_validation_n50_20261116",
    "results/minimal_validation_mixed_20261116",
    "results/five_causes_20261117",
]
TYPES = ("drop", "grasp", "placement", "stall", "transition", "regression", "verifier", "wrong_object")


def load(dirs):
    recs = []
    for d in dirs:
        p = ROOT / d
        if not p.exists():
            continue
        f = p / "summary.json"
        if f.exists():
            recs += json.loads(f.read_text(encoding="utf-8"))["records"]
        else:
            recs += [json.loads(x.read_text(encoding="utf-8"))
                     for x in sorted(p.glob("runs/*/*/*/*/record.json"))]
    return recs


def features(rec):
    prof = rec["decisions"][0].get("profile") or {}
    per = prof.get("per_subgoal") or {}
    ft = prof.get("failure_types") or {}
    total_t = sum(ft.values()) or 1
    feats = [
        1000.0 * float(prof.get("residual") or 0.0),
        float(prof.get("concentration") or 0.0),
        float(len(per)),
        float(sum(per.values())),
    ]
    feats += [ft.get(t, 0) / total_t for t in TYPES]
    return feats


def fit_softmax(X, y, k, epochs=400, lr=0.3, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    W = np.zeros((Z.shape[1], k))
    Y = np.eye(k)[y]
    for _ in range(epochs):
        logits = Z @ W
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        W -= lr * (Z.T @ (P - Y) / len(y) + l2 * W)
    return lambda A: np.argmax(np.c_[(A - mu) / sd, np.ones(len(A))] @ W, axis=1)


def main() -> int:
    recs = load(SOURCES)
    recs = [r for r in recs if r.get("decisions")]
    causes = sorted({r["cause"] for r in recs})
    idx = {c: i for i, c in enumerate(causes)}
    X = np.array([features(r) for r in recs], float)
    y = np.array([idx[r["cause"]] for r in recs])

    pred = np.zeros(len(y), int)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        if len(set(y[keep])) < len(causes):
            pred[i] = y[i]
            continue
        clf = fit_softmax(X[keep], y[keep], len(causes))
        pred[i] = clf(X[i:i + 1])[0]

    acc = float((pred == y).mean())
    lines = ["# Five-class diagnosability (leave-one-out)", "",
             f"samples={len(y)}  classes={len(causes)}  accuracy={acc:.1%}", "",
             "| true \\ predicted | " + " | ".join(c[:10] for c in causes) + " |",
             "|---|" + "---|" * len(causes)]
    for i, c in enumerate(causes):
        lines.append(f"| {c} | " + " | ".join(str(int(((y == i) & (pred == j)).sum()))
                                               for j in range(len(causes))) + " |")
    per_class = {c: float(((pred == i) & (y == i)).sum() / max((y == i).sum(), 1))
                 for i, c in enumerate(causes)}
    lines += ["", "| cause | n | recall |", "|---|---:|---:|"]
    for i, c in enumerate(causes):
        lines.append(f"| {c} | {int((y == i).sum())} | {per_class[c]:.0%} |")
    text = "\n".join(lines)
    out = ROOT / "results/diagnosability5_20261117"
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosability5.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
