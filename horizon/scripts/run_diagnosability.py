"""Advisor §四.4 step 4: can the failure cause be recovered from deployment feedback alone?

Builds a feature matrix from recorded feedback profiles (label = the injected cause), fits a simple
classifier with leave-one-out cross-validation, and prints a confusion matrix. Also reports an
interpretable one-threshold rule. If this fails, the whole line is not viable - that is the point.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_records(dirs):
    out = []
    for d in dirs:
        p = Path(d) / "summary.json"
        if p.exists():
            out += json.loads(p.read_text(encoding="utf-8"))["records"]
        else:
            out += [json.loads(x.read_text(encoding="utf-8"))
                    for x in sorted(Path(d).glob("runs/*/*/*/*/record.json"))]
    return out


def features(rec):
    prof = rec["decisions"][0].get("profile") or {}
    per = prof.get("per_subgoal") or {}
    total = sum(per.values()) or 1
    return [
        1000.0 * float(prof.get("residual") or 0.0),      # calibration residual, mm
        float(prof.get("concentration") or 0.0),          # share of the worst subgoal
        float(len(per)),                                  # how many subgoals failed
        float(total),                                     # failures observed
    ]


def fit_logreg(X, y, epochs=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y)) / len(y) + 1e-4 * w
    return lambda A: (1.0 / (1.0 + np.exp(-np.c_[(A - mu) / sd, np.ones(len(A))] @ w))) > 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = load_records(args.dirs)
    X = np.array([features(r) for r in recs], float)
    causes = sorted({r["cause"] for r in recs})
    y = np.array([1.0 if r["cause"] == causes[1] else 0.0 for r in recs])
    labels = [causes[int(v)] for v in y]

    # leave-one-out logistic regression
    pred = []
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        clf = fit_logreg(X[keep], y[keep])
        pred.append(causes[1] if clf(X[i:i + 1])[0] else causes[0])

    # interpretable baseline: one threshold on the residual
    thr = 4.0
    rule_pred = [causes[1] if f[0] > thr else causes[0] for f in X]

    def confusion(preds):
        cm = defaultdict(Counter)
        for t, p in zip(labels, preds):
            cm[t][p] += 1
        return cm

    lines = ["# Diagnosability of the failure cause from deployment feedback", "",
             f"samples={len(X)}  causes={causes}  features=[residual_mm, concentration, n_failed_subgoals, n_failures]", ""]
    for name, preds in (("logistic regression (leave-one-out)", pred),
                        (f"threshold rule: residual > {thr:.0f} mm", rule_pred)):
        cm = confusion(preds)
        acc = sum(1 for t, p in zip(labels, preds) if t == p) / len(labels)
        lines += [f"## {name}", "", f"accuracy = {acc:.1%}", "",
                  "| true \\ predicted | " + " | ".join(causes) + " |",
                  "|---|" + "---|" * len(causes)]
        for t in causes:
            lines.append(f"| {t} | " + " | ".join(str(cm[t][p]) for p in causes) + " |")
        lines.append("")
    means = {c: X[[i for i, r in enumerate(recs) if r["cause"] == c]].mean(0) for c in causes}
    lines += ["## feature means", "", "| cause | residual mm | concentration | n_subgoals | n_failures |",
              "|---|---:|---:|---:|---:|"]
    for c in causes:
        lines.append(f"| {c} | " + " | ".join(f"{v:.2f}" for v in means[c]) + " |")
    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
