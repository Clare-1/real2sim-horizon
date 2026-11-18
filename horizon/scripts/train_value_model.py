"""Fit the learned operation-value model on the counterfactual sweep.

    python scripts/train_value_model.py --sweep results/operation_sweep_20261118/sweep.jsonl

Reports leave-one-bank-out R^2 and action-ranking accuracy, and writes the JSON consumed by
``LearnedValueSelector``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACTIONS = ("calibrate_geometry", "regenerate_uniform", "regenerate_targeted",
           "add_recovery_data", "dagger_round")


def design(row: dict) -> list[float]:
    onehot = [1.0 if row["action"] == a else 0.0 for a in ACTIONS]
    return list(row["profile"]) + onehot + [row["label_cost"] / 100_000.0]


def fit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    return {"mu": mu, "sd": sd, "w": w}


def predict(model, rows) -> np.ndarray:
    X = np.array([design(r) for r in rows], float)
    Z = np.c_[(X - model["mu"]) / model["sd"], np.ones(len(X))]
    return Z @ model["w"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/operation_sweep_20261118/sweep.jsonl")
    ap.add_argument("--out", default="results/operation_sweep_20261118/value_model.json")
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(ROOT / args.sweep).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        sys.exit("empty sweep")
    groups = sorted({(r["scenario"], r["bank_index"]) for r in rows})
    print(f"rows={len(rows)} groups={len(groups)} actions={len(ACTIONS)}")

    y = np.array([r["delta"] for r in rows], float)
    loo = np.zeros(len(rows))
    for g in groups:
        test = np.array([(r["scenario"], r["bank_index"]) == g for r in rows])
        train_rows = [r for r, m in zip(rows, test) if not m]
        test_rows = [r for r, m in zip(rows, test) if m]
        if len(set(y[~test])) < 2:
            continue
        model = fit(np.array([design(r) for r in train_rows], float), y[~test])
        loo[test] = predict(model, test_rows)
    ss_res = float(((y - loo) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    print(f"leave-one-bank-out R2={1 - ss_res / ss_tot:.3f}  RMSE={np.sqrt(ss_res / len(y)):.3f}")

    hits = 0
    for g in groups:
        idx = [i for i, r in enumerate(rows) if (r["scenario"], r["bank_index"]) == g]
        if not idx:
            continue
        pick = idx[int(np.argmax(loo[idx]))]
        best_actual = idx[int(np.argmax([rows[i]["delta"] for i in idx]))]
        hits += int(pick == best_actual)
    print(f"action-ranking accuracy (LOO argmax matches best actual): {hits}/{len(groups)}")

    final = fit(np.array([design(r) for r in rows], float), y)
    mu, sd, w = final["mu"], final["sd"], final["w"]
    model = {"actions": list(ACTIONS), "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(),
             "metrics": {"rows": len(rows), "groups": len(groups),
                         "loo_r2": 1 - ss_res / ss_tot, "ranking": f"{hits}/{len(groups)}"}}
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"[written] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
