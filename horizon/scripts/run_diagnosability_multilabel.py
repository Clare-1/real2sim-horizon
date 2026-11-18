"""Multi-label diagnosability: can the observable profile light up *both* components of a mixture?

The multi-class diagnoser in run_diagnosability_bank.py must choose one label, so every mixture
collapses to its dominant component (mixed recall 0%). This script trains one binary logistic
model per component and reports per-label recall/precision plus exact-set accuracy. Mixtures are
the point: a good model should predict both {geometry, coverage} for mixed_gc.

    python scripts/run_diagnosability_multilabel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_diagnosability_bank import features  # noqa: E402

SOURCES = [
    "results/minimal_validation_20261112/stage1",
    "results/minimal_validation_20261112/stage2",
    "results/minimal_validation_n50_20261116",
    "results/minimal_validation_mixed_20261116",
    "results/four_causes_20261118",
    "results/mixed_grid_20261118",
]
LABELS = ("geometry_bias", "coverage_deficit", "recovery_deficit")
COMPONENTS = {
    "geometry_bias": {"geometry_bias"},
    "coverage_deficit": {"coverage_deficit"},
    "recovery_deficit": {"recovery_deficit"},
    "healthy": set(),
    "mixed_cause": {"geometry_bias", "coverage_deficit"},
    "mixed_gc": {"geometry_bias", "coverage_deficit"},
    "mixed_gr": {"geometry_bias", "recovery_deficit"},
    "mixed_cr": {"coverage_deficit", "recovery_deficit"},
    "mixed_gcr": {"geometry_bias", "coverage_deficit", "recovery_deficit"},
}


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
            if r.get("cause") not in COMPONENTS:
                continue
            if r.get("decisions") and r["decisions"][0].get("profile"):
                r["_source"] = d
                out.append(r)
    prof_path = ROOT / "results/diagnosis_profiles_20261118/profiles.jsonl"
    if prof_path.exists():
        for line in prof_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out.append({"cause": row["cause"], "_source": "profiles",
                            "bank_index": row["bank_index"],
                            "decisions": [{"profile": row["profile"]}]})
    # dedupe: one profile per (cause, source, bank); selectors share it
    seen, units = set(), []
    for r in out:
        key = (r["cause"], r["_source"], r.get("bank_index", 0),
               json.dumps(r["decisions"][0]["profile"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            units.append(r)
    return units


def fit_logreg(X, y, epochs=1500, lr=0.5, l2=1e-2):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
    return {"mu": mu, "sd": sd, "w": w}


def predict(model, X) -> np.ndarray:
    Z = np.c_[(X - model["mu"]) / model["sd"], np.ones(len(X))]
    return 1.0 / (1.0 + np.exp(-Z @ model["w"]))


def main() -> int:
    units = load()
    groups = sorted({(u["cause"], u["_source"], u.get("bank_index", 0)) for u in units})
    X = np.array([features(u) for u in units], float)
    Y = np.array([[1.0 if lab in COMPONENTS[u["cause"]] else 0.0 for lab in LABELS]
                  for u in units])
    print(f"distinct profiles={len(units)} groups={len(groups)} labels={LABELS}")

    probs = np.zeros_like(Y)
    for g in groups:
        test = np.array([(u["cause"], u["_source"], u.get("bank_index", 0)) == g
                         for u in units])
        if test.sum() == 0 or (~test).sum() < 5:
            continue
        for j in range(len(LABELS)):
            model = fit_logreg(X[~test], Y[~test, j])
            probs[test, j] = predict(model, X[test])
    pred = (probs >= 0.5).astype(float)

    lines = ["# Multi-label diagnosability (bank-grouped leave-one-out)", "",
             f"profiles={len(units)}  groups={len(groups)}", "",
             "| label | n | recall | precision |", "|---|---:|---:|---:|"]
    for j, lab in enumerate(LABELS):
        n = int(Y[:, j].sum())
        rec = float((pred[Y[:, j] == 1, j] == 1).mean()) if n else float("nan")
        denom = int(pred[:, j].sum())
        pre = float((Y[pred[:, j] == 1, j] == 1).mean()) if denom else float("nan")
        lines.append(f"| {lab} | {n} | {rec:.0%} | {pre:.0%} |")

    exact = float(((pred >= 0.5) == (Y >= 0.5)).all(1).mean())
    mixed = [i for i, u in enumerate(units) if len(COMPONENTS[u["cause"]]) >= 2]
    any_hit = sum(1 for i in mixed if ((pred[i] >= 0.5) & (Y[i] >= 0.5)).any()) / len(mixed) if mixed else 0.0
    all_hit = sum(1 for i in mixed if ((pred[i] >= 0.5) == (Y[i] >= 0.5)).all()) / len(mixed) if mixed else 0.0
    lines += ["", f"exact-set accuracy (incl. healthy): {exact:.0%}",
              f"mixtures n={len(mixed)}: any component hit {any_hit:.0%}, "
              f"all components hit {all_hit:.0%}", ""]
    text = "\n".join(lines)
    out = ROOT / "results/diagnosability_multilabel_20261118"
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosability_multilabel.md").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    print(f"[written] {out/'diagnosability_multilabel.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
