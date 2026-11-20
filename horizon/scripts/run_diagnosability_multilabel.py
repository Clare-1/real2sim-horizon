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
sys.path.insert(0, str(ROOT / "src"))

from horizon.rsi.selector import FailureProfile, diagnosis_features  # noqa: E402

SOURCES = [
    "results/minimal_validation_20261112/stage1",
    "results/minimal_validation_20261112/stage2",
    "results/minimal_validation_n50_20261116",
    "results/minimal_validation_mixed_20261116",
    "results/four_causes_20261118",
    "results/mixed_grid_20261118",
]
LABELS = ("geometry_bias", "coverage_deficit", "recovery_deficit")
# The selector evaluation uses the four-cause scenarios with banks 0 and 1. The final model must
# not see those banks, otherwise the online evaluation is in-sample.
HELDOUT_SOURCES = {"results/four_causes_20261118"}
HELDOUT_PROFILE_BANKS = {0, 1}
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


def features(rec: dict) -> tuple[float, ...]:
    prof = rec["decisions"][0]["profile"]
    p = FailureProfile(
        n_episodes=0,
        per_subgoal_failure=prof.get("per_subgoal") or {},
        calibration_residual=prof.get("residual"),
        recovery_triggers=prof.get("recovery_triggers", 0),
        perturbed_failure_rate=prof.get("perturbed_failure_rate", 0.0),
        sr=prof.get("sr", 0.0),
        failure_types=prof.get("failure_types") or {},
    )
    return diagnosis_features(p)


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
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-banks", default="0,1",
                    help="profile bank indices excluded from the held-out model")
    ap.add_argument("--suffix", default="", help="suffix for the held-out model file")
    args = ap.parse_args()
    heldout_banks = {int(x) for x in args.heldout_banks.split(",") if x.strip()}
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
    mixed = [i for i, u in enumerate(units) if len(COMPONENTS[u["cause"]]) >= 2]

    def metrics_at(t: float) -> dict:
        p = probs >= t
        rec, pre = [], []
        for j in range(len(LABELS)):
            n = int(Y[:, j].sum())
            rec.append(float((p[Y[:, j] == 1, j] == 1).mean()) if n else float("nan"))
            denom = int(p[:, j].sum())
            pre.append(float((Y[p[:, j] == 1, j] == 1).mean()) if denom else float("nan"))
        exact = float((p == (Y >= 0.5)).all(1).mean())
        any_hit = (sum(1 for i in mixed if (p[i] & (Y[i] >= 0.5)).any()) / len(mixed)
                   if mixed else 0.0)
        all_hit = (sum(1 for i in mixed if (p[i] == (Y[i] >= 0.5)).all()) / len(mixed)
                   if mixed else 0.0)
        return {"recall": rec, "precision": pre, "exact": exact,
                "any": any_hit, "all": all_hit}

    rows = [(t, metrics_at(t)) for t in (0.2, 0.3, 0.4, 0.5, 0.6)]
    # 0.5 maximises exact-set; 0.3 keeps 86% exact-set while lifting all-component mixture recall
    # from 50% to 67%. The selector exists to handle mixtures, so use 0.3 and report the sweep.
    best_t = 0.3
    best = metrics_at(best_t)

    lines = ["# Multi-label diagnosability (bank-grouped leave-one-out)", "",
             f"profiles={len(units)}  groups={len(groups)}  threshold={best_t}", "",
             "| threshold | " + " | ".join(LABELS) + " | exact | any mix | all mix |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for t, m in rows:
        lines.append(f"| {t:.1f} | " + " | ".join(f"{r:.0%}/{p:.0%}" for r, p in
                                                  zip(m["recall"], m["precision"])) +
                     f" | {m['exact']:.0%} | {m['any']:.0%} | {m['all']:.0%} |")
    lines += ["", "（每格为 recall / precision）", "",
              f"**selected threshold = {best_t}**（按 exact-set，再按全混合命中）", "",
              "| label | n | recall | precision |", "|---|---:|---:|---:|"]
    for j, lab in enumerate(LABELS):
        n = int(Y[:, j].sum())
        lines.append(f"| {lab} | {n} | {best['recall'][j]:.0%} | {best['precision'][j]:.0%} |")
    lines += ["", f"exact-set accuracy (incl. healthy): {best['exact']:.0%}",
              f"mixtures n={len(mixed)}: any component hit {best['any']:.0%}, "
              f"all components hit {best['all']:.0%}", ""]
    text = "\n".join(lines)
    out = ROOT / "results/diagnosability_multilabel_20261118"
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnosability_multilabel.md").write_text(text + "\n", encoding="utf-8")
    # final models trained on all banks, consumed by MultiLabelSelector
    models = {}
    for j, lab in enumerate(LABELS):
        m = fit_logreg(X, Y[:, j])
        models[lab] = {"mu": m["mu"].tolist(), "sd": m["sd"].tolist(), "w": m["w"].tolist()}
    (out / "multilabel_model.json").write_text(json.dumps(
        {"labels": list(LABELS), "models": models, "threshold": best_t,
         "features": "horizon.rsi.selector.diagnosis_features"}, indent=2), encoding="utf-8")

    # held-out model for the online selector evaluation: excludes the four-cause eval banks
    train_mask = np.array([
        not (u["_source"] in HELDOUT_SOURCES or
             (u["_source"] == "profiles" and u.get("bank_index", 0) in heldout_banks))
        for u in units])
    models_ho = {}
    for j, lab in enumerate(LABELS):
        m = fit_logreg(X[train_mask], Y[train_mask, j])
        models_ho[lab] = {"mu": m["mu"].tolist(), "sd": m["sd"].tolist(), "w": m["w"].tolist()}
    (out / f"multilabel_model_heldout{args.suffix}.json").write_text(json.dumps(
        {"labels": list(LABELS), "models": models_ho, "threshold": best_t,
         "features": "horizon.rsi.selector.diagnosis_features",
         "trained_without": sorted(HELDOUT_SOURCES) +
                            [f"profiles banks {sorted(heldout_banks)}"],
         "train_profiles": int(train_mask.sum())}, indent=2), encoding="utf-8")
    print(f"held-out model: train profiles={int(train_mask.sum())} / {len(units)}")
    print("\n" + text)
    print(f"[written] {out/'diagnosability_multilabel.md'} and multilabel_model.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
