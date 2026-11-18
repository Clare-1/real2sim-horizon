"""Paired, no-training intervention on the four never-observed neighbour inputs.

Use the original healthy E1 initial policies, all three seeds. A new development
bank is materialized before any outcomes. Only x_sd[37:41] changes, from epsilon
to one; weights, means, task graph, judge and physical world are identical.
This is a diagnostic of a numerical amplification, not a learned new method.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from horizon.harness.artifacts import canonical_hash, file_hash, load_blocks, load_policy, save_policy, write_json
from horizon.harness.experiment import TaskStore, evaluate, source_hashes, verify_artifacts
from horizon.real2sim.recon_error import ReconModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results/harness_e1_20261026")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root, source = Path(args.out).resolve(), Path(args.source).resolve()
    root.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "experiment.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(original["config"])
    cfg.update(H_values=[2, 4, 6, 8, 10], final_n_per=10)
    cfg["banks"]["final"] = 72_000_000
    inputs = {"source_fingerprint": original["fingerprint"], "source": str(source),
              "config": cfg, "source_sha256": source_hashes(), "script_sha256": file_hash(__file__),
              "workers": args.workers, "seeds": [0, 1, 2], "variants": ["original", "neighbor_unit_scale"],
              "intervention_indices": [37, 38, 39, 40], "training_steps": 0,
              "calibration_attempts": 0, "stage": "post-hoc development diagnostic"}
    definition = {**inputs, "fingerprint": canonical_hash(inputs)}
    manifest = root / "definition.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != definition:
        raise ValueError("definition changed: choose a new output directory")
    write_json(manifest, definition)
    store = TaskStore(root / "tasks", cfg, Path.cwd())
    # Materialize all identities before inspecting any outcome.
    for H in cfg["H_values"]:
        for i in range(cfg["final_n_per"]):
            store.get("pack", H, cfg["banks"]["final"] + H * 1000 + i)
    records = []
    for seed in inputs["seeds"]:
        initial = json.loads((source / "shared" / "healthy" / f"s{seed}" / "initial.json").read_text(encoding="utf-8"))
        verify_artifacts(initial)
        blocks = load_blocks(initial["dataset"])
        # Exact observed absence, not an epsilon threshold or evaluation-data fit.
        if not all(np.all(b.x_hier[:, 37:41] == 0) for b in blocks):
            raise ValueError("the four coordinates are not absent in source training")
        params = load_policy(initial["policy"])
        if not np.allclose(params["x_sd"][37:41], 1e-3, rtol=0, atol=1e-9):
            raise ValueError("source checkpoint does not have the expected epsilon scale")
        for variant in inputs["variants"]:
            output = root / f"{variant}_s{seed}.json"
            if output.exists():
                record = json.loads(output.read_text(encoding="utf-8"))
                if record["fingerprint"] != definition["fingerprint"]:
                    raise ValueError("resume fingerprint changed")
                verify_artifacts(record)
                records.append(record)
                continue
            intervention = copy.deepcopy(params)
            if variant == "neighbor_unit_scale":
                intervention["x_sd"][37:41] = 1.0
            policy_ref = save_policy(root / "policies" / f"{variant}_s{seed}.npz", intervention)
            result, ledger = evaluate(intervention, cfg, store, ReconModel.from_dict(initial["recon"]),
                                      cfg["banks"]["final"], cfg["final_n_per"], args.workers)
            if source_hashes() != inputs["source_sha256"]:
                raise RuntimeError("source changed during diagnostic")
            record = {"fingerprint": definition["fingerprint"], "seed": seed, "variant": variant,
                      "source_initial_policy": initial["policy"], "source_training_data": initial["dataset"],
                      "policy": policy_ref, "evaluation": result, "ledger": ledger}
            write_json(output, record)
            records.append(record)
            print(f"DONE {variant}/s{seed}: {result['nominal']['overall']:.3f}/{result['perturbed']['overall']:.3f}", flush=True)
    lines = ["# Constant-neighbour normalization diagnostic", "",
             "Three original E1 healthy initial policies; no training or calibration. Only the four absent neighbour scales change from 0.001 to 1.0. All weights and all other features are identical.", "",
             "Pack, 50 shared new development task identities; 600 actual rollouts, not 600 independent tasks. Seed mean ± sample SD; no significance claim. H2/4 were in training; H6/8/10 are extrapolation.", "",
             "| H | Mode | Original SR (%) | Unit-scale SR (%) | Paired delta (pp) |", "|---|---|---:|---:|---:|"]
    summary = []
    for H in cfg["H_values"]:
        for mode in ("nominal", "perturbed"):
            scores = {}
            identities = set()
            for variant in inputs["variants"]:
                values = []
                for seed in inputs["seeds"]:
                    record = next(r for r in records if r["seed"] == seed and r["variant"] == variant)
                    rows = [e for e in record["evaluation"][mode]["episodes"] if e["H"] == H]
                    identities.add(tuple((e["task_sha256"], e["rollout_seed"]) for e in rows))
                    values.append(100 * mean(e["success"] for e in rows))
                scores[variant] = values
            if len(identities) != 1:
                raise ValueError("unpaired evaluation identities")
            deltas = [b-a for a,b in zip(scores["original"], scores["neighbor_unit_scale"])]
            summary.append({"H": H, "mode": mode, "seed_scores_percent": scores, "paired_delta_pp": deltas})
            fmt = lambda x: f"{mean(x):.2f} ± {stdev(x):.2f}"
            lines.append(f"| {H} | {mode} | {fmt(scores['original'])} | {fmt(scores['neighbor_unit_scale'])} | {fmt(deltas)} |")
    lines += ["", "This isolates the scale intervention on the saved checkpoint; it does not show that unseen obstacles have been learned, identify all long-horizon failure causes, or establish agentic/physical transfer.", ""]
    write_json(root / "summary.json", {"fingerprint": definition["fingerprint"], "rows": summary,
        "actual_rollouts": sum(r["ledger"]["episodes"] for r in records), "complete": True})
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
