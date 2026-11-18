"""Paired legacy/observed gripper-input diagnostic on frozen D0 trajectories.

Re-execute every recorded D0 expert job of the three-seed full-H replication
(healthy condition). The expert acts on true state, so the trajectories, labels,
subgoals and sample selection are unchanged; only the BC input gripper channel
differs. Legacy features must reproduce the stored D0 bytes exactly, otherwise
the run stops. Both inputs are trained with the frozen config and seed, then
evaluated on a new development bank materialized before any outcome.
No calibration, DAgger, task replacement, retry or tuning is performed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from horizon.datagen.parallel import Job, Runner
from horizon.datagen.rollout import PerturbationConfig
from horizon.harness.artifacts import canonical_hash, file_hash, load_blocks, load_policy, save_blocks, write_json
from horizon.harness.experiment import (TaskStore, assert_source, check_results, evaluate, rollout_cost,
                                        source_hashes, train_version, verify_artifacts)
from horizon.monitor.verifier import VerifierConfig
from horizon.policy.dataset import episode_samples
from horizon.real2sim.recon_error import ReconModel
from horizon.sim.env import EnvConfig

SOURCE = "results/long_horizon_replication_20261028/fullH_dagger"
CONDITION = "healthy"
SEEDS = (0, 1, 2)
VARIANTS = ("legacy", "observed")
BANK = 94_000_000
N_PER = 20
FIELDS = ("x_hier", "x_flat", "y", "k", "stage")
BATCH = 48


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rebuild_job(record: dict, payload: dict, cfg: dict, recon: dict) -> Job:
    """Recreate one recorded open-loop D0 job; legacy features are computed in the worker."""
    task = record["task"]
    require(record["controller"] == "expert" and record["beta"] == 0.0, "D0 job is not an expert job")
    require(record["observation_mode"] == "legacy" and record["max_samples"] is None, "Unexpected D0 job fields")
    return Job(task["family"], task["H"], task["task_seed"], record["rollout_seed"], controller="expert",
               pert=PerturbationConfig(**record["perturbation"]), env_cfg=EnvConfig(latency=record["latency"]),
               verifier_cfg=VerifierConfig(tol_scale=1.0), expert_noise=record["expert_noise"],
               provenance=record["provenance"], only_success=True, task_payload=payload,
               world=cfg["world"], recon=recon, real=False, keep_log=True,
               observation_mode="legacy", tag={"task_artifact": task, "twin_sha256": canonical_hash(recon)})


def same_block(a, b, fields=FIELDS) -> bool:
    return (a.key, a.family, a.H, a.provenance) == (b.key, b.family, b.H, b.provenance) and all(
        getattr(a, f).dtype == getattr(b, f).dtype and np.array_equal(getattr(a, f), getattr(b, f)) for f in fields)


def regenerate(initial: dict, collection: dict, store: TaskStore, cfg: dict, workers: int):
    """Re-run accepted D0 jobs; return legacy check blocks, observed blocks and cost."""
    stored = load_blocks(initial["dataset"])
    accepted = [j for j in collection["jobs"] if j["accepted_labels"] > 0]
    require(len(accepted) == len(stored), "Accepted D0 jobs and stored blocks differ")
    recon = initial["recon"]
    jobs = []
    for record in accepted:
        task = record["task"]
        payload, ref = store.get(task["family"], task["H"], task["task_seed"])
        require(ref["sha256"] == task["sha256"], f"Task artifact differs: {task['key']}")
        jobs.append(rebuild_job(record, payload, cfg, recon))
    observed, ledger, column_diff = [], {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}, None
    with Runner(workers) as runner:
        for start in range(0, len(jobs), BATCH):
            results = runner.map(jobs[start:start + BATCH])
            check_results(results)
            for k, v in rollout_cost(results).items():
                ledger[k] += v
            for offset, r in enumerate(results):
                i = start + offset
                take = accepted[i]["accepted_labels"]
                require(r.success and r.samples is not None and r.samples.n == accepted[i]["available_labels"],
                        f"D0 job {i} did not reproduce its success/label count")
                legacy = r.samples.take(take)
                require(same_block(legacy, stored[i]), f"D0 job {i} legacy features differ from stored bytes")
                obs = episode_samples(r.log, max_samples=r.job.max_samples, seed=r.job.seed,
                                      observation_mode="observed").take(take)
                require(same_block(obs, stored[i], ("y", "k", "stage")), f"D0 job {i} labels changed")
                obs.meta = {**obs.meta, **{k: stored[i].meta[k] for k in ("task_sha256", "twin_sha256", "rollout_seed")}}
                diff = np.abs(obs.x_hier.astype(np.float64) - legacy.x_hier).max(axis=0)
                column_diff = diff if column_diff is None else np.maximum(column_diff, diff)
                observed.append(obs)
            print(f"  regenerated {min(start + BATCH, len(jobs))}/{len(jobs)}", flush=True)
    return stored, observed, ledger, column_diff



def definition_for(source: Path, workers: int) -> dict:
    original = json.loads((source / "experiment.json").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(original["config"])
    require(all(abs(BANK - b) >= 1_000_000 for b in cfg["banks"].values()), "Bank overlaps source")
    cfg["banks"]["final"] = BANK
    cfg["final_n_per"] = N_PER
    inputs = {"stage": "development diagnostic; no confirmation bank", "source": str(source),
              "source_fingerprint": original["fingerprint"], "condition": CONDITION, "config": cfg,
              "source_sha256": source_hashes(), "script_sha256": file_hash(__file__), "workers": workers,
              "seeds": list(SEEDS), "variants": list(VARIANTS), "bank": BANK, "n_per": N_PER,
              "changed_input": "hier gripper xyz from actor observation (0.5 mm proprio jitter)",
              "unchanged": ["trajectories", "labels", "sample selection", "train config/seed", "evaluation"],
              "calibration_attempts": 0, "dagger_rounds": 0}
    return {**inputs, "fingerprint": canonical_hash(inputs)}



def run_seed(root: Path, source: Path, definition: dict, store: TaskStore, seed: int, workers: int) -> list[dict]:
    cfg = definition["config"]
    shared = source / "shared" / CONDITION / f"s{seed}"
    initial = json.loads((shared / "initial.json").read_text(encoding="utf-8"))
    verify_artifacts(initial)
    outputs = {v: root / "runs" / f"{v}_s{seed}.json" for v in VARIANTS}
    if all(p.exists() for p in outputs.values()):
        records = [json.loads(p.read_text(encoding="utf-8")) for p in outputs.values()]
        for r in records:
            require(r["fingerprint"] == definition["fingerprint"], "Resume fingerprint changed")
            verify_artifacts(r)
        return records
    collection = json.loads((shared / "collection.json").read_text(encoding="utf-8"))
    stored, observed, regen_cost, column_diff = regenerate(initial, collection, store, cfg, workers)
    data_dir = root / "data" / f"s{seed}"
    observed_ref = save_blocks(data_dir / "observed.npz", observed)
    channel = {"max_abs_x_hier_difference_by_column": column_diff.tolist(),
               "changed_columns": [int(i) for i in np.nonzero(column_diff > 0)[0]]}
    records = []
    for variant, blocks, data_ref in (("legacy", stored, initial["dataset"]), ("observed", observed, observed_ref)):
        if outputs[variant].exists():
            record = json.loads(outputs[variant].read_text(encoding="utf-8"))
            require(record["fingerprint"] == definition["fingerprint"], "Resume fingerprint changed")
            verify_artifacts(record)
            records.append(record)
            continue
        params, policy_ref, stats = train_version(blocks, cfg, seed, root / "policies" / f"{variant}_s{seed}.npz")
        result, ledger = evaluate(params, cfg, store, ReconModel.from_dict(initial["recon"]), BANK, N_PER, workers)
        assert_source(definition["source_sha256"])
        record = {"fingerprint": definition["fingerprint"], "seed": seed, "variant": variant,
                  "source_initial": {"path": str(shared / "initial.json"), "sha256": file_hash(shared / "initial.json")},
                  "training_data": data_ref, "policy": policy_ref, "train_stats": stats,
                  "retrained_legacy_equals_source_pi0": policy_ref["sha256"] == initial["policy_sha256"]
                  if variant == "legacy" else None,
                  "channel_difference": channel, "evaluation": result,
                  "ledger": {"evaluation": ledger, "d0_regeneration_shared": regen_cost,
                             "optimizer_steps": cfg["train"]["steps"]}}
        write_json(outputs[variant], record)
        records.append(record)
        print(f"DONE {variant}/s{seed}: {result['nominal']['overall']:.3f}/{result['perturbed']['overall']:.3f}",
              flush=True)
    return records



def rate(record, mode, keep):
    rows = [e for e in record["evaluation"][mode]["episodes"] if keep(e)]
    return 100 * mean(e["success"] for e in rows), sorted((e["task_sha256"], e["rollout_seed"]) for e in rows)


def summarize(records, cfg):
    scopes = [("all", lambda e: True), ("H>=8", lambda e: e["H"] >= 8)]
    scopes += [(f"{f} all", lambda e, f=f: e["family"] == f) for f in cfg["families"]]
    scopes += [(f"{f} H>=8", lambda e, f=f: e["family"] == f and e["H"] >= 8) for f in cfg["families"]]
    rows = []
    for mode in ("nominal", "perturbed"):
        for name, keep in scopes:
            scores = {v: [] for v in VARIANTS}
            for seed in SEEDS:
                ids = set()
                for v in VARIANTS:
                    r = next(x for x in records if x["seed"] == seed and x["variant"] == v)
                    value, identity = rate(r, mode, keep)
                    scores[v].append(value)
                    ids.add(tuple(identity))
                require(len(ids) == 1, "Unpaired evaluation identities")
            delta = [b - a for a, b in zip(scores["legacy"], scores["observed"])]
            rows.append({"mode": mode, "scope": name, "seed_scores_percent": scores, "paired_delta_pp": delta})
    return rows


def report(rows, records):
    fmt = lambda x: f"{mean(x):.2f} ± {stdev(x):.2f}"
    lines = ["# Legacy / observed gripper-input diagnostic", "",
             "Frozen D0 (80k labels) of the healthy full-H replication, re-executed; legacy features reproduced the stored "
             "bytes exactly. Only the BC gripper input channel differs. Same train config/seed; no DAgger or calibration.", "",
             f"Evaluation: new development bank {BANK}, pack/sort/stack × H=2..10 × {N_PER} tasks, nominal and perturbed; "
             "shared task and rollout seeds across variants and seeds. Mean ± sample SD over 3 training seeds; no significance claim.", "",
             "| Mode | Scope | legacy SR (%) | observed SR (%) | observed − legacy (pp) | per seed |", "|---|---|---:|---:|---:|---|"]
    for r in rows:
        s, d = r["seed_scores_percent"], r["paired_delta_pp"]
        lines.append(f"| {r['mode']} | {r['scope']} | {fmt(s['legacy'])} | {fmt(s['observed'])} | {fmt(d)} | "
                     + " / ".join(f"{x:+.2f}" for x in d) + " |")
    legacy = [r for r in records if r["variant"] == "legacy"]
    lines += ["", "Retrained legacy policy byte-identical to source π0: "
              + ", ".join(f"s{r['seed']}={r['retrained_legacy_equals_source_pi0']}" for r in legacy),
              "Changed x_hier columns: " + str(legacy[0]["channel_difference"]["changed_columns"]), "",
              "Development diagnostic of π0 only; does not test DAgger rounds, calibration, or physical transfer.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root, source = Path(args.out).resolve(), Path(args.source).resolve()
    root.mkdir(parents=True, exist_ok=True)
    definition = definition_for(source, args.workers)
    manifest = root / "definition.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != definition["fingerprint"]:
        raise ValueError("definition changed: choose a new output directory")
    cfg = definition["config"]
    store = TaskStore(root / "tasks", cfg, Path.cwd())
    # Materialize every evaluation identity before any outcome is produced.
    for family in cfg["families"]:
        for H in cfg["H_values"]:
            for i in range(N_PER):
                store.get(family, H, BANK + {"pack": 0, "sort": 1, "stack": 2}[family] * 100_000 + H * 1000 + i)
    write_json(manifest, definition)
    records = []
    for seed in SEEDS:
        records += run_seed(root, source, definition, store, seed, args.workers)
    rows = summarize(records, cfg)
    write_json(root / "summary.json", {"fingerprint": definition["fingerprint"], "rows": rows, "complete": True,
                                       "actual_rollouts": sum(r["ledger"]["evaluation"]["episodes"] for r in records)
                                       + sum(r["ledger"]["d0_regeneration_shared"]["episodes"]
                                             for r in records if r["variant"] == "observed")})
    (root / "RESULTS.md").write_text(report(rows, records), encoding="utf-8")
    print(f"Saved {root / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()

