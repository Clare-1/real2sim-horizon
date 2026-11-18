"""One bounded, frozen-checkpoint pack H=6/8/10 development diagnostic.

The source E1 experiment must be complete (24 runs). No training or calibration
is called. Run with PYTHONPATH pointing to the source experiment's frozen src/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev
import tempfile
import time


ARMS = ("S00", "S01", "S10", "S11")
CONDITIONS = ("healthy", "height_bias")
SEEDS = (0, 1, 2)
HORIZONS = (6, 8, 10)
N_PER = 20
BANK = 68_000_000
MODES = ("nominal", "perturbed")
COLORS = {"S00": "#777777", "S01": "#2878b5", "S10": "#de8f05", "S11": "#20958a"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def source_manifest(source):
    manifest = read_json(source / "experiment.json")
    inputs = {k: v for k, v in manifest.items() if k != "fingerprint"}
    require(digest(inputs) == manifest["fingerprint"], "Source experiment fingerprint is invalid")
    snapshot = source / "source"
    module_root = snapshot / "src" / "horizon"
    require(module_root.is_dir(), "Frozen source snapshot is missing")
    actual_source = {p.relative_to(module_root).as_posix(): file_digest(p) for p in sorted(module_root.rglob("*.py"))}
    runner = snapshot / "scripts" / "run_harness.py"
    if runner.exists():
        actual_source["scripts/run_harness.py"] = file_digest(runner)
    require(actual_source == manifest["source_sha256"], "Saved source snapshot differs from executed source hashes")
    cfg = manifest["config"]
    require(cfg["families"] == ["pack"] and sorted(cfg["H_values"]) == [2, 4],
            "This diagnostic is only defined for pack policies trained at H=2/4")
    require(set(cfg["arms"]) == set(ARMS) and sorted(cfg["seeds"]) == list(SEEDS), "Expected four arms and three source seeds")
    require(set(cfg["conditions"]) == set(CONDITIONS), "Expected healthy and height_bias conditions")
    require(cfg.get("source", {}).get("kind", "procedural") == "procedural", "Expected E1 procedural task source")
    require(all(abs(BANK - value) >= 1_000_000 for value in cfg["banks"].values()), "Transfer bank overlaps a source bank")
    require(not 40_000_000 <= BANK < 41_000_000, "Reserved confirmation bank cannot be used")
    return manifest


def definition(source, manifest):
    cfg = copy.deepcopy(manifest["config"])
    cfg["H_values"] = list(HORIZONS)
    cfg["final_n_per"] = N_PER
    cfg["banks"]["final"] = BANK
    cfg["name"] = "frozen_pack_horizon_transfer_development"
    inputs = {
        "schema_version": 1, "source_run": str(source), "source_experiment_fingerprint": manifest["fingerprint"],
        "source_config_sha256": digest(manifest["config"]), "evaluation_config": cfg,
        "evaluation_bank": BANK, "H_values": list(HORIZONS), "n_per": N_PER,
        "modes": {"nominal": 0.0, "perturbed": 0.15}, "script_sha256": file_digest(__file__),
        "source_sha256": manifest["source_sha256"], "training_steps": 0, "calibration_attempts": 0,
    }
    return {**inputs, "fingerprint": digest(inputs)}


def plan_text(plan):
    return f"""# Bounded frozen-checkpoint horizon transfer diagnostic

This development diagnostic was designed after observing high healthy-condition
success on the source E1 short tasks. It is not a preregistered confirmation test.
No transfer outcome was used to select this fixed matrix. All results will be
reported; no additional horizons, conditions, perturbations, or tuned thresholds
will be added to search for a positive S11 result.

Source: `{plan['source_run']}`
Source experiment fingerprint: `{plan['source_experiment_fingerprint']}`
Evaluation definition fingerprint: `{plan['fingerprint']}`

- Source training covered only pack H=2/4, with the source's saved data budgets,
  architecture and training settings. Source 24-run completion is required before evaluation.
- Use all four final systems S00/S01/S10/S11, healthy and height_bias, seeds 0/1/2.
- Freeze each source final policy and final reconstruction. Do not train, calibrate,
  relabel data, select checkpoints, or update the systems during this diagnostic.
- Evaluate pack H=6/8/10, 20 tasks per H and mode, nominal and perturbation p=0.15.
- Task bank is [68M,69M), disjoint from source training/feedback/curve/final and
  reserved confirmation. All systems share task artifacts and rollout identities.
- Planned total: 24 systems × 3 horizons × 20 tasks × 2 modes = 2,880 rollouts.
  There are 60 distinct task identities, shared across systems and modes.
- Report full-task success by H and the equal-weight H macro average, seed mean
  and sample SD, and paired seed differences S11−S10, S11−S01 and other fixed contrasts.
  SD is not a confidence interval; do not claim significance from three seeds.
- This is same-MuJoCo proxy compositional extrapolation from H=2/4 training. It is
  not the original H=2..10 training/scaling experiment, physical sim-to-real transfer,
  new-skill transfer, or evidence that an agent makes better learning decisions.
- Do not fit an exponential decay coefficient alpha to this bounded diagnostic.
- Keep source and transfer scores separate; new task-bank scores are not a paired
  longitudinal change from the source final task bank.

Resume requires the same script, source experiment/config, final model bytes,
final reconstruction and evaluation definition, plus intact saved results and task
artifacts. The run ledger records actual evaluation cost. No confirmation task is used.

Interpretation is fixed before transfer execution: saturation suggests weak task
discrimination; uniformly low performance suggests extrapolation/coverage failure;
non-saturated results reveal room for a later H=2..10 learning experiment. None of
these outcomes alone establishes or refutes the original agentic-learning claim.
"""


def prepare_plan(out, plan):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "evaluation_definition.json"
    if path.exists():
        require(read_json(path) == plan, "Output belongs to a different transfer definition; use a new directory")
    else:
        require(not list((out / "runs").glob("*/*_s*.json")), "Existing run files have no evaluation definition")
        write_json(path, plan)
    text = plan_text(plan)
    plan_path = out / "PLAN.md"
    if plan_path.exists():
        require(plan_path.read_text(encoding="utf-8") == text, "Saved transfer plan changed")
    else:
        plan_path.write_text(text, encoding="utf-8")


def load_complete_source(source, manifest):
    expected = {(c, a, s) for c in CONDITIONS for a in ARMS for s in SEEDS}
    runs = {}
    for path in sorted((source / "runs").glob("*/*_s*.json")):
        value = read_json(path)
        key = (value["condition"], value["arm"], int(value["seed"]))
        require(key not in runs, f"Duplicate source run: {key}")
        require(value.get("status") == "complete", f"Source run is not complete: {key}")
        require(value["experiment"] == manifest, f"Source run differs from experiment manifest: {key}")
        require(len(value["history"]) == manifest["config"]["rounds"], f"Incomplete source history: {key}")
        require(value["final_policy"]["sha256"] == value["history"][-1]["policy_after"], f"Final source policy mismatch: {key}")
        require(value["final_recon"] == value["history"][-1]["recon_after"], f"Final source reconstruction mismatch: {key}")
        require(file_digest(value["final_policy"]["path"]) == value["final_policy"]["sha256"], f"Source model bytes changed: {key}")
        runs[key] = (value, {"path": str(path.resolve()), "sha256": file_digest(path)})
    require(set(runs) == expected, f"Source matrix is incomplete or unexpected: missing={sorted(expected-set(runs))}, extra={sorted(set(runs)-expected)}")
    return runs


def run_inputs(plan, source_run, source_ref):
    return {"evaluation_definition_fingerprint": plan["fingerprint"],
            "source_experiment_fingerprint": plan["source_experiment_fingerprint"],
            "source_config_sha256": plan["source_config_sha256"], "script_sha256": plan["script_sha256"],
            "source_result": source_ref, "source_final_policy": source_run["final_policy"],
            "source_final_recon": source_run["final_recon"], "evaluation_config_sha256": digest(plan["evaluation_config"])}


def validate_evaluation(evaluation, out=None):
    identities = {}
    for mode in MODES:
        item = evaluation[mode]
        require(not item.get("errors"), f"Rollout errors in {mode}")
        episodes = item["episodes"]
        expected = Counter({("pack", H): N_PER for H in HORIZONS})
        require(Counter((e["family"], int(e["H"])) for e in episodes) == expected, f"Wrong transfer grid: {mode}")
        ids = []
        for e in episodes:
            task_seed = int(e["task_seed"])
            offset = task_seed - BANK - int(e["H"])*1000
            require(0 <= offset < N_PER and int(e["rollout_seed"]) == task_seed+17, f"Unexpected task/rollout identity: {mode}")
            require(e["success"] in (True, False, 0, 1), "Success must be binary")
            ids.append((e["family"], int(e["H"]), task_seed, int(e["rollout_seed"]), e["task_sha256"]))
            if out is not None:
                task_path = out / "tasks" / f"pack_H{e['H']}_s{task_seed}.json"
                require(task_path.exists() and file_digest(task_path) == e["task_sha256"], f"Transfer task artifact changed: {task_path}")
                sidecar = task_path.with_name(task_path.stem + ".identity.json")
                require(sidecar.exists(), f"Transfer task identity record missing: {sidecar}")
                require(read_json(sidecar) == {"key": task_path.stem, "sha256": e["task_sha256"]},
                        f"Transfer task identity record changed: {sidecar}")
        require(len(set(ids)) == len(ids), "Duplicate transfer task identity")
        require(math.isclose(mean(int(e["success"]) for e in episodes), item["overall"], abs_tol=1e-12), "Stored transfer SR disagrees with episodes")
        identities[mode] = tuple(sorted(ids))
    return identities


def validate_saved(value, expected_inputs, out):
    require(value.get("status") == "complete" and value.get("schema_version") == 1, "Invalid saved transfer status/schema")
    require(value.get("record_sha256") == digest({k: v for k, v in value.items() if k != "record_sha256"}),
            "Saved transfer record checksum differs (including ledger)")
    require(value["inputs"] == expected_inputs and value["fingerprint"] == digest(expected_inputs), "Saved transfer inputs changed")
    require(value["evaluation_sha256"] == digest(value["evaluation"]), "Saved transfer evaluation changed")
    require(value["ledger"]["episodes"] == len(HORIZONS)*N_PER*len(MODES), "Wrong transfer rollout cost count")
    require(type(value["ledger"]["sim_steps"]) is int and value["ledger"]["sim_steps"] >= 0,
            "Invalid transfer simulation-step ledger")
    require(type(value["ledger"]["worker_seconds"]) in (int, float) and
            math.isfinite(value["ledger"]["worker_seconds"]) and value["ledger"]["worker_seconds"] >= 0,
            "Invalid transfer worker-time ledger")
    require(value["training_steps"] == value["calibration_attempts"] == 0, "Transfer result contains adaptation")
    return validate_evaluation(value["evaluation"], out)


def score(evaluation, H):
    episodes = [e for e in evaluation["episodes"] if H == "macro" or int(e["H"]) == H]
    require(bool(episodes), "Empty score cell")
    return mean(int(e["success"]) for e in episodes), sum(int(e["success"]) for e in episodes), len(episodes)


def stats(values):
    return {"mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None, "values": values}


def summarize(runs):
    rows, paired, aggregates, comparisons = [], [], [], []
    for (condition, arm, seed), value in sorted(runs.items()):
        for mode in MODES:
            for H in (*HORIZONS, "macro"):
                sr, k, n = score(value["evaluation"][mode], H)
                rows.append({"condition": condition, "arm": arm, "seed": seed, "mode": mode, "H": H, "k": k, "n": n, "success": sr})
    index = {(r["condition"], r["arm"], r["seed"], r["mode"], r["H"]): r["success"] for r in rows}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for mode in MODES:
                for H in (*HORIZONS, "macro"):
                    s = {arm: index[(condition, arm, seed, mode, H)] for arm in ARMS}
                    differences = {"S11-S10": s["S11"]-s["S10"], "S11-S01": s["S11"]-s["S01"],
                                   "S01-S00": s["S01"]-s["S00"], "S10-S00": s["S10"]-s["S00"],
                                   "S11-S00": s["S11"]-s["S00"], "interaction": s["S11"]-s["S10"]-s["S01"]+s["S00"]}
                    for contrast, difference in differences.items():
                        paired.append({"condition": condition, "seed": seed, "mode": mode, "H": H,
                                       "contrast": contrast, "delta_pp": 100*difference})
    for data, keys, measure, target in ((rows, ("condition", "arm", "mode", "H"), "success", aggregates),
                                         (paired, ("condition", "mode", "H", "contrast"), "delta_pp", comparisons)):
        groups = defaultdict(list)
        for row in data:
            groups[tuple(row[k] for k in keys)].append(row)
        for key, group in groups.items():
            group.sort(key=lambda r: r["seed"])
            require([r["seed"] for r in group] == list(SEEDS), "Summary requires all three paired seeds")
            target.append({**dict(zip(keys, key)), "seeds": [r["seed"] for r in group], **stats([r[measure] for r in group])})
    return rows, paired, aggregates, comparisons


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(plan, aggregates, comparisons, ledger):
    lines = ["# Frozen pack checkpoints: long-horizon development extrapolation", "",
             "This diagnostic was designed after observing the healthy short-horizon ceiling in E1. "
             "It is not a preregistered confirmation experiment. The fixed matrix in PLAN.md was not extended after transfer results.", "",
             "Training covered only pack H=2/4. Evaluation uses H=6/8/10 in the same MuJoCo deployment proxy, "
             "with frozen source final policies and reconstruction. No training or calibration occurred. "
             "This is not the original H=2..10 training/scaling experiment, real-robot transfer, or evidence of agent decision improvements. "
             "No exponential decay coefficient is fitted.", "",
             "S00=fixed twin/frozen policy; S01=fixed twin/DAgger; S10=calibrated twin/frozen policy; "
             "S11=calibrated twin/DAgger. These labels describe the source history; all four systems are frozen here.", "",
             f"Completed: 24 systems, {ledger['episodes']} evaluation rollouts, 60 unique task identities. "
             "The same identities are reused across seeds, arms, conditions and modes; these are not thousands of independent tasks. "
             "Numbers are seed means ± sample SD. SD is not a confidence interval; no statistical significance is claimed.", "",
             "## Full-task success", "", "Macro is the equal-weight average over H=6/8/10, each with 20 tasks per mode.", "",
             "| Condition | Arm | Mode | H | Success mean ± SD (%) | Seed values (%) |", "|---|---|---|---|---:|---|"]
    for row in aggregates:
        seeds = ", ".join(f"s{s}: {100*v:.2f}" for s, v in zip(row["seeds"], row["values"]))
        lines.append(f"| {row['condition']} | {row['arm']} | {row['mode']} | {row['H']} | {100*row['mean']:.2f} ± {100*row['sample_sd']:.2f} | {seeds} |")
    lines += ["", "## Paired contrasts", "",
              "Main contrasts are S11−S10 (source learning with calibrated twin) and S11−S01 (source calibration with learning). "
              "They compare complete source adaptive histories, not an intervention made during transfer. "
              "The full set of H-specific and macro contrasts is retained in transfer_summary.json and transfer_paired.csv.", "",
              "| Condition | Mode | H | Contrast | Mean ± SD (pp) | Seed differences (pp) |", "|---|---|---|---|---:|---|"]
    for row in comparisons:
        if row["contrast"] not in ("S11-S10", "S11-S01"):
            continue
        seeds = ", ".join(f"s{s}: {v:+.2f}" for s, v in zip(row["seeds"], row["values"]))
        lines.append(f"| {row['condition']} | {row['mode']} | {row['H']} | {row['contrast']} | {row['mean']:+.2f} ± {row['sample_sd']:.2f} | {seeds} |")
    lines += ["", "## Interpretation and cost", "",
              "Saturation would indicate limited discrimination in these tasks; near-zero scores would indicate poor "
              "compositional extrapolation or missing training coverage. Neither alone establishes or refutes a long-horizon "
              "agentic-learning advantage. Source and transfer final task banks differ, so their numerical difference is not a paired learning gain.", "",
              f"Actual simulation steps: {ledger['sim_steps']}; summed rollout worker seconds: {ledger['worker_seconds']:.2f}. "
              "Worker seconds sum concurrent work and are not wall-clock elapsed time. Training steps=0; calibration attempts=0.", "",
              f"Source experiment fingerprint: `{plan['source_experiment_fingerprint']}`.",
              f"Transfer definition fingerprint: `{plan['fingerprint']}`.", "",
              "Saved outputs: transfer_per_seed.csv, transfer_paired.csv, transfer_summary.json, validation.json, "
              "transfer_success.png/.pdf, transfer_paired.png/.pdf, and individual immutable run records.", ""]
    return "\n".join(lines)


def plot(out, aggregates, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ci, condition in enumerate(CONDITIONS):
        for mi, mode in enumerate(MODES):
            ax = axes[ci, mi]
            for arm in ARMS:
                group = sorted([r for r in aggregates if r["condition"] == condition and r["mode"] == mode and
                                r["arm"] == arm and r["H"] != "macro"], key=lambda r: r["H"])
                for si in range(len(SEEDS)):
                    ax.plot(HORIZONS, [r["values"][si] for r in group], color=COLORS[arm], alpha=.23, linewidth=.8)
                ax.plot(HORIZONS, [r["mean"] for r in group], color=COLORS[arm], marker="o", label=arm)
            ax.set(title=f"{condition} · {mode}", xlabel="Evaluation subgoals H", ylabel="Full-task success", ylim=(0, 1.03), xticks=HORIZONS)
            ax.grid(axis="y", alpha=.2)
            ax.legend(frameon=False, ncol=2)
    fig.suptitle("Frozen pack H=2/4-trained checkpoints · same-MuJoCo development extrapolation\nThin lines: individual seeds; thick lines: seed means", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"transfer_success.{extension}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ci, condition in enumerate(CONDITIONS):
        for mi, mode in enumerate(MODES):
            ax = axes[ci, mi]
            for offset, contrast, color in ((-.08, "S11-S10", "#20958a"), (.08, "S11-S01", "#2878b5")):
                group = [next(r for r in comparisons if r["condition"] == condition and r["mode"] == mode and
                              r["H"] == H and r["contrast"] == contrast) for H in (*HORIZONS, "macro")]
                x = np.arange(4)+offset
                ax.plot(x, [r["mean"] for r in group], color=color, marker="D", markersize=4, label=contrast)
                for si in range(len(SEEDS)):
                    ax.scatter(x+(si-1)*.025, [r["values"][si] for r in group], color=color, s=20, alpha=.5)
            ax.axhline(0, color="#777777", linewidth=.8)
            ax.set(title=f"{condition} · {mode}", xlabel="Evaluation H / macro", ylabel="Paired success difference (pp)")
            ax.set_xticks(range(4), ["6", "8", "10", "Macro"])
            ax.grid(axis="y", alpha=.2)
            ax.legend(frameon=False)
    fig.suptitle("Paired frozen-system contrasts · diamonds: seed mean · dots: seed differences\nThree seeds; descriptive results, no significance claim", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(out / f"transfer_paired.{extension}", dpi=200)
    plt.close(fig)


def report_all(out, plan, sources):
    runs, paired_identities = {}, defaultdict(set)
    ledger = {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}
    for key, (source_run, source_ref) in sorted(sources.items()):
        condition, arm, seed = key
        path = out / "runs" / condition / f"{arm}_s{seed}.json"
        require(path.exists(), f"Transfer matrix is incomplete: {key}")
        value = read_json(path)
        require((value["condition"], value["arm"], value["seed"]) == key, f"Transfer identity mismatch: {path}")
        ids = validate_saved(value, run_inputs(plan, source_run, source_ref), out)
        for mode in MODES:
            paired_identities[mode].add(ids[mode])
        for field in ledger:
            ledger[field] += value["ledger"][field]
        runs[key] = value
    require(all(len(values) == 1 for values in paired_identities.values()), "Transfer systems used different task artifacts/rollout identities")
    require(len(runs) == 24 and ledger["episodes"] == 2880, "Incomplete fixed transfer matrix")
    rows, paired, aggregates, comparisons = summarize(runs)
    validation = {"passed": True, "partial": False, "runs": 24, "unique_task_identities": 60, "ledger": ledger,
                  "training_steps": 0, "calibration_attempts": 0, "confirmation_executed": False,
                  "checks": ["complete source and target matrices", "model bytes and source result hashes", "script/config/evaluation fingerprints",
                             "frozen final reconstruction", "evaluation content checksum", "task artifact hashes and paired identities",
                             "per-H score arithmetic and resource count"]}
    write_csv(out / "transfer_per_seed.csv", rows)
    write_csv(out / "transfer_paired.csv", paired)
    write_json(out / "transfer_summary.json", {"definition": plan, "success": aggregates, "paired": comparisons, "ledger": ledger})
    write_json(out / "validation.json", validation)
    (out / "TRANSFER.md").write_text(render_report(plan, aggregates, comparisons, ledger), encoding="utf-8")
    plot(out, aggregates, comparisons)
    print(f"Completed transfer report: {out / 'TRANSFER.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true", help="Write the fixed plan without requiring source completion or running rollouts")
    parser.add_argument("--report-only", action="store_true", help="Validate and report an already complete transfer matrix")
    args = parser.parse_args()
    require(args.workers > 0, "workers must be positive")
    require(not (args.plan_only and args.report_only), "Choose either plan-only or report-only")
    source, out = args.source_run.resolve(), args.out.resolve()
    require(source != out and source not in out.parents, "Transfer output must be separate from the immutable source experiment")
    manifest = source_manifest(source)
    plan = definition(source, manifest)
    prepare_plan(out, plan)
    if args.plan_only:
        print(f"Plan saved; no rollouts executed: {out / 'PLAN.md'}")
        return
    sources = load_complete_source(source, manifest)
    if args.report_only:
        report_all(out, plan, sources)
        return
    # Lazy imports let --plan-only and synthetic statistical checks run without a simulator.
    from horizon.harness.artifacts import load_policy
    from horizon.harness.experiment import TaskStore, evaluate, source_hashes
    from horizon.real2sim.recon_error import ReconModel
    require(source_hashes() == manifest["source_sha256"], "Active harness source differs from the frozen E1 source; set PYTHONPATH to source/src")
    cfg = plan["evaluation_config"]
    store = TaskStore(out / "tasks", cfg, Path(__file__).resolve().parents[1])
    for key, (source_run, source_ref) in sorted(sources.items()):
        condition, arm, seed = key
        expected_inputs = run_inputs(plan, source_run, source_ref)
        target = out / "runs" / condition / f"{arm}_s{seed}.json"
        if target.exists():
            saved = read_json(target)
            require((saved["condition"], saved["arm"], saved["seed"]) == key, f"Saved transfer identity mismatch: {target}")
            validate_saved(saved, expected_inputs, out)
            print(f"SKIP verified {condition}/{arm}/s{seed}", flush=True)
            continue
        started = time.perf_counter()
        params = load_policy(source_run["final_policy"])
        recon = ReconModel.from_dict(source_run["final_recon"])
        evaluation, ledger = evaluate(params, cfg, store, recon, BANK, N_PER, args.workers)
        validate_evaluation(evaluation, out)
        require(source_hashes() == manifest["source_sha256"], "Frozen source changed during transfer")
        require(file_digest(__file__) == plan["script_sha256"], "Transfer script changed during execution")
        require(file_digest(source_run["final_policy"]["path"]) == source_run["final_policy"]["sha256"], "Source model changed during transfer")
        require(file_digest(source_ref["path"]) == source_ref["sha256"], "Source result changed during transfer")
        require(recon.to_dict() == source_run["final_recon"], "Reconstruction changed during frozen evaluation")
        value = {"schema_version": 1, "status": "complete", "condition": condition, "arm": arm, "seed": seed,
                 "inputs": expected_inputs, "fingerprint": digest(expected_inputs), "evaluation": evaluation,
                 "evaluation_sha256": digest(evaluation), "ledger": ledger, "training_steps": 0, "calibration_attempts": 0,
                 "workers": args.workers, "wall_seconds": time.perf_counter()-started,
                 "limits": ["same-MuJoCo proxy", "development diagnostic designed after short-horizon ceiling",
                            "pack H2/4 training to H6/8/10 evaluation", "frozen systems, not learning scaling"]}
        value["record_sha256"] = digest(value)
        validate_saved(value, expected_inputs, out)
        write_json(target, value)
        print(f"DONE {condition}/{arm}/s{seed}: {evaluation['nominal']['overall']:.3f}/{evaluation['perturbed']['overall']:.3f}", flush=True)
    report_all(out, plan, sources)


if __name__ == "__main__":
    main()
