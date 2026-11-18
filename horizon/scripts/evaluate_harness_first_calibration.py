"""Post-hoc matched-belief diagnosis of saved E1 policy checkpoints.

Always use S11 round-one reconstruction, never select a best round. Compare pi0
with final S11 on the already observed source final development task bank. Reuse
an evaluation only when policy, reconstruction, config and task identities match.
No policy training or reconstruction calibration is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import math
from pathlib import Path
from statistics import mean, stdev
import time

from evaluate_harness_transfer import (
    ARMS, CONDITIONS, SEEDS, MODES, digest, file_digest, load_complete_source,
    read_json, require, source_manifest, write_json,
)


SYSTEMS = ("initial_policy", "final_S11_policy")
BANK = 57_000_000
HORIZONS = (2, 4)
N_PER = 20
N_EVALUATIONS = len(HORIZONS)*N_PER*len(MODES)


def evaluation_identities(evaluation, verify_files=True):
    identities = {}
    for mode in MODES:
        item = evaluation[mode]
        require(not item.get("errors"), "Evaluation contains rollout errors")
        episodes = item["episodes"]
        require(Counter((e["family"], int(e["H"])) for e in episodes) ==
                Counter({("pack", H): N_PER for H in HORIZONS}), "Wrong matched-belief task grid")
        ids = []
        for e in episodes:
            H, task_seed = int(e["H"]), int(e["task_seed"])
            require(0 <= task_seed-BANK-H*1000 < N_PER and int(e["rollout_seed"]) == task_seed+17,
                    "Wrong final-bank task or rollout seed")
            require(e["success"] in (True, False, 0, 1), "Nonbinary success")
            ref = e["task_artifact"]
            require(ref["sha256"] == e["task_sha256"], "Task artifact and evaluation hash differ")
            if verify_files:
                path = Path(ref["path"])
                require(file_digest(path) == e["task_sha256"], f"Task artifact changed: {path}")
                sidecar = path.with_name(path.stem+".identity.json")
                require(sidecar.exists() and read_json(sidecar) == {"key": path.stem, "sha256": e["task_sha256"]},
                        f"Task identity missing or changed: {sidecar}")
            ids.append((e["family"], H, task_seed, int(e["rollout_seed"]), e["task_sha256"]))
        require(len(ids) == len(set(ids)), "Duplicate task identity")
        require(math.isclose(mean(int(e["success"]) for e in episodes), item["overall"], abs_tol=1e-12),
                "Success rate does not match episode counts")
        identities[mode] = sorted(ids)
    return identities


def source_jobs(sources, manifest):
    cfg = manifest["config"]
    require(cfg["banks"]["final"] == BANK and cfg["final_n_per"] == N_PER,
            "This diagnostic requires the existing 57M bank with 20 tasks per H")
    all_ids = [evaluation_identities(run["final"]) for run, _ in sources.values()]
    require(all(ids == all_ids[0] for ids in all_ids), "Source final task identities differ")
    identity_hash = digest(all_ids[0])
    jobs = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            learned, learned_ref = sources[(condition, "S11", seed)]
            calibrated, calibrated_ref = sources[(condition, "S10", seed)]
            first = learned["history"][0]["recon_after"]
            require(first == calibrated["history"][0]["recon_after"],
                    f"S10/S11 first calibration differs: {condition}/s{seed}")
            require(learned["initial"]["policy_sha256"] == calibrated["initial"]["policy_sha256"] and
                    learned["initial"]["data_sha256"] == calibrated["initial"]["data_sha256"],
                    "S10/S11 do not share initial policy and data")
            stable = all(h["recon_after"] == first for h in learned["history"]) and all(
                h["recon_before"] == first for h in learned["history"][1:])
            for system in SYSTEMS:
                policy = learned["initial"]["policy"] if system == "initial_policy" else learned["final_policy"]
                require(file_digest(policy["path"]) == policy["sha256"], "Source checkpoint bytes changed")
                reuse = None
                preference = ("S10", "S00", "S11", "S01") if system == "initial_policy" else ("S11", "S01", "S10", "S00")
                for arm in preference:
                    candidate, candidate_ref = sources[(condition, arm, seed)]
                    if (candidate["final_policy"]["sha256"] == policy["sha256"] and candidate["final_recon"] == first and
                            digest(candidate["experiment"]["config"]) == digest(cfg) and
                            digest(evaluation_identities(candidate["final"])) == identity_hash):
                        reuse = {"source_result": candidate_ref, "field": "final", "source_arm": arm,
                                 "evaluation_sha256": digest(candidate["final"])}
                        break
                jobs.append({"condition": condition, "seed": seed, "system": system,
                             "policy": policy, "recon": first, "S11_source": learned_ref, "S10_source": calibrated_ref,
                             "S10_S11_first_recon_equal": True, "S11_post_first_recon_constant": stable,
                             "evaluation_identity_sha256": identity_hash, "reuse": reuse})
    return jobs


def make_plan(source, manifest, jobs):
    helper = Path(__file__).with_name("evaluate_harness_transfer.py")
    inputs = {"schema_version": 1, "source_run": str(source), "source_experiment_fingerprint": manifest["fingerprint"],
              "source_config_sha256": digest(manifest["config"]), "evaluation_config": manifest["config"],
              "script_sha256": file_digest(__file__), "helper_sha256": file_digest(helper),
              "source_sha256": manifest["source_sha256"], "jobs": jobs,
              "training_steps": 0, "calibration_attempts": 0, "bank": BANK,
              "H_values": list(HORIZONS), "n_per": N_PER,
              "planned_new_rollouts": sum(j["reuse"] is None for j in jobs)*N_EVALUATIONS}
    return {**inputs, "fingerprint": digest(inputs)}


def plan_text(plan):
    return f"""# Post-hoc first-calibration matched-belief diagnostic

This diagnosis was designed after observing repeated calibration damage in E1
S10/height_bias/seed 1: the inferred support height drifted far above the physical
support. It is a post-hoc mechanism diagnosis on an already observed development
bank, not a preregistered new method or independent confirmation experiment.

- Require the complete original 24-run E1 matrix, unchanged model bytes and source snapshot.
- Include both original conditions and all three seeds, without excluding the damaged seed.
- For every condition/seed, always take **S11 round-one recon_after**. Never choose
  the best-performing calibration round or choose a reconstruction using evaluation truth.
- Verify S10 and S11 have the same first-round reconstruction and initial policy/data.
- At that shared reconstruction compare the source initial policy pi0 against the
  source final S11 policy. This gives 12 frozen model/reconstruction systems.
- Verify whether all later S11 reconstructions remain identical to the first one.
  If yes, describe the original S11 path as learning after the first calibration
  under fixed belief. If no, still report the matched-belief checkpoint comparison
  but do not interpret it as training under a fixed belief.
- Evaluate only existing pack H=2/4 final development tasks: bank 57M, 20 tasks per H,
  nominal and perturbation p=0.15. Do not change the separately frozen 2,880-rollout
  H=6/8/10 transfer plan.
- Reuse saved source final evaluation only if policy bytes, reconstruction, complete
  evaluation config and exact task/rollout/artifact identities match. Cite the source
  result and evaluation hashes; charge **zero new rollout cost** for reuse.
- Only unmatched systems get new frozen evaluation. Planned new rollout count from
  exact matching is {plan['planned_new_rollouts']}; 12 systems represent 960 logical evaluation records,
  not 960 newly executed rollouts.
- Do not train a second set of policies, calibrate a reconstruction, fit alpha, or
  claim agent decision improvements. This compares already trained checkpoints.
- Report full-task success, per-seed differences and seed mean/sample SD; no significance
  claim. The 40 task identities are reused across systems and modes, not independent repeats.

Source experiment: `{plan['source_experiment_fingerprint']}`
Diagnostic definition: `{plan['fingerprint']}`

This diagnostic asks whether the source final policy still improves execution when
the initial policy is given the same, undamaged first-calibration belief. The source
S11−S10 contrast also includes path-dependent feedback and repeated-calibration effects;
this post-hoc result must not retrospectively turn that contrast into a pure learning estimate.
"""


def prepare_plan(out, plan):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "diagnostic_definition.json"
    if path.exists():
        require(read_json(path) == plan, "Diagnostic definition changed; use a new output directory")
    else:
        require(not list((out / "runs").glob("*/*.json")), "Existing diagnostics have no definition")
        write_json(path, plan)
    text = plan_text(plan)
    if (out / "PLAN.md").exists():
        require((out / "PLAN.md").read_text(encoding="utf-8") == text, "Diagnostic plan changed")
    else:
        (out / "PLAN.md").write_text(text, encoding="utf-8")


def target_path(out, job):
    return out / "runs" / job["condition"] / f"{job['system']}_s{job['seed']}.json"


def validate_record(record, job, plan):
    require(record.get("schema_version") == 1 and record.get("status") == "complete", "Incomplete diagnostic result")
    require(record["record_sha256"] == digest({k: v for k, v in record.items() if k != "record_sha256"}),
            "Diagnostic record checksum differs")
    require(record["job"] == job and record["definition_fingerprint"] == plan["fingerprint"], "Diagnostic inputs changed")
    require(file_digest(job["policy"]["path"]) == job["policy"]["sha256"], "Diagnostic source checkpoint changed")
    for name in ("S11_source", "S10_source"):
        require(file_digest(job[name]["path"]) == job[name]["sha256"], f"Diagnostic source record changed: {name}")
    require(record["evaluation_sha256"] == digest(record["evaluation"]), "Diagnostic evaluation changed")
    require(digest(evaluation_identities(record["evaluation"])) == job["evaluation_identity_sha256"], "Diagnostic task identities differ")
    ledger = record["actual_cost"]
    require(ledger["episodes"] == (0 if job["reuse"] else N_EVALUATIONS), "Wrong new rollout count")
    require(type(ledger["sim_steps"]) is int and ledger["sim_steps"] >= 0, "Invalid simulation cost")
    require(type(ledger["worker_seconds"]) in (int, float) and math.isfinite(ledger["worker_seconds"]) and ledger["worker_seconds"] >= 0,
            "Invalid worker time")
    if job["reuse"]:
        ref = job["reuse"]["source_result"]
        require(file_digest(ref["path"]) == ref["sha256"], "Reused source result changed")
        original = read_json(ref["path"])[job["reuse"]["field"]]
        require(digest(original) == job["reuse"]["evaluation_sha256"] == record["evaluation_sha256"], "Reused evaluation does not match citation")
        require(ledger == {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}, "Reuse incurred fabricated rollout costs")
    require(record["training_steps"] == record["calibration_attempts"] == 0, "Diagnostic adapted a system")


def summarize(records):
    per_seed, paired, summaries = [], [], []
    for record in records:
        job = record["job"]
        for mode in MODES:
            for H in (*HORIZONS, "macro"):
                episodes = [e for e in record["evaluation"][mode]["episodes"] if H == "macro" or e["H"] == H]
                k, n = sum(int(e["success"]) for e in episodes), len(episodes)
                per_seed.append({"condition": job["condition"], "seed": job["seed"], "system": job["system"],
                                 "mode": mode, "H": H, "k": k, "n": n, "success": k/n,
                                 "reused_evaluation": job["reuse"] is not None,
                                 "S11_post_first_recon_constant": job["S11_post_first_recon_constant"]})
    index = {(r["condition"], r["seed"], r["system"], r["mode"], r["H"]): r for r in per_seed}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for mode in MODES:
                for H in (*HORIZONS, "macro"):
                    initial = index[(condition, seed, "initial_policy", mode, H)]
                    final = index[(condition, seed, "final_S11_policy", mode, H)]
                    paired.append({"condition": condition, "seed": seed, "mode": mode, "H": H,
                                   "initial_success": initial["success"], "final_S11_success": final["success"],
                                   "delta_pp": 100*(final["success"]-initial["success"]),
                                   "S11_post_first_recon_constant": final["S11_post_first_recon_constant"]})
    groups = defaultdict(list)
    for row in paired:
        groups[(row["condition"], row["mode"], row["H"])].append(row)
    for (condition, mode, H), group in groups.items():
        group.sort(key=lambda r: r["seed"])
        require([r["seed"] for r in group] == list(SEEDS), "Paired summary needs all three seeds")
        values = [r["delta_pp"] for r in group]
        summaries.append({"condition": condition, "mode": mode, "H": H, "seeds": list(SEEDS),
                          "initial_mean": mean(r["initial_success"] for r in group),
                          "initial_sd": stdev(r["initial_success"] for r in group),
                          "final_mean": mean(r["final_S11_success"] for r in group),
                          "final_sd": stdev(r["final_S11_success"] for r in group),
                          "paired_mean_pp": mean(values), "paired_sd_pp": stdev(values), "paired_values_pp": values,
                          "all_S11_paths_constant_after_first": all(r["S11_post_first_recon_constant"] for r in group)})
    return per_seed, paired, summaries


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(out, plan):
    records = []
    for job in plan["jobs"]:
        path = target_path(out, job)
        require(path.exists(), f"Missing diagnostic result: {path}")
        value = read_json(path)
        validate_record(value, job, plan)
        records.append(value)
    require(len(records) == 12, "Expected all 12 frozen model/reconstruction systems")
    per_seed, paired, summaries = summarize(records)
    cost = {key: sum(r["actual_cost"][key] for r in records) for key in ("episodes", "sim_steps", "worker_seconds")}
    reused = sum(r["job"]["reuse"] is not None for r in records)
    write_csv(out / "diagnostic_per_seed.csv", per_seed)
    write_csv(out / "diagnostic_paired.csv", paired)
    write_json(out / "diagnostic_summary.json", {"definition": plan, "summary": summaries,
                "actual_cost": cost, "reused_system_evaluations": reused, "training_steps": 0, "calibration_attempts": 0})
    lines = ["# Post-hoc matched-belief checkpoint diagnosis", "",
             "Designed after the S10/height_bias/seed 1 repeated-calibration failure. This uses the already observed "
             "57M final development task bank, and is not a preregistered new method or an independent confirmation test.", "",
             "For all conditions and seeds, reconstruction is fixed to S11 round-one recon_after; S10's same-round reconstruction "
             "is verified identical. Compare the source initial policy with the source final S11 policy. No second set of policies "
             "was trained and no reconstruction was calibrated. Same-MuJoCo proxy, pack H=2/4 only; no long-horizon or real-robot claim.", "",
             f"All 12 systems completed. {reused} exact matching source evaluations were reused; actual new rollouts: {cost['episodes']}. "
             f"Logical evaluation records: 960; unique task identities: 40. Actual simulation steps: {cost['sim_steps']}; "
             f"summed worker seconds: {cost['worker_seconds']:.2f}. Reuse costs zero new rollouts.", "",
             "## Paired full-task success", "",
             "Mean ± sample SD across the same three seeds; SD is not a confidence interval. All per-seed values "
             "are retained in the CSVs. No significance claim. Macro equally weights H=2 and H=4.", "",
             "| Condition | Mode | H | Initial SR mean ± SD (%) | Final S11 SR mean ± SD (%) | Paired Δ mean ± SD (pp) | Seed Δ (pp) |",
             "|---|---|---|---:|---:|---:|---|"]
    for row in summaries:
        values = ", ".join(f"s{s}: {v:+.2f}" for s, v in zip(row["seeds"], row["paired_values_pp"]))
        lines.append(f"| {row['condition']} | {row['mode']} | {row['H']} | {100*row['initial_mean']:.2f} ± {100*row['initial_sd']:.2f} | "
                     f"{100*row['final_mean']:.2f} ± {100*row['final_sd']:.2f} | {row['paired_mean_pp']:+.2f} ± {row['paired_sd_pp']:.2f} | {values} |")
    lines += ["", "## Source-path check", "", "| Condition | Seed | S11 belief constant after round one | Initial evaluation source | Final policy evaluation source |",
              "|---|---|---|---|---|"]
    for condition in CONDITIONS:
        for seed in SEEDS:
            pair = [j for j in plan["jobs"] if j["condition"] == condition and j["seed"] == seed]
            labels = [f"reused {j['reuse']['source_arm']} final" if j["reuse"] else "new frozen evaluation" for j in pair]
            lines.append(f"| {condition} | {seed} | {pair[0]['S11_post_first_recon_constant']} | {labels[0]} | {labels[1]} |")
    lines += ["", "When the source-path check is true, the original S11 path continued learning after its first calibration "
              "without later belief changes. When false, the result remains a checkpoint comparison at matched evaluation belief; "
              "it cannot be described as training under a fixed belief. In both cases this diagnosis is post-hoc and does not "
              "retroactively remove path-dependent repeated-calibration effects from the original S11−S10 contrast.", "",
              "Every reused evaluation cites a source result path/hash and evaluation hash in diagnostic_definition.json and its run record. "
              "Reuse requires identical policy bytes, reconstruction, config and task/rollout/artifact identities. "
              "The separate fixed H=6/8/10 transfer experiment is unchanged.", ""]
    (out / "DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "validation.json", {"passed": True, "systems": 12, "reused_system_evaluations": reused,
                "actual_cost": cost, "post_hoc": True, "confirmation_executed": False,
                "checks": ["complete source matrix and source snapshot hashes", "first-round S10/S11 reconstruction equality",
                           "fixed first-round reconstruction selection", "model bytes", "source-path stability recorded",
                           "exact reuse signatures and source citations", "task identities and artifact sidecars", "ledger and record checksums"]})
    print(f"Matched-belief diagnostic reported: {out / 'DIAGNOSTIC.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    require(args.workers > 0 and not (args.plan_only and args.report_only), "Invalid execution options")
    source, out = args.source_run.resolve(), args.out.resolve()
    require(source != out and source not in out.parents, "Diagnostic output must not alter the source experiment")
    manifest = source_manifest(source)
    sources = load_complete_source(source, manifest)
    jobs = source_jobs(sources, manifest)
    plan = make_plan(source, manifest, jobs)
    prepare_plan(out, plan)
    if args.plan_only:
        print(f"Plan saved: {out / 'PLAN.md'}; expected new rollouts={plan['planned_new_rollouts']}; none executed")
        return
    if args.report_only:
        report(out, plan)
        return
    from horizon.harness.artifacts import load_policy
    from horizon.harness.experiment import TaskStore, evaluate, source_hashes
    from horizon.real2sim.recon_error import ReconModel
    require(source_hashes() == manifest["source_sha256"], "Set PYTHONPATH to the frozen source experiment src/")
    store = TaskStore(out / "tasks", manifest["config"], Path(__file__).resolve().parents[1])
    for job in jobs:
        path = target_path(out, job)
        if path.exists():
            validate_record(read_json(path), job, plan)
            print(f"SKIP verified {job['condition']}/{job['system']}/s{job['seed']}", flush=True)
            continue
        started = time.perf_counter()
        if job["reuse"]:
            evaluation = copy.deepcopy(read_json(job["reuse"]["source_result"]["path"])["final"])
            cost = {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}
        else:
            recon = ReconModel.from_dict(job["recon"])
            evaluation, cost = evaluate(load_policy(job["policy"]), manifest["config"], store, recon, BANK, N_PER, args.workers)
            require(recon.to_dict() == job["recon"], "Frozen reconstruction changed")
        require(file_digest(__file__) == plan["script_sha256"] and
                file_digest(Path(__file__).with_name("evaluate_harness_transfer.py")) == plan["helper_sha256"], "Diagnostic code changed")
        require(source_hashes() == manifest["source_sha256"], "Frozen source changed")
        require(file_digest(job["policy"]["path"]) == job["policy"]["sha256"], "Source model changed")
        value = {"schema_version": 1, "status": "complete", "job": job, "definition_fingerprint": plan["fingerprint"],
                 "evaluation": evaluation, "evaluation_sha256": digest(evaluation), "actual_cost": cost,
                 "training_steps": 0, "calibration_attempts": 0, "workers": args.workers,
                 "wall_seconds": time.perf_counter()-started}
        value["record_sha256"] = digest(value)
        validate_record(value, job, plan)
        write_json(path, value)
        print(f"DONE {job['condition']}/{job['system']}/s{job['seed']}: "
              f"{evaluation['nominal']['overall']:.3f}/{evaluation['perturbed']['overall']:.3f}; new rollouts={cost['episodes']}", flush=True)
    report(out, plan)


if __name__ == "__main__":
    main()
