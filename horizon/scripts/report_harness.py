"""Validate and report the E1 learning harness without running any rollouts.

Usage: python scripts/report_harness.py results/harness_e1
Only saved development results are read. Seed SD is descriptive, not a CI.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ARMS = ("S00", "S01", "S10", "S11")
LABELS = {
    "S00": "Fixed twin / frozen policy",
    "S01": "Fixed twin / policy learning",
    "S10": "Calibrated twin / frozen policy",
    "S11": "Calibrated twin / policy learning",
}
COLORS = {"S00": "#777777", "S01": "#2878b5", "S10": "#de8f05", "S11": "#20958a"}
MODES = ("nominal", "perturbed")
PHASES = ("pre", "post_cal", "post_train")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def describe(values):
    values = [float(v) for v in values]
    return {"values": values, "n_seeds": len(values), "mean": mean(values) if values else None,
            "sample_sd": stdev(values) if len(values) > 1 else None}


def identity(episode):
    return (episode["family"], int(episode["H"]), int(episode["task_seed"]),
            int(episode["rollout_seed"]), episode.get("task_sha256"))


def evaluated_rows(evaluation, horizon=None):
    return [e for e in evaluation["episodes"] if horizon is None or int(e["H"]) == horizon]


def score(evaluation, horizon=None):
    episodes = evaluated_rows(evaluation, horizon)
    require(bool(episodes), f"No evaluation episodes for H={horizon}")
    return mean(int(e["success"]) for e in episodes)


def validate_evaluation(evaluation, label, config=None, n_per=None):
    require(isinstance(evaluation, dict), f"Missing evaluation: {label}")
    ids = {}
    for mode in MODES:
        require(mode in evaluation, f"Missing mode: {label}/{mode}")
        item = evaluation[mode]
        require(not item.get("errors"), f"Rollout errors: {label}/{mode}")
        episodes = item["episodes"]
        require(bool(episodes), f"Empty evaluation: {label}/{mode}")
        keys = [identity(e) for e in episodes]
        require(len(keys) == len(set(keys)), f"Duplicate rollout identity: {label}/{mode}")
        require(all(e["success"] in (True, False, 0, 1) for e in episodes),
                f"Nonbinary success: {label}/{mode}")
        require(math.isclose(score(item), float(item["overall"]), abs_tol=1e-12),
                f"Stored success rate differs from episode counts: {label}/{mode}")
        if config is not None and n_per is not None:
            expected = Counter({(family, int(H)): int(n_per) for family in config["families"] for H in config["H_values"]})
            require(Counter((e["family"], int(e["H"])) for e in episodes) == expected,
                    f"Wrong family/horizon evaluation grid: {label}/{mode}")
        ids[mode] = tuple(sorted(keys))
    return ids


def load_and_validate(folder, allow_partial=False):
    files = sorted((folder / "runs").glob("*/*_s*.json"))
    require(bool(files), f"No run files under {folder / 'runs'}")
    runs, incomplete = {}, []
    for path in files:
        run = json.loads(path.read_text(encoding="utf-8"))
        key = (run["condition"], run["arm"], int(run["seed"]))
        require(key not in runs, f"Duplicate run: {key}")
        if run.get("status") != "complete":
            incomplete.append({"path": str(path), "status": run.get("status")})
            continue
        require(run.get("schema_version") == 1, f"Unsupported schema: {path}")
        require(run["arm"] in ARMS, f"Unknown arm: {key}")
        runs[key] = run
    require(bool(runs), "No complete runs to report")
    first = next(iter(runs.values()))
    config = first["experiment"]["config"]
    require(config.get("calibration_schedule", "every_round") in ("every_round", "first_round"),
            "Unknown calibration_schedule")
    require(config.get("learning_collection", "dagger") in ("dagger", "nominal_recovery"),
            "Unknown learning_collection")
    expected = {(c, a, int(s)) for c in config["conditions"] for a in config["arms"] for s in config["seeds"]}
    extra = sorted(set(runs) - expected)
    missing = sorted(expected - set(runs))
    require(not extra, f"Unexpected runs outside config: {extra}")
    require(allow_partial or not (missing or incomplete),
            f"Incomplete experiment: {len(missing)} missing runs; use --allow-partial for a development preview")
    initial, paired, fingerprints, source_versions, final_task_ids = defaultdict(set), defaultdict(set), set(), set(), set()
    evaluation_rollouts = 0
    for (condition, arm, seed), run in sorted(runs.items()):
        label = f"{condition}/{arm}/s{seed}"
        require(canonical(run["experiment"]["config"]) == canonical(config), f"Config mismatch: {label}")
        fingerprints.add(run["experiment"]["fingerprint"])
        source_versions.add(canonical(run["experiment"]["source_sha256"]))
        init = run["initial"]
        initial[(condition, seed)].add((init["data_sha256"], init["policy_sha256"], int(init["n"]), int(init["train_steps"])))
        require(int(init["n"]) == int(config["init_budget"]), f"Wrong initial data count: {label}")
        history = run["history"]
        require(len(history) == int(config["rounds"]), f"Wrong number of rounds: {label}")
        rounds = [int(h["round"]) for h in history]
        require(rounds == list(range(1, len(history) + 1)), f"Nonsequential round numbers: {label}")
        previous_n = int(init["n"])
        previous_policy = init["policy_sha256"]
        previous_recon = None
        previous_round = None
        for h in history:
            r = int(h["round"])
            require(int(h["n_before"]) == previous_n, f"Dataset continuity: {label}/r{r}")
            require(h["policy_before"] == previous_policy, f"Policy continuity: {label}/r{r}")
            if previous_recon is not None:
                require(canonical(h["recon_before"]) == previous_recon, f"Twin continuity: {label}/r{r}")
            calibration_due = arm[1] == "1" and (config.get("calibration_schedule", "every_round") == "every_round" or r == 1)
            pre_reused = config.get("reuse_pre_evaluation", False) and previous_round is not None
            require(h["reused_equal_state_evaluation"].get("pre", False) is pre_reused,
                    f"Pre-evaluation reuse flag mismatch: {label}/r{r}")
            if pre_reused:
                require(h["pre"] == previous_round["post_train"] and
                        h.get("pre_evaluation_source_round") == r - 1,
                        f"Cached pre-evaluation differs from preceding round: {label}/r{r}")
                require(h["ledger"]["pre_evaluation"] == {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0},
                        f"Reused pre-evaluation charged new work: {label}/r{r}")
            require(h["calibration_attempted"] is calibration_due, f"Calibration schedule mismatch: {label}/r{r}")
            if not calibration_due:
                require(canonical(h["recon_before"]) == canonical(h["recon_after"]),
                        f"Unscheduled calibration changed twin: {label}/r{r}")
                require(h["geometry_report"] == {} and h["ledger"]["calibration_seconds"] == 0,
                        f"Unscheduled calibration charged/reported work: {label}/r{r}")
            if arm[2] == "0":
                require(h["policy_before"] == h["policy_after"], f"Frozen policy changed: {label}/r{r}")
                require(int(h["n_after"]) == previous_n, f"Frozen arm collected training data: {label}/r{r}")
            else:
                require(int(h["n_after"]) - previous_n == int(config["round_budget"]),
                        f"Wrong round data budget: {label}/r{r}")
            phase_ids = {}
            for phase in PHASES:
                phase_ids[phase] = validate_evaluation(h[phase], f"{label}/r{r}/{phase}", config, config.get("curve_n_per"))
                for mode in MODES:
                    paired[(condition, seed, r, phase, mode)].add(phase_ids[phase][mode])
                    evaluation_rollouts += len(h[phase][mode]["episodes"])
            require(phase_ids["pre"] == phase_ids["post_cal"] == phase_ids["post_train"],
                    f"Stages use different rollout identities: {label}/r{r}")
            if not calibration_due:
                for mode in MODES:
                    require({identity(e): e["success"] for e in h["pre"][mode]["episodes"]} ==
                            {identity(e): e["success"] for e in h["post_cal"][mode]["episodes"]},
                            f"Fixed twin stage changed task outcomes: {label}/r{r}/{mode}")
            if arm[2] == "0":
                for mode in MODES:
                    require({identity(e): e["success"] for e in h["post_cal"][mode]["episodes"]} ==
                            {identity(e): e["success"] for e in h["post_train"][mode]["episodes"]},
                            f"Frozen policy stage changed task outcomes: {label}/r{r}/{mode}")
            previous_n = int(h["n_after"])
            previous_policy = h["policy_after"]
            previous_recon = canonical(h["recon_after"])
            previous_round = h
        ledger = run["ledger"]
        require(ledger["geometry_attempts"] == sum(h["calibration_attempted"] for h in history),
                f"Calibration attempt ledger mismatch: {label}")
        if "additional_labels" in ledger:
            require(int(ledger["additional_labels"]) == previous_n - int(init["n"]), f"Label ledger mismatch: {label}")
        if "additional_training_steps" in ledger:
            require(int(ledger["additional_training_steps"]) == sum(int(h["ledger"]["optimizer_steps"]) for h in history),
                    f"Training ledger mismatch: {label}")
            if arm[2] == "0":
                require(ledger["additional_training_steps"] == 0, f"Frozen arm optimized policy: {label}")
        if "feedback_episodes" in ledger:
            require(int(ledger["feedback_episodes"]) == len(history) * int(config["feedback_episodes"]),
                    f"Feedback ledger mismatch: {label}")
        final_ids = validate_evaluation(run["final"], f"{label}/final", config, config.get("final_n_per"))
        for mode in MODES:
            paired[(condition, seed, "final", mode)].add(final_ids[mode])
            final_task_ids.update((condition, *identity(e)[:3]) for e in run["final"][mode]["episodes"])
            evaluation_rollouts += len(run["final"][mode]["episodes"])
            curve_tasks = {identity(e)[:3] for h in history for e in h["pre"][mode]["episodes"]}
            require(not curve_tasks.intersection({identity(e)[:3] for e in run["final"][mode]["episodes"]}),
                    f"Curve/final development task overlap: {label}/{mode}")
    require(len(fingerprints) == 1, "Mixed experiment fingerprints")
    require(len(source_versions) == 1, "Mixed executed source versions")
    require(all(len(v) == 1 for v in initial.values()), "Initial data or policy differs between arms")
    require(all(len(v) == 1 for v in paired.values()), "Unpaired rollout identities between arms")
    validation = {
        "passed": True, "partial": bool(missing or incomplete), "complete_runs": len(runs),
        "expected_runs": len(expected), "missing_runs": [list(k) for k in missing], "incomplete_files": incomplete,
        "fingerprint": next(iter(fingerprints)), "source_versions": len(source_versions),
        "evaluation_records": evaluation_rollouts, "unique_final_tasks_including_condition": len(final_task_ids),
        "checks": ["configured run grid", "same initial data and policy per condition/seed", "same source fingerprint",
                   "dataset/policy/twin continuity", "treatment permissions", "per-episode score arithmetic",
                   "paired stage and arm rollout identities", "curve/final task separation"],
        "limits": ["Saved records checked; no rollouts rerun and no independent external validation",
                   "Source hashes are compared between runs; this report does not attest source contents",
                   "Same-MuJoCo deployment proxy; development results, not real-robot evidence",
                   "Episode records can include reused evaluations; actual physical costs are in the run ledgers"],
    }
    return runs, config, validation


def summarize(runs, config, validation):
    stages, final_rows, effects, summary_rows, effect_summaries = [], [], [], [], []
    horizons = [None, *sorted(map(int, config["H_values"]))]
    for (condition, arm, seed), run in sorted(runs.items()):
        for h in run["history"]:
            for mode in MODES:
                for horizon in horizons:
                    for phase in PHASES:
                        episodes = evaluated_rows(h[phase][mode], horizon)
                        stages.append({"condition": condition, "arm": arm, "seed": seed, "round": h["round"],
                                       "phase": phase, "mode": mode, "H": horizon if horizon is not None else "all",
                                       "k": sum(int(e["success"]) for e in episodes), "n": len(episodes),
                                       "success": score(h[phase][mode], horizon),
                                       "training_transitions": h["n_after"] if phase == "post_train" else h["n_before"]})
                    pre, post_cal, post_train = [score(h[p][mode], horizon) for p in PHASES]
                    for name, value in (("immediate_calibration_gain", post_cal - pre),
                                        ("learning_gain", post_train - post_cal),
                                        ("round_total_gain", post_train - pre)):
                        effects.append({"condition": condition, "arm": arm, "seed": seed, "round": h["round"],
                                        "mode": mode, "H": horizon if horizon is not None else "all",
                                        "effect": name, "delta_pp": 100 * value, "task_bank": "curve_development"})
        for mode in MODES:
            for horizon in horizons:
                episodes = evaluated_rows(run["final"][mode], horizon)
                final_rows.append({"condition": condition, "arm": arm, "seed": seed, "mode": mode,
                                   "H": horizon if horizon is not None else "all",
                                   "k": sum(int(e["success"]) for e in episodes), "n": len(episodes),
                                   "success": score(run["final"][mode], horizon),
                                   "training_transitions": run["history"][-1]["n_after"] if run["history"] else run["initial"]["n"],
                                   "initial_train_steps": run["initial"]["train_steps"],
                                   "ledger_json": canonical(run["ledger"])})
    by_final = {(r["condition"], r["arm"], r["seed"], r["mode"], r["H"]): r for r in final_rows}
    omitted = []
    for condition in config["conditions"]:
        for seed in map(int, config["seeds"]):
            if not all((condition, arm, seed) in runs for arm in ARMS):
                omitted.append({"condition": condition, "seed": seed, "reason": "Four complete factorial arms required"})
                continue
            for mode in MODES:
                for horizon in horizons:
                    H = horizon if horizon is not None else "all"
                    s00, s01, s10, s11 = [by_final[(condition, arm, seed, mode, H)]["success"] for arm in ARMS]
                    contrasts = {
                        "interaction": s11 - s10 - s01 + s00,
                        "learning_with_fixed_twin": s01 - s00,
                        "learning_with_calibrated_twin": s11 - s10,
                        "calibration_with_frozen_policy": s10 - s00,
                        "calibration_with_learning": s11 - s01,
                        "joint_total": s11 - s00,
                    }
                    for name, delta in contrasts.items():
                        effects.append({"condition": condition, "arm": "factorial", "seed": seed, "round": "final",
                                        "mode": mode, "H": H, "effect": name, "delta_pp": 100 * delta,
                                        "task_bank": "final_development"})
    grouped = defaultdict(list)
    for row in final_rows:
        grouped[(row["condition"], row["arm"], row["mode"], row["H"])].append(row)
    for key, rows in grouped.items():
        rows.sort(key=lambda r: r["seed"])
        summary_rows.append(dict(zip(("condition", "arm", "mode", "H"), key),
                                 seeds=[r["seed"] for r in rows], **describe([r["success"] for r in rows])))
    grouped_effects = defaultdict(list)
    for row in effects:
        grouped_effects[(row["condition"], row["arm"], row["round"], row["mode"], row["H"], row["effect"])].append(row)
    for key, rows in grouped_effects.items():
        rows.sort(key=lambda r: r["seed"])
        effect_summaries.append(dict(zip(("condition", "arm", "round", "mode", "H", "effect"), key),
                                    seeds=[r["seed"] for r in rows], **describe([r["delta_pp"] for r in rows])))
    summary = {"schema_version": 1, "validation": validation, "config": config,
               "final": summary_rows, "effects": effect_summaries, "omitted_factorial_contrasts": omitted,
               "resources": [{"condition": c, "arm": a, "seed": s, "initial": r["initial"], "ledger": r["ledger"]}
                             for (c, a, s), r in sorted(runs.items())]}
    return summary, stages, final_rows, effects


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def family_horizon_metrics(runs, config):
    """Additional per-cell rows; preserve the existing aggregate report schema."""
    rows = []
    for (condition, arm, seed), run in sorted(runs.items()):
        measurements = [(h["round"], phase, h[phase]) for h in run["history"] for phase in PHASES]
        measurements.append(("final", "final", run["final"]))
        for round_id, phase, evaluation in measurements:
            for mode in MODES:
                for family in config["families"]:
                    for H in config["H_values"]:
                        episodes = [e for e in evaluation[mode]["episodes"] if e["family"] == family and e["H"] == H]
                        require(bool(episodes), f"Empty family/H cell: {family}/{H}")
                        k = sum(int(e["success"]) for e in episodes)
                        row = {"condition": condition, "arm": arm, "seed": seed, "round": round_id,
                               "phase": phase, "mode": mode, "family": family, "H": H,
                               "k": k, "n": len(episodes), "success": k / len(episodes)}
                        for diagnostic in ("progress", "final_satisfied", "max_satisfied", "first_unsatisfied_index"):
                            row[f"mean_{diagnostic}"] = mean(e[diagnostic] for e in episodes) if all(
                                diagnostic in e for e in episodes) else None
                        rows.append(row)
    return rows


def formatted_stats(row, scale=100):
    value = f"{row['mean'] * scale:.2f}"
    return value + (f" ± {row['sample_sd'] * scale:.2f}" if row["sample_sd"] is not None else " (one seed)")


def render_markdown(summary):
    validation, config = summary["validation"], summary["config"]
    collection = config.get("learning_collection", "dagger")
    learner = "β-mixed DAgger" if collection == "dagger" else "Fresh nominal/recovery expert demonstrations + BC"
    calibration = "Observable-evidence calibration, " + (
        "first round only; then frozen" if config.get("calibration_schedule", "every_round") == "first_round" else "every round")
    lines = ["# Real2Sim2Real learning harness: development results", "",
             f"Status: {'PARTIAL PREVIEW' if validation['partial'] else 'complete configured run grid'}; "
             f"{validation['complete_runs']}/{validation['expected_runs']} runs.", "",
             "Same-MuJoCo deployment proxy. These results do not establish physical-robot transfer, "
             "an agent decision advantage, or recursive improvement of the improver.", "",
             "Success means full-task evaluation truth, not judged progress. Means and sample SD are across "
             "initialization/training seeds; SD is not a confidence interval. All seed differences are retained. "
             "No significance tests are claimed. Shared tasks and repeated rounds are not independent replications.", "",
             "## Factorial definition", "",
             "| Arm | Twin update | Policy learning |", "|---|---|---|",
             "| S00 | Fixed | Frozen |", f"| S01 | Fixed | {learner} |",
             f"| S10 | {calibration} | Frozen |",
             f"| S11 | {calibration} | {learner} |", "",
             f"Configured H: {config['H_values']}; families: {config['families']}; rounds: {config['rounds']}.",
             "Arms share the materialized initial dataset and policy within each condition/seed. Frozen-policy "
             "arms use less actual training and are mechanism controls, not matched-training-cost efficiency baselines. "
             "Different conditions can have different initial policies; do not interpret their absolute score "
             "difference as a calibration treatment effect. Nominal/recovery collection keeps successful expert "
             "episodes under the original recovery recipe; DAgger labels learner-visited states under expert mixing. "
             "Equal retained-label and optimizer budgets do not imply equal rollout cost.", "", "## Final development success", "",
             "| Condition | Arm | Mode | SR mean ± seed SD (%) | Seed SR (%) |", "|---|---|---|---:|---|"]
    for row in summary["final"]:
        if row["H"] == "all":
            values = ", ".join(f"s{s}: {100*v:.2f}" for s, v in zip(row["seeds"], row["values"]))
            lines.append(f"| {row['condition']} | {row['arm']} | {row['mode']} | {formatted_stats(row)} | {values} |")
    lines += ["", "## Paired final factorial effects", "",
              "Interaction = S11 − S10 − S01 + S00. It describes the full adaptive trajectories, "
              "including changed feedback and accumulated training data; it does not isolate data quality alone. "
              "Per-round gains on the separate curve bank must not be summed or spliced into this final-bank comparison.", "",
              "| Condition | Mode | Effect | Mean ± seed SD (pp) | Seed deltas (pp) |", "|---|---|---|---:|---|"]
    for row in summary["effects"]:
        if row["round"] == "final" and row["H"] == "all":
            values = ", ".join(f"s{s}: {v:+.2f}" for s, v in zip(row["seeds"], row["values"]))
            lines.append(f"| {row['condition']} | {row['mode']} | {row['effect']} | {formatted_stats(row, 1)} | {values} |")
    if summary["omitted_factorial_contrasts"]:
        lines += ["", "Factorial contrasts omitted for incomplete four-arm groups: " +
                  ", ".join(f"{r['condition']}/s{r['seed']}" for r in summary["omitted_factorial_contrasts"]) + "."]
    lines += ["", "## Per-round mechanism measurements", "",
              "`immediate_calibration_gain = post_cal − pre`; `learning_gain = post_train − post_cal`; "
              "`round_total_gain = post_train − pre`. All use the same task and rollout identities within a round. "
              "These local effects are measured along each arm's own history, not counterfactual trajectories "
              "with identical later feedback. Full per-H and per-seed values are in paired_effects.csv.", "",
              "## Resource accounting and interpretation", "",
              "per_seed.csv and summary.json preserve each arm's resource ledger. Initial data/policy costs "
              "are logically assigned to every arm although physically shared; use the runner's physical-cost "
              "record for actual experiment expense. Evaluation records may reuse an unchanged checkpoint evaluation, "
              "so their count is not automatically the number of executed rollouts. No synthetic human-minute "
              "or robot-minute conversion is applied.", "",
              "Previously collected training samples remain in the accumulated dataset after twin updates. "
              "Therefore joint learning can retain effects of earlier reconstruction errors. Calibration and "
              "learning gains should be read alongside per-round belief versions and dataset sizes.", "",
              "## Files", "",
              "- validation.json: grid, pairing, treatment and arithmetic checks.",
              "- summary.json: seed summaries, factorial contrasts, config and original resource ledgers.",
              "- per_seed.csv: final success counts by condition, arm, seed, mode and H.",
              "- stage_metrics.csv: pre / post-calibration / post-training curve-bank scores.",
              "- paired_effects.csv: per-seed local gains and full-path factorial contrasts.",
              "- family_horizon_metrics.csv: separate family × H × seed cells for every stage and final evaluation; optional final/max predicate diagnostics remain distinct from SR.",
              "- learning_curves.png / .pdf: development learning trajectories and final horizon curves (when plotting enabled).", ""]
    if any(r["round"] == "final" for r in summary["effects"]):
        lines.append("- factorial_effects.png / .pdf: paired seed effects and mean effects (when plotting enabled).")
    return "\n".join(lines)


def make_plots(folder, summary, stages):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = list(summary["config"]["conditions"])
    arms = [a for a in ARMS if a in summary["config"]["arms"]]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(len(conditions), 4, figsize=(15, 3.6 * len(conditions)), squeeze=False, layout="constrained")
    for ci, condition in enumerate(conditions):
        for mi, mode in enumerate(MODES):
            curve_ax, horizon_ax = axes[ci, 2*mi:2*mi+2]
            for arm in arms:
                selected = [r for r in stages if r["condition"] == condition and r["arm"] == arm and
                            r["mode"] == mode and r["H"] == "all"]
                seeds = sorted({r["seed"] for r in selected})
                if not seeds:
                    continue
                values = []
                for seed in seeds:
                    rows = [r for r in selected if r["seed"] == seed and r["phase"] == "post_train"]
                    rows.sort(key=lambda r: r["round"])
                    first = next(r for r in selected if r["seed"] == seed and r["round"] == 1 and r["phase"] == "pre")
                    x = [0, *[r["round"] for r in rows]]
                    y = [first["success"], *[r["success"] for r in rows]]
                    values.append(y)
                    curve_ax.plot(x, y, color=COLORS[arm], alpha=.2, linewidth=.8)
                curve_ax.plot(x, np.mean(values, axis=0), marker="o", markersize=3.5, color=COLORS[arm], label=arm)
                final = [r for r in summary["final"] if r["condition"] == condition and r["arm"] == arm and
                         r["mode"] == mode and r["H"] != "all"]
                final.sort(key=lambda r: int(r["H"]))
                horizon_ax.plot([r["H"] for r in final], [r["mean"] for r in final], color=COLORS[arm],
                                marker="o", markersize=3.5, label=arm)
                for seed in seeds:
                    horizon_ax.plot([r["H"] for r in final], [r["values"][r["seeds"].index(seed)] for r in final],
                                    color=COLORS[arm], alpha=.2, linewidth=.8)
            curve_ax.set(title=f"{condition} · {mode}\nCurve development bank", xlabel="Learning round")
            horizon_ax.set(title=f"{condition} · {mode}\nFinal development bank", xlabel="Subgoals H")
            horizon_ax.set_xticks(summary["config"]["H_values"])
            for ax in (curve_ax, horizon_ax):
                ax.set_ylim(0, 1.03)
                ax.set_ylabel("Full-task success")
                ax.grid(axis="y", alpha=.2)
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Same-MuJoCo proxy · development · thin lines are individual seeds" +
                 (" · PARTIAL" if summary["validation"]["partial"] else ""), fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"learning_curves.{extension}", dpi=200)
    plt.close(fig)

    if not any(r["round"] == "final" for r in summary["effects"]):
        return

    contrasts = ["learning_with_fixed_twin", "learning_with_calibrated_twin", "calibration_with_frozen_policy",
                 "calibration_with_learning", "interaction"]
    names = ["Learn, fixed twin", "Learn, calibrated twin", "Calibrate, frozen policy", "Calibrate, learning", "Interaction"]
    fig, axes = plt.subplots(len(conditions), 2, figsize=(12, 3.9*len(conditions)), squeeze=False, layout="constrained")
    for ci, condition in enumerate(conditions):
        for mi, mode in enumerate(MODES):
            ax = axes[ci, mi]
            for i, effect in enumerate(contrasts):
                found = [r for r in summary["effects"] if r["condition"] == condition and r["mode"] == mode and
                         r["round"] == "final" and r["H"] == "all" and r["effect"] == effect]
                if not found:
                    continue
                row = found[0]
                ax.barh(i, row["mean"], color="#2878b5" if effect != "interaction" else "#20958a", alpha=.6, height=.55)
                offsets = np.linspace(-.12, .12, len(row["values"])) if len(row["values"]) > 1 else [0]
                ax.scatter(row["values"], i+np.asarray(offsets), color="#202020", s=20, zorder=3)
            ax.axvline(0, color="#888888", linewidth=.8)
            ax.set_yticks(range(len(contrasts)), names)
            ax.invert_yaxis()
            ax.set(title=f"{condition} · {mode}", xlabel="Paired final success difference (pp)")
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("Full-path factorial effects · bars: seed mean · dots: paired seed deltas · no significance claim", fontsize=11)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"factorial_effects.{extension}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Report complete runs as an explicitly partial preview")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    runs, config, validation = load_and_validate(args.folder, args.allow_partial)
    summary, stages, final_rows, effects = summarize(runs, config, validation)
    for name, payload in (("summary.json", summary), ("validation.json", validation)):
        (args.folder / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_csv(args.folder / "stage_metrics.csv", stages)
    write_csv(args.folder / "per_seed.csv", final_rows)
    write_csv(args.folder / "paired_effects.csv", effects)
    write_csv(args.folder / "family_horizon_metrics.csv", family_horizon_metrics(runs, config))
    (args.folder / "RESULTS.md").write_text(render_markdown(summary), encoding="utf-8")
    if not args.no_plots:
        make_plots(args.folder, summary, stages)
    print(f"Reported {len(runs)}/{validation['expected_runs']} runs: {args.folder / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
