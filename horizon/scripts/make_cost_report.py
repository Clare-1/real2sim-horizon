"""Seed-level, cost-aware reporting for paired repair experiments.

Does not treat test episodes as independent training seeds. Reports descriptive
sample SD and paired seed differences, not a significance claim from two seeds.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from statistics import mean, stdev


def load_runs(folder: Path) -> dict:
    runs = defaultdict(dict)
    for p in sorted(folder.glob("*.json")):
        match = re.fullmatch(r"(.+)_s([0-9]+)\.json", p.name)
        if match is None:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("experiment", {}).get("source_unchanged") is False:
            raise ValueError(f"Source changed during run: {p}")
        c = r["config"]
        seed, method = c["seed"], c["name"]
        if (method, seed) != (match.group(1), int(match.group(2))):
            raise ValueError(f"Filename does not match method/seed: {p}")
        if seed in runs[method]:
            raise ValueError(f"Duplicate method/seed: {method}/{seed}")
        limit = c.get("repair_budget")
        if limit is not None and r["repair_cost_used"] > limit + 1e-8:
            raise ValueError(f"Repair budget exceeded: {p}")
        attempts = sum(h.get("repair_cost", 0.) for h in r["history"])
        if abs(attempts - r["repair_cost_used"]) > 1e-8:
            raise ValueError(f"Inconsistent repair cost ledger: {p}")
        runs[method][seed] = r
    if not runs:
        raise ValueError(f"No result JSON files in {folder}")
    protocols = {r.get("protocol", "training_loop") for rs in runs.values() for r in rs.values()}
    if len(protocols) != 1:
        raise ValueError("Do not combine fixed-policy probes and retraining runs")
    grids = {tuple(sorted((k, v["n"]) for k, v in r["final"]["real_nominal"]["by_family_H"].items()))
             for rs in runs.values() for r in rs.values()}
    if len(grids) != 1:
        raise ValueError("Mismatched evaluation grids or test budgets")
    resources = {(r.get("n"), r.get("train_steps")) for rs in runs.values() for r in rs.values()}
    if len(resources) != 1:
        raise ValueError("Mismatched training transition / step budgets")
    for key in ("real_nominal", "real_pert0.15", "sim_nominal"):
        tasks = {tuple((e["family"], e["H"], e["seed"]) for e in r["final"].get(key, {}).get("episodes", []))
                 for rs in runs.values() for r in rs.values()}
        if len(tasks) != 1:
            raise ValueError(f"Mismatched test episode seeds: {key}")
    fair_keys = ("init_budget", "round_budget", "rounds", "real_episodes", "audit_frac", "twin_episodes",
                 "repair_budget", "repair_costs", "paired_feedback", "world", "bias", "judge_scale", "grounding",
                 "eval_seed_base", "curve_seed_base", "feedback_seed_base", "evaluation_split", "confirmation_seed_base")
    signatures = {json.dumps({k: r["config"].get(k) for k in fair_keys}, sort_keys=True)
                  for rs in runs.values() for r in rs.values()}
    if len(signatures) != 1:
        raise ValueError("Mismatched budget / injected-error protocol")
    manifests = {json.dumps(r.get("evaluation_manifest"), sort_keys=True)
                 for rs in runs.values() for r in rs.values()}
    if len(manifests) != 1:
        raise ValueError("Mismatched evaluation manifests")
    for seed in {s for rs in runs.values() for s in rs}:
        matched = [rs[seed] for rs in runs.values() if seed in rs]
        feedback = {json.dumps([h.get("feedback_tasks") for h in r["history"]], sort_keys=True)
                    for r in matched if r["config"].get("route") not in ("none", "perfect")}
        if len(feedback) > 1:
            raise ValueError(f"Mismatched paired feedback tasks at seed {seed}")
        checkpoints = {r.get("provenance", {}).get("checkpoint_sha256") for r in matched}
        if len(checkpoints) != 1:
            raise ValueError(f"Different starting policies at seed {seed}")
    return dict(runs)


def dispersion(values: list[float]) -> str:
    return f"{100*mean(values):.1f}" + (f" ± {100*stdev(values):.1f}" if len(values) > 1 else " (single seed)")


def make_report(runs: dict, reference: str) -> tuple[str, dict]:
    first = next(iter(next(iter(runs.values())).values()))
    protocol = first.get("protocol", "training_loop")
    split = first.get("evaluation_manifest", {}).get("split", "legacy/unrecorded")
    lines = ["# Paired repair experiment", "", f"Protocol: **{protocol}**. Evaluation split: **{split}**.", "",
             "Success is measured in a second configuration of the same MuJoCo simulator.",
             "Audits are simulated from privileged labels; this is not a real-robot or SOTA result.", "",
             "Values are training-seed means ± sample SD in percentage points. Seeds, not",
             "test episodes, are the independent replicates. Small seed counts support developmental conclusions only.", "",
             "| Method | Seeds | Nominal (%) | Perturbed (%) | Per-seed nominal (%) |",
             "|---|---|---:|---:|---|"]
    summary = {"protocol": protocol, "methods": {}, "paired_differences": {}}
    for method, rs in runs.items():
        seeds = sorted(rs)
        nominal = [rs[s]["final"]["real_nominal"]["overall"] for s in seeds]
        pert = [rs[s]["final"]["real_pert0.15"]["overall"] for s in seeds]
        per_seed = ", ".join(f"s{s}: {100*v:.1f}" for s, v in zip(seeds, nominal))
        lines.append(f"| {method} | {', '.join(map(str, seeds))} | {dispersion(nominal)} | {dispersion(pert)} | {per_seed} |")
        summary["methods"][method] = {"seeds": seeds, "nominal_mean": mean(nominal),
                                       "perturbed_mean": mean(pert), "per_seed_nominal": dict(zip(seeds, nominal))}
    lines += ["", "## Resources per run", "",
              "Training transitions/steps are additional costs in this experiment. Fixed-policy",
              "probes reuse previously trained checkpoints; their historical pretraining is not free.", "",
              "Repair costs are configurable abstract units, including rejected fits. They are",
              "not measured operator minutes. Zero-feedback and oracle controls are identified by route.", "",
              "Feedback deployments exclude curve/final evaluations. Final evaluations below include",
              "nominal and perturbed deployment episodes; audits count episodes, not individual labels.", "",
              "| Method / seed | Route | Transitions | Steps | Feedback deployments | Audit episodes | Final evaluations | Curve evaluations | Repair cost | Seconds |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, rs in runs.items():
        for seed, r in sorted(rs.items()):
            final_eval = sum(len(r["final"][k].get("episodes", [])) for k in ("real_nominal", "real_pert0.15"))
            curve_eval = r.get("curve_evaluation_episodes", "unrecorded")
            lines.append(f"| {method} / {seed} | {r['config']['route']} | {r['n']} | {r.get('train_steps', 'unrecorded')} | "
                         f"{r['real_episodes_used']} | {r['audits_used']} | {final_eval} | {curve_eval} | "
                         f"{r['repair_cost_used']:.1f} | {r['wall']:.1f} |")
    if reference in runs:
        lines += ["", f"## Paired differences relative to {reference}", "",
                  "Each difference uses the same training seed and test grid. No statistical significance",
                  "is asserted. Pair counts can differ when a sweep is incomplete.", "",
                  "| Method | Paired seeds | Nominal difference (pp) | Perturbed difference (pp) |",
                  "|---|---|---:|---:|"]
        for method, rs in runs.items():
            if method == reference:
                continue
            seeds = sorted(rs.keys() & runs[reference].keys())
            if not seeds:
                continue
            deltas = {key: [rs[s]["final"][key]["overall"] - runs[reference][s]["final"][key]["overall"] for s in seeds]
                      for key in ("real_nominal", "real_pert0.15")}
            lines.append(f"| {method} | {', '.join(map(str, seeds))} | {dispersion(deltas['real_nominal'])} | "
                         f"{dispersion(deltas['real_pert0.15'])} |")
            summary["paired_differences"][method] = {"reference": reference, "seeds": seeds, **deltas}
    Hs = sorted(first["final"]["real_nominal"]["by_H"], key=int)
    lines += ["", "## Success by horizon", "", "Training-seed means (%); nominal and perturbed conditions are separate rows.", "",
              "| Method / condition | " + " | ".join("H=" + H for H in Hs) + " |",
              "|---|" + "---:|" * len(Hs)]
    for method, rs in runs.items():
        for key, label in (("real_nominal", "nominal"), ("real_pert0.15", "perturbed")):
            values = [100 * mean(r["final"][key]["by_H"][H]["k"] / r["final"][key]["by_H"][H]["n"]
                                  for r in rs.values()) for H in Hs]
            lines.append(f"| {method} / {label} | " + " | ".join(f"{v:.1f}" for v in values) + " |")
    if any("audit_keyframes_used" in r for rs in runs.values() for r in rs.values()):
        lines += ["", "## Annotation and replay accounting", "",
                  "Audit episodes do not imply identical numbers of keyframe/predicate labels. Older",
                  "baseline executions did not record these counts; those values remain unrecorded.", "",
                  "| Method / seed | Audit keyframes | Predicate labels | Twin replays |",
                  "|---|---:|---:|---:|"]
        for method, rs in runs.items():
            for seed, r in sorted(rs.items()):
                lines.append(f"| {method} / {seed} | {r.get('audit_keyframes_used', 'unrecorded')} | "
                             f"{r.get('audit_predicate_labels_used', 'unrecorded')} | {r.get('twin_replays_used', 'unrecorded')} |")
    lines += ["", "## Accepted repairs and attempted costs", "",
              "| Method / seed | Per-round accepted repairs | Attempts | Cumulative cost |",
              "|---|---|---|---:|"]
    for method, rs in runs.items():
        for seed, r in sorted(rs.items()):
            chosen = "; ".join(f"r{h['round']}: {','.join(h.get('repair_committed', h.get('chosen', []))) or 'none'}" for h in r["history"])
            attempted = "; ".join(",".join(h.get("repair_attempts", [])) or "none" for h in r["history"])
            lines.append(f"| {method} / {seed} | {chosen} | {attempted} | {r['repair_cost_used']:.1f} |")
    return "\n".join(lines) + "\n", summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--reference", default="routed")
    args = ap.parse_args()
    text, summary = make_report(load_runs(args.folder), args.reference)
    path = args.folder / "COST_RESULTS.md"
    path.write_text(text, encoding="utf-8")
    (args.folder / "cost_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
