"""Validate every frozen factorial run and report seed-paired outcomes.

Reads saved development results only. Never evaluates confirmation tasks.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import yaml
from make_cost_report import load_runs, make_report
from horizon.rsi.repair_memory import audit_partition, accept_candidate

KEYS = {"nominal": "real_nominal", "perturbed": "real_pert0.15"}
METHODS = ["sim_only", "routed", "gated", "fixed_gvg", "fixed_ggv", "fixed_grouped", "greedy_share", "greedy_ratio", "cost_gated", "cascade", "memory_current", "memory_accum", "memory_no_reject"]
REFERENCES = ["routed", "gated", "fixed_gvg", "fixed_ggv", "fixed_grouped", "greedy_share", "greedy_ratio", "memory_current", "memory_no_reject", "sim_only"]

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(values):
    return {"values": values, "mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None}

def verify(folder, partial=False):
    schedule = json.loads((folder / "schedule.json").read_text())
    status = json.loads((folder / "status.json").read_text())
    identity = lambda t: (t["condition"], t["method"], t["seed"])
    full_expected = Counter(map(identity, schedule["tasks"]))
    require(all(n == 1 for n in full_expected.values()), "Duplicate scheduled runs")
    completed = Counter(map(identity, status["completed"]))
    require(not (completed - full_expected), "Unscheduled or duplicate completed runs")
    require(status["total"] == len(full_expected), "Incorrect full schedule total")
    if partial:
        schedule = {**schedule, "tasks": [t for t in schedule["tasks"] if identity(t) in completed]}
    expected = Counter(map(identity, schedule["tasks"]))
    require(all(n == 1 for n in expected.values()), "Duplicate scheduled runs")
    require(Counter(map(identity, status["completed"])) == expected, "Incomplete or mismatched schedule")
    require(bool(expected), "No completed runs to validate")
    require(all(t["returncode"] == 0 for t in status["completed"]), "Failed runs")
    conditions = list(dict.fromkeys(t["condition"] for t in schedule["tasks"]))
    data = {c: load_runs(folder / c) for c in conditions}
    if partial:
        data = {c: {m: {s: r for s, r in runs.items() if (c,m,s) in expected}
                    for m, runs in methods.items() if any((c,m,s) in expected for s in runs)}
                for c, methods in data.items()}
    actual = Counter((c, m, s) for c, methods in data.items() for m, runs in methods.items() for s in runs)
    require(actual == expected, "Result grid differs from predeclared schedule")
    sources, policies, feedback, evaluations = set(), {}, {}, {}
    totals = Counter()
    for task in schedule["tasks"]:
        c, m, s = identity(task)
        run = data[c][m][s]
        label = f"{c}/{m}/s{s}"
        p, cfg, manifest = run["provenance"], run["config"], run["evaluation_manifest"]
        require(sha(Path(task["config"])) == task["config_sha256"] == p["config_sha256"], f"Config changed: {label}")
        conf = yaml.safe_load(Path(task["config"]).read_text())
        requested = {**conf["common"], **conf["methods"][m], "seed": s, "name": m}
        require(all(cfg[k] == v for k, v in requested.items()), f"Effective config mismatch: {label}")
        require(run["protocol"] == "fixed_policy_probe" and run["n"] == run["train_steps"] == 0, f"Unexpected training: {label}")
        require(manifest["split"] == "development" and not manifest["confirmation_executed"], f"Confirmation was used: {label}")
        payload = {k:v for k,v in manifest.items() if k != "sha256"}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        require(digest == manifest["sha256"], f"Manifest checksum: {label}")
        banks, span = manifest["banks"], manifest["namespace_span"]
        ranges = list(banks.items())
        for i, (name, base) in enumerate(ranges):
            require(all(max(base, b) >= min(base+span, b+span) for _, b in ranges[:i]), f"Overlapping banks: {label}")
        require(banks["reserved_confirmation"] == 40_000_000, f"Wrong reserved bank: {label}")
        require(sha(Path(p["checkpoint"])) == p["checkpoint_sha256"], f"Checkpoint changed: {label}")
        policies.setdefault(s, set()).add(p["checkpoint_sha256"])
        sources.add(json.dumps(p["source_sha256"], sort_keys=True))
        target = [(e["family"], e["H"], e["task_seed"]) for e in manifest["evaluation"]]
        require(len(target) == 18 and len(set(target)) == 18, f"Wrong final test size: {label}")
        require(Counter((family, horizon) for family, horizon, _ in target) == Counter({(family,horizon):2 for family in ("pack","sort","stack") for horizon in (2,6,10)}), f"Wrong task family/horizon grid: {label}")
        evaluations.setdefault(s, set()).add(json.dumps(target))
        for key in (*KEYS.values(), "sim_nominal"):
            ev = run["final"][key]
            require(not ev["errors"], f"Rollout errors: {label}/{key}")
            episodes = ev["episodes"]
            require([(e["family"], e["H"], e["seed"]) for e in episodes] == target, f"Actual tasks mismatch: {label}/{key}")
            require(abs(mean(e["success"] for e in episodes)-ev["overall"]) < 1e-12, f"Wrong score: {label}/{key}")
            require(all(banks["evaluation"] <= e["seed"] < banks["evaluation"]+span for e in episodes), f"Wrong task bank: {label}/{key}")
            for H, metric in ev["by_H"].items():
                selected = [e for e in episodes if e["H"] == int(H)]
                require(len(selected) == metric["n"] and sum(e["success"] for e in selected) == metric["k"], f"Wrong horizon totals: {label}")
        require(len(run["history"]) == 3 and run["curve_evaluation_episodes"] == 27, f"Curve count mismatch: {label}")
        require(len(manifest["curve"]) == 9 and all(banks["development_curve"] <= e["task_seed"] < banks["development_curve"]+span for e in manifest["curve"]), f"Curve manifest mismatch: {label}")
        active = cfg["route"] not in ("none", "perfect")
        require(run["real_episodes_used"] == (36 if active else 0), f"Deployment count mismatch: {label}")
        require(run["audits_used"] == (18 if active else 0), f"Audit count mismatch: {label}")
        require(run["twin_replays_used"] == (36 if active else 0), f"Twin count mismatch: {label}")
        for h in run["history"]:
            tasks = h.get("feedback_tasks", [])
            if active:
                require(len(tasks) == 12 and sum(t["audited"] for t in tasks) == 6 and sum(t["twin"] for t in tasks) == 12, f"Feedback ledger mismatch: {label}")
                require(all(banks["feedback_audit"] <= t["task_seed"] < banks["feedback_audit"]+span for t in tasks), f"Feedback bank mismatch: {label}")
                feedback.setdefault((s,h["round"]), set()).add(json.dumps(tasks, sort_keys=True))
            paid = sum(cfg["repair_costs"][x] for x in h.get("repair_attempts", []))
            require(abs(paid-h.get("repair_cost",0)) < 1e-9, f"Attempt cost mismatch: {label}")
            require(set(h.get("repair_committed",[])) <= set(h.get("repair_attempts",[])), f"Uncharged state change: {label}")
        require(abs(sum(h.get("repair_cost",0) for h in run["history"])-run["repair_cost_used"]) < 1e-9, f"Cumulative cost mismatch: {label}")
        require(run["repair_cost_used"] <= cfg["repair_budget"] + 1e-9, f"Exceeded repair budget: {label}")
        audit_keys = set()
        groups_seen = {}
        for i, h in enumerate(run["history"]):
            before = h["belief"]
            after = run["history"][i+1]["belief"] if i+1 < len(run["history"]) else run["final_belief"]
            changed = {op for op, field in (("geometry", "recon"), ("verifier", "judge_scale"), ("graph", "grounding")) if before[field] != after[field]}
            require(changed == set(h.get("repair_committed", [])), f"State/commit mismatch: {label}/r{i+1}")
            current_audits = {(t["family"], t["H"], t["task_seed"]) for t in h.get("feedback_tasks", []) if t["audited"]}
            audit_keys.update(current_audits)
            if "memory_audit" in h:
                mem = h["memory_audit"]
                require(set(mem["fit_ids"]).isdisjoint(mem["check_ids"]), f"Fit/check leakage: {label}")
                require(all(len(mem[group+"_ids"]) == len(set(mem[group+"_ids"])) for group in ("fit","check")), f"Duplicate audit identities: {label}")
                permitted = audit_keys if cfg["audit_memory"] else current_audits
                require(mem["stored"] == len(permitted), f"Memory episode accounting: {label}")
                for group in ("fit", "check"):
                    for audit_id in mem[group+"_ids"]:
                        family, horizon, seed, *_ = audit_id.split(":")
                        require((family, int(horizon), int(seed)) in permitted, f"Unaudited memory: {label}/{audit_id}")
                        require(family not in mem["unresolved"], f"Unresolved geometry entered calibration: {label}")
                        require(audit_partition(audit_id) == group, f"Unstable split: {label}/{audit_id}")
                        require(groups_seen.get(audit_id, group) == group, f"Cross-round leakage: {label}/{audit_id}")
                        groups_seen[audit_id] = group
                report = h.get("verifier_report")
                if report is not None:
                    supported = accept_candidate(report["before"], report["after"], cfg["validation_margin"])
                    require(supported == report["supported_candidate"], f"Wrong acceptance support: {label}")
                    require(report["accepted"] == (supported if cfg["reject_verifier_candidate"] else True), f"Wrong acceptance decision: {label}")
                    require("verifier" in h["repair_attempts"], f"Unpaid verifier fit: {label}")
                    require(report["changed"] == (report["accepted"] and report["chosen"] != before["judge_scale"]), f"Verifier change ledger mismatch: {label}")
                    require(after["judge_scale"] == (report["chosen"] if report["accepted"] else before["judge_scale"]), f"Verifier acceptance/state mismatch: {label}")
        for k in ("audit_keyframes", "audit_predicate_labels", "twin_replays"):
            require(sum(h.get(k,0) for h in run["history"]) == run[k+"_used"], f"Resource mismatch: {label}/{k}")
        for k in ("real_episodes_used", "audits_used", "twin_replays_used", "audit_keyframes_used", "audit_predicate_labels_used", "curve_evaluation_episodes"):
            totals[k] += run[k]
        totals["final_rollouts"] += 54
    require(len(sources) == 1, "Mixed source versions")
    require(all(len(v)==1 for v in policies.values()), "Mixed checkpoints per seed")
    require(all(len(v)==1 for v in evaluations.values()), "Unpaired final evaluation tasks")
    require(len(set().union(*evaluations.values())) == 1, "Evaluation tasks differ across checkpoint seeds")
    require(all(len(v)==1 for v in feedback.values()), "Unpaired feedback across methods or conditions")
    for filename, digest in json.loads(next(iter(sources))).items():
        require(sha(Path(filename)) == digest, f"Source modified: {filename}")
    validation = {"passed": True, "partial": partial, "runs": len(expected), "conditions": conditions, "checkpoint_seeds": sorted(policies), "source_versions": len(sources), "unique_final_task_identities": len(target), "evaluation_tasks_shared_across_checkpoints": True, "confirmation_executed": False, "totals": dict(totals), "checks": ["completed subset of predeclared grid" if partial else "complete predeclared grid", "config/source/checkpoint hashes", "disjoint task namespaces", "actual final task manifests and score arithmetic", "paired feedback tasks and audit flags", "attempt costs and committed repairs", "resource ledgers", "immutable audit partitions and repair state transitions"], "limits": ["Curve task identities are specified by the saved manifest and source, not individually logged rollout outcomes", "Distinct task seed banks do not establish an independent simulator or real-world distribution"]}
    return data, schedule, validation


def summarize(data, schedule):
    summary = {}
    lines = ["# v7: eight error combinations and feasible repair controls", "",
        "Fixed policies, three checkpoint seeds. Same-MuJoCo deployment proxy; privileged synthetic audits.",
        "Fresh development banks, 6 audit episodes/round, H=2/6/10, 18 tasks per final condition.",
        "The same 18 final task identities are shared across all three checkpoints; 54 rollouts do not mean 54 independent tasks.",
        "Mean +/- sample SD; no confidence interval or significance claim. Do not compare scores directly with v6.",
        "Error bits: geometry / verifier / graph. Costs are abstract units, not measured human minutes.",
        "No new policy training. Confirmation was not executed.", ""]
    for condition, runs in data.items():
        summary[condition] = {"methods": {}, "paired": {}}
        lines += [f"## {condition}", "", "| Method | Nominal % | Perturbed % | Paid cost |", "|---|---:|---:|---:|"]
        for method in METHODS:
            if method not in runs:
                continue
            rs = runs[method]
            seeds = sorted(rs)
            result = {"seeds": seeds, "cost": stats([rs[s]["repair_cost_used"] for s in seeds])}
            for mode, key in KEYS.items():
                result[mode] = stats([100*rs[s]["final"][key]["overall"] for s in seeds])
            result["by_H"] = {mode: {h: stats([100*rs[s]["final"][key]["by_H"][h]["rate"][0] for s in seeds])
                                    for h in rs[seeds[0]]["final"][key]["by_H"]} for mode,key in KEYS.items()}
            result["curve"] = [stats([100*rs[s]["history"][i]["real_test"] for s in seeds]) for i in range(3)]
            summary[condition]["methods"][method] = result
            fmt = lambda x: f"{x['mean']:.1f} +/- {x['sample_sd']:.1f}"
            lines.append(f"| {method} | {fmt(result['nominal'])} | {fmt(result['perturbed'])} | {result['cost']['mean']:.2f} |")
        lines += ["", "### Paired seed differences (percentage points)", "", "| Method - reference | Nominal mean [s0,s1,s2] | Perturbed mean [s0,s1,s2] |", "|---|---|---|"]
        pairs = [(m, ref) for m in runs for ref in REFERENCES if ref in runs and m != ref]
        for method, ref in pairs:
            seeds = sorted(runs[method].keys() & runs[ref].keys())
            paired = {mode: stats([100*(runs[method][s]["final"][key]["overall"]-runs[ref][s]["final"][key]["overall"]) for s in seeds]) for mode,key in KEYS.items()}
            summary[condition]["paired"][method+" - "+ref] = paired
            if (method,ref) in [("memory_accum","memory_current"),("memory_accum","memory_no_reject"),("memory_accum","gated"),("memory_accum","fixed_grouped"),("fixed_grouped","fixed_ggv"),("greedy_share","routed"),("greedy_ratio","routed"),("cost_gated","greedy_share"),("cost_gated","greedy_ratio")]:
                fmt = lambda x: f"{x['mean']:+.1f} ["+", ".join(f"{v:+.1f}" for v in x['values'])+"]"
                lines.append(f"| {method} - {ref} | {fmt(paired['nominal'])} | {fmt(paired['perturbed'])} |")
        lines += ["", f"[All methods, horizons and resource counts]({condition}/COST_RESULTS.md)", ""]
    lines += ["## Interpretation limits", "", "All nine conditions are development experiments with three reused checkpoints. Results are paired across methods, conditions and evaluation modes. Do not treat these cells or episodes as independent policy seeds.", "", "Current and accumulated audit controllers share all rules except the memory window. Their checks are adaptive across rounds, not independent confirmation. No-reject disables verifier acceptance only; geometry and eligibility rules remain. The old cascade differs in more than accumulation, so its difference cannot isolate memory.", "", "Greedy, exact subset and gated can execute multiple repairs per round. Fixed grouped attempts geometry+graph in round 1 and verifier in round 2; one-operator fixed schedules act in three rounds. Attempt costs include rejected and unchanged fits. Audit episode budgets are matched; annotation label counts can differ with trajectory length.", "", "No noisy labels, independent simulator, image policy or new skill is evaluated in this matrix. Only pick_place-based families are covered. Confirmation remains reserved."]
    return summary, "\n".join(lines)+"\n"


def mechanism_summary(data):
    """Descriptive ledgers; a changed parameter is not automatically a harmful repair."""
    summary = {}
    lines = ["# Repair mechanisms and costs", "", "Counts sum over three reused checkpoint seeds. They are descriptive, not independent replications.",
             "A rejected candidate equal to the current parameter cannot establish prevention of a harmful change.",
             "Clean-component edits are flagged for inspection; their harm requires paired outcome evidence.", ""]
    for condition, methods in data.items():
        summary[condition] = {}
        bits = condition.split("_")[1]
        clean_ops = {op for bit, op in zip(bits, ("geometry","verifier","graph")) if bit == "0"}
        lines += [f"## {condition}", "", "| Method | Attempts / commits | Verifier fits: accepted / rejected | Rejected changed / no-op | Clean-component commits | Unresolved geometry rounds |", "|---|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            if method not in methods:
                continue
            counts = Counter()
            per_seed = {}
            for seed, run in sorted(methods[method].items()):
                ledger = Counter()
                for h in run["history"]:
                    attempts, commits = h.get("repair_attempts",[]), h.get("repair_committed",[])
                    ledger["attempts"] += len(attempts)
                    ledger["commits"] += len(commits)
                    ledger["paid_no_commit"] += len(set(attempts)-set(commits))
                    ledger["clean_component_commits"] += len(set(commits)&clean_ops)
                    for op in ("geometry","verifier","graph"):
                        ledger[op+"_attempts"] += op in attempts
                        ledger[op+"_commits"] += op in commits
                    mem = h.get("memory_audit")
                    if mem is not None:
                        ledger["memory_rounds"] += 1
                        ledger["eligible_rounds"] += bool(mem["eligible"])
                        ledger["unresolved_geometry_rounds"] += bool(mem["unresolved"])
                        ledger["eligible_unpaid_rounds"] += mem["eligible"] and "verifier" not in attempts
                    report = h.get("verifier_report")
                    if report is not None and "accepted" in report:
                        ledger["verifier_fits"] += 1
                        accepted = bool(report["accepted"])
                        changed_candidate = report["chosen"] != h["belief"]["judge_scale"]
                        ledger["accepted"] += accepted
                        ledger["rejected"] += not accepted
                        ledger["rejected_changed"] += not accepted and changed_candidate
                        ledger["rejected_noop"] += not accepted and not changed_candidate
                        ledger["accepted_unsupported"] += accepted and not report["supported_candidate"]
                        ledger["accepted_unsupported_changed"] += accepted and not report["supported_candidate"] and changed_candidate
                counts.update(ledger)
                per_seed[seed] = {**dict(ledger), "paid_cost": run["repair_cost_used"]}
            summary[condition][method] = {"totals": dict(counts), "per_seed": per_seed}
            lines.append(f"| {method} | {counts['attempts']} / {counts['commits']} | {counts['accepted']} / {counts['rejected']} | {counts['rejected_changed']} / {counts['rejected_noop']} | {counts['clean_component_commits']} | {counts['unresolved_geometry_rounds']} |")
    return summary, "\n".join(lines)+"\n"


def macro_summary(summary):
    """Equal weight per error cell, then paired across the same three policies."""
    conditions = [f"factorial_{i:03b}" for i in range(8)]
    common = [m for m in METHODS if all(m in summary[c]["methods"] for c in conditions)]
    macro = {m: {mode: stats([mean(summary[c]["methods"][m][mode]["values"][s] for c in conditions) for s in range(3)]) for mode in KEYS} for m in common}
    lines = ["# Equal-weight descriptive average over all eight error combinations", "", "This weighting is a development summary, not an estimated deployment error distribution. Cells share the same checkpoint seeds and task banks.", "", "| Method | Nominal mean +/- seed SD % | Perturbed mean +/- seed SD % | Average repair cost |", "|---|---:|---:|---:|"]
    for m in common:
        macro[m]["cost"] = mean(summary[c]["methods"][m]["cost"]["mean"] for c in conditions)
        n, p = macro[m]["nominal"], macro[m]["perturbed"]
        lines.append(f"| {m} | {n['mean']:.2f} +/- {n['sample_sd']:.2f} | {p['mean']:.2f} +/- {p['sample_sd']:.2f} | {macro[m]['cost']:.3f} |")
    pairs = {}
    lines += ["", "## Paired comparison of accumulated audits with common controls", "", "Counts are wins/ties/losses across 24 paired cell-seed entries (correlated).", "", "| Reference | Nominal delta pp [three seed macro deltas]; W/T/L | Perturbed delta pp [three seed macro deltas]; W/T/L |", "|---|---|---|"]
    for ref in common:
        if ref == "memory_accum":
            continue
        pairs[ref] = {}
        formatted = []
        for mode in KEYS:
            deltas = [a-b for a,b in zip(macro["memory_accum"][mode]["values"],macro[ref][mode]["values"])]
            cells = [summary[c]["methods"]["memory_accum"][mode]["values"][s]-summary[c]["methods"][ref][mode]["values"][s] for c in conditions for s in range(3)]
            wtl = [sum(v>1e-9 for v in cells),sum(abs(v)<=1e-9 for v in cells),sum(v< -1e-9 for v in cells)]
            pairs[ref][mode] = {**stats(deltas), "cell_seed_wins_ties_losses": wtl}
            formatted.append(f"{mean(deltas):+.2f} ["+", ".join(f"{d:+.2f}" for d in deltas)+f"]; {wtl[0]}/{wtl[1]}/{wtl[2]}")
        lines.append(f"| {ref} | {formatted[0]} | {formatted[1]} |")
    return {"methods":macro,"memory_accum_paired":pairs}, "\n".join(lines)+"\n"


def outcome_diagnostics(data):
    results = {}
    for condition, methods in data.items():
        results[condition] = {}
        for method, runs in methods.items():
            results[condition][method] = {}
            for mode, key in KEYS.items():
                per_seed = {}
                for seed, run in sorted(runs.items()):
                    episodes = run["final"][key]["episodes"]
                    per_seed[seed] = {
                        "n": len(episodes),
                        "successes": sum(bool(e["success"]) for e in episodes),
                        "declared_done": sum(bool(e["judged_done"]) for e in episodes),
                        "false_done": sum(bool(e["judged_done"]) and not e["success"] for e in episodes),
                        "missed_done": sum(not e["judged_done"] and bool(e["success"]) for e in episodes),
                        "by_family": {family: {"n": sum(e["family"] == family for e in episodes), "successes": sum(e["family"] == family and bool(e["success"]) for e in episodes)} for family in ("pack", "sort", "stack")},
                        "execution_failure_types": dict(Counter(e["oracle"]["ftype"] for e in episodes if not e["success"] and e.get("oracle"))),
                    }
                results[condition][method][mode] = per_seed
    return results



def pairwise_diagnostics(data):
    """Compare matched final episodes, costs, beliefs and attempted repair schedules."""
    pairs = [('greedy_share','cost_gated'), ('greedy_ratio','cost_gated'),
             ('gated','greedy_share'), ('fixed_grouped','fixed_gvg'),
             ('fixed_grouped','fixed_ggv'), ('memory_accum','memory_current'),
             ('memory_accum','memory_no_reject'), ('memory_accum','fixed_grouped')]
    results = {}
    lines = ['# Paired endpoint and execution diagnostics', '',
             'A/B denote the first and second method. Episode wins/losses are paired and correlated, not independent training replicates.',
             'Matching mean scores do not imply matching beliefs or repair execution. Healthy changes alone do not establish harm.', '']
    for condition, methods in data.items():
        results[condition] = {}
        lines += [f'## {condition}', '',
                  '| A / B | Same final belief seeds | Same attempted schedule seeds | Nominal A-only / B-only successes | Perturbed A-only / B-only successes |',
                  '|---|---:|---:|---:|---:|']
        for a,b in pairs:
            if a not in methods or b not in methods:
                continue
            values = {}
            counts = Counter()
            for seed in sorted(set(methods[a]) & set(methods[b])):
                ra,rb = methods[a][seed],methods[b][seed]
                attempted = lambda r: [h.get('repair_attempts',h.get('chosen',[])) for h in r['history']]
                item = {'same_final_belief':ra['final_belief']==rb['final_belief'],
                        'same_attempted_schedule':attempted(ra)==attempted(rb),
                        'cost_difference':ra['repair_cost_used']-rb['repair_cost_used']}
                counts['same_final_belief'] += item['same_final_belief']
                counts['same_attempted_schedule'] += item['same_attempted_schedule']
                for mode,key in KEYS.items():
                    ea,eb = ra['final'][key]['episodes'],rb['final'][key]['episodes']
                    ident = lambda e:(e['family'],e['H'],e['seed'])
                    require(list(map(ident,ea)) == list(map(ident,eb)),f'Pair order mismatch: {condition}/{a}/{b}/{seed}')
                    aw = sum(bool(x['success']) and not y['success'] for x,y in zip(ea,eb))
                    bw = sum(bool(y['success']) and not x['success'] for x,y in zip(ea,eb))
                    item[mode] = {'n':len(ea),'a_only_success':aw,'b_only_success':bw,
                                  'same_success_vector':all(x['success']==y['success'] for x,y in zip(ea,eb)),
                                  'same_declared_done_vector':all(x['judged_done']==y['judged_done'] for x,y in zip(ea,eb)),
                                  'delta_pp':100*(aw-bw)/len(ea)}
                    counts[mode+'_a_only'] += aw
                    counts[mode+'_b_only'] += bw
                values[seed] = item
            results[condition][a+'__vs__'+b] = {'a':a,'b':b,'per_seed':values,'totals':dict(counts)}
            lines.append(f"| {a} / {b} | {counts['same_final_belief']} | {counts['same_attempted_schedule']} | {counts['nominal_a_only']} / {counts['nominal_b_only']} | {counts['perturbed_a_only']} / {counts['perturbed_b_only']} |")
    return results, '\n'.join(lines)+'\n'


def geometry_gate_summary(data):
    """Expose small-batch evidence exclusions without changing the frozen gate."""
    from inspect import signature
    from horizon.rsi.simreal import flagged_families
    params = signature(flagged_families).parameters
    min_family = params['min_family'].default
    gap_threshold = params['gap_threshold'].default
    results = {'min_family':min_family,'gap_threshold':gap_threshold,'conditions':{}}
    lines = ['# Geometry evidence gates under the reduced feedback budget', '',
             f'The frozen family-level twin gate requires n >= {min_family} and progress gap > {gap_threshold}. Each round has only 12 feedback episodes split across three task families.',
             'The table describes accumulated-audit runs. A blocked geometric candidate is not proof that it would be correct; it identifies an observable disagreement between estimators. No threshold was changed after seeing these results.', '',
             '| Condition | Family-round gaps above threshold but too few samples | Rounds with robust geometry candidate but no twin support | Verifier fits in those rounds | Geometry commits |',
             '|---|---:|---:|---:|---:|']
    for condition, methods in data.items():
        results['conditions'][condition] = {}
        for method, runs in methods.items():
            counts = Counter()
            per_seed = {}
            for seed, run in sorted(runs.items()):
                rows = []
                for h in run['history']:
                    twin = h.get('twin',{})
                    small = {f:v for f,v in twin.items() if v['gap'] > gap_threshold and v['n'] < min_family}
                    candidate = any(v.get('changed',False) for v in h.get('geometry_report',{}).values() if isinstance(v,dict))
                    blocked = candidate and h.get('geometry_supported') is False
                    fitted = 'verifier' in h.get('repair_attempts',[])
                    counts['low_count_positive_gap_family_rounds'] += len(small)
                    counts['robust_candidate_without_twin_support_rounds'] += blocked
                    counts['verifier_fits_with_blocked_geometry'] += blocked and fitted
                    counts['geometry_commits'] += 'geometry' in h.get('repair_committed',[])
                    rows.append({'round':h['round'],'family_evidence':twin,'positive_gaps_below_min_count':small,
                                 'robust_candidate_without_twin_support':blocked,
                                 'verifier_fitted':fitted,'unresolved_for_calibration':h.get('memory_audit',{}).get('unresolved')})
                per_seed[seed] = rows
            results['conditions'][condition][method] = {'totals':dict(counts),'per_seed':per_seed}
            if method == 'memory_accum':
                lines.append(f"| {condition} | {counts['low_count_positive_gap_family_rounds']} | {counts['robust_candidate_without_twin_support_rounds']} | {counts['verifier_fits_with_blocked_geometry']} | {counts['geometry_commits']} |")
    lines += ['', 'A follow-up should compare a budget-compatible evidence design against the frozen gate under matched feedback and strong fixed/greedy controls. Lowering this threshold is a new development experiment, not a correction to these saved scores.']
    return results, '\n'.join(lines)+'\n'

def cost_progress_summary(data):
    """Align pre-repair curve scores with costs already paid before that evaluation."""
    result = {}
    lines = ['# Development success versus repair cost', '',
             "Each point is evaluated BEFORE that round's repair on the same nine curve tasks. The x coordinate excludes the current round's later charge. Values are three-checkpoint means.",
             'These are observed trajectories under budget 3, not separate budget sweeps. Do not append the 18-task final evaluation to these nine-task curves.', '',
             '| Condition / method | Round 1: cost / success % | Round 2: cost / success % | Round 3: cost / success % |',
             '|---|---:|---:|---:|']
    for condition, methods in data.items():
        result[condition] = {}
        for method in METHODS:
            if method not in methods:
                continue
            by_seed = {}
            for seed, run in sorted(methods[method].items()):
                paid = 0.0
                points = []
                for h in run['history']:
                    points.append({'round': h['round'], 'cost_before_evaluation': paid,
                                   'nominal_curve_success_pct': 100*h['real_test'],
                                   'evaluation_episodes': h['real_test_episode_count']})
                    paid += h.get('repair_cost', 0.0)
                require(abs(paid-run['repair_cost_used']) < 1e-9, 'Curve cost mismatch')
                by_seed[seed] = points
            points = [{'round':i+1,
                       'cost':stats([p[i]['cost_before_evaluation'] for p in by_seed.values()]),
                       'success':stats([p[i]['nominal_curve_success_pct'] for p in by_seed.values()])}
                      for i in range(3)]
            result[condition][method] = {'per_seed':by_seed,'points':points}
            cells = [f"{p['cost']['mean']:.2f} / {p['success']['mean']:.2f}" for p in points]
            lines.append('| '+condition+' / '+method+' | '+' | '.join(cells)+' |')
    return result, '\n'.join(lines)+'\n'


def plot_cost_progress(progress, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,5.7),layout='constrained',sharey=True)
    selected = ['gated','fixed_ggv','fixed_grouped','greedy_share','memory_current','memory_accum']
    styles = ['-','--','-.',':','--','-']
    markers = ['o','s','^','v','D','P']
    conditions = ['factorial_000','factorial_111','factorial_111_cost']
    for ax,c in zip(axes,conditions):
        for m,style,marker in zip(selected,styles,markers):
            if m not in progress[c]:
                continue
            pts = progress[c][m]['points']
            ax.plot([p['cost']['mean'] for p in pts],[p['success']['mean'] for p in pts],
                    linestyle=style,marker=marker,label=m,alpha=.8,markersize=7)
        ax.set_title(c.replace('factorial_',''))
        ax.set_xlabel('Repair cost paid BEFORE evaluation')
        ax.set_xlim(-.1,3.1);ax.set_xticks([0,1,2,3]);ax.set_ylim(-3,103);ax.grid(alpha=.2)
    axes[0].set_ylabel('Nominal development success (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=10)
    fig.suptitle('Three pre-repair evaluations on nine shared curve tasks | seed means\nObserved cost trajectories, not budget sweeps; overlaps retained; final test excluded',fontsize=12)
    for ext in ('png','pdf'):
        fig.savefig(folder/f'factorial_cost_progress.{ext}',dpi=180)
    plt.close(fig)


def plot(summary, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    conditions = [f"factorial_{i:03b}" for i in range(8)]
    common = [m for m in METHODS if all(m in summary[c]["methods"] for c in conditions)]
    fig, axes = plt.subplots(1,2,figsize=(14,7),layout="constrained")
    for ax, mode in zip(axes, KEYS):
        values = np.array([[summary[c]["methods"][m][mode]["mean"] for c in conditions] for m in common])
        im=ax.imshow(values,vmin=0,vmax=100,cmap="viridis",aspect="auto")
        ax.set_xticks(range(8),[c.split("_")[-1] for c in conditions]); ax.set_yticks(range(len(common)),common)
        ax.set_xlabel("Injected errors: geometry / verifier / graph")
        ax.set_title(mode.capitalize()+" success (%)")
        for i in range(len(common)):
            for j in range(8):
                ax.text(j,i,f"{values[i,j]:.1f}",ha="center",va="center",color="black" if values[i,j]>60 else "white",fontsize=9)
    fig.colorbar(im,ax=axes,label="Three checkpoint seed mean (%)",shrink=.75)
    fig.suptitle("Eight error combinations | development proxy | 6 audits/round\nH=2,6,10; 18 final tasks/seed/mode; no confirmation",fontsize=13)
    for ext in ("png","pdf"): fig.savefig(folder/f"factorial_success.{ext}",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,7),layout="constrained",sharey=True)
    selected=["gated","fixed_gvg","fixed_ggv","fixed_grouped","greedy_share","greedy_ratio","memory_current","memory_accum"]
    for ax,c in zip(axes.flat,conditions):
        for method in selected:
            cells=summary[c]["methods"][method]["by_H"]["perturbed"]
            hs=sorted(cells,key=int)
            ax.plot(list(map(int,hs)),[cells[h]["mean"] for h in hs],marker="o",label=method,alpha=.75)
        ax.set_title(c.split("_")[-1]);ax.set_xticks([2,6,10]);ax.set_ylim(-3,103);ax.grid(alpha=.2)
        ax.set_xlabel("H")
    axes[0,0].set_ylabel("Perturbed success (%)");axes[1,0].set_ylabel("Perturbed success (%)")
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="outside lower center",ncol=4,fontsize=9)
    fig.suptitle("All evaluated horizons | seed means; overlapping curves retained")
    for ext in ("png","pdf"):fig.savefig(folder/f"factorial_horizons.{ext}",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--validate-completed", action="store_true", help="Validate completed runs only; do not write final outputs")
    parser.add_argument("--preview", action="store_true", help="Print incomplete progress only; never write validated output")
    args=parser.parse_args()
    if args.validate_completed:
        data, _, validation = verify(args.folder, partial=True)
        pairs, _ = pairwise_diagnostics(data)
        geometry_gate_summary(data)
        cost_progress_summary(data)
        validation["paired_diagnostic_comparisons"] = sum(len(v) for v in pairs.values())
        print(json.dumps(validation, indent=2))
        return
    if args.preview:
        status=json.loads((args.folder/"status.json").read_text())
        print(json.dumps({"complete":len(status["completed"]),"total":status["total"],"failed":sum(t["returncode"]!=0 for t in status["completed"]),"provisional":True}))
        conditions=list(dict.fromkeys(t["condition"] for t in status["completed"]))
        for c in conditions:
            runs=load_runs(args.folder/c)
            for m, rs in runs.items():
                if len(rs)==3:
                    print(c,m,round(100*mean(r["final"]["real_nominal"]["overall"] for r in rs.values()),2),round(100*mean(r["final"]["real_pert0.15"]["overall"] for r in rs.values()),2))
        return
    data,schedule,validation=verify(args.folder)
    require(len(schedule["tasks"])==258,"Wrong frozen matrix size")
    require(set(data)=={*(f"factorial_{i:03b}" for i in range(8)),"factorial_111_cost"},"Missing factorial conditions")
    summary,text=summarize(data,schedule)
    for c,runs in data.items():
        report,compact=make_report(runs,"routed" if "routed" in runs else "gated")
        (args.folder/c/"COST_RESULTS.md").write_text(report,encoding="utf-8")
        (args.folder/c/"cost_summary.json").write_text(json.dumps(compact,indent=2),encoding="utf-8")
    diagnostics={c:{m:{s:{"final_belief":r["final_belief"],"cost":r["repair_cost_used"],"rounds":[{k:h[k] for k in ("round","shares","active","repair_attempts","repair_committed","repair_cost","geometry_report","memory_audit","verifier_report","calibration_gate") if k in h} for h in r["history"]]} for s,r in rs.items()} for m,rs in runs.items()} for c,runs in data.items()}
    (args.folder/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    (args.folder/"factorial_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (args.folder/"factorial_diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    (args.folder/"FACTORIAL_RESULTS.md").write_text(text,encoding="utf-8")
    outcomes = outcome_diagnostics(data)
    (args.folder/"factorial_outcomes.json").write_text(json.dumps(outcomes,indent=2),encoding="utf-8")
    mechanisms, mechanism_text = mechanism_summary(data)
    macro, macro_text = macro_summary(summary)
    for name, value in (("factorial_mechanisms", mechanisms), ("factorial_macro", macro)):
        (args.folder/(name+".json")).write_text(json.dumps(value,indent=2),encoding="utf-8")
    (args.folder/"MECHANISMS.md").write_text(mechanism_text,encoding="utf-8")
    (args.folder/"MACRO_RESULTS.md").write_text(macro_text,encoding="utf-8")
    paired, paired_text = pairwise_diagnostics(data)
    (args.folder/"factorial_pairwise.json").write_text(json.dumps(paired,indent=2),encoding="utf-8")
    (args.folder/"PAIRED_DIAGNOSTICS.md").write_text(paired_text,encoding="utf-8")
    gates, gate_text = geometry_gate_summary(data)
    (args.folder/"factorial_geometry_gates.json").write_text(json.dumps(gates,indent=2),encoding="utf-8")
    (args.folder/"GEOMETRY_GATES.md").write_text(gate_text,encoding="utf-8")
    progress, progress_text = cost_progress_summary(data)
    (args.folder/"factorial_cost_progress.json").write_text(json.dumps(progress,indent=2),encoding="utf-8")
    (args.folder/"COST_PROGRESS.md").write_text(progress_text,encoding="utf-8")
    plot(summary,args.folder)
    plot_cost_progress(progress,args.folder)
    print(json.dumps(validation,indent=2))
    print("Complete factorial screen validated and reported; confirmation not executed.")

if __name__=="__main__":
    main()
