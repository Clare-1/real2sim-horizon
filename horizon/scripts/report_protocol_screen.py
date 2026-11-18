"""Validate and summarize the complete, predeclared development screen.

Run from horizon/: python scripts/report_protocol_screen.py results/protocol_screen_20261025
Only saved results are read; this never executes confirmation tasks or rollouts.
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

KEYS = {"nominal": "real_nominal", "perturbed": "real_pert0.15"}
LABELS = {"triple_unit": "Three defects / unit costs", "triple_cost": "Three defects / costs 3:1:2", "healthy": "No injected defects / unit costs"}
COLORS = {"sim_only":"#7d8790", "perfect":"#17835c", "routed":"#2563ad", "gated":"#8c59ae", "cascade":"#df602f", "fixed_gvg":"#bd8c20", "fixed_ggv":"#26a7ad", "repair_all":"#ce477c", "random_valid":"#697728", "cost_gated":"#8654bd"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {"values": values, "mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None}


def verify(folder):
    schedule = json.loads((folder / "schedule.json").read_text())
    status = json.loads((folder / "status.json").read_text())
    identity = lambda t: (t["condition"], t["method"], t["seed"])
    expected = Counter(map(identity, schedule["tasks"]))
    require(all(n == 1 for n in expected.values()), "Duplicate scheduled runs")
    require(Counter(map(identity, status["completed"])) == expected, "Incomplete or mismatched schedule")
    require(status["total"] == len(expected), "Incorrect total")
    require(all(t["returncode"] == 0 for t in status["completed"]), "Failed runs")
    conditions = list(dict.fromkeys(t["condition"] for t in schedule["tasks"]))
    data = {c: load_runs(folder / c) for c in conditions}
    actual = Counter((c, m, s) for c, methods in data.items() for m, runs in methods.items() for s in runs)
    require(actual == expected, "Result grid differs from predeclared schedule")
    sources, policies, feedback = set(), {}, {}
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
        require(len(target) == 60 and len(set(target)) == 60, f"Wrong final test size: {label}")
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
        require(len(run["history"]) == 3 and run["curve_evaluation_episodes"] == 45, f"Curve count mismatch: {label}")
        require(len(manifest["curve"]) == 15 and all(banks["development_curve"] <= e["task_seed"] < banks["development_curve"]+span for e in manifest["curve"]), f"Curve manifest mismatch: {label}")
        active = cfg["route"] not in ("none", "perfect")
        require(run["real_episodes_used"] == (90 if active else 0), f"Deployment count mismatch: {label}")
        require(run["audits_used"] == (45 if active else 0), f"Audit count mismatch: {label}")
        require(run["twin_replays_used"] == (72 if active else 0), f"Twin count mismatch: {label}")
        for h in run["history"]:
            tasks = h.get("feedback_tasks", [])
            if active:
                require(len(tasks) == 30 and sum(t["audited"] for t in tasks) == 15 and sum(t["twin"] for t in tasks) == 24, f"Feedback ledger mismatch: {label}")
                require(all(banks["feedback_audit"] <= t["task_seed"] < banks["feedback_audit"]+span for t in tasks), f"Feedback bank mismatch: {label}")
                feedback.setdefault((s,h["round"]), set()).add(json.dumps(tasks, sort_keys=True))
            paid = sum(cfg["repair_costs"][x] for x in h.get("repair_attempts", []))
            require(abs(paid-h.get("repair_cost",0)) < 1e-9, f"Attempt cost mismatch: {label}")
            require(set(h.get("repair_committed",[])) <= set(h.get("repair_attempts",[])), f"Uncharged state change: {label}")
        for k in ("audit_keyframes", "audit_predicate_labels", "twin_replays"):
            require(sum(h.get(k,0) for h in run["history"]) == run[k+"_used"], f"Resource mismatch: {label}/{k}")
        for k in ("real_episodes_used", "audits_used", "twin_replays_used", "audit_keyframes_used", "audit_predicate_labels_used", "curve_evaluation_episodes"):
            totals[k] += run[k]
        totals["final_rollouts"] += 180
    for seed in policies:
        healthy = data["healthy"]["sim_only"][seed]
        oracle = data["triple_unit"]["perfect"][seed]
        for key in KEYS.values():
            require(healthy["final"][key]["episodes"] == oracle["final"][key]["episodes"], f"Equivalent healthy/oracle controls differ: seed {seed}, {key}")
    require(len(sources) == 1, "Mixed source versions")
    require(all(len(v)==1 for v in policies.values()), "Mixed checkpoints per seed")
    require(all(len(v)==1 for v in feedback.values()), "Unpaired feedback across methods or conditions")
    for filename, digest in json.loads(next(iter(sources))).items():
        require(sha(Path(filename)) == digest, f"Source modified: {filename}")
    validation = {"passed": True, "runs": len(expected), "conditions": conditions, "checkpoint_seeds": sorted(policies), "source_versions": len(sources), "confirmation_executed": False, "totals": dict(totals), "checks": ["complete predeclared grid", "config/source/checkpoint hashes", "disjoint task namespaces", "actual final task manifests and score arithmetic", "paired feedback tasks and audit flags", "attempt costs and committed repairs", "resource ledgers", "equivalent healthy/oracle controls"], "limits": ["Curve task identities are specified by the saved manifest and source, not individually logged rollout outcomes", "Distinct task seed banks do not establish an independent simulator or real-world distribution"]}
    return data, schedule, validation


def summarize(data, schedule):
    summary = {}
    lines = ["# Complete development screen", "", "Three existing DAgger checkpoint seeds; fixed policies; same-MuJoCo proxy.", "All numbers are descriptive. Mean +/- sample SD is not a confidence interval.", "Reserved confirmation tasks were not run.", ""]
    for condition, runs in data.items():
        methods = list(dict.fromkeys(t["method"] for t in schedule["tasks"] if t["condition"] == condition))
        summary[condition] = {"methods": {}, "paired": {}}
        lines += [f"## {LABELS[condition]}", "", "| Method | Nominal % (mean ± SD) | Perturbed % (mean ± SD) | Mean paid cost | Mean state changes |", "|---|---:|---:|---:|---:|"]
        for m in methods:
            seeds = sorted(runs[m])
            ss = {name: stats([100*runs[m][s]["final"][key]["overall"] for s in seeds]) for name,key in KEYS.items()}
            ss["seeds"] = seeds
            ss["subgoal_fraction_diagnostic"] = {name: stats([100*mean(e["n_sat"]/e["H"] for e in runs[m][s]["final"][key]["episodes"]) for s in seeds]) for name,key in KEYS.items()}
            ss["cost"] = stats([runs[m][s]["repair_cost_used"] for s in seeds])
            ss["state_changes"] = stats([sum(len(h.get("repair_committed",[])) for h in runs[m][s]["history"]) for s in seeds])
            summary[condition]["methods"][m] = ss
            lines.append(f"| {m} | {ss['nominal']['mean']:.1f} ± {ss['nominal']['sample_sd']:.1f} | {ss['perturbed']['mean']:.1f} ± {ss['perturbed']['sample_sd']:.1f} | {ss['cost']['mean']:.2f} | {ss['state_changes']['mean']:.2f} |")
        lines += ["", "Auxiliary pre-round development curves (%; 15 tasks/seed, separate from the final 60-task bank):", "", "| Method | Before round 1 | Before round 2 | Before round 3 |", "|---|---:|---:|---:|"]
        for m in methods:
            curve = [100*mean(r["history"][i]["real_test"] for r in runs[m].values()) for i in range(3)]
            summary[condition]["methods"][m]["pre_round_curve_mean"] = curve
            lines.append("| " + m + " | " + " | ".join(f"{v:.1f}" for v in curve) + " |")
        lines += ["", "Fixed schedules attempt one operator per round; gated/cascade may attempt several. Early recovery differences therefore also reflect scheduling, not only attribution."]
        references = [r for r in ("routed", "fixed_ggv", "fixed_gvg", "sim_only") if r in runs]
        for ref in references:
            lines += ["", f"Paired differences relative to **{ref}** (percentage points; seed order 0,1,2):", "", "| Method | Nominal per seed | Mean ± SD | Perturbed per seed | Mean ± SD |", "|---|---|---:|---|---:|"]
            summary[condition]["paired"][ref] = {}
            for m in methods:
                if m == ref: continue
                seeds = sorted(runs[m])
                delta = {name:stats([100*(runs[m][s]["final"][key]["overall"]-runs[ref][s]["final"][key]["overall"]) for s in seeds]) for name,key in KEYS.items()}
                summary[condition]["paired"][ref][m] = delta
                a,b = delta["nominal"],delta["perturbed"]
                av,bv = (", ".join(f"{v:+.1f}" for v in x["values"]) for x in (a,b))
                lines.append(f"| {m} | {av} | {a['mean']:+.1f} ± {a['sample_sd']:.1f} | {bv} | {b['mean']:+.1f} ± {b['sample_sd']:.1f} |")
        lines += [""]
    return summary, "\n".join(lines)


def plot(data, summary, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), height_ratios=[9,4,4], layout="constrained")
    for row, (condition, item) in enumerate(summary.items()):
        methods = list(item["methods"])
        for col, condition_key in enumerate(KEYS):
            ax = axes[row,col]
            for y,m in enumerate(methods):
                stat = item["methods"][m][condition_key]
                ax.errorbar(stat["mean"],y,xerr=stat["sample_sd"],fmt="D",color=COLORS[m],capsize=3,markersize=5,zorder=3)
                ax.scatter(stat["values"], y+np.array([-.13,0,.13]),s=14,color=COLORS[m],alpha=.65,zorder=4)
                ax.text(103,y,f"{stat['mean']:.1f}",va="center",fontsize=9)
            ax.set_yticks(range(len(methods)),methods if col==0 else [""]*len(methods))
            ax.set_ylim(len(methods)-.5,-.5)
            ax.set_xlim(-2,111)
            ax.set_xticks(range(0,101,20))
            ax.set_title(f"{LABELS[condition]} | {condition_key}", fontsize=11)
            ax.set_xlabel("Task success (%)")
            ax.grid(axis="x",alpha=.18)
    fig.suptitle("Development screen: fixed DAgger policies, 3 checkpoint seeds\nDiamonds: means; bars: sample SD; dots: seeds. Same-MuJoCo proxy, no confirmation.",fontsize=12)
    for ext in ("png","pdf"): fig.savefig(folder/f"screen_success.{ext}",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(15,8.5),layout="constrained")
    for col,(condition,item) in enumerate(summary.items()):
        for row,(label,key) in enumerate(KEYS.items()):
            ax=axes[row,col]
            for m in item["methods"]:
                runs=list(data[condition][m].values())
                Hs=sorted(runs[0]["final"][key]["by_H"],key=int)
                vals=[100*mean(r["final"][key]["by_H"][h]["k"]/r["final"][key]["by_H"][h]["n"] for r in runs) for h in Hs]
                ax.plot(list(map(int,Hs)),vals,label=m,color=COLORS[m],marker="o",markersize=3,linewidth=1.5,linestyle="--" if m.startswith("fixed") else "-")
            ax.set_title(LABELS[condition]+" | "+label,fontsize=10)
            ax.set_xticks([2,4,6,8,10]); ax.set_ylim(-3,103)
            ax.set_xlabel("Subgoals (H)"); ax.set_ylabel("Mean success (%)")
            ax.grid(alpha=.18)
            if row==0: ax.legend(fontsize=8,loc="upper right" if condition=="triple_cost" else "lower left",ncol=2 if col==0 else 1)
    fig.suptitle("All horizons, nominal and perturbed | Development only\n12 episodes per H per checkpoint; curves are descriptive means",fontsize=12)
    for ext in ("png","pdf"): fig.savefig(folder/f"screen_horizons.{ext}",dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder",type=Path)
    args=ap.parse_args()
    data,schedule,validation=verify(args.folder)
    summary,text=summarize(data,schedule)
    for c,runs in data.items():
        report,compact=make_report(runs,"sim_only" if c=="healthy" else "routed")
        (args.folder/c/"COST_RESULTS.md").write_text(report,encoding="utf-8")
        (args.folder/c/"cost_summary.json").write_text(json.dumps(compact,indent=2),encoding="utf-8")
    (args.folder/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    (args.folder/"screen_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (args.folder/"SCREEN_RESULTS.md").write_text(text+"\n",encoding="utf-8")
    diagnostics = {c: {m: {s: {"final_belief": r["final_belief"], "paid": r["repair_cost_used"], "rounds": [{k: h[k] for k in ("round", "real_test", "shares", "repair_attempts", "repair_committed", "repair_cost", "calibration_gate", "verifier_report", "geometry_report", "graph_report") if k in h} for h in r["history"]]} for s,r in runs.items()} for m,runs in methods.items()} for c,methods in data.items()}
    (args.folder/"screen_diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    plot(data,summary,args.folder)
    print(json.dumps(validation,indent=2))
    print(f"Reports and PNG/PDF figures written to {args.folder}")


if __name__ == "__main__":
    main()
