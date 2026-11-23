"""Operation conflict / ordering analysis from existing records (no new rollouts).

For every recorded decision we get (action, actions seen so far, sr_before, sr_after).  This script
aggregates the marginal effect of each action per cause, and the same effect conditioned on whether
the twin had already been calibrated (i.e. ordering), which is the "operations conflict" question.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/aurora/real2sim/horizon")
SOURCES = [
    "results/four_causes_paper8_20261120",
    "results/experiment1_table1_20261122",
    "results/mixed_grid_20261118",
    "results/four_causes_llm_20261122",
    "results/four_causes_20261118",
]
CALIB = ("calibrate_geometry",)


def records():
    seen = set()
    for src in SOURCES:
        for p in sorted(glob.glob(str(ROOT / src / "runs/*/*/*/*/record.json"))):
            key = (src, p)
            if key in seen:
                continue
            seen.add(key)
            yield src.split("/")[1], json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    marg = defaultdict(list)          # (cause, action) -> [dSR]
    conditioned = defaultdict(list)   # (cause, action, calibrated_before) -> [dSR]
    first_action = defaultdict(list)  # (cause, first_action) -> [final_sr]
    seq_gain = defaultdict(list)      # (cause, tuple(seq)) -> final_sr
    n = 0
    for _, r in records():
        ds = r.get("decisions") or []
        if not ds:
            continue
        n += 1
        cause = r["cause"]
        first_action[cause, ds[0]["action"]].append(float(r["final_sr"]))
        seq_gain[cause, tuple(d["action"] for d in ds)].append(float(r["final_sr"]))
        seen = []
        for d in ds:
            delta = float(d["sr_after"]) - float(d["sr_before"])
            marg[cause, d["action"]].append(delta)
            conditioned[cause, d["action"], any(a in CALIB for a in seen)].append(delta)
            seen.append(d["action"])

    print(f"records with decisions: {n}\n")
    print("## marginal dSR per action (all positions)\n")
    causes = sorted({c for c, _ in marg})
    acts = sorted({a for _, a in marg})
    print("| cause | " + " | ".join(acts) + " |")
    print("|---|" + "---:|" * len(acts))
    for c in causes:
        cells = []
        for a in acts:
            v = marg.get((c, a), [])
            cells.append(f"{100*sum(v)/len(v):+.1f} (n={len(v)})" if v else "-")
        print(f"| {c} | " + " | ".join(cells) + " |")

    print("\n## same action, calibrated before vs not\n")
    print("| cause | action | calibrated before | not calibrated |")
    print("|---|---|---:|---:|")
    for c in causes:
        for a in acts:
            y = conditioned.get((c, a, True), [])
            no = conditioned.get((c, a, False), [])
            if not y and not no:
                continue
            fy = f"{100*sum(y)/len(y):+.1f} (n={len(y)})" if y else "-"
            fn = f"{100*sum(no)/len(no):+.1f} (n={len(no)})" if no else "-"
            print(f"| {c} | {a} | {fy} | {fn} |")

    print("\n## final SR by first action\n")
    print("| cause | first action | mean final SR | n |")
    print("|---|---|---:|---:|")
    for (c, a), v in sorted(first_action.items()):
        print(f"| {c} | {a} | {sum(v)/len(v):.3f} | {len(v)} |")

    print("\n## sequences seen (mean final SR)\n")
    print("| cause | sequence | mean final SR | n |")
    print("|---|---|---:|---:|")
    for (c, seq), v in sorted(seq_gain.items(), key=lambda kv: (kv[0][0], -sum(kv[1]) / len(kv[1]))):
        if len(seq) < 2:
            continue
        short = "→".join(s.replace("_geometry", "").replace("_targeted", "-tgt")
                          .replace("_uniform", "-uni").replace("regenerate", "regen")
                          .replace("dagger_round", "dagger") for s in seq)
        print(f"| {c} | {short} | {sum(v)/len(v):.3f} | {len(v)} |")


if __name__ == "__main__":
    main()
