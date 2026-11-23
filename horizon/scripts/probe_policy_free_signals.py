"""Policy-independent diagnostic signals (no training, no LLM).

The twin and the policy are coupled: the policy is optimised inside the twin, so success measured
inside the twin is circular.  This probe measures only quantities the learned policy cannot influence:

    A2 expert_privileged(deployment)  expert plans on the true scene        -> is the task solvable?
    A  expert_with_twin(deployment)   expert plans on the twin, acts in reality
    B  expert_with_twin(twin)         expert plans on the twin, acts in the twin (self-consistent = 1)
    C  policy(deployment)             learned policy, true physics, twin belief
    D  policy(twin)                   learned policy, twin physics       -> the circular in-twin check

Signals per cell:
    twin_cost      = A2 - A   how much the reconstruction costs an optimal actor
    knowledge_gap  = A  - C   what the policy still misses *given the same twin*
    selfcheck_bias = D  - C   how much the in-twin check overstates deployment success

``twin_cost`` separates "the twin is wrong" from ``knowledge_gap`` = "the data/policy is not good
enough", and ``selfcheck_bias`` exposes the circular signal directly.

    python scripts/probe_policy_free_signals.py [--banks 0,1] [--episodes 5] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from horizon.datagen.parallel import Runner
from horizon.datagen.rollout import PerturbationConfig
from horizon.harness.artifacts import load_params
from horizon.harness.experiment import TaskStore, check_results, job_for
from horizon.real2sim.recon_error import ReconModel

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/four_causes_paper8_20261120"


def build_jobs(store, cfg, recon, bank, n, pert, controller, physics=None, tag="", believe=True):
    # An identity reconstruction makes the belief equal the true scene: a privileged controller
    # without needing job_for(recon=None), which is not supported.
    recon_arg = recon if believe else ReconModel()
    jobs = []
    for i in range(n):
        ts = cfg["banks"]["curve"] + i
        jobs.append(job_for(store, "pack", 4, ts, ts + 17, cfg, recon_arg,
                            real=True, controller=controller,
                            pert=PerturbationConfig.suite(pert), physics=physics,
                            want_samples=False, keep_log=True, provenance=f"probe:{tag}"))
    return jobs


def sr(results):
    check_results(results)
    return float(np.mean([bool(r.log and r.log.success) for r in results]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", default="0,1")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/policy_free_signals_20261123")
    args = ap.parse_args()
    banks = [int(b) for b in args.banks.split(",")]

    cfg = json.loads((SRC / "experiment.json").read_text(encoding="utf-8"))["config"]
    cfg = {**cfg, "H_values": [4]}
    scen_cfg = {s["name"]: s for s in cfg["scenarios"]}
    rows = []
    for name, sc in scen_cfg.items():
        bias = (sc.get("conditions", {}).get("recon", {}) or {}).get("bias", {}) or {}
        recon = ReconModel(bias=dict(bias))
        params = load_params(str(SRC / "shared" / name / "s0" / "policy_0.npz"))
        store = TaskStore(SRC / "tasks_deploy" / name, cfg, ROOT)
        with Runner(args.workers, policy={"params": params}) as runner:
            for bank in banks:
                row = {"scenario": name, "cause": sc["cause"], "bank": bank}
                for pert_tag, pert in (("nominal", 0.0), ("perturbed", 0.7)):
                    n = args.episodes if pert_tag == "nominal" else max(3, args.episodes // 2)
                    a2 = sr(runner.map(build_jobs(store, cfg, recon, bank, n, pert, "expert", None,
                                                  f"expert-priv-{pert_tag}", believe=False)))
                    a = sr(runner.map(build_jobs(store, cfg, recon, bank, n, pert, "expert", None,
                                                 f"expert-dep-{pert_tag}")))
                    b = sr(runner.map(build_jobs(store, cfg, recon, bank, n, pert, "expert",
                                                 recon.to_dict(), f"expert-twin-{pert_tag}")))
                    c = sr(runner.map(build_jobs(store, cfg, recon, bank, n, pert, "hier", None,
                                                 f"policy-dep-{pert_tag}")))
                    d = sr(runner.map(build_jobs(store, cfg, recon, bank, n, pert, "hier",
                                                 recon.to_dict(), f"policy-twin-{pert_tag}")))
                    row[pert_tag] = {"expert_privileged": a2, "expert_dep": a, "expert_twin": b,
                                     "policy_dep": c, "policy_twin": d,
                                     "twin_cost": a2 - a, "knowledge_gap": a - c,
                                     "selfcheck_bias": d - c}
                    print(f"{name:12s} bank{bank} {pert_tag:9s} "
                          f"A2={a2:.2f} A={a:.2f} B={b:.2f} C={c:.2f} D={d:.2f} | "
                          f"twin_cost={a2-a:+.2f} knowledge_gap={a-c:+.2f} "
                          f"selfcheck_bias={d-c:+.2f}", flush=True)
                rows.append(row)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "signals.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # separability of the three signals, averaged over banks, nominal mode
    lines = ["# 策略无关诊断信号（nominal）", "",
             "A2 = 专家用真值规划；A = 专家用 twin 规划、在真值里执行；B = 专家用 twin 规划、在 twin 里执行；",
             "C = 策略在真值里执行；D = 策略在 twin 里执行（自检）。",
             "",
             "| 原因 | A2 | A | B | C | D | twin_cost (A2−A) | knowledge_gap (A−C) | selfcheck_bias (D−C) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cause in sorted({r["cause"] for r in rows}):
        rs = [r["nominal"] for r in rows if r["cause"] == cause]
        m = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        lines.append(f"| {cause} | {m['expert_privileged']:.2f} | {m['expert_dep']:.2f} | "
                     f"{m['expert_twin']:.2f} | {m['policy_dep']:.2f} | {m['policy_twin']:.2f} | "
                     f"**{m['twin_cost']:+.2f}** | **{m['knowledge_gap']:+.2f}** | "
                     f"**{m['selfcheck_bias']:+.2f}** |")
    text = "\n".join(lines) + "\n"
    (out / "signals.md").write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"[written] {out/'signals.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
