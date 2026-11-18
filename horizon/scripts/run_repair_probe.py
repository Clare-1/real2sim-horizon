"""Isolate repair quality using paired, frozen pretrained policies (no retraining).

This is a diagnostic on the MuJoCo proxy, not an end-to-end SOTA comparison.
A separate checkpoint for each training seed is shared across repair variants.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import yaml

from horizon.datagen.generator import TaskDistribution
from horizon.datagen.parallel import Runner
from horizon.rsi.loop import load_params
from horizon.eval.protocol import evaluation_manifest
from horizon.rsi.repair_cascade import cascade_repairs
from horizon.rsi.repair_memory import AuditMemory, memory_repairs
from horizon.rsi.simreal import (Belief, SimRealConfig, budgeted_repairs,
                                deploy_and_attribute, evaluate_sr)


def source_hashes() -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*Path("src/horizon").rglob("*.py"), Path(__file__)])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", default="results/main_v1/dagger_beta_s{seed}_policy.npz")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    raw = Path(args.config).read_bytes()
    conf = yaml.safe_load(raw)
    out = Path(args.out or f"results/{conf['name']}")
    out.mkdir(parents=True, exist_ok=True)
    d = conf["dist"]
    dist = TaskDistribution(families=tuple(d["families"]), H_values=tuple(d["H_values"]), seed_base=d.get("seed_base", 0))
    grid = {"families": dist.families, "H_values": dist.H_values}
    methods = args.methods.split(",") if args.methods else list(conf["methods"])
    for seed in map(int, args.seeds.split(",")):
        checkpoint = Path(args.checkpoints.format(seed=seed))
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        params = load_params(checkpoint)
        for method in methods:
            path = out / f"{method}_s{seed}.json"
            if args.skip_existing and path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))["provenance"]
                if (saved["config_sha256"] != hashlib.sha256(raw).hexdigest()
                        or saved["checkpoint_sha256"] != checkpoint_hash
                        or saved["source_sha256"] != source_hashes()):
                    raise ValueError(f"Existing result has different inputs/source: {path}; use a new output directory")
                print(f"skip {path}", flush=True)
                continue
            cfg = SimRealConfig(name=method, seed=seed,
                                **{**conf["common"], **conf["methods"][method]})
            manifest = evaluation_manifest(cfg, dist, conf["test"]["n_per"], conf["test"]["curve_n_per"])
            b = Belief.initial(cfg)
            started, hashes = time.time(), source_hashes()
            history, repairs = [], []
            audit_memory = AuditMemory()
            spent, real_used, audit_used = 0.0, 0, 0
            audit_frames, audit_labels, twins, curve_episodes = 0, 0, 0, 0
            with Runner(args.workers, policy={"params": params}) as runner:
                for r in range(1, cfg.rounds + 1):
                    h = {"round": r, "n": 0, "belief": b.to_dict(), "verifier_scale": b.judge_scale}
                    pre = evaluate_sr(runner, cfg, b, conf["test"]["curve_n_per"], 0., True, seed_base=cfg.curve_seed_base, **grid)
                    h["real_test"] = pre["overall"]
                    h["real_test_episode_count"] = len(pre["episodes"])
                    curve_episodes += len(pre["episodes"])
                    if cfg.route not in ("none", "perfect"):
                        rng = np.random.default_rng(np.random.SeedSequence([seed, r, 937]))
                        remaining = math.inf if cfg.repair_budget is None else max(0., cfg.repair_budget - spent)
                        ev = deploy_and_attribute(runner, cfg, b, dist, rng, remaining, round_index=r)
                        real_used += len(ev["real_logs"])
                        audit_used += len(ev["audited"])
                        audit_frames += ev["audit_keyframes"]
                        audit_labels += ev["audit_predicate_labels"]
                        twins += ev["twin_replays"]
                        h.update({k: ev[k] for k in ("twin", "flagged", "audit", "shares", "active",
                                                    "real_judged_done", "real_success", "feedback_tasks", "audit_keyframes",
                                                    "audit_predicate_labels", "twin_replays")})
                        if cfg.route == "memory":
                            h.update(memory_repairs(b, ev, cfg, remaining, audit_memory))
                        elif cfg.route == "cascade":
                            h.update(cascade_repairs(b, ev, cfg, remaining))
                        else:
                            h.update(budgeted_repairs(b, ev["chosen"], ev, cfg, remaining))
                        spent += h["repair_cost"]
                        h["repair_cost_used"] = spent
                        repairs += h["chosen"]
                    history.append(h)
                    print(f"[{method} s{seed}] round {r}: pre {h['real_test']:.3f}, "
                          f"repairs {h.get('chosen', [])}, cost {spent:.1f}, judge {b.judge_scale}", flush=True)
                final = {key: evaluate_sr(runner, cfg, b, conf["test"]["n_per"], pert, real, **grid)
                         for key, pert, real in (("real_nominal", 0., True),
                                                 ("real_pert0.15", .15, True),
                                                 ("sim_nominal", 0., False))}
            if hashes != source_hashes():
                raise RuntimeError("Source changed during probe; rerun before reporting this result")
            result = {"config": asdict(cfg), "history": history, "final": final,
                      "repairs": repairs, "final_belief": b.to_dict(), "repair_cost_used": spent,
                      "real_episodes_used": real_used, "audits_used": audit_used, "n": 0,
                      "train_steps": 0, "wall": time.time() - started,
                      "protocol": "fixed_policy_probe", "distribution": asdict(dist),
                      "evaluation_manifest": manifest, "test_config": conf["test"], "audit_keyframes_used": audit_frames,
                      "audit_predicate_labels_used": audit_labels, "twin_replays_used": twins,
                      "curve_evaluation_episodes": curve_episodes,
                      "provenance": {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                                     "config_sha256": hashlib.sha256(raw).hexdigest(),
                                     "source_sha256": hashes, "python": platform.python_version(),
                                     "workers": args.workers}}
            path.write_text(json.dumps(result, indent=1), encoding="utf-8")
            print(f"[{method} s{seed}] final nominal {final['real_nominal']['overall']:.3f}, "
                  f"perturbed {final['real_pert0.15']['overall']:.3f}", flush=True)


if __name__ == "__main__":
    main()
