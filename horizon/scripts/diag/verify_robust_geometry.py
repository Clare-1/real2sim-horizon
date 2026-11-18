"""Verify the robust geometry estimator against the historical divergence mode.

The 2026-10-26 failure moved a 19.84 mm belief to 93.54 mm because two episodes with
oscillating descents contributed 44 of 66 windows, and the estimator pooled frames.
This script compares the shipped estimator with the repaired one on the same logs.
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np

from horizon.datagen.rollout import ExpertController, rollout
from horizon.real2sim.recon_error import DEPLOY_WORLD, single_view_bias
from horizon.rsi.simreal import geometry_repair, support_observations, support_observations_per_episode
from horizon.tasks.factory import make_task


def build_logs(family="pack", H=4, seeds=(0, 1, 2)):
    recon = single_view_bias()
    logs = []
    for s in seeds:
        task = make_task(family, H, s)
        real = DEPLOY_WORLD.apply(task.scene)
        task = dataclasses.replace(task, scene=real, belief=recon.apply(real))
        logs.append(rollout(task, ExpertController(), seed=s))
    return recon, logs


def main() -> int:
    recon, logs = build_logs()
    print(f"logs={len(logs)}  believed support tops: "
          f"{ {f.name: round(f.top_z*1000, 2) for f in logs[0].meta and [] or []} }")

    per_ep = support_observations_per_episode(logs)
    flat = support_observations(logs)
    print("\n=== A. per-episode medians vs pooled frames (mm) ===")
    for name in sorted(per_ep):
        ep = np.array(per_ep[name]) * 1000
        fl = np.array(flat[name]) * 1000
        print(f"{name:6s} episodes={len(ep):2d} per-episode={np.round(ep, 2).tolist()}")
        print(f"{'':6s} pooled n={len(fl):3d} median={np.median(fl):7.2f}  p10={np.percentile(fl,10):7.2f} "
              f"p90={np.percentile(fl,90):7.2f}")

    print("\n=== B. shipped estimator vs robust estimator on the same logs ===")
    old, old_rep = geometry_repair(recon, logs)
    new, new_rep = geometry_repair(recon, logs, robust=True)
    for name in sorted(set(old_rep) | set(new_rep)):
        o, r = old_rep.get(name, {}), new_rep.get(name, {})
        print(f"{name:6s} old est={o.get('estimated', float('nan'))*1000:7.2f}mm n={o.get('n')}"
              f" | robust est={r.get('estimated', float('nan'))*1000:7.2f}mm n={r.get('n')}"
              f" spread={r.get('spread', 0)*1000:5.2f}mm ratio={r.get('applied_ratio', 1):.3f}")

    print("\n=== C. frame weighting: why the shipped estimator drifts ===")
    counts = {}
    for i, log in enumerate(logs):
        for name, vals in support_observations([log]).items():
            counts.setdefault(name, []).append((i, len(vals), float(np.median(vals)) * 1000))
    for name, rows in counts.items():
        total = sum(c for _, c, _ in rows)
        print(f"{name}: total frames={total}")
        for i, c, med in rows:
            print(f"   episode {i}: frames={c:4d} ({100*c/total:5.1f}% of frames)  median={med:6.2f} mm"
                  f"   -> vote share in robust = {100/len(rows):.1f}%")
        hi = max(rows, key=lambda r: r[1])
        print(f"   dominant episode {hi[0]} holds {100*hi[1]/total:.1f}% of the weight in the shipped"
              f" estimator but only {100/len(rows):.1f}% in the robust one")

    print("\n=== D. step cap: believed 6 mm, truth 20 mm ===")
    for cap in (1e9, 2.0):
        tgt, cur = 0.020, 0.006
        ratio = tgt / cur
        applied = float(np.clip(ratio, 1 / cap, cap))
        print(f"step_cap={cap if cap < 1e9 else 'off':>4}: raw ratio={ratio:.2f}x -> applied={applied:.2f}x "
              f"-> next belief={cur*applied*1000:.2f} mm")
    print("\n=== E. synthetic: one chatty episode dominates the pooled median ===")
    # emulates the 2026-10-26 case: 2 episodes supplied 44 of 66 candidate windows,
    # with high values from dip-then-rise descents.
    good = [[19.8, 20.1, 20.2, 20.0], [20.3, 20.2, 20.4, 20.1], [20.0, 19.9, 20.2, 20.1]]
    chatty = [[93.5] * 44]
    pooled = [v for ep in (good + chatty) for v in ep]
    per_ep = [float(np.median(ep)) for ep in (good + chatty)]
    print(f"pooled: n={len(pooled)} median={np.median(pooled):6.2f} mm   <- shipped")
    print(f"per-episode: {[round(v, 2) for v in per_ep]} median={np.median(per_ep):6.2f} mm   <- robust")
    spread = float(np.median(np.abs(np.array(per_ep) - np.median(per_ep))))
    arr = np.array(per_ep)
    inliers = arr[np.abs(arr - np.median(arr)) <= 5.0]   # max_dev = 5 mm
    print(f"per-episode spread(MAD)={spread:.2f} mm (max_dev gate = 5 mm)")
    print(f"after outlier drop: kept={[round(v, 2) for v in inliers]} -> estimate={np.median(inliers):.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
