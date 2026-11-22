"""Deployment loop and logging for the real-robot track.

The policy is any callable ``PoseSample -> action (5,)``. The integration wraps the state MLP:

    feat = TaskFeaturizer(scene, graph, verifier)
    policy_fn = lambda s: mlp.act(feat.hier(s.to_world_state(), k))

Every step is logged as JSONL so failures can be attributed and the twin can be recalibrated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .interfaces import Robot


def run_episodes(robot: Robot, policy_fn, episodes: int, max_steps: int,
                 log_path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for ep in range(int(episodes)):
        sample = robot.reset()
        for t in range(int(max_steps)):
            action = np.asarray(policy_fn(sample), dtype=float).reshape(-1)
            sample = robot.step(action)
            rows.append({
                "episode": ep, "t": t, "action": action.tolist(),
                "grip_pos": np.asarray(sample.grip_pos, dtype=float).tolist(),
                "grip_open": float(sample.grip_open), "held": int(sample.held),
                "success": bool(sample.success),
                "obj_pos": np.asarray(sample.obj_pos, dtype=float).tolist(),
            })
            if sample.success:
                break
    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows
