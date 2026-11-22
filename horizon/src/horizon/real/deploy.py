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
                 log_path: str | Path, success_fn=None) -> list[dict]:
    rows: list[dict] = []
    for ep in range(int(episodes)):
        sample = robot.reset()
        for t in range(int(max_steps)):
            action = np.asarray(policy_fn(sample), dtype=float).reshape(-1)
            sample = robot.step(action)
            ok = bool(sample.success)
            if not ok and success_fn is not None:
                ok = bool(success_fn(sample))
            rows.append({
                "episode": ep, "t": int(sample.t), "action": action.tolist(),
                "grip_pos": np.asarray(sample.grip_pos, dtype=float).tolist(),
                "grip_yaw": float(sample.grip_yaw),
                "grip_open": float(sample.grip_open),
                "grip_open_vel": float(sample.grip_open_vel),
                "held": int(sample.held),
                "success": ok,
                "obj_pos": np.asarray(sample.obj_pos, dtype=float).tolist(),
                "obj_quat": np.asarray(sample.obj_quat, dtype=float).tolist(),
            })
            if ok:
                break
    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows


def rows_to_episode(rows: list[dict], task):
    """Build an ``EpisodeLog``-compatible record from deployment rows.

    ``phase`` / ``valid`` are filled with placeholders: real deployment does not know the expert
    phase, and supervision masks need manual/anomaly annotations. Attribution that depends on
    phases should be treated as approximate until those fields are annotated.
    """
    from ..datagen.rollout import EpisodeLog

    if not rows:
        raise ValueError("cannot build an EpisodeLog from zero rows")
    T = len(rows)
    H = int(task.H)
    grip = np.asarray([[r["grip_pos"][0], r["grip_pos"][1], r["grip_pos"][2],
                        r["grip_yaw"], r["grip_open"], r["grip_open_vel"]] for r in rows], dtype=float)
    obj_obs = np.asarray([np.concatenate([np.asarray(r["obj_pos"], float),
                                          np.asarray(r["obj_quat"], float)], axis=1) for r in rows],
                         dtype=float)
    action = np.asarray([r["action"] for r in rows], dtype=float)
    held = np.asarray([r["held"] for r in rows], dtype=int)
    return EpisodeLog(
        key=task.key, family=task.family, H=H, seed=int(task.seed),
        scene=task.scene.to_dict(), graph=task.graph.to_dict(),
        grip=grip, obj_obs=obj_obs, action=action, expert_action=None,
        k_sel=np.full(T, -1), k_true=np.full(T, -1),
        sat_true=np.zeros((T, H), dtype=bool), sat_ver=np.zeros((T, H), dtype=bool),
        held_true=held, held_est=held, phase=np.zeros(T, dtype=int),
        valid=np.ones(T, dtype=bool), events=[],
        success=any(bool(r["success"]) for r in rows),
        success_ver=any(bool(r["success"]) for r in rows),
        steps=T, provenance="real", meta={"source": "real_deploy"},
    )
