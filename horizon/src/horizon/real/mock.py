"""Mock and CSV-replay robots for the real-robot track (no hardware, no SDK)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .interfaces import PoseSample


def sample_from_dict(row: dict) -> PoseSample:
    return PoseSample(
        t=int(row.get("t", 0)),
        obj_pos=np.asarray(row["obj_pos"], dtype=float),
        obj_quat=np.asarray(row["obj_quat"], dtype=float),
        grip_pos=np.asarray(row["grip_pos"], dtype=float),
        grip_yaw=float(row["grip_yaw"]),
        grip_open=float(row["grip_open"]),
        grip_open_vel=float(row.get("grip_open_vel", 0.0)),
        held=int(row.get("held", -1)),
        success=bool(row.get("success", False)),
    )


def make_mock_robot(frames: int = 20) -> "ScriptedRobot":
    """A deterministic 2-object trajectory: gripper moves down, closes, lifts, marks success."""
    out = []
    for t in range(frames):
        z = 0.20 - 0.10 * min(1.0, t / max(frames - 1, 1))
        out.append(PoseSample(
            t=t,
            obj_pos=np.array([[0.0, 0.0, 0.02], [0.15, 0.0, 0.02]]),
            obj_quat=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            grip_pos=np.array([0.0, 0.0, z]),
            grip_yaw=0.0,
            grip_open=0.08 if t < frames // 2 else 0.02,
            grip_open_vel=0.0,
            held=0 if t >= frames // 2 else -1,
            success=t == frames - 1,
        ))
    return ScriptedRobot(out)


class ScriptedRobot:
    """Replay a fixed frame list; ``step`` ignores the action and advances one frame."""

    def __init__(self, frames: list[PoseSample]):
        if not frames:
            raise ValueError("ScriptedRobot needs at least one frame")
        self.frames = frames
        self.i = 0

    def reset(self) -> PoseSample:
        self.i = 0
        return self.frames[0]

    def step(self, action: np.ndarray) -> PoseSample:
        self.i = min(self.i + 1, len(self.frames) - 1)
        return self.frames[self.i]

    def close(self) -> None:
        return None


def load_jsonl_robot(path: str | Path) -> ScriptedRobot:
    """Replay a recorded deployment log (one JSON object per line, PoseSample schema)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"empty replay file: {path}")
    return ScriptedRobot([sample_from_dict(r) for r in rows])
