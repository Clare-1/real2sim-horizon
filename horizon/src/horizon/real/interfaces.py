"""Hardware-facing interfaces for the real-robot track.

A real integration implements two small protocols (PoseSource, Robot) and reuses the state policy
and the deployment logger. Nothing here imports a robot SDK; the mock/CSV replay in ``mock.py``
lets the integration be tested before hardware is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class PoseSample:
    """One perceived state, matching the fields the state policy needs."""

    t: int
    obj_pos: np.ndarray   # (N, 3) object centres, metres
    obj_quat: np.ndarray  # (N, 4) wxyz
    grip_pos: np.ndarray  # (3,) tool centre point
    grip_yaw: float
    grip_open: float
    grip_open_vel: float = 0.0
    held: int = -1
    success: bool = False

    def to_world_state(self):
        from ..sim.state import WorldState
        return WorldState(self.obj_pos, self.obj_quat, self.grip_pos, self.grip_yaw,
                          self.grip_open, self.grip_open_vel, self.held, self.t)


class PoseSource(Protocol):
    """AprilTag / pose estimator / ROS topic adapter: return the current perceived state."""

    def read(self) -> PoseSample: ...


class Robot(Protocol):
    """Minimal robot adapter: reset to a fixed initial layout, execute one action, close."""

    def reset(self) -> PoseSample: ...

    def step(self, action: np.ndarray) -> PoseSample: ...

    def close(self) -> None: ...
