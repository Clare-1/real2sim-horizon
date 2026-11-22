"""LIBERO environment protocol and a mock env for tests (no LIBERO install required)."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class LiberoEnv(Protocol):
    def reset(self, task_id: int) -> np.ndarray: ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...


class MockLiberoEnv:
    """Deterministic env: success after ``horizon`` steps if the action is finite."""

    def __init__(self, horizon: int = 5, image_shape=(128, 128, 3)):
        self.horizon = int(horizon)
        self.image_shape = image_shape
        self.t = 0

    def reset(self, task_id: int) -> np.ndarray:
        self.t = 0
        return np.zeros(self.image_shape, dtype=np.uint8)

    def step(self, action: np.ndarray):
        self.t += 1
        obs = np.zeros(self.image_shape, dtype=np.uint8)
        done = self.t >= self.horizon and np.isfinite(action).all()
        return obs, float(done), bool(done), {"task_success": bool(done)}
