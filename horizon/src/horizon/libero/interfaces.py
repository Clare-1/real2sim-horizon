"""LIBERO environment protocol, official-env adapter, and a mock env for tests.

The official adapter is best-effort: it imports ``libero.libero.envs.OffScreenRenderEnv`` at
construction time and raises a clear error when the package is missing. Camera/state key names
follow the official release; override ``_image`` if your build differs.
"""
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


class LiberoEnvAdapter:
    """Official LIBERO env wrapper (requires the ``libero`` package and a BDDL file)."""

    def __init__(self, bddl_file: str, camera_height: int = 128, camera_width: int = 128,
                 init_state=None, settle_steps: int = 10):
        try:
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "LIBERO is not installed. Install the official libero package and pass --bddl."
            ) from exc
        self.env = OffScreenRenderEnv(bddl_file_name=bddl_file,
                                      camera_heights=int(camera_height),
                                      camera_widths=int(camera_width))
        self.init_state = init_state
        self.settle_steps = int(settle_steps)

    def reset(self, task_id: int = 0) -> np.ndarray:
        obs = self.env.reset()
        if self.init_state is not None:
            self.env.set_init_state(np.asarray(self.init_state))
            for _ in range(self.settle_steps):
                obs, _, _, _ = self.env.step(np.zeros(7, dtype=float))
        return self._image(obs)

    def step(self, action: np.ndarray):
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=float))
        return self._image(obs), float(reward), bool(done), {
            "task_success": bool(info.get("success", done)), "raw_info": info}

    @staticmethod
    def _image(obs) -> np.ndarray:
        for key in ("agentview_image", "agentview_rgb", "image"):
            if key in obs:
                img = np.asarray(obs[key], dtype=np.uint8)
                return img[..., ::-1] if img.shape[-1] == 3 and img[..., 0].mean() > 200 else img
        raise KeyError(f"no image key in LIBERO obs: {list(obs)[:12]}")
