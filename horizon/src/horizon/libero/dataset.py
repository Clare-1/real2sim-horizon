"""LIBERO data adapter: read HDF5 demonstrations into plain arrays.

Tested against the official layout ``data/demo_i/obs/agentview_rgb``, ``obs/ee_states``,
``actions``. Key names vary between releases, so both are configurable in the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Demo:
    images: np.ndarray   # (T, H, W, 3) uint8
    state: np.ndarray    # (T, D) float32
    actions: np.ndarray  # (T, A) float32
    task: str = ""
    success: bool = True


def load_hdf5_demos(path: str | Path, image_key: str = "obs/agentview_rgb",
                    state_key: str = "obs/ee_states", action_key: str = "actions",
                    max_demos: int | None = None) -> list[Demo]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("LIBERO reading needs h5py: pip install h5py") from exc
    demos: list[Demo] = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else 0)
        for key in keys[:max_demos] if max_demos else keys:
            g = data[key]
            images = np.asarray(g[image_key][:], dtype=np.uint8)
            state = np.asarray(g[state_key][:], dtype=np.float32)
            actions = np.asarray(g[action_key][:], dtype=np.float32)
            if len(images) != len(actions) or len(state) != len(actions):
                raise ValueError(f"{path}:{key} has mismatched lengths")
            demos.append(Demo(images=images, state=state, actions=actions))
    return demos


def concat_demos(demos: list[Demo]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = np.concatenate([d.images for d in demos], 0)
    state = np.concatenate([d.state for d in demos], 0)
    actions = np.concatenate([d.actions for d in demos], 0)
    return images, state, actions
