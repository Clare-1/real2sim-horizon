"""LIBERO scaffold: layout prior, mock env, HDF5 reader and image-BC smoke (no LIBERO install)."""
from __future__ import annotations

import numpy as np
import pytest

from horizon.libero import (ImageBC, MockLiberoEnv, apply_layout_prior, concat_demos,
                            layout_prior, load_hdf5_demos)


def test_layout_prior_and_apply_do_not_mutate():
    prior = layout_prior([np.array([[0.0, 0.0], [1.0, 1.0]]),
                          np.array([[0.2, 0.2], [1.2, 1.2]])])
    assert prior["n"] == 2
    assert np.allclose(prior["mean"], [[0.1, 0.1], [1.1, 1.1]])
    scene = {"objects": [1, 2]}
    out = apply_layout_prior(scene, prior)
    assert "layout_prior" not in scene and out["layout_prior"]["n"] == 2


def test_mock_env_succeeds_after_horizon():
    env = MockLiberoEnv(horizon=3)
    env.reset(0)
    done = False
    for _ in range(3):
        _, _, done, info = env.step(np.zeros(7))
    assert done and info["task_success"]


def test_hdf5_reader(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "demo.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("data/demo_0")
        g.create_dataset("obs/agentview_rgb", data=np.zeros((4, 16, 16, 3), dtype=np.uint8))
        g.create_dataset("obs/ee_states", data=np.zeros((4, 8), dtype=np.float32))
        g.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float32))
    demos = load_hdf5_demos(path)
    images, state, actions = concat_demos(demos)
    assert images.shape == (4, 16, 16, 3) and state.shape == (4, 8) and actions.shape == (4, 7)


def test_image_bc_smoke(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(4, 64, 64, 3), dtype=np.uint8)
    actions = rng.normal(size=(4, 7)).astype(np.float32)
    model = ImageBC(action_dim=7)
    model.fit(images, actions, epochs=1, batch=2)
    assert model.act(images[0]).shape == (7,)
    path = tmp_path / "bc.pt"
    model.save(path)
    assert ImageBC.load(path).act(images[0]).shape == (7,)
