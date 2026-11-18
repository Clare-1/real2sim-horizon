"""Regression for short-H padded inputs becoming occupied at longer horizons."""
import numpy as np
import pytest

from horizon.policy.model import NumpyMLP, TrainConfig, input_normalization, train_bc


def test_unit_scale_prevents_unseen_neighbour_epsilon_amplification():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1000, 55)).astype(np.float32)
    x[:, 37:41] = 0  # fourth near-TCP slot, absent with at most four task objects
    x[:, 54] = 0.2
    old_mu, old_sd = input_normalization(x)
    mu, sd = input_normalization(x, "constant_safe")
    assert np.all(sd[37:41] == 1)
    assert sd[54] == 1 and mu[54] == x[0, 54]
    np.testing.assert_array_equal(mu[:37], old_mu[:37])
    np.testing.assert_array_equal(sd[:37], old_sd[:37])
    obs = x[0].copy()
    obs[37:41] = [0.4, -0.2, -0.1, -0.08]
    assert np.max(np.abs((obs - old_mu)[37:41] / old_sd[37:41])) > 300
    assert np.max(np.abs((obs - mu)[37:41] / sd[37:41])) < 1


def test_legacy_normalization_is_unchanged():
    x = np.random.default_rng(2).normal(size=(200, 8)).astype(np.float32)
    x[:, 0] = 0
    mu, sd = input_normalization(x)
    np.testing.assert_array_equal(mu, x.mean(0))
    np.testing.assert_array_equal(sd, x.std(0) + 1e-3)


@pytest.mark.parametrize("bad", ["clip", "automatic"])
def test_unknown_normalization_is_rejected(bad):
    with pytest.raises(ValueError, match="unknown"):
        input_normalization(np.ones((2, 3)), bad)


def test_safe_normalization_exports_matching_numpy_policy():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(80, 55)).astype(np.float32)
    x[:, 37:41] = 0
    y = rng.uniform(-1, 1, (80, 5)).astype(np.float32)
    params, stats = train_bc(x, y, TrainConfig(hidden=16, layers=1, steps=3,
        batch=16, device="cpu", normalization="constant_safe"), verbose=False)
    assert stats["np_torch_maxdiff"] < 1e-5
    assert stats["constant_input_indices"] == [37, 38, 39, 40]
    assert np.isfinite(NumpyMLP(params)(x)).all()
