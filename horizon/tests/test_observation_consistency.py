"""The optional BC observation channel must preserve the actor's sampled input."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from horizon.datagen.parallel import Job, _run
from horizon.datagen.rollout import rollout
from horizon.harness.experiment import observed_feedback
from horizon.monitor.verifier import Verifier
from horizon.policy.dataset import episode_samples
from horizon.policy.features import TaskFeaturizer
from horizon.sim.env import TabletopEnv
from horizon.tasks.factory import make_task


class RecordingActor:
    privileged = False

    def reset(self, task):
        self.features = TaskFeaturizer(task.scene, task.graph, Verifier(task.scene))
        self.hier, self.flat, self.observed_grip, self.ideal_grip = [], [], [], []

    def act(self, obs, true):
        self.hier.append(self.features.hier(obs, 0))
        self.flat.append(self.features.flat(obs))
        for target, state in ((self.observed_grip, obs), (self.ideal_grip, true)):
            target.append([*state.grip_pos, state.grip_yaw, state.grip_open, state.grip_open_vel])
        # A short, deterministic trace remains on the first subgoal. Labels still
        # come from the state expert, independently of this actor's action.
        return np.array([0.1, 0.0, -0.1, 0.0, -1.0]), 0


@pytest.fixture(scope="module")
def recorded():
    actor = RecordingActor()
    log = rollout(make_task("pack", 2, 17), actor, seed=23, max_steps=16, label_expert=True)
    assert log.valid.all() and (log.k_true == 0).all()
    return log, actor


def test_observed_features_match_the_controller_input(recorded):
    log, actor = recorded
    block = episode_samples(log, observation_mode="observed")
    np.testing.assert_array_equal(log.grip_obs, np.asarray(actor.observed_grip, dtype=np.float32))
    # Object and gripper logs retain the historical float32 storage precision;
    # geometry features are recomputed from those values rather than resampled.
    np.testing.assert_allclose(block.x_hier, actor.hier, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(block.x_flat, actor.flat, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(block.y, log.expert_action)
    assert block.meta == {"observation_mode": "observed", "gripper_channel": "actor_observation"}


def test_default_features_and_geometry_keep_the_ideal_encoder(recorded):
    log, actor = recorded
    default = episode_samples(log)
    explicit = episode_samples(log, observation_mode="legacy")
    observed = episode_samples(log, observation_mode="observed")
    for field in ("x_hier", "x_flat", "y", "k", "stage"):
        np.testing.assert_array_equal(getattr(default, field), getattr(explicit, field))
    np.testing.assert_array_equal(log.grip, np.asarray(actor.ideal_grip, dtype=np.float32))
    assert not np.array_equal(log.grip[:, :3], log.grip_obs[:, :3])
    assert not np.array_equal(default.x_hier, observed.x_hier)
    feedback = observed_feedback(log)
    assert feedback.grip is log.grip and not hasattr(feedback, "grip_obs")
    assert default.meta == {"observation_mode": "legacy", "gripper_channel": "ideal_encoder"}


@pytest.mark.parametrize("missing_attribute", [False, True])
def test_historical_logs_require_explicit_legacy_mode(recorded, missing_attribute):
    log, _ = recorded
    fields = dict(vars(log))
    if missing_attribute:
        fields.pop("grip_obs")
    else:
        fields["grip_obs"] = None
    old = SimpleNamespace(**fields)
    np.testing.assert_array_equal(episode_samples(old).x_hier, episode_samples(log).x_hier)
    with pytest.raises(ValueError, match="requires logged grip_obs"):
        episode_samples(old, observation_mode="observed")


@pytest.mark.parametrize("malformed", ["shape", "nan"])
def test_observed_channel_rejects_malformed_data(recorded, malformed):
    log, _ = recorded
    values = log.grip_obs.copy()
    if malformed == "shape":
        values = values[:, :5]
    else:
        values[0, 0] = np.nan
    bad = replace(log, grip_obs=values)
    with pytest.raises(ValueError, match="finite.*T, 6"):
        episode_samples(bad, observation_mode="observed")
    # Legacy readers do not depend on the optional channel.
    np.testing.assert_array_equal(episode_samples(bad).x_hier, episode_samples(log).x_hier)


@pytest.mark.parametrize("mode", ["auto", "ideal", None])
def test_unknown_observation_mode_is_rejected(recorded, mode):
    with pytest.raises(ValueError, match="unknown observation_mode"):
        episode_samples(recorded[0], observation_mode=mode)


def test_worker_mode_changes_only_sample_features_and_records_mode(monkeypatch):
    reads = []
    observe = TabletopEnv.observe

    def counted_observe(env):
        reads.append(env.t)
        return observe(env)

    monkeypatch.setattr(TabletopEnv, "observe", counted_observe)
    job = Job("pack", 2, 17, 23, controller="expert", max_steps=16,
              max_samples=7, keep_log=True, only_success=False)
    legacy = _run(job)
    observed = _run(replace(job, observation_mode="observed"))
    assert legacy.error is observed.error is None
    # Exactly reset + each env.step + the final judge observation; logging adds
    # no sensor calls and therefore no new observation-noise RNG draws.
    assert len(reads) == legacy.log.steps + observed.log.steps + 4
    for field in ("grip", "grip_obs", "obj_obs", "action", "expert_action", "k_sel", "k_true",
                  "sat_true", "sat_ver", "held_true", "held_est", "phase", "valid"):
        np.testing.assert_array_equal(getattr(legacy.log, field), getattr(observed.log, field))
    assert (legacy.success, legacy.log.success_ver, legacy.sim_steps, legacy.log.events) == (
        observed.success, observed.log.success_ver, observed.sim_steps, observed.log.events)
    for field in ("y", "k", "stage"):
        np.testing.assert_array_equal(getattr(legacy.samples, field), getattr(observed.samples, field))
    assert legacy.samples.n == observed.samples.n == 7
    assert not np.array_equal(legacy.samples.x_hier, observed.samples.x_hier)
    assert observed.samples.meta["observation_mode"] == "observed"
    assert observed.samples.take(3).meta == observed.samples.meta


def test_worker_rejects_unknown_mode_before_running(monkeypatch):
    import horizon.tasks.factory as factory

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid mode must fail before task construction")

    monkeypatch.setattr(factory, "make_task", forbidden)
    result = _run(Job("pack", 2, 17, 23, observation_mode="auto"))
    assert "unknown observation_mode" in result.error
    assert result.log is None and result.sim_steps == 0
