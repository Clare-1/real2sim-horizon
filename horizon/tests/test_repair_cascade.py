"""Repair decisions must respect cost, use observed evidence, and avoid label leakage."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from horizon.datagen.rollout import ExpertController, rollout
from horizon.monitor.verifier import VerifierConfig
from horizon.real2sim.recon_error import DEPLOY_WORLD, ReconModel, single_view_bias
from horizon.rsi.repair_cascade import (
    cascade_repairs, replay_audits, robust_geometry_candidate, select_cost_subset, split_audits,
)
from horizon.rsi.simreal import Belief, SimRealConfig, route
from horizon.taskgraph.grounding import GroundingError
from horizon.tasks.factory import make_task


def test_subset_solves_knapsack_rather_than_value_greedy():
    shares = {"geometry": 0.9, "verifier": 0.6, "graph": 0.6}
    costs = {"geometry": 3, "verifier": 2, "graph": 2}
    assert select_cost_subset(shares, costs, 4, 0.05) == ["verifier", "graph"]
    assert select_cost_subset(shares, costs, 0, 0.05) == []
    assert select_cost_subset({"geometry": 0.01}, costs, 100, 0.05) == []


@pytest.mark.parametrize("kwargs", [
    {"repair_costs": {"geometry": 0}}, {"repair_costs": {"graph": float("nan")}},
    {"repair_budget": -1}, {"repair_budget": float("inf")}, {"route": "typo"},
    {"audit_frac": 1.1}, {"geometry_min_episodes": 0}, {"keyframe_stride": 0},
])
def test_invalid_cost_and_feedback_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SimRealConfig(name="bad", **kwargs)


def test_cost_router_does_not_exceed_remaining_budget():
    cfg = SimRealConfig(name="cost", route="cost_gated", repair_costs={"geometry": 2})
    twin = {"pack": {"n": 10, "gap": 0.9}}
    audit = {"audited": 10, "judge": 4, "plan": 3}
    chosen, _ = route(cfg, twin, audit, ["pack"], np.random.default_rng(0), 1)
    assert chosen == ["verifier"]


def test_geometry_episode_votes_resist_one_long_false_contact(monkeypatch):
    import horizon.rsi.simreal as sr
    fixture = SimpleNamespace(name="tray", kind="plate", top_z=0.006)
    monkeypatch.setattr(sr, "_believed", lambda log: SimpleNamespace(fixtures=[fixture]))
    monkeypatch.setattr(sr, "support_observations", lambda logs: {"tray": logs[0].values})
    logs = [SimpleNamespace(values=[0.02] * 5) for _ in range(4)]
    logs += [SimpleNamespace(values=[0.10] * 10000)]
    recon = ReconModel({"tray": 0.3})
    new, report = robust_geometry_candidate(recon, logs)
    assert new.factor("tray") == pytest.approx(1)
    assert report["tray"]["episodes"] == 5
    assert recon.factor("tray") == pytest.approx(0.3)
    assert robust_geometry_candidate(recon, logs[:1])[0] == recon


def test_geometry_does_not_move_a_correct_support(monkeypatch):
    import horizon.rsi.simreal as sr
    fixture = SimpleNamespace(name="tray", kind="plate", top_z=0.02)
    monkeypatch.setattr(sr, "_believed", lambda log: SimpleNamespace(fixtures=[fixture]))
    monkeypatch.setattr(sr, "support_observations", lambda logs: {"tray": logs[0].values})
    logs = [SimpleNamespace(values=[v]) for v in (0.019, 0.020, 0.021, 0.0195)]
    recon = ReconModel()
    assert robust_geometry_candidate(recon, logs)[0] == recon


@pytest.fixture(scope="module")
def observed_logs():
    logs = []
    for seed in range(3):
        task = make_task("pack", 2, seed)
        real = DEPLOY_WORLD.apply(task.scene)
        task = replace(task, scene=real, belief=single_view_bias().apply(real))
        logs.append(rollout(task, ExpertController(), seed=seed, verifier_cfg=VerifierConfig(tol_scale=0.4)))
    return logs


def test_replay_uses_belief_and_observations_without_privileged_geometry(observed_logs):
    old = single_view_bias()
    new = ReconModel(dict(old.bias), {"tray": 1 / old.factor("tray"), "plate": 1 / old.factor("plate")})
    poisoned = [replace(l, scene={"privileged_geometry": "unavailable"},
                        held_true=np.full_like(l.held_true, 99), k_true=np.full_like(l.k_true, 99))
                for l in observed_logs]
    replayed = replay_audits(poisoned, old, new, 1.0)
    for original, replay in zip(observed_logs, replayed):
        tray = next(f for f in replay.meta["belief"]["fixtures"] if f["name"] == "tray")
        assert tray["size"][2] == pytest.approx(0.01)
        np.testing.assert_array_equal(original.sat_true, replay.sat_true)
        assert original.meta["belief"] != replay.meta["belief"]
    assert next(f for f in observed_logs[0].meta["belief"]["fixtures"] if f["name"] == "tray")["size"][2] == pytest.approx(0.003)


def test_audit_split_never_shares_frames_of_an_episode(observed_logs):
    fit, check = split_audits(observed_logs)
    assert len(fit) == 2 and len(check) == 1
    assert not {l.key for l in fit} & {l.key for l in check}


def test_exhausted_budget_cannot_mutate_belief(observed_logs):
    cfg = SimRealConfig(name="cascade", route="cascade", judge_scale=0.4)
    belief = Belief(single_view_bias(), 0.4, GroundingError())
    before = belief.to_dict()
    ev = {"real_logs": observed_logs, "audited": observed_logs, "flagged": ["pack"]}
    result = cascade_repairs(belief, ev, cfg, budget=0)
    assert belief.to_dict() == before
    assert result["repair_cost"] == 0 and result["repair_attempts"] == []


@pytest.mark.parametrize("candidate,new_accuracy,accepted", [(0.4, 0.8, False), (1.0, 0.7, False), (1.0, 0.9, True)])
def test_every_eligible_verifier_fit_costs_budget(monkeypatch, candidate, new_accuracy, accepted):
    import horizon.rsi.repair_cascade as rc
    import horizon.rsi.simreal as sr
    monkeypatch.setattr(rc, "robust_geometry_candidate", lambda recon, *args: (recon, {}))
    monkeypatch.setattr(rc, "replay_audits", lambda logs, *args: logs)
    monkeypatch.setattr(rc, "split_audits", lambda logs: ([1, 2], [3]))
    monkeypatch.setattr(sr, "verifier_recal", lambda *args: (candidate, {"frames": 12}))
    calls = iter((0.8, new_accuracy))
    monkeypatch.setattr(rc, "audit_accuracy", lambda *args: next(calls))
    monkeypatch.setattr(sr, "audit_test", lambda *args: {"plan": 0, "judge": 2})
    cfg = SimRealConfig(name="cascade", route="cascade", judge_scale=0.4)
    belief = Belief.initial(cfg)
    out = rc.cascade_repairs(belief, {"real_logs": [], "audited": [], "flagged": []}, cfg, budget=1)
    assert out["repair_cost"] == 1 and out["repair_attempts"] == ["verifier"]
    assert out["chosen"] == (["verifier"] if accepted else [])
    assert belief.judge_scale == (candidate if accepted else 0.4)


def test_zero_audit_budget_and_common_rollout_randomness(observed_logs):
    from horizon.datagen.generator import TaskDistribution
    from horizon.rsi.simreal import deploy_and_attribute

    class RecordedRunner:
        def __init__(self):
            self.calls = []

        def map(self, jobs):
            self.calls.append(jobs)
            return [SimpleNamespace(error=None, log=replace(observed_logs[0], seed=j.task_seed)) for j in jobs]

    cfg = SimRealConfig(name="paired", real_episodes=3, twin_episodes=3, audit_frac=0,
                        paired_feedback=True)
    runner = RecordedRunner()
    ev = deploy_and_attribute(runner, cfg, Belief.initial(cfg),
                              TaskDistribution(families=("pack",), H_values=(2,)), np.random.default_rng(7))
    assert ev["audited"] == [] and ev["audit"]["audited"] == 0
    physical = {j.task_seed: j.seed for j in runner.calls[0]}
    assert all(j.seed == physical[j.task_seed] for j in runner.calls[1])


@pytest.mark.parametrize("error,log", [("simulation failed", None), (None, None)])
def test_missing_deployment_evidence_is_not_silently_dropped(error, log):
    from horizon.datagen.generator import TaskDistribution
    from horizon.rsi.simreal import deploy_and_attribute
    runner = SimpleNamespace(map=lambda jobs: [SimpleNamespace(error=error, log=log) for j in jobs])
    cfg = SimRealConfig(name="bad", real_episodes=2)
    with pytest.raises(RuntimeError, match="Deployment rollout failed"):
        deploy_and_attribute(runner, cfg, Belief.initial(cfg), TaskDistribution(), np.random.default_rng(0))


def test_custom_evaluation_grid_is_respected():
    from horizon.rsi.simreal import test_jobs as jobs_for_test
    cfg = SimRealConfig(name="grid")
    jobs = jobs_for_test(cfg, Belief.initial(cfg), 2, 0, True, families=("sort",), H_values=(2, 6))
    assert len(jobs) == 4
    assert {(j.family, j.H) for j in jobs} == {("sort", 2), ("sort", 6)}


@pytest.mark.parametrize("fit_misreads,check_misreads", [(2, 0), (0, 2), (1, 0), (0, 0)])
def test_uninformative_audits_preserve_budget_for_later_rounds(monkeypatch, fit_misreads, check_misreads):
    import horizon.rsi.repair_cascade as rc
    import horizon.rsi.simreal as sr
    monkeypatch.setattr(rc, "robust_geometry_candidate", lambda recon, *args: (recon, {}))
    monkeypatch.setattr(rc, "replay_audits", lambda logs, *args: logs)
    fit, check = [1, 2], [3]
    monkeypatch.setattr(rc, "split_audits", lambda logs: (fit, check))
    monkeypatch.setattr(sr, "audit_test", lambda logs: {"plan": 0, "judge": (
        fit_misreads if logs is fit else check_misreads if logs is check else 0)})
    def forbidden_fit(*args):
        raise AssertionError("An uninformative fit must not be attempted")
    monkeypatch.setattr(sr, "verifier_recal", forbidden_fit)
    cfg = SimRealConfig(name="cascade", route="cascade")
    out = rc.cascade_repairs(Belief.initial(cfg), {"real_logs": [], "audited": [], "flagged": []}, cfg, budget=1)
    assert out["repair_cost"] == 0 and out["repair_attempts"] == []
    assert not out["calibration_gate"]["eligible"]
