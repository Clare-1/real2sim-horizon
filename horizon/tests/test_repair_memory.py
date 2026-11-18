"""Checks for feasible controls, immutable audit splits, and paid rejection."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from horizon.real2sim.recon_error import ReconModel, single_view_bias
from horizon.rsi.simreal import SimRealConfig, Belief, route
from horizon.rsi.repair_memory import (
    AuditMemory, episode_id, audit_partition, select_feasible_greedy, accept_candidate, memory_repairs,
)
from test_repair_cascade import observed_logs


@pytest.mark.parametrize("ratio", [False, True])
def test_greedy_skips_unaffordable_first_choice(ratio):
    assert select_feasible_greedy({"geometry": .9, "verifier": .2, "graph": .1},
                                  {"geometry": 3, "verifier": 1, "graph": 2}, 1, .05, ratio) == ["verifier"]
    assert select_feasible_greedy({"geometry": .9}, {"geometry": 3}, 0, .05, ratio) == []


def test_greedy_variants_distinct_with_matched_multi_operator_budget():
    shares = {"geometry": .9, "verifier": .6, "graph": .6}
    costs = {"geometry": 3, "verifier": 2, "graph": 2}
    assert select_feasible_greedy(shares, costs, 4, .05) == ["geometry"]
    assert select_feasible_greedy(shares, costs, 4, .05, True) == ["geometry"]  # stable equal-density tie
    costs["geometry"] = 3.1
    assert select_feasible_greedy(shares, costs, 4, .05, True) == ["verifier", "graph"]


def test_fixed_grouped_schedule_does_not_repeat_operators():
    cfg = SimRealConfig(name="grouped", route="fixed", repair_order=("geometry", "graph", "verifier"), fixed_repairs_per_round=2)
    rounds = [route(cfg, {}, {"audited": 0, "judge": 0, "plan": 0}, [], np.random.default_rng(0), 3, r)[0] for r in (1,2,3)]
    assert rounds == [["geometry", "graph"], ["verifier"], []]


def test_memory_partitions_stay_fixed_and_replay_each_original_belief(observed_logs):
    memory = AuditMemory()
    old = single_view_bias()
    current = ReconModel(dict(old.bias), {"tray": 1 / old.factor("tray"), "plate": 1 / old.factor("plate")})
    memory.observe(observed_logs[:2], old)
    assignments = {episode_id(log): audit_partition(episode_id(log)) for log in observed_logs[:2]}
    memory.observe(observed_logs, old)
    assert len(memory.entries) == 3
    fit, check = memory.replay(current, 1.0)
    assert not {episode_id(log) for log in fit} & {episode_id(log) for log in check}
    for log in fit + check:
        tray = next(f for f in log.meta["belief"]["fixtures"] if f["name"] == "tray")
        assert tray["size"][2] == pytest.approx(.01)
        if episode_id(log) in assignments:
            assert audit_partition(episode_id(log)) == assignments[episode_id(log)]
    fit2, check2 = memory.replay(current, 1.0)
    assert [log.meta["belief"] for log in fit + check] == [log.meta["belief"] for log in fit2 + check2]
    assert next(f for f in observed_logs[0].meta["belief"]["fixtures"] if f["name"] == "tray")["size"][2] == pytest.approx(.003)
    memory.observe(observed_logs[-1:], old, accumulate=False)
    assert len(memory.entries) == 1


@pytest.mark.parametrize("new_score,new_fp,positive,expected", [(.9,0,2,True),(.8,0,2,False),(.9,1,2,False),(.9,0,0,False)])
def test_rejection_requires_improvement_without_new_false_positives(new_score,new_fp,positive,expected):
    before = {"positive_labels":positive,"negative_labels":4,"balanced_accuracy":.8,"fp":0}
    after = {"balanced_accuracy":new_score,"fp":new_fp}
    assert accept_candidate(before, after, .005) == expected


@pytest.mark.parametrize("reject,expected_scale", [(True,.4),(False,1.2)])
def test_rejected_candidate_is_paid_and_ablation_only_changes_acceptance(monkeypatch,reject,expected_scale):
    import horizon.rsi.repair_memory as rm
    import horizon.rsi.simreal as sr
    monkeypatch.setattr(rm,"robust_geometry_candidate",lambda recon,*a:(recon,{}))
    monkeypatch.setattr(rm,"replay_audits",lambda logs,*a: logs)
    monkeypatch.setattr(sr,"audit_test",lambda logs:{"plan":0})
    monkeypatch.setattr(sr,"verifier_recal",lambda *a:(1.2,{"frames":10}))
    fake = [SimpleNamespace(family="pack",H=2,seed=i,key=str(i)) for i in range(4)]
    class Memory:
        entries = {"a":1,"b":2,"c":3,"d":4}
        def observe(self,*a,**kw): pass
        def replay(self,*a): return fake[:2],fake[2:]
    metrics = {"error_episodes":2,"positive_labels":5,"negative_labels":5,"balanced_accuracy":.8,"fp":0}
    monkeypatch.setattr(rm,"keyframe_metrics",lambda *a:dict(metrics))
    cfg=SimRealConfig(name="memory",route="memory",judge_scale=.4,reject_verifier_candidate=reject)
    belief=Belief.initial(cfg)
    out=memory_repairs(belief,{"real_logs":[],"audited":[],"flagged":[],"shares":{}},cfg,1,Memory())
    assert out["repair_cost"] == 1 and out["repair_attempts"] == ["verifier"]
    assert belief.judge_scale == expected_scale
    assert out["repair_committed"] == ([] if reject else ["verifier"])


def test_memory_exhausted_budget_cannot_change_state(observed_logs):
    cfg=SimRealConfig(name="memory",route="memory",judge_scale=.4,audit_memory=True)
    belief=Belief.initial(cfg)
    before=belief.to_dict()
    ev={"real_logs":observed_logs,"audited":observed_logs,"flagged":["pack"],"shares":{"geometry":1.0}}
    out=memory_repairs(belief,ev,cfg,0,AuditMemory())
    assert belief.to_dict() == before
    assert out["repair_cost"] == 0 and out["repair_attempts"] == []
