"""Prevent evaluation leakage and misleading repair-baseline accounting."""
from dataclasses import replace
import numpy as np
import pytest

from horizon.datagen.generator import TaskDistribution
from horizon.eval.protocol import evaluation_manifest
from horizon.rsi import simreal as sr


def explicit(**kwargs):
    return sr.SimRealConfig(name="probe", evaluation_split="development",
                           eval_seed_base=30_000_000, curve_seed_base=20_000_000,
                           feedback_seed_base=2_000_000, **kwargs)


def test_development_curve_feedback_and_confirmation_are_disjoint():
    cfg = explicit()
    manifest = evaluation_manifest(cfg, TaskDistribution(), 20, 4)
    final = {e["task_seed"] for e in manifest["evaluation"]}
    curve = {e["task_seed"] for e in manifest["curve"]}
    assert len(final) == 300 and len(curve) == 60
    assert final.isdisjoint(curve)
    assert not manifest["confirmation_executed"]
    assert manifest["banks"]["reserved_confirmation"] == 40_000_000
    jobs = sr.test_jobs(cfg, sr.Belief.initial(cfg), 20, 0, True)
    assert {j.task_seed for j in jobs} == final
    assert all(30_000_000 <= s < 31_000_000 for s in final)


@pytest.mark.parametrize("field,value", [("eval_seed_base", 20_000_000),
                                          ("feedback_seed_base", 0),
                                          ("curve_seed_base", 10_000_000),
                                          ("confirmation_seed_base", 30_000_000)])
def test_explicit_protocol_rejects_leaked_or_overlapping_banks(field, value):
    cfg = replace(explicit(), **{field: value})
    with pytest.raises(ValueError, match="overlap|Overlapping"):
        evaluation_manifest(cfg, TaskDistribution(), 4, 1)


def test_family_subset_preserves_task_identity_and_legacy_bank():
    cfg = explicit()
    b = sr.Belief.initial(cfg)
    full = [j.task_seed for j in sr.test_jobs(cfg, b, 2, 0, True) if j.family == "sort"]
    subset = [j.task_seed for j in sr.test_jobs(cfg, b, 2, 0, True, families=("sort",))]
    assert full == subset
    old = sr.SimRealConfig(name="legacy")
    assert sr.test_jobs(old, sr.Belief.initial(old), 1, 0, True)[0].task_seed == 10_002_000


def test_fixed_schedule_does_not_use_attribution_or_repeat_after_completion():
    cfg = sr.SimRealConfig(name="fixed", route="fixed", repair_order=("geometry", "graph", "verifier"))
    evidence = {"audited": 0, "judge": 0, "plan": 0}
    chosen = [sr.route(cfg, {}, evidence, [], np.random.default_rng(0), round_index=r)[0]
              for r in range(1, 5)]
    assert chosen == [["geometry"], ["graph"], ["verifier"], []]


def test_random_real_operators_excludes_decoy_without_using_error_labels():
    cfg = sr.SimRealConfig(name="random", route="random", random_include_decoy=False)
    rng = np.random.default_rng(4)
    choices = {sr.route(cfg, {}, {"audited": 0, "judge": 0, "plan": 0}, [], rng)[0][0]
               for _ in range(100)}
    assert choices == {"geometry", "verifier", "graph"}


def test_noop_attempts_are_charged_and_not_reported_as_committed(monkeypatch):
    cfg = sr.SimRealConfig(name="fixed", route="fixed", repair_costs={"geometry": 3, "verifier": 1, "graph": 2})
    belief = sr.Belief.initial(cfg)
    calls = []
    def fake_apply(b, chosen, ev, config):
        calls.extend(chosen)
        if chosen == ["graph"]:
            b.judge_scale = 1.0
        return {}
    monkeypatch.setattr(sr, "apply_repairs", fake_apply)
    out = sr.budgeted_repairs(belief, ["geometry", "verifier", "graph"], {}, cfg, budget=3)
    assert calls == ["geometry"]
    assert out["repair_attempts"] == ["geometry"] and out["repair_committed"] == []
    assert out["repair_cost"] == 3
