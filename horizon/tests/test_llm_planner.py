"""LLM planner plumbing: leak-free prompt, strict parsing, hard failure without a key."""
from __future__ import annotations

import json

import pytest

from horizon.rsi.llm_planner import LLMPlannerSelector, build_user_prompt, parse_decision
from horizon.rsi.selector import ACTIONS, Budget, FailureProfile, Observation


def obs(**kw):
    return Observation(FailureProfile(**{**{"n_episodes": 24, "sr": 0.4}, **kw}),
                       Budget(deploy_spent=24, deploy_total=72, rounds_done=0))


def test_prompt_exposes_only_observable_signals():
    text = build_user_prompt(obs(calibration_residual=0.0093, per_subgoal_failure={"slot0": 5}))
    payload = json.loads(text)
    assert payload["calibration_residual_mm"] == 9.3
    assert "condition" not in text and "true_cause" not in text and "bias" not in text
    assert "height_bias" not in text


def test_parser_rejects_actions_outside_the_set():
    assert parse_decision('{"action": "calibrate_geometry", "target": null, "reason": "x"}').action \
        == "calibrate_geometry"
    with pytest.raises(ValueError, match="outside the action set"):
        parse_decision('{"action": "make_coffee"}')
    with pytest.raises(ValueError, match="no JSON object"):
        parse_decision("I think you should calibrate.")
    assert set(ACTIONS)  # non-empty action set


def test_missing_key_fails_loudly_instead_of_degrading(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sel = LLMPlannerSelector()
    assert not sel.available()
    with pytest.raises(RuntimeError, match="refusing to substitute"):
        sel.decide(obs())


def test_budget_exhaustion_still_stops_without_calling_the_api(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sel = LLMPlannerSelector()
    done = Observation(FailureProfile(sr=0.3), Budget(deploy_spent=72, deploy_total=72))
    assert sel.decide(done).action == "stop"  # guard fires before any API need


def test_client_injection_allows_deterministic_pilot():
    """A pilot can run without a key by injecting a transport; usage is still recorded."""
    def fake(system: str, user: str):
        assert "deployment feedback" in system
        return '{"action": "calibrate_geometry", "target": null, "reason": "mock"}', {
            "input_tokens": 10, "output_tokens": 5}

    sel = LLMPlannerSelector(client=fake)
    assert sel.decide(obs()).action == "calibrate_geometry"
    assert sel.usage == [{"input_tokens": 10, "output_tokens": 5}]
