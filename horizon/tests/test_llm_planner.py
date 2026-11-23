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


def test_client_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(system: str, user: str):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient")
        return '{"action": "dagger_round", "target": null, "reason": "retry"}', {
            "input_tokens": 1, "output_tokens": 1}

    sel = LLMPlannerSelector(client=flaky, max_retries=1, retry_backoff=0)
    assert sel.decide(obs()).action == "dagger_round"
    assert calls["n"] == 2


def test_call_cap_stops_before_overspending():
    def fake(system: str, user: str):
        return '{"action": "stop", "target": null, "reason": "x"}', {
            "input_tokens": 1, "output_tokens": 1}

    sel = LLMPlannerSelector(client=fake, max_calls=1)
    sel.decide(obs())
    with pytest.raises(RuntimeError, match="call cap"):
        sel.decide(obs())


def test_memory_is_visible_to_the_llm_row():
    text = build_user_prompt(obs(), memory=("past coverage scenario: targeted helped",))
    assert "memory" in text and "targeted helped" in text
    assert "true_cause" not in text


def test_withheld_actions_are_not_offered_and_not_accepted():
    """Agent ablation ("no specialist tools"): the prompt hides them and the parser refuses them."""
    allowed = ("calibrate_geometry", "dagger_round", "stop")
    seen = {}

    def fake(system: str, user: str):
        seen["system"] = system
        return '{"action": "regenerate_targeted", "target": null, "reason": "x"}', {
            "input_tokens": 1, "output_tokens": 1}

    sel = LLMPlannerSelector(client=fake, allowed_actions=allowed)
    with pytest.raises(ValueError, match="withheld"):
        sel.decide(obs())
    assert "regenerate_targeted" not in seen["system"]
    assert "calibrate_geometry" in seen["system"]


def test_full_action_set_is_offered_by_default():
    seen = {}

    def fake(system: str, user: str):
        seen["system"] = system
        return '{"action": "stop", "target": null, "reason": "x"}', {"input_tokens": 1, "output_tokens": 1}

    LLMPlannerSelector(client=fake).decide(obs())
    for action in ACTIONS:
        assert action in seen["system"]


def test_per_arm_overrides_reach_make_selector():
    """Table 10 ablation: llm_minimal hides tools, llm_cheap changes the model, llm_memory adds lessons."""
    from horizon.harness.selector_experiment import make_selector

    cfg = {
        "llm": {"provider": "openai", "base_url": "https://x/v1", "model": "big",
                "api_key_env": "HORIZON_LLM_KEY"},
        "llm_overrides": {
            "llm_minimal": {"allowed_actions": ["calibrate_geometry", "dagger_round", "stop"]},
            "llm_cheap": {"model": "small"},
            "llm_memory": {"memory": ["past: targeted helped"]},
        },
    }
    assert make_selector("llm", "healthy", cfg).model == "big"
    assert make_selector("llm_cheap", "healthy", cfg).model == "small"
    assert make_selector("llm_minimal", "healthy", cfg).allowed_actions == (
        "calibrate_geometry", "dagger_round", "stop")
    assert make_selector("llm_memory", "healthy", cfg).memory == ("past: targeted helped",)
