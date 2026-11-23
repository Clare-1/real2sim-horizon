"""Policy-free diagnosis: the signal -> action map and the selector built on it."""

import pytest

from horizon.rsi.policy_free import (PolicyFreeSelector, ProbeSignals, ProbeThresholds, diagnose)
from horizon.rsi.selector import Budget, FailureProfile, Observation


def obs():
    return Observation(FailureProfile(n_episodes=24, sr=0.3), Budget(deploy_spent=24, deploy_total=72))


def test_measured_signals_map_to_the_expected_actions():
    """The four causes as measured on 2026-11-23 at 3 banks x 10 episodes (README 5.12)."""
    geometry = ProbeSignals(twin_cost=1.00, knowledge_gap=-0.30, perturbation_drop=0.30,
                            selfcheck_bias=0.50)
    coverage = ProbeSignals(twin_cost=0.00, knowledge_gap=0.90, perturbation_drop=-0.10,
                            selfcheck_bias=0.00)
    recovery = ProbeSignals(twin_cost=0.00, knowledge_gap=0.00, perturbation_drop=1.00,
                            selfcheck_bias=0.00)
    healthy = ProbeSignals(twin_cost=0.00, knowledge_gap=0.30, perturbation_drop=0.30,
                           selfcheck_bias=0.00)
    assert diagnose(geometry)[0] == "calibrate_geometry"
    assert diagnose(coverage)[0] == "regenerate_targeted"
    assert diagnose(recovery)[0] == "add_recovery_data"
    assert diagnose(healthy)[0] == "stop"


def test_thresholds_are_explicit_and_configurable():
    sig = ProbeSignals(twin_cost=0.2, knowledge_gap=0.4)
    assert diagnose(sig)[0] == "dagger_round"          # between stop (0.35) and targeted (0.5)
    assert diagnose(sig, ProbeThresholds(knowledge=0.3))[0] == "regenerate_targeted"
    assert diagnose(sig, ProbeThresholds(healthy=0.5))[0] == "stop"


def test_missing_signals_fail_loudly_instead_of_degrading():
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        diagnose(ProbeSignals())
    with pytest.raises(RuntimeError, match="needs a probe"):
        PolicyFreeSelector().decide(obs())


def test_selector_calls_the_probe_each_round_so_a_mixed_cause_is_re_diagnosed():
    seen = {"round": 0}

    def probe(_):
        seen["round"] += 1
        # round 1: the twin is wrong; after calibration the twin is fine but data is missing
        return ProbeSignals(twin_cost=1.0 if seen["round"] == 1 else 0.0,
                            knowledge_gap=0.0 if seen["round"] == 1 else 0.9,
                            perturbation_drop=0.0)

    sel = PolicyFreeSelector(probe=probe)
    assert sel.decide(obs()).action == "calibrate_geometry"
    assert sel.decide(obs()).action == "regenerate_targeted"
    assert seen["round"] == 2
