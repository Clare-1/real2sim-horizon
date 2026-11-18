"""Selector interface: leak-proof input, and the two causes being distinguishable from feedback."""
from __future__ import annotations

import pytest

from horizon.rsi.selector import (
    ACTIONS, CAUSE_ACTION, Budget, Decision, FailureProfile, FixedRecipeSelector, HistoryEntry,
    Observation, OracleSelector, RulePlannerSelector, first_step_correct,
)


def obs(residual=None, conc=None, labels=0, sr=0.4, rounds=0, pert=0.0, recovery=0, history=()):
    return Observation(
        profile=FailureProfile(n_episodes=24, per_subgoal_failure=conc or {}, calibration_residual=residual,
                               recovery_triggers=recovery, perturbed_failure_rate=pert, sr=sr),
        budget=Budget(deploy_spent=0, deploy_total=72, labels_spent=labels, labels_total=200_000, rounds_done=rounds),
        history=tuple(history),
    )


def test_observation_rejects_leaked_condition():
    """A fixture that passes the injected cause must fail loudly, not be quietly ignored."""
    with pytest.raises(ValueError, match="leaks forbidden keys"):
        Observation.from_dict({"profile": {}, "budget": {}, "condition": "height_bias"})
    with pytest.raises(ValueError, match="leaks forbidden keys"):
        Observation.from_dict({"profile": {}, "budget": {}, "true_cause": "geometry_bias"})
    ok = Observation.from_dict({"profile": {"sr": 0.5}, "budget": {"deploy_total": 72}})
    assert ok.profile.sr == 0.5


def test_two_causes_are_distinguishable_from_feedback_alone():
    """Geometry shows a large calibration residual; coverage shows concentration with no residual."""
    rule = RulePlannerSelector()
    geometry = obs(residual=0.014, conc={"slot0": 8, "slot1": 2})
    coverage = obs(residual=0.0004, conc={"slot0": 2, "slot3": 12})
    assert rule.decide(geometry).action == "calibrate_geometry"
    assert rule.decide(coverage).action == "regenerate_targeted"
    assert rule.decide(geometry).action != rule.decide(coverage).action


def test_fixed_recipe_never_diagnoses():
    """The baseline the agent must beat: same response regardless of the cause."""
    fixed = FixedRecipeSelector()
    geometry, coverage = obs(residual=0.014), obs(residual=0.0004, conc={"slot3": 12})
    assert fixed.decide(geometry).action == fixed.decide(coverage).action == "calibrate_geometry"
    assert fixed.decide(obs(rounds=1)).action == "dagger_round"


def test_oracle_is_correct_by_construction():
    for cause, action in CAUSE_ACTION.items():
        sel = OracleSelector(true_cause=cause)
        if cause == "healthy":
            assert sel.decide(obs(sr=0.3)).action == "stop"
        assert first_step_correct(Decision(action, None, "", 0), cause)


def test_rule_beats_fixed_on_coverage_deficit():
    """The core claim of the minimal validation, at the decision level."""
    rule, fixed = RulePlannerSelector(), FixedRecipeSelector()
    coverage = obs(residual=0.0004, conc={"slot3": 12})
    assert first_step_correct(rule.decide(coverage), "coverage_deficit")
    assert not first_step_correct(fixed.decide(coverage), "coverage_deficit")


def test_stops_when_budget_exhausted_or_stalled():
    rule = RulePlannerSelector(patience=2)
    assert rule.decide(Observation(FailureProfile(sr=0.3), Budget(deploy_spent=72, deploy_total=72))).action == "stop"
    # two consecutive rounds without any gain -> stop
    stalled = (HistoryEntry("dagger_round", 0.3, 0.3), HistoryEntry("dagger_round", 0.3, 0.3))
    assert rule.decide(obs(history=stalled)).action == "stop"
    # a single improving round must not be treated as stalled; use a profile that is clearly
    # localised so the "nothing localised -> stop" branch does not pre-empt it
    improving = (HistoryEntry("dagger_round", 0.3, 0.3), HistoryEntry("dagger_round", 0.3, 0.31))
    localised = obs(conc={"slot0": 9, "slot1": 1}, labels=50_000, history=improving)
    assert rule.decide(localised).action == "dagger_round"


def test_healthy_profile_stops_instead_of_updating():
    """A healthy twin shows no geometry residual and diffuse failures: do nothing."""
    rule = RulePlannerSelector()
    healthy = obs(conc={"slot0": 3, "slot1": 3, "slot2": 3}, sr=0.9)
    assert rule.decide(healthy).action == "stop"


def test_slip_failures_do_not_trigger_randomization():
    """This platform has no gradable dynamics cause, so the rule must not claim one.

    Friction (1.0->0.01), finger friction (1.5->0.1) and mass (1x->8x) leave SR unchanged and
    latency is binary, so a slip-heavy profile is indistinguishable from an already-correct twin.
    Keeping a slip branch only produced misfires on diffuse failures (see fig. 4 chains).
    """
    rule = RulePlannerSelector()
    slipping = Observation(
        FailureProfile(n_episodes=24, per_subgoal_failure={"slot0": 12}, sr=0.4,
                       failure_types={"drop": 8, "grasp": 5, "stall": 2}),
        Budget(deploy_total=72), ())
    assert rule.decide(slipping).action != "randomize_physics"


def test_oracle_follows_a_sequence_not_a_single_action():
    """A single-action oracle never trains on a wrong twin and scores below the baselines."""
    geo = OracleSelector(true_cause="geometry_bias")
    first = geo.decide(obs(residual=0.014))
    assert first.action == "calibrate_geometry"
    after = geo.decide(obs(residual=0.0004, history=(HistoryEntry("calibrate_geometry", 0.0, 0.0),)))
    assert after.action == "regenerate_uniform", "oracle must move on to data once the twin is fixed"
    later = geo.decide(obs(residual=0.0004, history=(
        HistoryEntry("calibrate_geometry", 0.0, 0.0), HistoryEntry("regenerate_uniform", 0.0, 0.2))))
    assert later.action == "regenerate_uniform", "then it repeats the cause's best operation"
    cov = OracleSelector(true_cause="coverage_deficit")
    assert cov.decide(obs()).action == "regenerate_targeted", "coverage needs no calibration"
    healthy = OracleSelector(true_cause="healthy")
    assert healthy.decide(obs(sr=0.3)).action == "stop"


def test_v2_reads_perturbation_pressure_from_the_log_not_from_unset_metadata():
    """The legacy signal is dead: log.meta["perturbed"] is never written and no "recover" event
    is ever emitted. v2 must therefore use the measured failure rate under perturbations."""
    v2 = RulePlannerSelector(recovery_signal="v2")
    prof = FailureProfile(n_episodes=24, per_subgoal_failure={"slot0": 8, "slot1": 2},
                          perturbation_events=11, perturbation_failure_rate=0.64,
                          sr=0.4, calibration_residual=0.0005)
    assert v2.decide(Observation(prof, Budget(deploy_total=72))).action == "add_recovery_data"
    # the same numbers with the legacy signal must not produce a recovery action
    legacy = RulePlannerSelector()
    assert legacy.decide(Observation(prof, Budget(deploy_total=72))).action != "add_recovery_data"


def test_v2_does_not_trigger_recovery_when_perturbations_are_survived():
    v2 = RulePlannerSelector(recovery_signal="v2")
    prof = FailureProfile(n_episodes=24, per_subgoal_failure={"slot0": 3, "slot1": 3, "slot2": 3},
                          perturbation_events=11, perturbation_failure_rate=0.18,
                          sr=0.8, calibration_residual=0.0005)
    assert v2.decide(Observation(prof, Budget(deploy_total=72))).action == "stop"
    # and a profile with no perturbations at all cannot claim a recovery gap
    quiet = FailureProfile(n_episodes=24, per_subgoal_failure={"slot0": 3, "slot1": 3},
                           perturbation_events=0, perturbation_failure_rate=0.0, sr=0.8,
                           calibration_residual=0.0005)
    assert v2.decide(Observation(quiet, Budget(deploy_total=72))).action != "add_recovery_data"


def test_every_action_is_reachable_and_valid():
    assert set(CAUSE_ACTION.values()) <= set(ACTIONS)
    with pytest.raises(ValueError, match="unknown action"):
        Decision("make_coffee", None, "", 0)
