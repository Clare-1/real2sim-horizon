"""Operation interaction (conflict / order) tests: pre-registered two-operation sequences."""

from horizon.rsi.selector import Budget, FailureProfile, Observation
from horizon.rsi.sequences import SequenceSelector, parse_sequence_name


def obs(rounds_done=0):
    return Observation(FailureProfile(n_episodes=24, sr=0.4),
                       Budget(deploy_spent=24 * rounds_done, deploy_total=72, rounds_done=rounds_done))


def test_sequence_plays_actions_in_order_then_stops():
    sel = SequenceSelector(("calibrate_geometry", "regenerate_targeted"))
    assert sel.decide(obs(0)).action == "calibrate_geometry"
    first = sel.decide(obs(0))
    from horizon.rsi.selector import HistoryEntry
    o1 = Observation(obs(0).profile, obs(0).budget,
                     history=(HistoryEntry(first.action, 0.4, 0.7),))
    assert sel.decide(o1).action == "regenerate_targeted"
    o2 = Observation(obs(0).profile, obs(0).budget,
                     history=(HistoryEntry("calibrate_geometry", 0.4, 0.7),
                              HistoryEntry("regenerate_targeted", 0.7, 0.9)))
    assert sel.decide(o2).action == "stop"


def test_sequence_name_parsing_and_validation():
    assert parse_sequence_name("seq:calibrate_geometry+regenerate_targeted") == (
        "calibrate_geometry", "regenerate_targeted")
    import pytest
    with pytest.raises(ValueError, match="unknown actions"):
        parse_sequence_name("seq:calibrate_geometry+make_coffee")
    with pytest.raises(ValueError, match="not a sequence"):
        parse_sequence_name("fixed")
