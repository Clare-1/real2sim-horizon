"""Pre-registered operation sequences (conflict / ordering experiments).

The interesting question is not "which single operation is best" but "does the order matter".
A ``SequenceSelector`` executes a fixed list of operations, one per round, and then stops.  Running
the same operations in different orders on the same cause measures whether they are additive,
super-additive (calibration enables targeted regeneration) or destructive (targeted without
calibration).

Selector names take the form ``seq:action1+action2`` (see ``parse_sequence_name``), so a config can
pre-register e.g. ``seq:calibrate_geometry+regenerate_targeted`` next to
``seq:regenerate_targeted+calibrate_geometry`` without new code.
"""
from __future__ import annotations

from dataclasses import dataclass

from .selector import ACTIONS, Decision, Observation


def parse_sequence_name(name: str) -> tuple[str, ...]:
    """``seq:a+b`` -> ("a", "b"); unknown actions are rejected at construction time."""
    if not name.startswith("seq:"):
        raise ValueError(f"not a sequence selector name: {name!r}")
    actions = tuple(a for a in name.split(":", 1)[1].split("+") if a)
    bad = [a for a in actions if a not in ACTIONS]
    if bad:
        raise ValueError(f"unknown actions in sequence {name!r}: {bad}")
    return actions


@dataclass
class SequenceSelector:
    """Plays a fixed list of operations, one per round, then stops."""

    actions: tuple[str, ...]
    name: str = "sequence"

    def __post_init__(self) -> None:
        bad = [a for a in self.actions if a not in ACTIONS]
        if bad:
            raise ValueError(f"unknown actions in sequence: {bad}")

    def decide(self, obs: Observation) -> Decision:
        i = len(obs.history)
        if i < len(self.actions):
            return Decision(self.actions[i], None, f"pre-registered step {i + 1}/{len(self.actions)}", 0)
        return Decision("stop", None, "pre-registered sequence exhausted", 0)
