"""Executable success predicates (paper Appendix A) and subgoal bookkeeping.

The same functions run on the privileged simulator state (ground truth, used for oracle
attribution and final scoring) and on the perceived state with the verifier's own
tolerances (used by System 2 and by automatic attribution).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sim.scene import BIN_WALL, SceneSpec
from ..sim.state import Geometry, WorldState
from .schema import Pred, Subgoal, TaskGraph


@dataclass
class Tolerances:
    slot_xy: float = 0.02
    stack_xy: float = 0.02
    plate_xy: float = 0.025
    rest_z: float = 0.012
    tilt: float = math.radians(15)
    bin_margin: float = 0.003
    retreat_dz: float = 0.03  # TCP height above the placed object before the subgoal counts as done

    def scaled(self, k: float) -> "Tolerances":
        """Uniformly loosen (k > 1) or tighten (k < 1) the geometric tolerances."""
        return Tolerances(self.slot_xy * k, self.stack_xy * k, self.plate_xy * k, self.rest_z * k,
                          self.tilt * k, self.bin_margin, self.retreat_dz)


class PredicateEngine:
    def __init__(self, scene: SceneSpec, tol: Tolerances | None = None):
        self.scene = scene
        self.geom = Geometry(scene)
        self.tol = tol or Tolerances()
        self.obj_idx = {o.name: i for i, o in enumerate(scene.objects)}
        self.fix = {f.name: f for f in scene.fixtures}

    # ---------------------------------------------------------------- primitives
    def _xy(self, s: WorldState, i: int) -> np.ndarray:
        return s.obj_pos[i, :2]

    def at_slot(self, s: WorldState, obj: str, slot: str) -> bool:
        i = self.obj_idx[obj]
        f = self.fix[slot]
        t = self.tol
        if s.held == i:
            return False
        if np.linalg.norm(self._xy(s, i) - np.array(f.pos[:2])) > t.slot_xy:
            return False
        if abs(self.geom.bottom(i, s) - f.top_z) > t.rest_z:
            return False
        return self.geom.tilt_state(i, s) < t.tilt

    def in_bin(self, s: WorldState, obj: str, bin_name: str) -> bool:
        i = self.obj_idx[obj]
        f = self.fix[bin_name]
        if s.held == i:
            return False
        ihx, ihy, wall_h = f.size
        rel = self._xy(s, i) - np.array(f.pos[:2])
        c, sn = math.cos(-f.yaw), math.sin(-f.yaw)
        lx, ly = c * rel[0] - sn * rel[1], sn * rel[0] + c * rel[1]
        m = self.tol.bin_margin
        if abs(lx) > ihx - m or abs(ly) > ihy - m:
            return False
        return f.pos[2] - 0.01 < s.obj_pos[i, 2] < f.pos[2] + wall_h + BIN_WALL

    def on(self, s: WorldState, obj: str, support: str) -> bool:
        i = self.obj_idx[obj]
        t = self.tol
        if s.held == i:
            return False
        if support in self.fix:
            f = self.fix[support]
            sxy, stop, tol_xy = np.array(f.pos[:2]), f.top_z, t.plate_xy
        else:
            j = self.obj_idx[support]
            sxy, stop, tol_xy = self._xy(s, j), self.geom.top(j, s), t.stack_xy
        if np.linalg.norm(self._xy(s, i) - sxy) > tol_xy:
            return False
        if abs(self.geom.bottom(i, s) - stop) > t.rest_z:
            return False
        return self.geom.tilt_state(i, s) < t.tilt

    def clear(self, s: WorldState, obj: str) -> bool:
        """Nothing rests on top of ``obj``."""
        i = self.obj_idx[obj]
        top = self.geom.top(i, s)
        for j in range(self.geom.n):
            if j == i or s.held == j:
                continue
            if np.linalg.norm(self._xy(s, j) - self._xy(s, i)) < self.geom.radius[i] * 0.9 and \
                    abs(self.geom.bottom(j, s) - top) < 0.015:
                return False
        return True

    def eval(self, s: WorldState, p: Pred) -> bool:
        if p.name == "at_slot":
            return self.at_slot(s, *p.args)
        if p.name == "in_bin":
            return self.in_bin(s, *p.args)
        if p.name == "on":
            return self.on(s, *p.args)
        if p.name == "clear":
            return self.clear(s, *p.args)
        if p.name == "held":
            return s.held == self.obj_idx[p.args[0]]
        raise ValueError(f"unknown predicate {p.name}")

    # ---------------------------------------------------------------- subgoals
    def effect(self, s: WorldState, g: Subgoal) -> bool:
        return self.eval(s, g.effect)

    def retreated(self, s: WorldState, g: Subgoal) -> bool:
        i = self.obj_idx[g.obj]
        return s.grip_pos[2] > self.geom.top(i, s) + self.tol.retreat_dz

    def preconditions(self, s: WorldState, g: Subgoal) -> list[Pred]:
        """Return the violated preconditions of ``g`` (empty list if all hold)."""
        return [p for p in g.pre if not self.eval(s, p)]

    def satisfied_vector(self, s: WorldState, graph: TaskGraph) -> np.ndarray:
        return np.array([self.effect(s, g) for g in graph.subgoals], dtype=bool)

    def task_success(self, s: WorldState, graph: TaskGraph) -> bool:
        return s.held < 0 and bool(self.satisfied_vector(s, graph).all())


def select_subgoal(eng: PredicateEngine, s: WorldState, graph: TaskGraph, current: int | None,
                   order: list[int] | None = None, sat: np.ndarray | None = None) -> int | None:
    """System-2 subgoal selection over a task graph.

    Keeps working on the subgoal whose object is in hand; otherwise returns the first
    subgoal (topological order) that is unsatisfied, or the current subgoal while its
    gripper retreat is unfinished. Earlier subgoals that regress (e.g. a knocked-over
    object) are therefore revisited automatically. Returns None when the task is complete.
    ``sat`` optionally supplies (debounced) effect values instead of evaluating them on ``s``.
    """
    order = order if order is not None else graph.order()

    def done(k: int) -> bool:
        return bool(sat[k]) if sat is not None else eng.effect(s, graph.subgoals[k])

    if s.held >= 0:
        for k in order:
            g = graph.subgoals[k]
            if eng.obj_idx[g.obj] == s.held and not done(k):
                return k
    for k in order:
        g = graph.subgoals[k]
        if not done(k):
            return k
        if k == current and not eng.retreated(s, g):
            return k
    return None


class Debouncer:
    """Holds each predicate's value until a change persists for ``steps`` consecutive reads.

    Perception noise makes a predicate near its tolerance flicker; without debouncing,
    System 2 chatters between an "almost placed" subgoal and the next one and finishes neither.
    """

    def __init__(self, steps: int = 4):
        self.steps = steps
        self.value: np.ndarray | None = None
        self.count: np.ndarray | None = None

    def reset(self) -> None:
        self.value = self.count = None

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=bool)
        if self.value is None:
            self.value = raw.copy()
            self.count = np.zeros(len(raw), dtype=int)
            return self.value.copy()
        diff = raw != self.value
        self.count = np.where(diff, self.count + 1, 0)
        flip = self.count >= self.steps
        self.value = np.where(flip, raw, self.value)
        self.count = np.where(flip, 0, self.count)
        return self.value.copy()
