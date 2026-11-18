"""State-based expert for the pick_place skill, with built-in recovery behaviour.

Like ARSTAG's expert programs (approach, grasp, lift, transfer, release with targets
computed from current object states), but written as a *Markov* policy: the phase is
re-derived from the privileged state at every step. That gives three things for free:
  * recovery primitives (re-grasp after a slip, re-place after a bad placement, release
    of a wrong object) without extra code paths,
  * relabelling of any state a learned policy visits (DAgger-style supervision),
  * corrective continuations from restored snapshots (RSI regeneration operators).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..sim.env import EnvConfig
from ..sim.mjcf import MAX_FINGER, TCP_Z_MIN
from ..sim.scene import BIN_WALL, SceneSpec
from ..sim.state import WorldState, quat_yaw, wrap_angle
from ..taskgraph.predicates import PredicateEngine, select_subgoal
from ..taskgraph.schema import Subgoal, TaskGraph

MAX_OPEN = 2 * MAX_FINGER
YAW_LIMIT = math.pi / 2 + 0.1
PRESHAPE_MARGIN = 0.022  # pre-grasp opening beyond the grasp width
FINGER_THICK = 0.01
FINGER_HALF_W = 0.014  # half width of the finger strip across the closing axis, plus margin


def preshape_cmd(width: float) -> float:
    """Gripper command that opens the fingers to ``width + PRESHAPE_MARGIN`` (opening = MAX_OPEN/2 * (1 - g))."""
    return float(np.clip(1.0 - (width + PRESHAPE_MARGIN) / MAX_FINGER, -1.0, 0.0))


@dataclass
class ExpertConfig:
    # decision tolerances sit above the perception noise (bias 3 mm + jitter 1.5 mm, yaw 0.03 rad) so that
    # labels are consistent from the policy's point of view; the 11 mm pre-shape clearance absorbs them
    pos_tol: float = 0.008  # xy alignment before descending to grasp / place
    yaw_tol: float = 0.12
    pregrasp_dz: float = 0.07
    approach_far: float = 0.05  # beyond this xy distance, travel at safe height
    place_clear: float = 0.004  # release when object bottom is this far above its support
    bin_drop: float = 0.012  # release height above bin wall top
    open_margin: float = 0.008  # opening beyond grasp width before descending (pre-shape targets +0.022)
    travel_margin: float = 0.035
    min_travel_z: float = 0.13
    settle_vel: float = 0.004  # finger closing speed below which the grasp is "settled"


def _move(cur: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    v = goal - cur
    n = float(np.linalg.norm(v))
    if n > max_step:
        v = v * (max_step / n)
    return v / max_step


class PickPlaceExpert:
    def __init__(self, scene: SceneSpec, env_cfg: EnvConfig | None = None, cfg: ExpertConfig | None = None,
                 eng: PredicateEngine | None = None):
        self.scene = scene
        self.env_cfg = env_cfg or EnvConfig()
        self.cfg = cfg or ExpertConfig()
        self.eng = eng or PredicateEngine(scene)
        self.geom = self.eng.geom

    # ---------------------------------------------------------------- geometry
    def _finger_conflicts(self, i: int, s: WorldState, yaw: float) -> float:
        """Penetration depth of the pre-shaped fingers into neighbouring objects (0 = collision-free).

        Mirrors ARSTAG's c_scene grasp screening: the pre-shaped fingers sweep a strip along the
        gripper's closing axis; neighbours are approximated by their footprint circles.
        """
        axis = np.array([-math.sin(yaw), math.cos(yaw)])
        perp = np.array([math.cos(yaw), math.sin(yaw)])
        c = s.obj_pos[i, :2]
        psi = quat_yaw(s.obj_quat[i])
        reach = (self.geom.grasp_width(i, yaw, psi) + PRESHAPE_MARGIN) / 2 + FINGER_THICK + 0.004
        worst = 0.0
        for j in range(self.geom.n):
            if j == i or s.held == j:
                continue
            rel = s.obj_pos[j, :2] - c
            along = abs(float(rel @ axis))
            across = abs(float(rel @ perp))
            rj = self.geom.radius[j]
            if across < FINGER_HALF_W + rj:
                worst = max(worst, (reach + rj) - along)
        return max(0.0, worst)

    def grasp_yaw(self, i: int, s: WorldState) -> float:
        """Top-down grasp yaw: collision-free with neighbours first, then least rotation."""
        if self.geom.shape[i] == "cylinder":
            base = float(np.clip(s.grip_yaw, -YAW_LIMIT, YAW_LIMIT))
            cands = [base + m * math.pi / 4 for m in range(-4, 5)]
            psi = 0.0
        else:
            psi = quat_yaw(s.obj_quat[i])
            cands = [psi + k * math.pi / 2 for k in range(-3, 4)]
        best, best_key = float(s.grip_yaw), (1e9, 1e9)
        for c in cands:
            if abs(c) > YAW_LIMIT:
                continue
            if self.geom.shape[i] == "box" and self.geom.grasp_width(i, c, psi) > MAX_OPEN - 0.015:
                continue
            pen = self._finger_conflicts(i, s, c)
            key = (round(pen, 3), abs(c - s.grip_yaw))
            if key < best_key:
                best, best_key = float(c), key
        return best

    def grasp_z(self, i: int, s: WorldState) -> float:
        z, top = float(s.obj_pos[i, 2]), self.geom.top(i, s)
        return max(z, top - 0.02, TCP_Z_MIN)

    def needed_width(self, i: int, s: WorldState) -> float:
        return self.geom.grasp_width(i, s.grip_yaw, quat_yaw(s.obj_quat[i]))

    def travel_z(self, s: WorldState, carry: int) -> float:
        tops = [self.geom.top(j, s) for j in range(self.geom.n) if j != carry]
        for f in self.scene.fixtures:
            if f.kind == "bin":
                tops.append(f.pos[2] + f.size[2] + BIN_WALL)
            elif f.kind == "plate":
                tops.append(f.top_z)
        hi = max(tops) if tops else 0.0
        drop = 0.03
        if carry >= 0:
            drop = max(drop, float(s.grip_pos[2]) - self.geom.bottom(carry, s))
        return max(self.cfg.min_travel_z, hi + drop + self.cfg.travel_margin)

    def place_target(self, g: Subgoal, s: WorldState) -> tuple[np.ndarray, float, bool]:
        """(object xy goal, support top z, is_drop) for subgoal g in state s."""
        if g.target in self.eng.fix:
            f = self.eng.fix[g.target]
            if f.kind == "bin":
                return self._bin_spot(f, s, g), f.pos[2] + f.size[2] + BIN_WALL, True
            return np.array(f.pos[:2]), f.top_z, False
        j = self.eng.obj_idx[g.target]
        return s.obj_pos[j, :2].copy(), self.geom.top(j, s), False

    def _bin_spot(self, f, s: WorldState, g: Subgoal) -> np.ndarray:
        """Least-occupied quadrant of the bin, so later objects do not pile up."""
        ihx, ihy, _ = f.size
        c = np.array(f.pos[:2])
        spots = [c + np.array([dx, dy]) for dx in (-ihx / 2, ihx / 2) for dy in (-ihy / 2, ihy / 2)]
        me = self.eng.obj_idx[g.obj]
        others = [s.obj_pos[j, :2] for j in range(self.geom.n) if j != me and s.held != j]
        def crowd(p):
            return sum(math.exp(-np.sum((p - o) ** 2) / (2 * 0.03 ** 2)) for o in others)
        # deterministic tie-break keeps the expert Markov
        return min(spots, key=lambda p: (round(crowd(p), 6), p[0], p[1]))

    # ---------------------------------------------------------------- policy
    def act(self, s: WorldState, g: Subgoal) -> tuple[np.ndarray, str, bool]:
        """Return (action[5], phase, subgoal_done) for subgoal ``g`` in true state ``s``."""
        c = self.cfg
        ec = self.env_cfg
        i = self.eng.obj_idx[g.obj]
        tcp = s.grip_pos
        a = np.zeros(5)

        if s.held >= 0 and s.held != i:
            # wrong object in hand: put it down where it is, then lift
            a[4] = -1.0
            a[2] = 0.5
            return a, "release_wrong", False

        if s.held == i:
            if s.grip_open_vel < -c.settle_vel:
                a[4] = 1.0
                return a, "settle_grasp", False
            a[4] = 1.0
            goal_xy, support_top, is_drop = self.place_target(g, s)
            d_xy = goal_xy - s.obj_pos[i, :2]
            tz = self.travel_z(s, i)
            place_yaw = self._place_yaw(g, s)
            a[3] = float(np.clip(wrap_angle(place_yaw - s.grip_yaw) / ec.max_yaw_step, -1, 1))
            dist = float(np.linalg.norm(d_xy))
            if dist > c.pos_tol + (0.01 if is_drop else 0.0):
                if dist > 0.03 and tcp[2] < tz - 0.01:
                    goal = np.array([tcp[0], tcp[1], tz])
                    phase = "lift"
                elif dist > 0.03:
                    goal = np.array([tcp[0] + d_xy[0], tcp[1] + d_xy[1], tz])
                    phase = "transport"
                else:
                    goal = np.array([tcp[0] + d_xy[0], tcp[1] + d_xy[1], tcp[2]])
                    phase = "align_place"
                a[:3] = _move(tcp, goal, ec.max_step)
                return a, phase, False
            bottom = self.geom.bottom(i, s)
            want = support_top + (c.bin_drop if is_drop else c.place_clear)
            dz = want - bottom
            if dz < -0.004:
                goal = np.array([tcp[0] + d_xy[0], tcp[1] + d_xy[1], tcp[2] + dz])
                a[:3] = _move(tcp, goal, ec.max_step)
                return a, "descend_place", False
            a[4] = preshape_cmd(self.needed_width(i, s))
            return a, "release", False

        # object not in hand
        need = self.needed_width(i, s)
        if self.eng.effect(s, g):
            a[4] = preshape_cmd(need)
            top = self.geom.top(i, s)
            done = tcp[2] > top + self.eng.tol.retreat_dz and s.grip_open > need + 0.008
            goal = np.array([tcp[0], tcp[1], max(tcp[2], top + self.eng.tol.retreat_dz + 0.02)])
            a[:3] = _move(tcp, goal, ec.max_step)
            return a, "retreat", bool(done)

        if s.obj_vel is not None and float(np.linalg.norm(s.obj_vel[i])) > 0.05:
            # object still falling / sliding (after a release or a slip): wait with open fingers
            a[4] = preshape_cmd(need)
            a[2] = 0.2
            return a, "wait_settle", False
        gyaw = self.grasp_yaw(i, s)
        need = self.geom.grasp_width(i, gyaw, quat_yaw(s.obj_quat[i]))
        gz = self.grasp_z(i, s)
        obj_xy = s.obj_pos[i, :2]
        xy_err = float(np.linalg.norm(tcp[:2] - obj_xy))
        yaw_err = abs(wrap_angle(gyaw - s.grip_yaw))
        a[3] = float(np.clip(wrap_angle(gyaw - s.grip_yaw) / ec.max_yaw_step, -1, 1))
        a[4] = preshape_cmd(need)
        low = tcp[2] < gz + c.pregrasp_dz - 0.015
        aligned = (xy_err <= c.pos_tol and yaw_err <= c.yaw_tol) or \
            (low and xy_err <= 1.4 * c.pos_tol and yaw_err <= 1.5 * c.yaw_tol)
        if not aligned:
            far = xy_err > c.approach_far
            tz = self.travel_z(s, -1) if far else gz + c.pregrasp_dz
            if far and tcp[2] < tz - 0.02:
                goal = np.array([tcp[0], tcp[1], tz])  # rise before travelling
                phase = "rise"
            elif not far and tcp[2] < gz + 0.03 and xy_err > 0.025:
                goal = np.array([tcp[0], tcp[1], gz + c.pregrasp_dz])
                phase = "rise"
            else:
                goal = np.array([obj_xy[0], obj_xy[1], tz])
                phase = "approach"
            a[:3] = _move(tcp, goal, ec.max_step)
            return a, phase, False
        if s.grip_open < need + c.open_margin and tcp[2] > gz + 0.004:
            a[:3] = 0.0
            return a, "open", False
        if tcp[2] > gz + 0.004:
            goal = np.array([obj_xy[0], obj_xy[1], gz])
            a[:3] = _move(tcp, goal, ec.max_step)
            return a, "descend_grasp", False
        if s.grip_open < need - 0.006:
            # fingers closed past the object: missed grasp -> reopen in place
            a[2] = 0.3
            return a, "reopen", False
        a[:2] = _move(tcp[:2], obj_xy, ec.max_step)[:2] * 0.5
        a[4] = 1.0
        return a, "close", False

    def _place_yaw(self, g: Subgoal, s: WorldState) -> float:
        if g.place_yaw is None:
            return float(s.grip_yaw)
        best = g.place_yaw
        for cand in (g.place_yaw, g.place_yaw + math.pi, g.place_yaw - math.pi):
            if abs(cand) <= YAW_LIMIT and abs(cand - s.grip_yaw) < abs(best - s.grip_yaw):
                best = cand
        return float(best)


class TaskExpert:
    """Task-level expert: privileged System 2 (``select_subgoal``) + skill expert."""

    def __init__(self, scene: SceneSpec, graph: TaskGraph, env_cfg: EnvConfig | None = None,
                 cfg: ExpertConfig | None = None, eng: PredicateEngine | None = None):
        self.graph = graph
        self.eng = eng or PredicateEngine(scene)
        self.skill = PickPlaceExpert(scene, env_cfg, cfg, self.eng)
        self.order = graph.order()
        self.current: int | None = None

    def reset(self) -> None:
        self.current = None

    def select(self, s: WorldState) -> int | None:
        k = select_subgoal(self.eng, s, self.graph, self.current, self.order)
        self.current = k
        return k

    def act(self, s: WorldState, k: int | None = None) -> tuple[np.ndarray, int | None, str]:
        if k is None:
            k = self.select(s)
        if k is None:
            a = np.zeros(5)
            a[4] = -1.0
            a[2] = 0.3 if s.grip_pos[2] < 0.15 else 0.0
            return a, None, "done"
        a, phase, _ = self.skill.act(s, self.graph.subgoals[k])
        return a, k, phase
