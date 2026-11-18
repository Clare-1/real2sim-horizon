"""Object-centric observation features for System 1 (hierarchical) and flat policies.

All features are computed from the *perceived* state with deterministic geometric
grounding (grasp yaw from a top-down grasp planner, target spot and travel height from the
scene graph) - the same information ARSTAG's expert programs and grasp proposals expose.
Nothing privileged (true poses, contacts, phases) enters a policy input.

The hierarchical policy is conditioned on one subgoal chosen by System 2; the flat policy
sees every subgoal's object and target in task order and must infer progress itself.
"""

from __future__ import annotations

import math

import numpy as np

from ..sim.scene import SceneSpec
from ..sim.state import WorldState, quat_yaw, wrap_angle
from ..skills.expert import PickPlaceExpert
from ..taskgraph.schema import TaskGraph

FAMILY_IDS = {"pack": 0, "sort": 1, "stack": 2}
H_NORM = 10.0
K_NEAR_TCP = 4
K_NEAR_SPOT = 3
MAX_SUBGOALS = 10
FLAT_SLOT = 10


class TaskFeaturizer:
    """Per-task feature builder (static geometry cached at construction)."""

    def __init__(self, scene: SceneSpec, graph: TaskGraph, verifier=None):
        self.scene, self.graph = scene, graph
        self.geo = PickPlaceExpert(scene)  # geometry helpers only; never queried with true state here
        self.geom = self.geo.geom
        self.verifier = verifier
        self.obj_of = [self.geo.eng.obj_idx[g.obj] for g in graph.subgoals]
        self.order = graph.order()
        self.fam = np.zeros(3)
        self.fam[FAMILY_IDS[graph.family]] = 1.0

    # ------------------------------------------------------------------ helpers
    def _held_est(self, s: WorldState) -> int:
        if self.verifier is not None:
            return self.verifier.estimate_held(s)
        return -1

    def _proprio(self, s: WorldState) -> list[float]:
        return [*s.grip_pos, math.sin(s.grip_yaw), math.cos(s.grip_yaw), s.grip_open * 10.0, s.grip_open_vel]

    def hier(self, s: WorldState, k: int) -> np.ndarray:
        g = self.graph.subgoals[k]
        i = self.obj_of[k]
        geo, geom = self.geo, self.geom
        tcp = s.grip_pos
        held = self._held_est(s)
        gyaw = geo.grasp_yaw(i, s)
        width = geom.grasp_width(i, gyaw, quat_yaw(s.obj_quat[i]))
        gz = geo.grasp_z(i, s)
        top_i = geom.top(i, s)
        bot_i = geom.bottom(i, s)
        spot, support_top, is_drop = geo.place_target(g, s)
        pyaw = geo._place_yaw(g, s)
        tz = geo.travel_z(s, held if held == i else -1)
        f: list[float] = self._proprio(s)
        f += [*(s.obj_pos[i] - tcp), wrap_angle(gyaw - s.grip_yaw), width * 10.0, geom.half_h[i] * 10.0,
              top_i - tcp[2], gz - tcp[2], 1.0 if held == i else 0.0]
        f += [spot[0] - tcp[0], spot[1] - tcp[1], support_top - tcp[2], spot[0] - s.obj_pos[i, 0],
              spot[1] - s.obj_pos[i, 1], bot_i - support_top, wrap_angle(pyaw - s.grip_yaw), float(is_drop)]
        f += [tz - tcp[2]]
        others = [j for j in range(geom.n) if j != i]
        near = sorted(others, key=lambda j: float(np.linalg.norm(s.obj_pos[j, :2] - tcp[:2])))[:K_NEAR_TCP]
        for j in near:
            f += [*(s.obj_pos[j] - tcp), geom.top(j, s) - tcp[2]]
        f += [0.0, 0.0, 0.0, 0.0, ] * (K_NEAR_TCP - len(near))
        near_s = sorted(others, key=lambda j: float(np.linalg.norm(s.obj_pos[j, :2] - spot)))[:K_NEAR_SPOT]
        for j in near_s:
            f += [s.obj_pos[j, 0] - spot[0], s.obj_pos[j, 1] - spot[1], geom.top(j, s) - support_top]
        f += [0.5, 0.5, 0.0] * (K_NEAR_SPOT - len(near_s))
        f += [*self.fam, k / H_NORM, self.graph.H / H_NORM]
        return np.asarray(f, dtype=np.float32)

    def flat(self, s: WorldState) -> np.ndarray:
        geo, geom = self.geo, self.geom
        tcp = s.grip_pos
        held = self._held_est(s)
        f: list[float] = self._proprio(s)
        f += [geo.travel_z(s, held) - tcp[2], 1.0 if held >= 0 else 0.0]
        for slot in range(MAX_SUBGOALS):
            if slot < len(self.order):
                k = self.order[slot]
                g = self.graph.subgoals[k]
                i = self.obj_of[k]
                gyaw = geo.grasp_yaw(i, s)
                spot, support_top, is_drop = geo.place_target(g, s)
                f += [*(s.obj_pos[i] - tcp), wrap_angle(gyaw - s.grip_yaw),
                      geom.grasp_width(i, gyaw, quat_yaw(s.obj_quat[i])) * 10.0, geo.grasp_z(i, s) - tcp[2],
                      spot[0] - s.obj_pos[i, 0], spot[1] - s.obj_pos[i, 1], geom.bottom(i, s) - support_top,
                      1.0 if held == i else 0.0]
            else:
                f += [0.0] * FLAT_SLOT
        f += [*self.fam, self.graph.H / H_NORM]
        return np.asarray(f, dtype=np.float32)


def state_from_log(grip_row: np.ndarray, obj_row: np.ndarray, t: int = 0) -> WorldState:
    return WorldState(
        obj_pos=obj_row[:, :3].astype(float), obj_quat=obj_row[:, 3:7].astype(float),
        grip_pos=grip_row[:3].astype(float), grip_yaw=float(grip_row[3]), grip_open=float(grip_row[4]),
        grip_open_vel=float(grip_row[5]), held=-1, t=t, noisy=True,
    )


HIER_DIM = 7 + 9 + 8 + 1 + 4 * K_NEAR_TCP + 3 * K_NEAR_SPOT + 5
FLAT_DIM = 7 + 2 + FLAT_SLOT * MAX_SUBGOALS + 4
