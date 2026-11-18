"""Perception-side verifier (paper Sec. 5.6).

Evaluates subgoal predicates on the *perceived* state (biased + jittered poses) with its
own tolerances, and estimates which object is in hand from proprioception. Its errors
(false success / false failure near tolerance boundaries) are a real failure source for
System 2, which is why verifier recalibration is one of the RSI operators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.scene import SceneSpec
from ..sim.state import WorldState, quat_yaw
from ..taskgraph.predicates import PredicateEngine, Tolerances
from ..taskgraph.schema import TaskGraph


@dataclass
class VerifierConfig:
    tol_scale: float = 1.0  # >1 loosens tolerances (more false successes), <1 tightens
    held_xy: float = 0.03
    held_z: float = 0.045
    held_width_slack: float = 0.012


class Verifier:
    def __init__(self, scene: SceneSpec, cfg: VerifierConfig | None = None, base_tol: Tolerances | None = None):
        self.cfg = cfg or VerifierConfig()
        self.base_tol = base_tol or Tolerances()
        self.eng = PredicateEngine(scene, self.base_tol.scaled(self.cfg.tol_scale))
        self.geom = self.eng.geom

    def set_scale(self, k: float) -> None:
        self.cfg.tol_scale = k
        self.eng.tol = self.base_tol.scaled(k)

    def estimate_held(self, s: WorldState) -> int:
        """Object between the fingers, judged from opening width and relative pose."""
        c = self.cfg
        best, best_d = -1, 1e9
        for i in range(self.geom.n):
            d_xy = float(np.linalg.norm(s.obj_pos[i, :2] - s.grip_pos[:2]))
            d_z = abs(float(s.obj_pos[i, 2] - s.grip_pos[2]))
            if d_xy > c.held_xy or d_z > c.held_z:
                continue
            w = self.geom.grasp_width(i, s.grip_yaw, quat_yaw(s.obj_quat[i]))
            if not (w - c.held_width_slack < s.grip_open < w + 0.004):
                continue
            if d_xy < best_d:
                best, best_d = i, d_xy
        return best

    def perceive(self, obs: WorldState) -> WorldState:
        s = obs.copy()
        s.held = self.estimate_held(obs)
        return s

    def satisfied(self, obs: WorldState, graph: TaskGraph) -> np.ndarray:
        return self.eng.satisfied_vector(self.perceive(obs), graph)

    def task_success(self, obs: WorldState, graph: TaskGraph) -> bool:
        s = self.perceive(obs)
        return s.held < 0 and bool(self.eng.satisfied_vector(s, graph).all())
