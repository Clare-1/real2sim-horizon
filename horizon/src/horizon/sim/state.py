"""World state containers and geometry helpers shared by predicates, experts and policies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .scene import SceneSpec


@dataclass
class WorldState:
    """Snapshot of the quantities a verifier, expert or policy may read.

    The same container is used for the privileged simulator state (``noisy=False``) and for
    the perceived state (noisy poses, estimated ``held``). Positions are object centres.
    """

    obj_pos: np.ndarray  # (N, 3)
    obj_quat: np.ndarray  # (N, 4) wxyz
    grip_pos: np.ndarray  # (3,) tool centre point
    grip_yaw: float
    grip_open: float  # finger opening width (m)
    grip_open_vel: float  # d(opening)/dt (m/s)
    held: int  # index of the object between the fingers, -1 if none
    t: int = 0
    noisy: bool = False
    obj_vel: np.ndarray | None = None  # (N, 3) linear velocities (privileged state only)
    # per-state cache of rotation matrices / vertical extents (never copied; states are not mutated)
    _cache: dict | None = field(default=None, repr=False, compare=False)

    def copy(self) -> "WorldState":
        return WorldState(
            self.obj_pos.copy(), self.obj_quat.copy(), self.grip_pos.copy(), float(self.grip_yaw),
            float(self.grip_open), float(self.grip_open_vel), int(self.held), int(self.t), self.noisy,
            None if self.obj_vel is None else self.obj_vel.copy(),
        )


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_mat_batch(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz -> (N, 3, 3); elementwise identical to :func:`quat_to_mat`."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def quat_yaw(q: np.ndarray) -> float:
    R = quat_to_mat(q)
    return math.atan2(R[1, 0], R[0, 0])


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


class Geometry:
    """Static per-object geometry for a scene, with pose-dependent extent queries."""

    def __init__(self, scene: SceneSpec):
        self.scene = scene
        self.n = len(scene.objects)
        self.shape = [o.shape for o in scene.objects]
        self.size = [tuple(o.size) for o in scene.objects]
        self.half_h = np.array([o.half_height for o in scene.objects])
        self.radius = np.array([o.footprint_radius for o in scene.objects])
        self.cube_like = [
            o.shape == "box" and min(o.size) / max(o.size) > 0.7 for o in scene.objects
        ]
        self._is_box = np.array([sh == "box" for sh in self.shape], dtype=bool)
        sz = np.zeros((self.n, 3))
        for i, (sh, size) in enumerate(zip(self.shape, self.size)):
            sz[i] = size if sh == "box" else (size[0], size[1], 0.0)  # cylinder: (radius, half height)
        self._sz = sz

    def _cached(self, s: WorldState) -> dict:
        c = s._cache
        if c is None or c.get("geom") is not self:
            R = quat_to_mat_batch(np.asarray(s.obj_quat, dtype=float).reshape(-1, 4))
            box = np.abs(R[:, 2, 0]) * self._sz[:, 0] + np.abs(R[:, 2, 1]) * self._sz[:, 1] + np.abs(R[:, 2, 2]) * self._sz[:, 2]
            cc = np.minimum(1.0, np.abs(R[:, 2, 2]))
            cyl = cc * self._sz[:, 1] + np.sqrt(np.maximum(0.0, 1 - cc * cc)) * self._sz[:, 0]
            c = {"geom": self, "R": R, "ext": np.where(self._is_box, box, cyl)}
            s._cache = c
        return c

    def half_z_extent(self, i: int, quat: np.ndarray) -> float:
        R = quat_to_mat(quat)
        if self.shape[i] == "box":
            hx, hy, hz = self.size[i]
            return abs(R[2, 0]) * hx + abs(R[2, 1]) * hy + abs(R[2, 2]) * hz
        r, hh = self.size[i]
        c = min(1.0, abs(R[2, 2]))
        return c * hh + math.sqrt(max(0.0, 1 - c * c)) * r

    def bottom(self, i: int, s: WorldState) -> float:
        return float(s.obj_pos[i, 2] - self._cached(s)["ext"][i])

    def top(self, i: int, s: WorldState) -> float:
        return float(s.obj_pos[i, 2] + self._cached(s)["ext"][i])

    def tilt_state(self, i: int, s: WorldState) -> float:
        """Same as :meth:`tilt` for object i of state s, using the per-state rotation cache."""
        R = self._cached(s)["R"][i]
        if self.cube_like[i]:
            c = max(abs(R[2, 0]), abs(R[2, 1]), abs(R[2, 2]))
        else:
            c = abs(R[2, 2])
        return math.acos(min(1.0, c))

    def tilt(self, i: int, quat: np.ndarray) -> float:
        """Angle (rad) between the resting face normal and world z."""
        R = quat_to_mat(quat)
        if self.cube_like[i]:
            c = max(abs(R[2, 0]), abs(R[2, 1]), abs(R[2, 2]))
        else:
            c = abs(R[2, 2])
        return math.acos(min(1.0, c))

    def grasp_width(self, i: int, grip_yaw: float, obj_yaw: float) -> float:
        """Opening needed to grasp object i with the given gripper yaw (nearest candidate)."""
        if self.shape[i] == "cylinder":
            return 2.0 * self.size[i][0]
        rel = wrap_angle(grip_yaw - obj_yaw)
        hx, hy = self.size[i][0], self.size[i][1]
        # rel ~ 0 or pi squeezes y; rel ~ +-pi/2 squeezes x
        return 2.0 * (hy if abs(math.cos(rel)) > abs(math.sin(rel)) else hx)

    def min_width(self, i: int) -> float:
        if self.shape[i] == "cylinder":
            return 2.0 * self.size[i][0]
        return 2.0 * min(self.size[i][0], self.size[i][1])
