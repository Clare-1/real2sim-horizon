"""Scene specification: the hand-off format between real2sim reconstruction and simulation.

A ``SceneSpec`` is what a reconstruction agent (ARSTAG-style task-scoped reconstruction,
Agentic RSR, Real2Gym, ...) would hand to the data generator: task-relevant movable
objects with physical parameters, static fixtures (slots, bins, plates) and the table.
Procedural task families build the same structure, so every downstream module is
agnostic to whether a scene came from a photograph or from a sampler.

Frame convention: table top is the plane z = 0, the table is centred at the origin and
the (virtual) robot base sits at x = -0.5 facing +x.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Shape = Literal["box", "cylinder"]
FixtureKind = Literal["slot", "bin", "plate"]


@dataclass
class ObjectSpec:
    """A movable rigid object (free joint in simulation)."""

    name: str
    shape: Shape
    # box: half extents (hx, hy, hz); cylinder: (radius, half_height)
    size: tuple[float, ...]
    pos: tuple[float, float, float]
    yaw: float = 0.0
    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)
    mass: float = 0.08
    friction: float = 1.0
    category: str = "block"
    color: str = "red"

    @property
    def half_height(self) -> float:
        return float(self.size[2] if self.shape == "box" else self.size[1])

    @property
    def footprint_radius(self) -> float:
        if self.shape == "box":
            return float(math.hypot(self.size[0], self.size[1]))
        return float(self.size[0])

    def grasp_candidates(self) -> list[tuple[float, float]]:
        """(yaw offset relative to object yaw, finger opening needed) for a top-down grasp.

        Fingers close along the gripper's local y axis. A box grasped with gripper yaw equal
        to the object yaw squeezes the object's y extent; rotating by pi/2 squeezes x.
        Cylinders are rotationally symmetric, so a single candidate covers all yaws.
        """
        if self.shape == "cylinder":
            return [(0.0, 2.0 * self.size[0])]
        hx, hy = self.size[0], self.size[1]
        return [
            (0.0, 2.0 * hy),
            (math.pi / 2, 2.0 * hx),
            (math.pi, 2.0 * hy),
            (-math.pi / 2, 2.0 * hx),
        ]


@dataclass
class FixtureSpec:
    """A static scene element used as a manipulation target."""

    name: str
    kind: FixtureKind
    pos: tuple[float, float, float]
    # slot: (hx, hy) footprint; bin: (inner_hx, inner_hy, wall_height); plate: (hx, hy, hz)
    size: tuple[float, ...]
    yaw: float = 0.0
    rgba: tuple[float, float, float, float] = (0.3, 0.6, 0.3, 0.35)
    color: str = ""

    @property
    def top_z(self) -> float:
        """Height of the surface an object placed on this fixture rests on."""
        if self.kind == "plate":
            return float(self.pos[2] + 2.0 * self.size[2])
        if self.kind == "bin":
            return float(self.pos[2] + BIN_FLOOR)
        return float(self.pos[2])


BIN_FLOOR = 0.006  # bin floor thickness (m)
BIN_WALL = 0.006  # bin wall thickness (m)


@dataclass
class SceneSpec:
    scene_id: str
    objects: list[ObjectSpec]
    fixtures: list[FixtureSpec] = field(default_factory=list)
    table_half: tuple[float, float] = (0.45, 0.6)
    gripper_home: tuple[float, float, float, float] = (0.0, 0.0, 0.25, 0.0)
    # global physics multipliers; reconstruction-error studies perturb these
    physics: dict = field(default_factory=dict)
    provenance: str = "procedural"

    def object_index(self, name: str) -> int:
        for i, o in enumerate(self.objects):
            if o.name == name:
                return i
        raise KeyError(f"unknown object {name!r}")

    def fixture(self, name: str) -> FixtureSpec:
        for f in self.fixtures:
            if f.name == name:
                return f
        raise KeyError(f"unknown fixture {name!r}")

    def has(self, name: str) -> bool:
        return any(o.name == name for o in self.objects) or any(f.name == name for f in self.fixtures)

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "SceneSpec":
        objs = [ObjectSpec(**{**o, "size": tuple(o["size"]), "pos": tuple(o["pos"]), "rgba": tuple(o["rgba"])}) for o in d["objects"]]
        fixs = [FixtureSpec(**{**f, "size": tuple(f["size"]), "pos": tuple(f["pos"]), "rgba": tuple(f["rgba"])}) for f in d.get("fixtures", [])]
        return SceneSpec(
            scene_id=d["scene_id"],
            objects=objs,
            fixtures=fixs,
            table_half=tuple(d.get("table_half", (0.45, 0.6))),
            gripper_home=tuple(d.get("gripper_home", (0.0, 0.0, 0.25, 0.0))),
            physics=dict(d.get("physics", {})),
            provenance=d.get("provenance", "procedural"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> "SceneSpec":
        return SceneSpec.from_dict(json.loads(Path(path).read_text()))
