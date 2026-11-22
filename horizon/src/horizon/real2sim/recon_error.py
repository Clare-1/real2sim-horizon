"""Systematic reconstruction errors of a single-image real2sim pipeline, and their correction.

``RealWorld`` defines the physical scene the robot is deployed in: compared with the procedural
task families it adds height to the supporting surfaces - a raised tray under the packing slots
and a thick cutting board instead of a thin plate.

A ``ReconModel`` turns that true scene into the scene the robot *believes* in: the one its
simulator is built from, its verifier measures with and its features are computed against.
Errors are per category (object categories, or the support fixtures "plate" / "tray") and
multiplicative on the vertical extent - support heights are what single-view reconstruction
gets wrong most (ARSTAG Table III: 34 mm median support gap before repair). ``corrections``
multiply on top of the biases and are what scene repair estimates; the ideal correction for a
category is 1 / bias.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..sim.scene import FixtureSpec, SceneSpec
from ..tasks.families import LAYERS

THIN_CATEGORIES = tuple(LAYERS)  # sandwich layers
PLATE = "plate"
TRAY = "tray"


@dataclass
class RealWorld:
    """Physical modifications of the procedural scenes (the deployment world)."""

    plate_half: float = 0.0  # plate half thickness override (0: keep); e.g. 0.01 = a 2 cm board
    tray_height: float = 0.0  # raised tray under the packing slots (0: slots on the table)
    # Global multiplier on object friction in the deployment world. The twin keeps the nominal
    # value, so 1.0 means "no dynamics gap" and e.g. 0.55 means the real table is much slipperier.
    friction_scale: float = 1.0
    # Rigid translation (metres, +x) of every target fixture in the deployment world. The twin
    # keeps the nominal layout, so this is a *position* gap rather than a height gap.
    lateral_offset: float = 0.0
    # Global multiplier on object mass in the deployment world. Heavier objects defeat the gripper's
    # force closure, which is the dynamics gap that actually moves the success rate here.
    mass_scale: float = 1.0
    # Control steps of actuation delay in the deployment world (ARSTAG-style latency). The twin
    # assumes zero delay, so this is a dynamics gap the calibration operators cannot see.
    latency: int = 0
    # Gripper pad friction in the deployment world. Objects are held by finger friction, so this -
    # not the object-table friction - is the knob that creates a real dynamics gap.
    finger_friction: float | None = None

    def apply(self, scene: SceneSpec) -> SceneSpec:
        # NOTE: every block below must appear exactly once. Duplicated copies of this method used to
        # add ``lateral_offset`` twice, so a configured 2 cm position gap was deployed as 4 cm.
        sc = copy.deepcopy(scene)
        physics = dict(sc.physics or {})
        if self.friction_scale != 1.0:
            physics["friction_scale"] = float(self.friction_scale)
        if self.mass_scale != 1.0:
            physics["mass_scale"] = float(self.mass_scale)
        if physics:
            sc.physics = physics
        if self.lateral_offset:
            for f in sc.fixtures:
                if f.kind in ("plate", "bin", "slot"):
                    f.pos = (f.pos[0] + float(self.lateral_offset), f.pos[1], f.pos[2])
        if self.plate_half > 0:
            for f in sc.fixtures:
                if f.kind == "plate":
                    f.size = (f.size[0], f.size[1], self.plate_half)
        slots = [f for f in sc.fixtures if f.kind == "slot"]
        if self.tray_height > 0 and slots:
            xs = [f.pos[0] for f in slots]
            ys = [f.pos[1] for f in slots]
            hx = (max(xs) - min(xs)) / 2 + 0.05
            hy = (max(ys) - min(ys)) / 2 + 0.05
            sc.fixtures.append(FixtureSpec(name=TRAY, kind="plate", pos=((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0.0),
                                           size=(hx, hy, self.tray_height / 2), rgba=(0.55, 0.45, 0.35, 1.0)))
            for f in slots:
                f.pos = (f.pos[0], f.pos[1], self.tray_height)
        return sc

    def to_dict(self) -> dict:
        return {"plate_half": self.plate_half, "tray_height": self.tray_height,
                "friction_scale": self.friction_scale, "lateral_offset": self.lateral_offset,
                "mass_scale": self.mass_scale, "latency": self.latency,
                "finger_friction": self.finger_friction}

    @staticmethod
    def from_dict(d: dict | None) -> "RealWorld":
        return RealWorld(**(d or {}))


@dataclass
class ReconModel:
    bias: dict[str, float] = field(default_factory=dict)  # category (or "plate") -> recon/real height factor
    corrections: dict[str, float] = field(default_factory=dict)  # estimated by scene repair

    def factor(self, category: str) -> float:
        return float(self.bias.get(category, 1.0)) * float(self.corrections.get(category, 1.0))

    def apply(self, real: SceneSpec) -> SceneSpec:
        """Believed scene: heights scaled per category, objects re-seated on the table (support repair)."""
        sc = copy.deepcopy(real)
        for o in sc.objects:
            f = self.factor(o.category)
            if f == 1.0:
                continue
            if o.shape == "box":
                o.size = (o.size[0], o.size[1], o.size[2] * f)
                z = o.size[2]
            else:
                o.size = (o.size[0], o.size[1] * f)
                z = o.size[1]
            o.pos = (o.pos[0], o.pos[1], z)
        for fx in sc.fixtures:
            if fx.kind != "plate":
                continue
            f = self.factor(TRAY if fx.name == TRAY else PLATE)
            if f != 1.0:
                fx.size = (fx.size[0], fx.size[1], fx.size[2] * f)
            if fx.name == TRAY:  # slots sit on the (believed) tray top
                for s in sc.fixtures:
                    if s.kind == "slot":
                        s.pos = (s.pos[0], s.pos[1], fx.top_z)
        sc.provenance = f"recon({real.provenance})"
        return sc

    def is_identity(self) -> bool:
        return all(abs(self.factor(c) - 1.0) < 1e-9 for c in set(self.bias) | set(self.corrections))

    def to_dict(self) -> dict:
        return {"bias": dict(self.bias), "corrections": dict(self.corrections)}

    @staticmethod
    def from_dict(d: dict | None) -> "ReconModel":
        d = d or {}
        return ReconModel(dict(d.get("bias", {})), dict(d.get("corrections", {})))


def single_view_bias(thin: float = 1.0, plate: float = 0.3, tray: float = 0.3) -> ReconModel:
    """The injected gap: support surfaces (cutting board, tray) reconstructed far too low."""
    b = {c: thin for c in THIN_CATEGORIES if thin != 1.0}
    b[PLATE] = plate
    b[TRAY] = tray
    return ReconModel(bias=b)


DEPLOY_WORLD = RealWorld(plate_half=0.01, tray_height=0.02)
