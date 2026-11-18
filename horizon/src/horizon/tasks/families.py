"""Long-horizon task families with a controllable number of stages H (paper Sec. 6.2).

* ``pack``  (装箱/摆盘): H objects into H tray slots, in order; later items are placed
  next to earlier ones, so clutter grows with the stage index.
* ``sort``  (分类回收): H objects into colour-matched bins, in order.
* ``stack`` (做三明治): H thin layers stacked on a plate in order; placement error
  compounds because every layer rests on the previous one.

Each sampler draws a scene under *task-consistent randomisation* (ARSTAG Sec. III-B):
pick objects live in lateral regions, destinations in the shared central region, and
object size / mass / friction / layout offset are jittered without breaking task
relations. The instruction is then compiled into a task graph by the compiler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..sim.scene import FixtureSpec, ObjectSpec, SceneSpec

PALETTE = {
    "red": (0.85, 0.2, 0.2, 1.0),
    "green": (0.2, 0.7, 0.3, 1.0),
    "blue": (0.2, 0.35, 0.85, 1.0),
    "yellow": (0.9, 0.8, 0.2, 1.0),
    "purple": (0.6, 0.3, 0.75, 1.0),
    "orange": (0.95, 0.55, 0.15, 1.0),
    "cyan": (0.2, 0.75, 0.8, 1.0),
}

LAYERS = {
    "bread": ((0.035, 0.035, 0.008), (0.87, 0.72, 0.48, 1.0)),
    "ham": ((0.033, 0.033, 0.005), (0.92, 0.55, 0.6, 1.0)),
    "cheese": ((0.032, 0.032, 0.004), (0.98, 0.85, 0.25, 1.0)),
    "lettuce": ((0.034, 0.034, 0.004), (0.35, 0.75, 0.3, 1.0)),
    "tomato": ((0.031, 0.031, 0.005), (0.85, 0.2, 0.15, 1.0)),
    "egg": ((0.03, 0.03, 0.006), (0.97, 0.95, 0.85, 1.0)),
}

FAMILIES = ("pack", "sort", "stack")
H_MAX = {"pack": 10, "sort": 10, "stack": 10}


@dataclass
class RandomizationConfig:
    size_jitter: float = 0.1
    mass_jitter: float = 0.3
    friction_jitter: float = 0.2
    layout_offset: float = 0.02  # whole-layout shift (geometry randomisation)
    min_spacing: float = 0.1
    distractors: int = 0
    lateral_y: tuple[float, float] = (0.27, 0.42)
    x_range: tuple[float, float] = (-0.28, 0.28)
    # Object placement support. "full" is the historical behaviour; the half variants restrict
    # the sampler so a dataset can be deliberately incomplete while the twin stays correct.
    layout_prior: str = "full"


@dataclass
class SceneDraft:
    family: str
    H: int
    scene: SceneSpec
    instruction: str
    meta: dict = field(default_factory=dict)


def _jit(rng: np.random.Generator, x: float, frac: float) -> float:
    return float(x * (1.0 + rng.uniform(-frac, frac)))


def _sample_points(rng: np.random.Generator, n: int, regions: list[tuple[tuple[float, float], tuple[float, float]]],
                   spacing: float, avoid: list[np.ndarray] | None = None, max_tries: int = 4000) -> list[np.ndarray]:
    pts: list[np.ndarray] = []
    avoid = list(avoid or [])
    tries = 0
    while len(pts) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not place objects with the requested spacing")
        (x0, x1), (y0, y1) = regions[int(rng.integers(len(regions)))]
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if all(np.linalg.norm(p - q) >= spacing for q in pts + avoid):
            pts.append(p)
    return pts


def _lateral_regions(r: RandomizationConfig, off: np.ndarray) -> list:
    (x0, x1), (y0, y1) = r.x_range, r.lateral_y
    if r.layout_prior == "left_half":
        x1 = x0 + 0.5 * (x1 - x0)
    elif r.layout_prior == "right_half":
        x0 = x0 + 0.5 * (x1 - x0)
    elif r.layout_prior != "full":
        raise ValueError(f"unknown layout_prior: {r.layout_prior!r}")
    return [((x0 + off[0], x1 + off[0]), (y0 + off[1], y1 + off[1])),
            ((x0 + off[0], x1 + off[0]), (-y1 + off[1], -y0 + off[1]))]


def _small_object(rng, r: RandomizationConfig, name: str, xy: np.ndarray, color: str) -> ObjectSpec:
    if rng.random() < 0.5:
        h = _jit(rng, 0.022, r.size_jitter)
        size = (h, h, h)
        shape, cat = "box", "cube"
        z = h
    else:
        rad = _jit(rng, 0.021, r.size_jitter)
        hh = _jit(rng, 0.024, r.size_jitter)  # squat cans: stable when set down
        size = (rad, hh)
        shape, cat = "cylinder", "can"
        z = hh
    return ObjectSpec(
        name=name, shape=shape, size=size, pos=(float(xy[0]), float(xy[1]), z),
        yaw=float(rng.uniform(-math.pi, math.pi)), rgba=PALETTE[color],
        mass=_jit(rng, 0.08, r.mass_jitter), friction=_jit(rng, 1.0, r.friction_jitter),
        category=cat, color=color,
    )


def _distractors(rng, r: RandomizationConfig, taken: list[np.ndarray], off: np.ndarray) -> list[ObjectSpec]:
    if r.distractors <= 0:
        return []
    pts = _sample_points(rng, r.distractors, _lateral_regions(r, off), r.min_spacing, taken)
    return [_small_object(rng, r, f"distractor{k}", p, "cyan") for k, p in enumerate(pts)]


# ---------------------------------------------------------------------- families
def sample_pack(H: int, rng: np.random.Generator, r: RandomizationConfig) -> SceneDraft:
    assert 1 <= H <= H_MAX["pack"]
    off = rng.uniform(-r.layout_offset, r.layout_offset, 2)
    cols = [-0.17, -0.085, 0.0, 0.085, 0.17]
    rows = [-0.045, 0.075]
    slots = []
    for k in range(10):
        x, y = rows[k // 5] + off[0], cols[k % 5] + off[1]
        slots.append(FixtureSpec(name=f"slot{k}", kind="slot", pos=(x, y, 0.0), size=(0.032, 0.032)))
    pts = _sample_points(rng, H, _lateral_regions(r, off), r.min_spacing)
    colors = list(PALETTE)
    objs = [_small_object(rng, r, f"obj{k}", pts[k], colors[k % len(colors)]) for k in range(H)]
    objs += _distractors(rng, r, pts, off)
    scene = SceneSpec(scene_id=f"pack_H{H}", objects=objs, fixtures=slots[:H])
    clauses = [f"place {o.name} in slot{k}" for k, o in enumerate(objs[:H])]
    instr = "Pack the tray in order: " + ", then ".join(clauses) + "."
    return SceneDraft("pack", H, scene, instr)


def sample_sort(H: int, rng: np.random.Generator, r: RandomizationConfig) -> SceneDraft:
    assert 1 <= H <= H_MAX["sort"]
    off = rng.uniform(-r.layout_offset, r.layout_offset, 2)
    bin_colors = ["red", "green", "blue"]
    bins = [
        FixtureSpec(name=f"bin_{c}", kind="bin", pos=(0.2 + off[0], y + off[1], 0.0), size=(0.07, 0.07, 0.05),
                    rgba=tuple(0.6 * v for v in PALETTE[c][:3]) + (1.0,), color=c)
        for c, y in zip(bin_colors, (-0.2, 0.0, 0.2))
    ]
    regions = [((-0.3 + off[0], 0.06 + off[0]), (-0.42 + off[1], 0.42 + off[1]))]
    pts = _sample_points(rng, H, regions, r.min_spacing * 0.9)
    colors = [bin_colors[k % 3] for k in range(H)]
    rng.shuffle(colors)
    objs = [_small_object(rng, r, f"obj{k}", pts[k], colors[k]) for k in range(H)]
    objs += _distractors(rng, r, pts, off)
    scene = SceneSpec(scene_id=f"sort_H{H}", objects=objs, fixtures=bins)
    order = ", ".join(o.name for o in objs[:H])
    instr = f"Sort the objects into the bins matching their colours, in this order: {order}."
    return SceneDraft("sort", H, scene, instr)


def sample_stack(H: int, rng: np.random.Generator, r: RandomizationConfig) -> SceneDraft:
    assert 1 <= H <= H_MAX["stack"]
    off = rng.uniform(-r.layout_offset, r.layout_offset, 2)
    plate = FixtureSpec(name="plate", kind="plate", pos=(0.04 + off[0], 0.0 + off[1], 0.0), size=(0.06, 0.06, 0.003),
                        rgba=(0.95, 0.95, 0.95, 1.0))
    kinds = ["bread"] + [str(k) for k in rng.choice(["ham", "cheese", "lettuce", "tomato", "egg"], size=max(0, H - 2))]
    if H >= 2:
        kinds.append("bread")
    kinds = kinds[:H]
    pts = _sample_points(rng, H, _lateral_regions(r, off), r.min_spacing + 0.01)
    objs = []
    for k, (kind, p) in enumerate(zip(kinds, pts)):
        (hx, hy, hz), rgba = LAYERS[kind]
        s = 1.0 + rng.uniform(-0.05, 0.05)
        size = (hx * s, hy * s, hz)
        objs.append(ObjectSpec(
            name=f"obj{k}", shape="box", size=size, pos=(float(p[0]), float(p[1]), hz),
            yaw=float(rng.uniform(-math.pi, math.pi)), rgba=rgba, mass=_jit(rng, 0.05, r.mass_jitter),
            friction=_jit(rng, 1.0, r.friction_jitter), category=kind, color=kind,
        ))
    objs += _distractors(rng, r, pts, off)
    scene = SceneSpec(scene_id=f"stack_H{H}", objects=objs, fixtures=[plate])
    order = ", ".join(o.name for o in objs[:H])
    instr = f"Make a sandwich on the plate, stacking from bottom to top: {order}."
    return SceneDraft("stack", H, scene, instr)


SAMPLERS = {"pack": sample_pack, "sort": sample_sort, "stack": sample_stack}


def sample_scene(family: str, H: int, seed: int, rcfg: RandomizationConfig | None = None) -> SceneDraft:
    rng = np.random.default_rng(seed)
    draft = SAMPLERS[family](H, rng, rcfg or RandomizationConfig())
    draft.scene.scene_id = f"{family}_H{H}_s{seed}"
    draft.meta["seed"] = seed
    return draft
