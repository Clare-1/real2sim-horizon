"""Feasibility checks for compiled task graphs (paper Sec. 5.1 step 4).

Three layers, cheapest first:
  1. structural  - schema validation (unknown entities, cycles, duplicate objects);
  2. symbolic    - STRIPS-style roll-out of effects to verify every precondition holds at
                   its turn (e.g. a sandwich layer is not buried before it is moved);
  3. geometric   - reachability of pick/place poses for a virtual arm (stand-in for
                   ARSTAG's IK screening), grasp width vs. gripper opening, initial-layout
                   overlap and container capacity.
Expert screening in simulation (layer 4) lives in :mod:`horizon.datagen.screening`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..sim.mjcf import MAX_FINGER
from ..sim.scene import SceneSpec
from .schema import TaskGraph

REACH_BASE = np.array([-0.55, 0.0])
REACH_MAX = 0.93
REACH_MIN = 0.25
MAX_OPEN = 2 * MAX_FINGER


@dataclass
class Issue:
    kind: str  # structure | precondition | unreachable_pick | unreachable_place | too_wide | overlap | capacity
    sid: str | None
    entity: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.sid or '-'} {self.entity}: {self.detail}"


@dataclass
class FeasibilityReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def kinds(self) -> set[str]:
        return {i.kind for i in self.issues}


def reachable_xy(xy) -> bool:
    d = float(np.linalg.norm(np.asarray(xy[:2]) - REACH_BASE))
    return REACH_MIN <= d <= REACH_MAX


def symbolic_issues(graph: TaskGraph) -> list[Issue]:
    issues: list[Issue] = []
    covered: dict[str, str] = {}  # object -> what rests on it
    placed: set[str] = set()
    try:
        order = graph.order()
    except ValueError as e:
        return [Issue("structure", None, graph.task_id, str(e))]
    stack_targets = {g.target for g in graph.subgoals if g.effect.name == "on"}
    moved_later = {g.obj for g in graph.subgoals}
    for k in order:
        g = graph.subgoals[k]
        for p in g.pre:
            if p.name == "clear" and p.args[0] in covered:
                issues.append(Issue("precondition", g.sid, p.args[0],
                                    f"{p} violated: {covered[p.args[0]]} already rests on it"))
        if g.effect.name == "on" and g.target in moved_later and g.target not in placed:
            issues.append(Issue("precondition", g.sid, g.target,
                                f"stacking onto {g.target} before {g.target} itself is placed"))
        if g.effect.name == "on" and g.target in stack_targets:
            covered[g.target] = g.obj
        placed.add(g.obj)
    return issues


def geometric_issues(scene: SceneSpec, graph: TaskGraph) -> list[Issue]:
    issues: list[Issue] = []
    objs = {o.name: o for o in scene.objects}
    fixes = {f.name: f for f in scene.fixtures}
    for g in graph.subgoals:
        o = objs[g.obj]
        if not reachable_xy(o.pos):
            issues.append(Issue("unreachable_pick", g.sid, g.obj, f"pick at {np.round(o.pos[:2], 3)} out of reach"))
        if g.target in fixes and not reachable_xy(fixes[g.target].pos):
            issues.append(Issue("unreachable_place", g.sid, g.target, "place pose out of reach"))
        widths = [w for _, w in o.grasp_candidates()]
        if min(widths) > MAX_OPEN - 0.015:
            issues.append(Issue("too_wide", g.sid, g.obj, f"min grasp width {min(widths):.3f} > opening"))
    names = list(objs)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            oa, ob = objs[names[a]], objs[names[b]]
            if math.dist(oa.pos[:2], ob.pos[:2]) < 0.8 * (oa.footprint_radius + ob.footprint_radius):
                issues.append(Issue("overlap", None, f"{oa.name}/{ob.name}", "initial poses overlap"))
    load: dict[str, float] = {}
    for g in graph.subgoals:
        if g.effect.name == "in_bin":
            r = objs[g.obj].footprint_radius
            load[g.target] = load.get(g.target, 0.0) + math.pi * r * r
    for b, area in load.items():
        ihx, ihy, _ = fixes[b].size
        if area > 0.9 * 4 * ihx * ihy:
            issues.append(Issue("capacity", None, b, f"footprint {area:.4f} m^2 exceeds bin capacity"))
    return issues


def check(scene: SceneSpec, graph: TaskGraph) -> FeasibilityReport:
    errs = graph.validate({o.name for o in scene.objects}, {f.name for f in scene.fixtures})
    if errs:
        return FeasibilityReport([Issue("structure", None, graph.task_id, e) for e in errs])
    return FeasibilityReport(symbolic_issues(graph) + geometric_issues(scene, graph))
