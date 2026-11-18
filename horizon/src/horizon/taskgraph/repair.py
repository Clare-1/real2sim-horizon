"""Bounded repair loop for infeasible task graphs (paper Sec. 5.1 step 5).

Graph-level repairs are applied here; layout-level repairs (resampling object poses, as
ARSTAG does when no grasp or waypoint passes screening) are requested from the caller
via ``RepairResult.needs_resample`` because only the scene sampler can produce a new
task-consistent layout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..sim.scene import SceneSpec
from .feasibility import MAX_OPEN, FeasibilityReport, check, reachable_xy
from .schema import Pred, TaskGraph

LAYOUT_KINDS = {"unreachable_pick", "overlap", "capacity"}


@dataclass
class RepairResult:
    scene: SceneSpec
    graph: TaskGraph
    report: FeasibilityReport
    needs_resample: bool = False
    log: list[str] = field(default_factory=list)


def reorder_for_preconditions(graph: TaskGraph) -> TaskGraph | None:
    """Greedy re-sequencing so that supports are placed before anything is stacked on them."""
    g = copy.deepcopy(graph)
    remaining = list(range(g.H))
    seq: list[int] = []
    placed: set[str] = set()
    movers = {s.obj for s in g.subgoals}
    while remaining:
        pick = None
        for k in remaining:
            s = g.subgoals[k]
            if s.effect.name == "on" and s.target in movers and s.target not in placed:
                continue
            pick = k
            break
        if pick is None:
            return None
        seq.append(pick)
        placed.add(g.subgoals[pick].obj)
        remaining.remove(pick)
    subs = [g.subgoals[k] for k in seq]
    for i, s in enumerate(subs):
        s.deps = [subs[i - 1].sid] if i > 0 else []
    g.subgoals = subs
    return g


def reassign_unreachable_slots(scene: SceneSpec, graph: TaskGraph) -> bool:
    used = {s.target for s in graph.subgoals}
    free = [f for f in scene.fixtures if f.kind == "slot" and f.name not in used and reachable_xy(f.pos)]
    changed = False
    for s in graph.subgoals:
        f = next((x for x in scene.fixtures if x.name == s.target), None)
        if f is not None and f.kind == "slot" and not reachable_xy(f.pos) and free:
            alt = min(free, key=lambda x: float(np.linalg.norm(np.array(x.pos[:2]) - np.array(f.pos[:2]))))
            free.remove(alt)
            s.target = alt.name
            s.effect = Pred("at_slot", (s.obj, alt.name))
            changed = True
    return changed


def rescale_too_wide(scene: SceneSpec, names: set[str]) -> list[str]:
    """Reconstruction correction: shrink objects whose every grasp exceeds the opening."""
    log = []
    for o in scene.objects:
        if o.name not in names:
            continue
        widths = [w for _, w in o.grasp_candidates()]
        if min(widths) > MAX_OPEN - 0.015:
            k = (MAX_OPEN - 0.02) / min(widths)
            if o.shape == "box":
                o.size = (o.size[0] * k, o.size[1] * k, o.size[2])
            else:
                o.size = (o.size[0] * k, o.size[1])
            log.append(f"rescaled {o.name} footprint by {k:.2f}")
    return log


def repair(scene: SceneSpec, graph: TaskGraph, report: FeasibilityReport, max_rounds: int = 3) -> RepairResult:
    scene = copy.deepcopy(scene)
    graph = copy.deepcopy(graph)
    log: list[str] = []
    for _ in range(max_rounds):
        kinds = report.kinds()
        if not kinds:
            break
        acted = False
        if "precondition" in kinds:
            g2 = reorder_for_preconditions(graph)
            if g2 is not None:
                graph = g2
                log.append("reordered subgoals to satisfy preconditions")
                acted = True
        if "unreachable_place" in kinds and reassign_unreachable_slots(scene, graph):
            log.append("reassigned unreachable slots")
            acted = True
        if "too_wide" in kinds:
            names = {i.entity for i in report.issues if i.kind == "too_wide"}
            msgs = rescale_too_wide(scene, names)
            log += msgs
            acted = acted or bool(msgs)
        report = check(scene, graph)
        if not acted:
            break
    graph.version += 1 if log else 0
    graph.repair_log += log
    needs = bool(report.kinds() & LAYOUT_KINDS) or not report.ok
    return RepairResult(scene, graph, report, needs_resample=needs, log=log)
