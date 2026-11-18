"""JSON task payloads for executing reconstructed scenes without procedural sampling.

The payload is a complete simulator task, including its separate controller belief.
It is not an agent-visible observation: callers must use ``controller_view`` when
passing the task to an unprivileged controller. Feasibility is recomputed on load;
loading never repairs, resamples, or silently replaces the supplied scene or graph.
"""
from __future__ import annotations

import json

from ..sim.scene import SceneSpec
from ..taskgraph.feasibility import check
from ..taskgraph.schema import TaskGraph
from .factory import TaskInstance

PAYLOAD_VERSION = 1
_ARITY = {"clear": 1, "held": 1, "at_slot": 2, "in_bin": 2, "on": 2}


def _json_copy(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Task payload must be a JSON object")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Task payload must contain finite JSON values: {exc}") from exc


def _validate_scene(scene: SceneSpec, label: str) -> None:
    names = [x.name for x in [*scene.objects, *scene.fixtures]]
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError(f"{label}: entity names must be nonempty and unique")
    for obj in scene.objects:
        size_n = {"box": 3, "cylinder": 2}.get(obj.shape)
        if size_n is None or len(obj.size) != size_n or any(v <= 0 for v in obj.size):
            raise ValueError(f"{label}: invalid shape or size for {obj.name}")
        if len(obj.pos) != 3 or len(obj.rgba) != 4 or obj.mass <= 0 or obj.friction < 0:
            raise ValueError(f"{label}: invalid pose, color, mass or friction for {obj.name}")
    for fixture in scene.fixtures:
        size_n = {"slot": 2, "bin": 3, "plate": 3}.get(fixture.kind)
        if size_n is None or len(fixture.size) != size_n or any(v <= 0 for v in fixture.size):
            raise ValueError(f"{label}: invalid kind or size for {fixture.name}")
        if len(fixture.pos) != 3 or len(fixture.rgba) != 4:
            raise ValueError(f"{label}: invalid pose or color for {fixture.name}")
    if len(scene.table_half) != 2 or any(v <= 0 for v in scene.table_half) or len(scene.gripper_home) != 4:
        raise ValueError(f"{label}: invalid table or gripper home")


def _validate_graph(graph: TaskGraph, scene: SceneSpec, family: str, H: int, label: str) -> None:
    if graph.family != family or graph.H != H:
        raise ValueError(f"{label}: family/H disagree with task metadata")
    errors = graph.validate({o.name for o in scene.objects}, {f.name for f in scene.fixtures})
    for subgoal in graph.subgoals:
        for pred in [subgoal.effect, *subgoal.pre]:
            if pred.name in _ARITY and len(pred.args) != _ARITY[pred.name]:
                errors.append(f"{subgoal.sid}: wrong arity for {pred.name}")
    if errors:
        raise ValueError(f"{label}: " + "; ".join(errors))


def task_from_payload(payload: dict, *, expected_family: str | None = None,
                      expected_H: int | None = None, expected_seed: int | None = None) -> TaskInstance:
    """Load a JSON-compatible task, optionally checking the enclosing Job's identity.

    Malformed or structurally invalid tasks raise ``ValueError``. Geometric and
    symbolic feasibility findings are retained in ``task.report`` for the caller;
    unlike ``make_task``, this function does not substitute an easier scene.
    """
    p = _json_copy(payload)
    try:
        if type(p.get("version")) is not int or p["version"] != PAYLOAD_VERSION:
            raise ValueError("unsupported task payload version")
        family, H, seed = p["family"], p["H"], p["seed"]
        if not isinstance(family, str) or not family:
            raise ValueError("family must be a nonempty string")
        if type(H) is not int or H < 1 or type(seed) is not int or seed < 0:
            raise ValueError("H must be a positive integer and seed a nonnegative integer")
        for key, expected, actual in (("family", expected_family, family), ("H", expected_H, H),
                                      ("seed", expected_seed, seed)):
            if expected is not None and expected != actual:
                raise ValueError(f"task payload {key} mismatch: Job={expected!r}, payload={actual!r}")
        instruction = p["instruction"]
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("instruction must be a nonempty string")
        scene, graph = SceneSpec.from_dict(p["scene"]), TaskGraph.from_dict(p["graph"])
        _validate_scene(scene, "scene")
        _validate_graph(graph, scene, family, H, "graph")
        if graph.instruction != instruction:
            raise ValueError("graph instruction disagrees with task instruction")
        belief = SceneSpec.from_dict(p["belief"]) if p.get("belief") is not None else None
        belief_graph = TaskGraph.from_dict(p["belief_graph"]) if p.get("belief_graph") is not None else None
        if belief is not None:
            _validate_scene(belief, "belief")
            if [o.name for o in belief.objects] != [o.name for o in scene.objects]:
                raise ValueError("belief object identities/order must match the physical scene")
            if {f.name for f in belief.fixtures} != {f.name for f in scene.fixtures}:
                raise ValueError("belief fixture identities must match the physical scene")
        _validate_graph(belief_graph or graph, belief or scene, family, H, "controller graph")
        attempts, log = p.get("attempts", 1), p.get("log", [])
        if type(attempts) is not int or attempts < 1 or not isinstance(log, list) or any(not isinstance(x, str) for x in log):
            raise ValueError("invalid task attempts/log")
        return TaskInstance(family, H, seed, scene, graph, instruction, check(scene, graph), attempts,
                            list(log), belief, belief_graph)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise ValueError(f"Invalid task payload: {exc}") from exc


def task_to_payload(task: TaskInstance) -> dict:
    """Export an independent JSON object; never serialize executable Python objects."""
    payload = _json_copy({
        "version": PAYLOAD_VERSION, "family": task.family, "H": task.H, "seed": task.seed,
        "instruction": task.instruction, "scene": task.scene.to_dict(), "graph": task.graph.to_dict(),
        "belief": None if task.belief is None else task.belief.to_dict(),
        "belief_graph": None if task.belief_graph is None else task.belief_graph.to_dict(),
        "attempts": task.attempts, "log": list(task.log),
    })
    task_from_payload(payload)
    return payload
