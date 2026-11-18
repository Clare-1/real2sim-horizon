"""LLM front end for the Task Graph Compiler (paper Sec. 5.1 steps 1-3).

Claude grounds a free-form instruction in the scene graph and emits subgoals as JSON
constrained by a schema (structured outputs). The result is turned into a typed
``TaskGraph`` and goes through exactly the same validation, symbolic/geometric
feasibility checks and repair loop as the template compiler, so an LLM mistake is caught
as a feasibility issue rather than discovered during data collection.

Requires ``pip install anthropic`` and credentials (``ANTHROPIC_API_KEY`` or an
``ant auth login`` profile). Experiments never call this module.
"""

from __future__ import annotations

import json
import math

from ..sim.scene import SceneSpec
from .feasibility import FeasibilityReport, check
from .repair import repair
from .schema import Pred, Subgoal, TaskGraph

MODEL = "claude-opus-5-5"

GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "obj": {"type": "string", "description": "name of the object to move"},
                    "relation": {"type": "string", "enum": ["at_slot", "in_bin", "on"]},
                    "target": {"type": "string", "description": "slot / bin / plate name, or object name to stack on"},
                    "after": {"type": "array", "items": {"type": "integer"},
                              "description": "indices of subgoals that must finish first"},
                },
                "required": ["obj", "relation", "target", "after"],
                "additionalProperties": False,
            },
        },
        "unresolved": {"type": "array", "items": {"type": "string"},
                       "description": "phrases in the instruction that could not be grounded unambiguously"},
    },
    "required": ["subgoals", "unresolved"],
    "additionalProperties": False,
}

SYSTEM = """You compile robot manipulation instructions into subgoal graphs for a tabletop pick-and-place robot.
The only skill is pick_place: move one object so that a relation holds afterwards.
Relations: at_slot(object, slot) - object rests inside a tray slot; in_bin(object, bin) - object is inside a bin;
on(object, support) - object rests on a plate or on another object (stacking).
Use only entity names that appear in the scene graph. Every object is moved at most once.
List subgoals in execution order and fill "after" with the indices that must be complete first
(for stacking, a layer must come after the layer it rests on).
If the instruction refers to something that is absent or matches several entities, put that phrase in
"unresolved" instead of guessing."""


class ClarificationNeeded(ValueError):
    """The instruction could not be grounded; ARSTAG asks the user in this situation."""


def scene_graph_text(scene: SceneSpec) -> str:
    lines = ["objects:"]
    for o in scene.objects:
        lines.append(f"  - {o.name}: {o.color} {o.category} ({o.shape}), at x={o.pos[0]:.2f} y={o.pos[1]:.2f}")
    lines.append("fixtures:")
    for f in scene.fixtures:
        extra = f" colour={f.color}" if f.color else ""
        lines.append(f"  - {f.name}: {f.kind}{extra}, at x={f.pos[0]:.2f} y={f.pos[1]:.2f}")
    return "\n".join(lines)


def graph_from_json(data: dict, scene: SceneSpec, instruction: str, task_id: str) -> TaskGraph:
    fixtures = {f.name: f for f in scene.fixtures}
    subs = []
    items = data["subgoals"]
    for k, it in enumerate(items):
        obj, rel, tgt = it["obj"], it["relation"], it["target"]
        pre = [Pred("clear", (obj,))]
        if rel == "on" and tgt not in fixtures:
            pre.append(Pred("clear", (tgt,)))
        deps = [f"g{a}" for a in it.get("after", []) if 0 <= a < len(items) and a != k]
        subs.append(Subgoal(sid=f"g{k}", skill="pick_place", obj=obj, target=tgt, effect=Pred(rel, (obj, tgt)),
                            pre=pre, deps=deps, place_yaw=math.pi / 2 if rel == "at_slot" else None))
    fam = {"at_slot": "pack", "in_bin": "sort", "on": "stack"}.get(items[0]["relation"], "pack") if items else "pack"
    return TaskGraph(task_id=task_id, family=fam, instruction=instruction, subgoals=subs)


class LLMCompiler:
    def __init__(self, client=None, model: str = MODEL, effort: str = "high"):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def _request(self, scene: SceneSpec, instruction: str, image_b64: str | None) -> dict:
        content: list[dict] = []
        if image_b64:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}})
        content.append({"type": "text", "text": f"Scene graph:\n{scene_graph_text(scene)}\n\nInstruction: {instruction}"})
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": GRAPH_SCHEMA}},
            messages=[{"role": "user", "content": content}],
        )
        if resp.stop_reason == "refusal":
            raise ClarificationNeeded(f"model declined the request: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("compiler response truncated (max_tokens)")
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    def compile(self, scene: SceneSpec, instruction: str, task_id: str | None = None,
                image_b64: str | None = None) -> tuple[TaskGraph, FeasibilityReport]:
        data = self._request(scene, instruction, image_b64)
        if data.get("unresolved"):
            raise ClarificationNeeded("could not ground: " + "; ".join(data["unresolved"]))
        graph = graph_from_json(data, scene, instruction, task_id or scene.scene_id)
        report = check(scene, graph)
        if not report.ok:
            res = repair(scene, graph, report)
            graph, report = res.graph, res.report
        return graph, report
