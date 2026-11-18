"""Single image + instruction -> task-scoped SceneSpec draft (ARSTAG Sec. III-A, coarse).

ARSTAG's reconstruction agent scopes the scene to task-relevant objects, binds them to
SAM3 masks, lifts them to meshes with SAM 3D and repairs support relations. This module
implements the *task scoping* and a coarse *physical grounding* with a multimodal model:
Claude lists only the task-relevant objects and supports with approximate metric
footprints in the table frame, and we apply ARSTAG's support-consistency repair
(objects rest on the table plane, no floating or penetration).

For geometry-accurate scenes, reconstruct with an open real2sim pipeline (Agentic RSR,
Real2Gym) and ingest its MJCF with :mod:`horizon.real2sim.mjcf_import`.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from ..sim.scene import FixtureSpec, ObjectSpec, SceneSpec
from ..tasks.families import PALETTE

MODEL = "claude-opus-5-5"

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}, "category": {"type": "string"}, "color": {"type": "string"},
                "shape": {"type": "string", "enum": ["box", "cylinder"]},
                "x": {"type": "number"}, "y": {"type": "number"},
                "width": {"type": "number"}, "depth": {"type": "number"}, "height": {"type": "number"},
                "yaw_deg": {"type": "number"},
            },
            "required": ["name", "category", "color", "shape", "x", "y", "width", "depth", "height", "yaw_deg"],
            "additionalProperties": False}},
        "fixtures": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}, "kind": {"type": "string", "enum": ["slot", "bin", "plate"]},
                "color": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"},
                "width": {"type": "number"}, "depth": {"type": "number"}, "height": {"type": "number"},
            },
            "required": ["name", "kind", "color", "x", "y", "width", "depth", "height"],
            "additionalProperties": False}},
        "notes": {"type": "string"},
    },
    "required": ["objects", "fixtures", "notes"],
    "additionalProperties": False,
}

SYSTEM = """You turn a photo of a tabletop and a manipulation instruction into a task-scoped scene description.
Include only objects the instruction needs (objects to move, and the containers / plates / slots they go to);
omit everything else. Give each entity a unique snake_case name.
Frame: the table top is the plane z = 0 with its centre at the origin; the robot stands at x = -0.5 facing +x;
+y points to the robot's left. All lengths are metres. Estimate each object's centre (x, y) on the table and its
width (along its own x), depth (along its own y) and height; yaw_deg is the rotation about the vertical axis.
Bins are open containers; "height" of a bin is its wall height. Use "notes" for anything uncertain."""


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def scene_from_json(data: dict, scene_id: str, provenance: str) -> SceneSpec:
    objs = []
    for o in data["objects"]:
        w = _clamp(o["width"], 0.01, 0.12)
        d = _clamp(o["depth"], 0.01, 0.12)
        h = _clamp(o["height"], 0.005, 0.2)
        if o["shape"] == "cylinder":
            size = (max(w, d) / 2, h / 2)
        else:
            size = (w / 2, d / 2, h / 2)
        rgba = PALETTE.get(o["color"].lower(), (0.6, 0.6, 0.6, 1.0))
        # support-consistency repair: rest on the table plane instead of trusting estimated heights
        objs.append(ObjectSpec(name=o["name"], shape=o["shape"], size=size, pos=(o["x"], o["y"], h / 2),
                               yaw=o["yaw_deg"] * 3.14159265 / 180.0, rgba=rgba, category=o["category"],
                               color=o["color"].lower()))
    fixes = []
    for f in data["fixtures"]:
        w, d, h = _clamp(f["width"], 0.04, 0.4), _clamp(f["depth"], 0.04, 0.4), _clamp(f["height"], 0.002, 0.15)
        if f["kind"] == "bin":
            size = (w / 2 - 0.006, d / 2 - 0.006, h)
        elif f["kind"] == "plate":
            size = (w / 2, d / 2, max(0.002, h / 2))
        else:
            size = (w / 2, d / 2)
        fixes.append(FixtureSpec(name=f["name"], kind=f["kind"], pos=(f["x"], f["y"], 0.0), size=size,
                                 color=f["color"].lower()))
    return SceneSpec(scene_id=scene_id, objects=objs, fixtures=fixes, provenance=provenance)


def draft_scene(image_path: str | Path, instruction: str, client=None, model: str = MODEL) -> tuple[SceneSpec, str]:
    """Returns (scene draft, model notes)."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    p = Path(image_path)
    media = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    data = base64.standard_b64encode(p.read_bytes()).decode("utf-8")
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCENE_SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
            {"type": "text", "text": f"Instruction: {instruction}"},
        ]}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined the request: {getattr(resp, 'stop_details', None)}")
    out = json.loads(next(b.text for b in resp.content if b.type == "text"))
    return scene_from_json(out, scene_id=p.stem, provenance=f"vlm:{model}"), out.get("notes", "")
