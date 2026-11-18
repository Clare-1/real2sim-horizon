"""Systematic grounding errors of the instruction compiler (sim-real proxy, v3 report Sec. 7).

A compiler that grounds an instruction against a scene can be wrong in a way no amount of policy
data fixes: a colour confusion sends every green object to the blue bin. The robot then executes
and verifies the wrong plan perfectly - in its simulator and on the real system alike - and only a
human audit of the outcome reveals it. The repair is to recompile with the auditor's correction.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ..sim.scene import SceneSpec
from .schema import Pred, TaskGraph


@dataclass
class GroundingError:
    confuse: dict[str, str] = field(default_factory=dict)  # object colour -> colour of the bin it is sent to

    def is_identity(self) -> bool:
        return not self.confuse

    def apply(self, scene: SceneSpec, graph: TaskGraph) -> TaskGraph:
        """The plan the faulty compiler produces for this scene (the input graph if nothing is confused)."""
        if not self.confuse:
            return graph
        bins = {f.color: f.name for f in scene.fixtures if f.kind == "bin"}
        subgoals = []
        for g in graph.subgoals:
            if g.effect.name == "in_bin":
                wrong = bins.get(self.confuse.get(scene.objects[scene.object_index(g.obj)].color, ""))
                if wrong is not None and wrong != g.target:
                    g = dataclasses.replace(g, target=wrong, effect=Pred("in_bin", (g.obj, wrong)))
            subgoals.append(g)
        return dataclasses.replace(graph, subgoals=subgoals)

    def to_dict(self) -> dict:
        return {"confuse": dict(self.confuse)}

    @staticmethod
    def from_dict(d: dict | None) -> "GroundingError":
        return GroundingError(dict((d or {}).get("confuse", {})))
