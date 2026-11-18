"""Task Graph Compiler (paper Sec. 5.1): instruction + scene -> typed subgoal DAG.

``TemplateCompiler`` grounds the procedural task-family instructions deterministically
(used for all experiments, so results do not depend on API access). ``LLMCompiler`` in
:mod:`horizon.taskgraph.llm_compiler` produces the same JSON structure from free-form
instructions and images; both outputs go through the same validation, feasibility
checks and repair loop.
"""

from __future__ import annotations

import math
import re

from ..sim.scene import SceneSpec
from .schema import Pred, Subgoal, TaskGraph


class CompileError(ValueError):
    pass


class TemplateCompiler:
    PACK = re.compile(r"place (\w+) in (\w+)")
    ORDER = re.compile(r"(?:in this order|bottom to top):\s*(.+?)\.?$")

    def compile(self, scene: SceneSpec, instruction: str, task_id: str | None = None) -> TaskGraph:
        text = instruction.strip()
        low = text.lower()
        tid = task_id or scene.scene_id
        if low.startswith("pack"):
            pairs = self.PACK.findall(text)
            if not pairs:
                raise CompileError("no 'place X in Y' clauses found")
            return self._sequential(tid, "pack", text, [
                (o, s, Pred("at_slot", (o, s)), [Pred("clear", (o,))], math.pi / 2) for o, s in pairs
            ], scene)
        m = self.ORDER.search(text)
        if low.startswith("sort") and m:
            names = self._names(m.group(1))
            items = []
            for o in names:
                b = self._bin_for(scene, o)
                items.append((o, b, Pred("in_bin", (o, b)), [Pred("clear", (o,))], None))
            return self._sequential(tid, "sort", text, items, scene)
        if "sandwich" in low and m:
            names = self._names(m.group(1))
            plate = next((f.name for f in scene.fixtures if f.kind == "plate"), None)
            if plate is None:
                raise CompileError("stacking instruction but no plate in scene")
            items, support = [], plate
            for o in names:
                pre = [Pred("clear", (o,))]
                if support != plate:
                    pre.append(Pred("clear", (support,)))
                items.append((o, support, Pred("on", (o, support)), pre, None))
                support = o
            return self._sequential(tid, "stack", text, items, scene)
        raise CompileError(f"instruction not understood by the template compiler: {text!r}")

    @staticmethod
    def _names(s: str) -> list[str]:
        return [n.strip() for n in s.split(",") if n.strip()]

    @staticmethod
    def _bin_for(scene: SceneSpec, obj: str) -> str:
        color = scene.objects[scene.object_index(obj)].color
        for f in scene.fixtures:
            if f.kind == "bin" and f.color == color:
                return f.name
        raise CompileError(f"no bin matches the colour of {obj} ({color})")

    @staticmethod
    def _sequential(tid: str, family: str, text: str, items: list, scene: SceneSpec) -> TaskGraph:
        subgoals = []
        for k, (o, tgt, eff, pre, yaw) in enumerate(items):
            subgoals.append(Subgoal(
                sid=f"g{k}", skill="pick_place", obj=o, target=tgt, effect=eff, pre=list(pre),
                deps=[f"g{k - 1}"] if k > 0 else [], place_yaw=yaw,
            ))
        g = TaskGraph(task_id=tid, family=family, instruction=text, subgoals=subgoals)
        errs = g.validate({o.name for o in scene.objects}, {f.name for f in scene.fixtures})
        if errs:
            raise CompileError("; ".join(errs))
        return g
