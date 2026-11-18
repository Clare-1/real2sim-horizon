"""Typed task graph: the executable, verifiable, repairable plan representation (paper Sec. 5.1).

A task graph is a DAG of subgoals. Each subgoal binds one skill to object/target
arguments and carries preconditions, an effect (its success predicate) and dependencies.
It is not free-text planning: every field is checked by ``validate`` and every predicate
is executable by :mod:`horizon.taskgraph.predicates`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SKILLS = ("pick_place",)
PREDICATES = ("at_slot", "in_bin", "on", "clear", "held")


@dataclass(frozen=True)
class Pred:
    name: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"

    @staticmethod
    def parse(text: str) -> "Pred":
        name, rest = text.split("(", 1)
        args = tuple(a.strip() for a in rest.rstrip(")").split(",") if a.strip())
        return Pred(name.strip(), args)


@dataclass
class Subgoal:
    sid: str
    skill: str
    obj: str
    target: str  # fixture name, or object name for stacking
    effect: Pred
    pre: list[Pred] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    place_yaw: float | None = None  # optional yaw constraint on the gripper when placing

    def to_dict(self) -> dict:
        d = asdict(self)
        d["effect"] = str(self.effect)
        d["pre"] = [str(p) for p in self.pre]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Subgoal":
        return Subgoal(
            sid=d["sid"], skill=d["skill"], obj=d["obj"], target=d["target"],
            effect=Pred.parse(d["effect"]), pre=[Pred.parse(p) for p in d.get("pre", [])],
            deps=list(d.get("deps", [])), place_yaw=d.get("place_yaw"),
        )


@dataclass
class TaskGraph:
    task_id: str
    family: str
    instruction: str
    subgoals: list[Subgoal]
    version: int = 0
    repair_log: list[str] = field(default_factory=list)

    @property
    def H(self) -> int:
        return len(self.subgoals)

    def index(self, sid: str) -> int:
        for i, g in enumerate(self.subgoals):
            if g.sid == sid:
                return i
        raise KeyError(sid)

    def order(self) -> list[int]:
        """Stable topological order (Kahn's algorithm, ties broken by list position)."""
        idx = {g.sid: i for i, g in enumerate(self.subgoals)}
        indeg = [0] * self.H
        children: list[list[int]] = [[] for _ in range(self.H)]
        for i, g in enumerate(self.subgoals):
            for dep in g.deps:
                if dep in idx:
                    indeg[i] += 1
                    children[idx[dep]].append(i)
        ready = [i for i in range(self.H) if indeg[i] == 0]
        out: list[int] = []
        while ready:
            ready.sort()
            i = ready.pop(0)
            out.append(i)
            for c in children[i]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
        if len(out) != self.H:
            raise ValueError("task graph has a cycle")
        return out

    def validate(self, object_names: set[str], fixture_names: set[str]) -> list[str]:
        """Structural checks; returns a list of human-readable errors (empty if valid)."""
        errs: list[str] = []
        sids = [g.sid for g in self.subgoals]
        if len(set(sids)) != len(sids):
            errs.append("duplicate subgoal ids")
        known = object_names | fixture_names
        for g in self.subgoals:
            if g.skill not in SKILLS:
                errs.append(f"{g.sid}: unknown skill {g.skill}")
            if g.obj not in object_names:
                errs.append(f"{g.sid}: unknown object {g.obj}")
            if g.target not in known:
                errs.append(f"{g.sid}: unknown target {g.target}")
            for p in [g.effect, *g.pre]:
                if p.name not in PREDICATES:
                    errs.append(f"{g.sid}: unknown predicate {p.name}")
                for a in p.args:
                    if a not in known:
                        errs.append(f"{g.sid}: predicate {p} references unknown entity {a}")
            for dep in g.deps:
                if dep not in sids:
                    errs.append(f"{g.sid}: unknown dependency {dep}")
        objs = [g.obj for g in self.subgoals]
        if len(set(objs)) != len(objs):
            errs.append("an object is manipulated by more than one subgoal")
        try:
            self.order()
        except ValueError as e:
            errs.append(str(e))
        return errs

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "family": self.family, "instruction": self.instruction,
            "version": self.version, "repair_log": list(self.repair_log),
            "subgoals": [g.to_dict() for g in self.subgoals],
        }

    @staticmethod
    def from_dict(d: dict) -> "TaskGraph":
        return TaskGraph(
            task_id=d["task_id"], family=d["family"], instruction=d["instruction"],
            subgoals=[Subgoal.from_dict(g) for g in d["subgoals"]],
            version=int(d.get("version", 0)), repair_log=list(d.get("repair_log", [])),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
