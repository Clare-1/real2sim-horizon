"""Task instantiation: sample scene -> compile graph -> feasibility -> repair / resample."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ..sim.scene import SceneSpec
from ..taskgraph.compiler import TemplateCompiler
from ..taskgraph.feasibility import FeasibilityReport, check
from ..taskgraph.repair import repair
from ..taskgraph.schema import TaskGraph
from .families import RandomizationConfig, sample_scene


@dataclass
class TaskInstance:
    family: str
    H: int
    seed: int
    scene: SceneSpec
    graph: TaskGraph
    instruction: str
    report: FeasibilityReport
    attempts: int = 1
    log: list[str] = field(default_factory=list)
    # what the robot believes the scene is (reconstruction); None = it knows the physics scene exactly
    belief: SceneSpec | None = None
    # the plan the robot's compiler produced; None = the correct grounding of the instruction (``graph``)
    belief_graph: TaskGraph | None = None

    @property
    def key(self) -> str:
        return f"{self.family}_H{self.H}_s{self.seed}"

    def controller_view(self) -> "TaskInstance":
        """The task as the robot sees it: believed geometry and its own compiled plan."""
        if self.belief is None and self.belief_graph is None:
            return self
        return dataclasses.replace(self, scene=self.scene if self.belief is None else self.belief,
                                   graph=self.graph if self.belief_graph is None else self.belief_graph,
                                   belief=None, belief_graph=None)


class InfeasibleTask(RuntimeError):
    pass


def make_task(family: str, H: int, seed: int, rcfg: RandomizationConfig | None = None,
              compiler=None, max_attempts: int = 8) -> TaskInstance:
    compiler = compiler or TemplateCompiler()
    log: list[str] = []
    for attempt in range(max_attempts):
        layout_seed = seed + 104729 * attempt
        try:
            draft = sample_scene(family, H, layout_seed, rcfg)
        except RuntimeError as e:  # spacing could not be satisfied
            log.append(f"attempt {attempt}: sampler failed ({e})")
            continue
        graph = compiler.compile(draft.scene, draft.instruction, task_id=f"{family}_H{H}_s{seed}")
        report = check(draft.scene, graph)
        scene = draft.scene
        if not report.ok:
            res = repair(scene, graph, report)
            log += [f"attempt {attempt}: {m}" for m in res.log]
            scene, graph, report = res.scene, res.graph, res.report
            if res.needs_resample:
                log.append(f"attempt {attempt}: layout resampled ({', '.join(sorted(report.kinds()))})")
                continue
        return TaskInstance(family, H, seed, scene, graph, draft.instruction, report, attempt + 1, log)
    raise InfeasibleTask(f"{family} H={H} seed={seed}: no feasible layout after {max_attempts} attempts; {log[-3:]}")
