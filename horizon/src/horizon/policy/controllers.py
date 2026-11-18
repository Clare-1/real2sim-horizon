"""Learned controllers used at evaluation time.

* :class:`HierarchicalController` - System 2 selects the next subgoal from verifier
  predicates on the perceived state (revisiting regressed subgoals); System 1 is a
  subgoal-conditioned BC network.
* :class:`FlatController` - one network conditioned on the whole task; System 2 is only
  used to decide when the episode is finished and to log the subgoal being worked on.
"""

from __future__ import annotations

import numpy as np

from ..monitor.verifier import Verifier, VerifierConfig
from ..sim.state import WorldState
from ..taskgraph.predicates import Debouncer, select_subgoal
from ..tasks.factory import TaskInstance
from .features import TaskFeaturizer
from .model import NumpyMLP

HOVER = np.array([0.0, 0.0, 0.3, 0.0, -1.0], dtype=np.float32)


class HierarchicalController:
    privileged = False

    def __init__(self, params: dict, verifier_cfg: VerifierConfig | None = None, debounce: int = 4):
        self.net = NumpyMLP(params)
        self.verifier_cfg = verifier_cfg
        self.current: int | None = None
        self.debounce = Debouncer(debounce)

    def reset(self, task: TaskInstance) -> None:
        self.task = task
        self.ver = Verifier(task.scene, VerifierConfig(**vars(self.verifier_cfg)) if self.verifier_cfg else None)
        self.feat = TaskFeaturizer(task.scene, task.graph, self.ver)
        self.order = task.graph.order()
        self.current = None
        self.debounce.reset()

    def act(self, obs: WorldState, true: WorldState) -> tuple[np.ndarray, int]:
        perceived = self.ver.perceive(obs)
        sat = self.debounce(self.ver.eng.satisfied_vector(perceived, self.task.graph))
        k = select_subgoal(self.ver.eng, perceived, self.task.graph, self.current, self.order, sat=sat)
        self.current = k
        if k is None:
            return HOVER.copy(), -1
        return self.net.act(self.feat.hier(obs, k)), k


class FlatController:
    privileged = False

    def __init__(self, params: dict, verifier_cfg: VerifierConfig | None = None, debounce: int = 4):
        if str(params.get("arch", "mlp")) == "set":
            from .set_model import NumpySetPolicy

            self.net = NumpySetPolicy(params)
        else:
            self.net = NumpyMLP(params)
        self.verifier_cfg = verifier_cfg
        self.current: int | None = None
        self.debounce = Debouncer(debounce)

    def reset(self, task: TaskInstance) -> None:
        self.task = task
        self.ver = Verifier(task.scene, VerifierConfig(**vars(self.verifier_cfg)) if self.verifier_cfg else None)
        self.feat = TaskFeaturizer(task.scene, task.graph, self.ver)
        self.order = task.graph.order()
        self.current = None
        self.debounce.reset()

    def act(self, obs: WorldState, true: WorldState) -> tuple[np.ndarray, int]:
        perceived = self.ver.perceive(obs)
        sat = self.debounce(self.ver.eng.satisfied_vector(perceived, self.task.graph))
        k = select_subgoal(self.ver.eng, perceived, self.task.graph, self.current, self.order, sat=sat)
        self.current = k
        if k is None:
            return HOVER.copy(), -1
        return self.net.act(self.feat.flat(obs)), k


class MixedController:
    """DAgger collection policy: the expert acts with probability ``beta``, the learner otherwise."""

    privileged = False

    def __init__(self, learner, expert, beta: float, rng: np.random.Generator):
        self.learner, self.expert, self.beta, self.rng = learner, expert, float(beta), rng

    def reset(self, task: TaskInstance) -> None:
        self.learner.reset(task)
        self.expert.reset(task)

    def act(self, obs: WorldState, true: WorldState) -> tuple[np.ndarray, int]:
        a, k = self.learner.act(obs, true)
        if self.rng.random() < self.beta:
            a, _ = self.expert.act(obs, true)
        return a, k


def make_controller(kind: str, params: dict, verifier_cfg: VerifierConfig | None = None):
    if kind == "hier":
        return HierarchicalController(params, verifier_cfg)
    if kind in ("flat", "flat_set"):
        return FlatController(params, verifier_cfg)
    raise ValueError(kind)
