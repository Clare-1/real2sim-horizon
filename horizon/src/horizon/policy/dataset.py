"""Training samples from episode logs.

Features are extracted inside rollout workers right after an episode finishes (see
:func:`episode_samples`), so the learner only concatenates arrays. Supervision is the
expert action at each valid step; for policy rollouts (DAgger relabelling) the logged
``expert_action`` is the expert's response to the state the *policy* visited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..monitor.verifier import Verifier
from ..sim.scene import SceneSpec
from ..taskgraph.schema import TaskGraph
from .features import FLAT_DIM, HIER_DIM, TaskFeaturizer, state_from_log


@dataclass
class SampleBlock:
    """Training samples from one episode (or one regenerated segment)."""

    key: str
    family: str
    H: int
    provenance: str
    x_hier: np.ndarray  # (n, HIER_DIM)
    x_flat: np.ndarray  # (n, FLAT_DIM)
    y: np.ndarray  # (n, 5)
    k: np.ndarray  # (n,) subgoal index used for conditioning
    stage: np.ndarray  # (n,) position of that subgoal in topological order
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    def take(self, n: int) -> "SampleBlock":
        return SampleBlock(self.key, self.family, self.H, self.provenance, self.x_hier[:n], self.x_flat[:n],
                           self.y[:n], self.k[:n], self.stage[:n], dict(self.meta))


OFF_TABLE_Z = -0.05  # an object observed below the table plane has fallen off: unrecoverable


def episode_samples(log, t_mask: np.ndarray | None = None, stride: int = 1, max_samples: int | None = None,
                    seed: int = 0, *, observation_mode: str = "legacy") -> SampleBlock:
    """Select the gripper input channel without changing labels or sample selection.

    ``legacy`` preserves the historical ideal-encoder features. ``observed`` uses
    the actor's logged gripper observation and rejects old logs lacking that
    channel; it never silently substitutes privileged encoder values. Object
    observations, geometry and expert subgoal conditioning retain their original
    semantics. Logged observations have the existing float32 storage precision.
    """
    if observation_mode not in ("legacy", "observed"):
        raise ValueError(f"unknown observation_mode: {observation_mode}")
    grip = log.grip
    if observation_mode == "observed":
        grip = getattr(log, "grip_obs", None)
        if grip is None:
            raise ValueError("observation_mode='observed' requires logged grip_obs; no legacy fallback")
        grip = np.asarray(grip)
        if grip.shape != (len(log.action), 6) or not np.isfinite(grip).all():
            raise ValueError("observed grip_obs must be a finite (T, 6) array")
    scene = SceneSpec.from_dict(log.scene)
    graph = TaskGraph.from_dict(log.graph)
    feat = TaskFeaturizer(scene, graph, Verifier(scene))
    order = graph.order()
    labels = log.expert_action if log.expert_action is not None else log.action
    keep = log.valid & (log.k_true >= 0)
    if t_mask is not None:
        keep &= t_mask
    if max_samples is not None:
        fallen = np.nonzero((log.obj_obs[:, :, 2] < OFF_TABLE_Z).any(axis=1))[0]
        if len(fallen):
            keep[fallen[0]:] = False
    idx = np.nonzero(keep)[0][::stride]
    if max_samples is not None and len(idx) > max_samples:
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=max_samples, replace=False))
    xh = np.zeros((len(idx), HIER_DIM), dtype=np.float32)
    xf = np.zeros((len(idx), FLAT_DIM), dtype=np.float32)
    for r, t in enumerate(idx):
        s = state_from_log(grip[t], log.obj_obs[t], int(t))
        xh[r] = feat.hier(s, int(log.k_true[t]))
        xf[r] = feat.flat(s)
    k = log.k_true[idx].astype(np.int16)
    stage = np.array([order.index(int(v)) for v in k], dtype=np.int16)
    return SampleBlock(log.key, log.family, log.H, log.provenance, xh, xf, labels[idx].astype(np.float32), k, stage,
                       {"observation_mode": observation_mode,
                        "gripper_channel": "actor_observation" if observation_mode == "observed" else "ideal_encoder"})


class Dataset:
    def __init__(self, blocks: list[SampleBlock] | None = None):
        self.blocks: list[SampleBlock] = list(blocks or [])

    def add(self, blocks: list[SampleBlock]) -> None:
        self.blocks += blocks

    @property
    def n(self) -> int:
        return sum(b.n for b in self.blocks)

    def arrays(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.blocks:
            raise ValueError("empty dataset")
        X = np.concatenate([b.x_hier if kind == "hier" else b.x_flat for b in self.blocks])
        Y = np.concatenate([b.y for b in self.blocks])
        return X, Y

    def composition(self) -> dict:
        out: dict = {}
        for b in self.blocks:
            out[b.provenance] = out.get(b.provenance, 0) + b.n
        return out
