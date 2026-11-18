"""Execution monitor and subgoal-level failure attribution (paper Sec. 5.4).

Output for a failed episode: ``(subgoal, failure_type, cause, confidence)`` plus the time
of failure onset, which the RSI controller uses to pick restart snapshots.

``attribute_auto`` reads only what a deployed system can observe: the System-2 subgoal
trace, verifier predicates on perceived poses, the proprioceptive held estimate and the
commanded gripper. ``attribute_oracle`` reads privileged simulator truth and the log of
injected perturbations; it is the upper bound used for the "oracle attribution"
baseline and for the confusion matrix (paper Fig. 7).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..datagen.rollout import PHASE_ID, EpisodeLog
from ..sim.scene import SceneSpec
from ..taskgraph.schema import TaskGraph

FAILURE_TYPES = ("grasp", "drop", "placement", "regression", "wrong_object", "transition", "stall", "verifier")
GRASP_PHASES = {PHASE_ID[p] for p in ("approach", "rise", "open", "descend_grasp", "close", "reopen")}
PLACE_PHASES = {PHASE_ID[p] for p in ("align_place", "descend_place", "release")}
TRANSPORT_PHASES = {PHASE_ID[p] for p in ("lift", "transport", "settle_grasp")}
BOUNDARY = 30  # steps after a subgoal switch that count as the transition window


@dataclass
class Attribution:
    subgoal: int
    stage: int
    ftype: str
    cause: str
    confidence: float
    t_fail: int
    victim: int | None = None  # regressed subgoal (for regression failures)

    def to_dict(self) -> dict:
        return asdict(self)


def _segments(k: np.ndarray, j: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) windows where k == j."""
    out, start = [], None
    for t, v in enumerate(k):
        if v == j and start is None:
            start = t
        elif v != j and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, len(k)))
    return out


class _Ctx:
    def __init__(self, log: EpisodeLog):
        self.log = log
        self.graph = TaskGraph.from_dict(log.graph)
        self.scene = SceneSpec.from_dict(log.scene)
        self.order = self.graph.order()
        self.obj_of = [self.scene.object_index(g.obj) for g in self.graph.subgoals]

    def stage(self, j: int) -> int:
        return self.order.index(j)


def _first_unsat(sat_row: np.ndarray, order: list[int]) -> int | None:
    for j in order:
        if not sat_row[j]:
            return j
    return None


def _classify(c: _Ctx, j: int, k_trace: np.ndarray, sat: np.ndarray, held: np.ndarray, grip_cmd: np.ndarray,
              grip: np.ndarray, obj: np.ndarray) -> tuple[str, int, float, int | None]:
    """Shared event logic over a window of subgoal j. Returns (type, t_fail, confidence, victim)."""
    oi = c.obj_of[j]
    segs = _segments(k_trace, j)
    if not segs:
        # never worked on: the episode stalled earlier in the order or ran out of time at a boundary
        prev_t = int(np.nonzero(k_trace >= 0)[0][-1]) if (k_trace >= 0).any() else 0
        return "transition", prev_t, 0.4, None
    w0, w1 = segs[-1]
    first_entry = segs[0][0]
    # wrong object held while working on j
    wrong = [t for t in range(w0, w1) if held[t] >= 0 and held[t] != oi]
    # regression of an earlier subgoal during j's windows
    victim, t_reg = None, None
    for a, b in segs:
        for t in range(max(a, 1), b):
            for i in c.order[: c.stage(j)]:
                if sat[t - 1, i] and not sat[t, i]:
                    victim, t_reg = i, t
                    break
            if victim is not None:
                break
        if victim is not None:
            break
    # held transitions of the target object
    drops, releases_at_target = [], []
    for a, b in segs:
        for t in range(max(a, 1), b):
            if held[t - 1] == oi and held[t] != oi:
                # a commanded opening is a placement attempt; losing the object while the
                # gripper is commanded closed is a drop (slip, collision, weak grasp)
                commanded = grip_cmd[t - 1] < 0 or grip_cmd[t] < 0
                (releases_at_target if commanded else drops).append(t)
    grasp_attempts = sum(1 for a, b in segs for t in range(max(a, 1), b)
                         if grip_cmd[t] > 0 and grip_cmd[t - 1] <= 0 and np.linalg.norm(grip[t, :2] - obj[t, oi, :2]) < 0.04)
    ever_held = any(held[t] == oi for a, b in segs for t in range(a, b))
    if wrong:
        return "wrong_object", wrong[0], 0.85, None
    if victim is not None and not sat[-1, victim]:
        return "regression", int(t_reg), 0.8, victim
    if drops:
        return "drop", drops[-1], 0.8, None
    if releases_at_target and not sat[-1, j]:
        return "placement", releases_at_target[-1], 0.75, None
    if grasp_attempts > 0 and not ever_held:
        return "grasp", w0, 0.75, None
    if first_entry > 0 and (w1 - first_entry) < BOUNDARY and len(segs) == 1:
        return "transition", first_entry, 0.5, None
    if not ever_held and grasp_attempts == 0:
        return "transition" if (w0 - first_entry) < BOUNDARY else "stall", w0, 0.45, None
    return "stall", max(w0, w1 - 60), 0.4, None


def attribute_auto(log: EpisodeLog) -> Attribution | None:
    """Attribute a failed episode from observable signals only."""
    if log.success:
        return None
    c = _Ctx(log)
    sat = log.sat_ver
    k = log.k_sel.astype(int)
    j = _first_unsat(sat[-1], c.order)
    if j is None:
        # verifier believes the task is complete but the outcome audit says it failed
        jj = int(k[k >= 0][-1]) if (k >= 0).any() else c.order[-1]
        return Attribution(jj, c.stage(jj), "verifier", "false_success", 0.6, log.T - 1)
    ftype, t_fail, conf, victim = _classify(c, j, k, sat, log.held_est.astype(int), log.action[:, 4],
                                            log.grip, log.obj_obs)
    return Attribution(j, c.stage(j), ftype, "policy", conf, int(t_fail), victim)


def attribute_oracle(log: EpisodeLog) -> Attribution | None:
    """Ground-truth attribution from privileged state and injected-perturbation records."""
    if log.success:
        return None
    c = _Ctx(log)
    sat = log.sat_true
    k = log.k_true.astype(int)
    j = _first_unsat(sat[-1], c.order)
    held = log.held_true.astype(int)
    if j is None:
        if log.meta.get("final_held", -1) >= 0:
            jj = int(c.obj_of.index(log.meta["final_held"])) if log.meta["final_held"] in c.obj_of else c.order[-1]
            return Attribution(jj, c.stage(jj), "stall", "policy", 1.0, log.T - 1)
        return Attribution(c.order[-1], c.stage(c.order[-1]), "stall", "policy", 1.0, log.T - 1)
    # verifier false success: the controller moved on while the subgoal was not truly done
    ks = log.k_sel.astype(int)
    if (ks >= 0).any():
        st_j = c.stage(j)
        moved_on = [t for t in range(len(ks)) if ks[t] >= 0 and c.stage(int(ks[t])) > st_j and not sat[t, j]
                    and log.sat_ver[t, j]]
        if len(moved_on) > 20:
            return Attribution(j, st_j, "verifier", "false_success", 1.0, moved_on[0])
    ftype, t_fail, _, victim = _classify(c, j, k, sat, held, log.action[:, 4], log.grip, log.obj_obs)
    cause = "policy"
    for (t, kind, info) in log.events:
        if info.get("subgoal") == j or (victim is not None and kind == "knock"):
            if (kind == "slip" and ftype == "drop") or (kind == "knock" and ftype == "regression") or \
                    (kind in ("displace", "target", "traj") and ftype in ("grasp", "stall", "transition")):
                cause = f"perturbation:{kind}"
    return Attribution(j, c.stage(j), ftype, cause, 1.0, int(t_fail), victim)


def confusion(pairs: list[tuple[Attribution | None, Attribution | None]]) -> dict:
    """Agreement between automatic and oracle attribution (subgoal and type)."""
    n = 0
    sg_ok = 0
    ty_ok = 0
    mat = {a: {b: 0 for b in FAILURE_TYPES} for a in FAILURE_TYPES}
    for auto, orc in pairs:
        if auto is None or orc is None:
            continue
        n += 1
        sg_ok += int(auto.subgoal == orc.subgoal)
        ty_ok += int(auto.ftype == orc.ftype)
        mat[orc.ftype][auto.ftype] += 1
    return {"n": n, "subgoal_acc": sg_ok / max(1, n), "type_acc": ty_ok / max(1, n), "matrix": mat}
