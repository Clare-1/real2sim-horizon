"""Episode execution with subgoal-resolution logging, perturbation injection and snapshots.

One loop serves every consumer: expert demonstration generation (nominal and ARSTAG-style
recovery demos), policy evaluation, DAgger relabelling (``label_expert``) and RSI
regeneration from restored snapshots (``start``). Every step records the privileged
truth (true predicates, true held object, true System-2 subgoal) next to what the
controller perceived, which is what lets the monitor compare automatic and oracle
failure attribution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..monitor.verifier import Verifier, VerifierConfig
from ..sim.env import EnvConfig, Snapshot, TabletopEnv
from ..sim.state import WorldState
from ..skills.expert import ExpertConfig, TaskExpert
from ..taskgraph.predicates import PredicateEngine
from ..tasks.factory import TaskInstance

PHASES = ("approach", "rise", "open", "descend_grasp", "close", "reopen", "settle_grasp", "lift", "transport",
          "align_place", "descend_place", "release", "retreat", "release_wrong", "wait_settle", "done", "policy")
PHASE_ID = {p: i for i, p in enumerate(PHASES)}
PERT_KINDS = ("slip", "displace", "knock", "traj", "target")


@dataclass
class PerturbationConfig:
    """Per-subgoal probabilities of injected perturbations.

    ``traj`` and ``target`` reproduce ARSTAG's recovery-demo perturbations (arm displaced
    during approach; grasp pose offset right before closing). ``slip``, ``displace`` and
    ``knock`` form the evaluation perturbation suite (paper Sec. 6.8).
    """

    slip: float = 0.0
    displace: float = 0.0
    knock: float = 0.0
    traj: float = 0.0
    target: float = 0.0

    def any(self) -> bool:
        return any(getattr(self, k) > 0 for k in PERT_KINDS)

    @staticmethod
    def suite(p: float) -> "PerturbationConfig":
        return PerturbationConfig(slip=p, displace=p, knock=p * 0.5)


class Controller(Protocol):
    privileged: bool

    def reset(self, task: TaskInstance) -> None: ...

    def act(self, obs: WorldState, true: WorldState) -> tuple[np.ndarray, int]: ...


@dataclass
class EpisodeLog:
    key: str
    family: str
    H: int
    seed: int
    scene: dict
    graph: dict
    grip: np.ndarray  # (T, 6) ideal encoder: tcp xyz, yaw, opening, opening velocity
    obj_obs: np.ndarray  # (T, N, 7) perceived xyz + quaternion (wxyz)
    action: np.ndarray  # (T, 5) executed (commanded) action
    expert_action: np.ndarray | None  # (T, 5) relabelled expert action
    k_sel: np.ndarray  # (T,) subgoal the controller worked on (-1: done / unknown)
    k_true: np.ndarray  # (T,) privileged System-2 subgoal
    sat_true: np.ndarray  # (T, H) true effect predicates
    sat_ver: np.ndarray  # (T, H) verifier effect predicates
    held_true: np.ndarray  # (T,)
    held_est: np.ndarray  # (T,)
    phase: np.ndarray  # (T,) expert phase id at each visited state
    valid: np.ndarray  # (T,) supervision mask (False on perturbation steps)
    events: list = field(default_factory=list)  # (t, kind, info)
    success: bool = False
    success_ver: bool = False
    steps: int = 0
    sim_steps: int = 0
    wall: float = 0.0
    provenance: str = "expert"
    start_t: int = 0
    snapshots: dict = field(default_factory=dict)  # t -> Snapshot
    meta: dict = field(default_factory=dict)
    sat_plan: np.ndarray | None = None  # (T, H) true effect predicates of the robot's own plan, if it differs from ``graph``
    grip_obs: np.ndarray | None = None  # (T, 6) actor observation; absent in historical logs

    @property
    def T(self) -> int:
        return int(self.action.shape[0])

    def n_valid(self) -> int:
        return int(self.valid.sum())

    def strip(self) -> "EpisodeLog":
        """Drop heavy fields not needed for training (snapshots)."""
        self.snapshots = {}
        return self


class ExpertController:
    privileged = True

    def __init__(self, env_cfg: EnvConfig | None = None, cfg: ExpertConfig | None = None, noise: float = 0.0,
                 rng: np.random.Generator | None = None):
        self.env_cfg, self.cfg, self.noise = env_cfg, cfg, noise
        self.rng = rng or np.random.default_rng(0)
        self.expert: TaskExpert | None = None

    def reset(self, task: TaskInstance) -> None:
        self.expert = TaskExpert(task.scene, task.graph, self.env_cfg, self.cfg)

    def act(self, obs: WorldState, true: WorldState) -> tuple[np.ndarray, int]:
        a, k, _ = self.expert.act(true)
        if self.noise > 0:
            a = a.copy()
            a[:4] += self.rng.normal(0, self.noise, 4)
        return a, (-1 if k is None else k)


class _Injector:
    """Schedules at most one perturbation of each kind per subgoal, triggered by true state."""

    def __init__(self, cfg: PerturbationConfig, rng: np.random.Generator, family: str):
        self.cfg, self.rng, self.family = cfg, rng, family
        self.armed: dict[int, set[str]] = {}
        self.slip_delay: dict[int, int] = {}
        self.lift_seen: dict[int, int] = {}
        self.pick_bottom: dict[int, float] = {}

    def arm(self, k: int) -> None:
        if k in self.armed or k < 0:
            return
        kinds = {kind for kind in PERT_KINDS if self.rng.random() < getattr(self.cfg, kind)}
        if self.family == "stack":
            kinds.discard("knock")  # knocking a mid-stack layer forces a full unstack; out of scope
        self.armed[k] = kinds
        self.slip_delay[k] = int(self.rng.integers(0, 3))

    def step(self, env: TabletopEnv, s: WorldState, k: int, task: TaskInstance, eng: PredicateEngine,
             grip_cmd: float) -> list[tuple[str, dict]]:
        if k < 0 or k not in self.armed or not self.armed[k]:
            return []
        fired: list[tuple[str, dict]] = []
        kinds = self.armed[k]
        g = task.graph.subgoals[k]
        i = eng.obj_idx[g.obj]
        rng = self.rng
        tcp = s.grip_pos
        d_xy = float(np.linalg.norm(tcp[:2] - s.obj_pos[i, :2]))
        geom = eng.geom
        if "knock" in kinds:
            done_objs = [eng.obj_idx[task.graph.subgoals[j].obj] for j in range(task.H)
                         if j != k and eng.effect(s, task.graph.subgoals[j])]
            kinds.discard("knock")
            if done_objs:
                j = int(rng.choice(done_objs))
                spot = env.free_spot(j, rng, 0.05, 0.10)
                if spot is not None:
                    env.displace_object(j, None, rng.uniform(-0.5, 0.5), to_xy=spot)
                    fired.append(("knock", {"obj": j, "to": spot.tolist()}))
        if "traj" in kinds and s.held < 0 and 0.08 < d_xy < 0.3:
            kinds.discard("traj")
            off = rng.uniform(-1, 1, 3) * np.array([0.06, 0.06, 0.03])
            env.nudge_gripper(off)
            fired.append(("traj", {"dpos": off.tolist()}))
        if "displace" in kinds and s.held < 0 and d_xy < 0.08 and tcp[2] < geom.top(i, s) + 0.12:
            kinds.discard("displace")
            spot = env.free_spot(i, rng, 0.03, 0.06)
            if spot is not None:
                env.displace_object(i, None, rng.uniform(-0.6, 0.6), to_xy=spot)
                fired.append(("displace", {"obj": i, "to": spot.tolist()}))
        if "target" in kinds and s.held < 0 and d_xy < 0.012 and tcp[2] < s.obj_pos[i, 2] + 0.02 and grip_cmd > 0:
            kinds.discard("target")
            ang = rng.uniform(0, 2 * np.pi)
            off = rng.uniform(0.012, 0.025) * np.array([np.cos(ang), np.sin(ang), 0.0])
            env.nudge_gripper(off)
            fired.append(("target", {"dpos": off.tolist()}))
        if "slip" in kinds and s.held == i:
            # a weak grasp lets go early in the lift, a few centimetres above where it was picked
            # (a drop from travel height would topple cans, which needs non-prehensile reorientation)
            rise = geom.bottom(i, s) - self.pick_bottom.setdefault(k, geom.bottom(i, s))
            if 0.012 < rise < 0.05:
                self.lift_seen[k] = self.lift_seen.get(k, 0) + 1
                if self.lift_seen[k] > self.slip_delay[k]:
                    kinds.discard("slip")
                    env.force_release(4)
                    fired.append(("slip", {"obj": i}))
        return fired


def rollout(task: TaskInstance, controller: Controller, *, seed: int = 0, env_cfg: EnvConfig | None = None,
            pert: PerturbationConfig | None = None, verifier_cfg: VerifierConfig | None = None,
            max_steps: int | None = None, label_expert: bool = False, snapshot_every: int = 0,
            snapshot_events: bool = False, start: Snapshot | None = None, stop_when: str = "done",
            stop_subgoal: int | None = None, provenance: str = "expert", expert_cfg: ExpertConfig | None = None,
            sim_cfg=None,
            done_patience: int = 8) -> EpisodeLog:
    """Run one episode.

    stop_when:
      * ``"done"``    - stop once the controller reports completion (subgoal -1) for
                        ``done_patience`` steps, or at ``max_steps``;
      * ``"subgoal"`` - stop once the privileged System 2 moves past ``stop_subgoal``
                        (used by regeneration operators that only need a segment).
    """
    env_cfg = env_cfg or EnvConfig()
    env = TabletopEnv(task.scene, env_cfg=env_cfg, sim_cfg=sim_cfg, seed=seed)
    rng = np.random.default_rng(seed + 7)
    if start is None:
        obs = env.reset()
    else:
        env.reset()
        obs = env.set_snapshot(start, reseed=seed)
    eng = PredicateEngine(task.scene)  # ground truth (physics scene)
    view = task.controller_view()  # what the robot believes; its verifier measures with this geometry
    ver = Verifier(view.scene, verifier_cfg)
    oracle = TaskExpert(task.scene, task.graph, env_cfg, expert_cfg, eng)
    order = task.graph.order()
    if start is not None:
        oracle.current = None
    controller.reset(view)
    injector = _Injector(pert or PerturbationConfig(), rng, task.family)
    H, N = task.H, env.n_obj
    max_steps = max_steps or (H * 180 + 60)
    plan = view.graph  # the robot's own plan: what its judge verifies (differs from task.graph only if mis-compiled)
    rows: dict[str, list] = {k: [] for k in ("grip", "grip_obs", "obj", "act", "exp", "ksel", "ktrue", "sat", "satv", "satp",
                                              "held", "heldv", "phase", "valid")}
    events: list = []
    snaps: dict[int, Snapshot] = {}
    t0 = time.perf_counter()
    done_count = 0
    prev_k = None
    for _ in range(max_steps):
        true = env.true_state()
        k_true = oracle.select(true)
        kt = -1 if k_true is None else k_true
        if snapshot_events and kt != prev_k:
            snaps[env.t] = env.get_snapshot()
        if snapshot_every and env.t % snapshot_every == 0:
            snaps.setdefault(env.t, env.get_snapshot())
        injector.arm(kt)
        # Copy the observation already delivered to the actor. Do not observe again:
        # another sensor read would advance the noise RNG and change the rollout.
        rows["grip_obs"].append([*obs.grip_pos, obs.grip_yaw, obs.grip_open, obs.grip_open_vel])
        a, k_sel = controller.act(obs, true)
        if label_expert or controller.privileged:
            ea, _, phase = oracle.act(true, k_true) if k_true is not None else (np.array([0, 0, 0.3, 0, -1.0]), None, "done")
        else:
            ea, phase = None, "policy"
            if k_true is not None:
                _, phase, _ = oracle.skill.act(true, task.graph.subgoals[k_true])
        fired = injector.step(env, true, kt, task, eng, float(a[4]))
        valid = True
        if fired:
            for kind, info in fired:
                events.append((env.t, kind, {**info, "subgoal": kt}))
            if snapshot_events:
                snaps[env.t] = env.get_snapshot()
            valid = False
        if env.forced_open > 0:
            valid = False
        perceived = ver.perceive(obs)
        rows["grip"].append([*true.grip_pos, true.grip_yaw, true.grip_open, true.grip_open_vel])
        rows["obj"].append(np.concatenate([obs.obj_pos, obs.obj_quat], axis=1) if N else np.zeros((0, 7)))
        rows["act"].append(a)
        rows["exp"].append(ea if ea is not None else np.zeros(5))
        rows["ksel"].append(k_sel)
        rows["ktrue"].append(kt)
        rows["sat"].append(eng.satisfied_vector(true, task.graph))
        rows["satv"].append(ver.eng.satisfied_vector(perceived, plan))
        if plan is not task.graph:
            rows["satp"].append(eng.satisfied_vector(true, plan))
        rows["held"].append(true.held)
        rows["heldv"].append(perceived.held)
        rows["phase"].append(PHASE_ID.get(phase, PHASE_ID["policy"]))
        rows["valid"].append(valid)
        prev_k = kt
        if stop_when == "subgoal" and stop_subgoal is not None:
            pos = order.index(stop_subgoal)
            if k_true is None or (k_true in order and order.index(k_true) > pos and true.held < 0):
                break
        if k_sel < 0:
            done_count += 1
            if done_count >= done_patience:
                break
        else:
            done_count = 0
        obs = env.step(a)
    true = env.true_state()
    log = EpisodeLog(
        key=task.key, family=task.family, H=H, seed=task.seed, scene=task.scene.to_dict(), graph=task.graph.to_dict(),
        grip=np.asarray(rows["grip"], dtype=np.float32).reshape(-1, 6),
        obj_obs=np.asarray(rows["obj"], dtype=np.float32).reshape(-1, N, 7),
        action=np.asarray(rows["act"], dtype=np.float32).reshape(-1, 5),
        expert_action=np.asarray(rows["exp"], dtype=np.float32).reshape(-1, 5) if (label_expert or controller.privileged) else None,
        k_sel=np.asarray(rows["ksel"], dtype=np.int8), k_true=np.asarray(rows["ktrue"], dtype=np.int8),
        sat_true=np.asarray(rows["sat"], dtype=bool).reshape(-1, H), sat_ver=np.asarray(rows["satv"], dtype=bool).reshape(-1, H),
        held_true=np.asarray(rows["held"], dtype=np.int8), held_est=np.asarray(rows["heldv"], dtype=np.int8),
        phase=np.asarray(rows["phase"], dtype=np.int8), valid=np.asarray(rows["valid"], dtype=bool), events=events,
        success=eng.task_success(true, task.graph), success_ver=ver.task_success(env.observe(), plan),
        steps=env.t, sim_steps=env.sim_steps, wall=time.perf_counter() - t0, provenance=provenance,
        start_t=0 if start is None else start.t, snapshots=snaps,
        meta={"final_sat": eng.satisfied_vector(true, task.graph).tolist(), "final_held": int(true.held),
              **({"belief": task.belief.to_dict()} if task.belief is not None else {}),
              **({"belief_graph": plan.to_dict(), "plan_final_sat": eng.satisfied_vector(true, plan).tolist()}
                 if plan is not task.graph else {})},
        sat_plan=np.asarray(rows["satp"], dtype=bool).reshape(-1, H) if plan is not task.graph else None,
        grip_obs=np.asarray(rows["grip_obs"], dtype=np.float32).reshape(-1, 6),
    )
    return log
