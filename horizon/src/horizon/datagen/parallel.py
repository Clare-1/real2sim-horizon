"""Process-pool execution of rollout jobs (Windows-safe: spawn + top-level worker fns).

A job fully describes one rollout: which task to instantiate (family, H, seed), which
controller (expert or a learned policy whose weights are shipped once per pool), the
perturbation / noise configuration, an optional start snapshot, and what to return
(features for training, snapshots for later regeneration).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..monitor.verifier import VerifierConfig
from ..sim.env import EnvConfig, Snapshot
from ..skills.expert import ExpertConfig
from .rollout import EpisodeLog, ExpertController, PerturbationConfig, rollout

_POLICY: dict = {}


@dataclass
class Job:
    family: str
    H: int
    task_seed: int
    seed: int
    controller: str = "expert"  # expert | hier | flat
    pert: PerturbationConfig = field(default_factory=PerturbationConfig)
    env_cfg: EnvConfig = field(default_factory=EnvConfig)
    verifier_cfg: VerifierConfig | None = None
    expert_noise: float = 0.0
    label_expert: bool = False
    snapshot_every: int = 0
    snapshot_events: bool = False
    start: Snapshot | None = None
    stop_subgoal: int | None = None
    max_steps: int | None = None  # cap for regenerated segments; None = full task budget
    beta: float = 0.0  # controller "mixed": probability of executing the expert's action
    learner: str = "hier"  # controller "mixed": learned policy kind
    max_samples: int | None = None  # cap on labelled states taken from one episode
    world: dict | None = None  # RealWorld dict: physical modifications of the deployment world
    recon: dict | None = None  # ReconModel dict: run in the reconstructed scene (sim) or believe it (real)
    real: bool = False  # True: physics = true scene, robot believes the reconstruction
    grounding: dict | None = None  # GroundingError dict: the robot's compiler mis-grounds the instruction
    physics: dict | None = None  # ReconModel dict: run inside a twin of the deployment world (real=True keeps the robot's belief)
    provenance: str = "expert"
    want_samples: bool = True
    keep_snapshots: bool = False
    keep_log: bool = False
    only_success: bool = False
    tag: dict = field(default_factory=dict)
    task_payload: dict | None = None  # versioned JSON task; bypasses procedural make_task when supplied
    observation_mode: str = "legacy"
    sim_cfg: dict | None = None  # SimConfig overrides (e.g. finger friction) for a dynamics gap  # training gripper channel: legacy ideal encoder | observed actor input


@dataclass
class JobResult:
    job: Job
    log: EpisodeLog | None
    samples: object | None
    success: bool
    n_valid: int
    sim_steps: int
    wall: float
    error: str | None = None


THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _init(policy: dict | None) -> None:
    for k in THREAD_VARS:
        os.environ[k] = "1"
    _POLICY.clear()
    if policy:
        _POLICY.update(policy)


def _run(job: Job) -> JobResult:
    from ..policy.controllers import make_controller
    from ..policy.dataset import episode_samples
    from ..tasks.factory import make_task

    try:
        if job.observation_mode not in ("legacy", "observed"):
            raise ValueError(f"unknown observation_mode: {job.observation_mode}")
        if job.task_payload is None:
            task = make_task(job.family, job.H, job.task_seed)
        else:
            from ..tasks.serialization import task_from_payload

            task = task_from_payload(job.task_payload, expected_family=job.family,
                                     expected_H=job.H, expected_seed=job.task_seed)
        if job.world is not None or job.recon is not None:
            import dataclasses

            from ..real2sim.recon_error import ReconModel, RealWorld

            _world = RealWorld.from_dict(job.world)
            task = dataclasses.replace(task, scene=_world.apply(task.scene))
            if _world.latency:
                job.env_cfg = dataclasses.replace(job.env_cfg, latency=int(_world.latency))
            if getattr(_world, "finger_friction", None) is not None:
                job.sim_cfg = {**(job.sim_cfg or {}), "finger_friction": float(_world.finger_friction)}
        _sim_cfg = None
        if job.sim_cfg:
            from ..sim.mjcf import SimConfig as _SC
            _sim_cfg = _SC(**job.sim_cfg)
        real_scene = task.scene
        if job.physics is not None:  # evaluate inside a twin: the twin's geometry stands in for the real scene
            import dataclasses

            from ..real2sim.recon_error import ReconModel

            task = dataclasses.replace(task, scene=ReconModel.from_dict(job.physics).apply(real_scene))
        if job.recon is not None:
            believed = ReconModel.from_dict(job.recon).apply(real_scene)
            task = dataclasses.replace(task, belief=believed) if job.real else dataclasses.replace(task, scene=believed)
        if job.grounding is not None:
            import dataclasses

            from ..taskgraph.grounding import GroundingError

            planned = GroundingError.from_dict(job.grounding).apply(task.scene, task.graph)
            task = dataclasses.replace(task, belief_graph=planned) if job.real else dataclasses.replace(task, graph=planned)
        if job.controller == "expert":
            ctrl = ExpertController(job.env_cfg, ExpertConfig(), noise=job.expert_noise,
                                    rng=np.random.default_rng(job.seed + 11))
        elif job.controller == "mixed":
            from ..policy.controllers import MixedController

            ctrl = MixedController(make_controller(job.learner, _POLICY["params"], job.verifier_cfg),
                                   ExpertController(job.env_cfg, ExpertConfig()), job.beta,
                                   np.random.default_rng(job.seed + 13))
        else:
            ctrl = make_controller(job.controller, _POLICY["params"], job.verifier_cfg)
        log = rollout(
            task, ctrl, seed=job.seed, env_cfg=job.env_cfg, pert=job.pert, verifier_cfg=job.verifier_cfg,
            label_expert=job.label_expert, snapshot_every=job.snapshot_every, snapshot_events=job.snapshot_events,
            start=job.start, stop_when="subgoal" if job.stop_subgoal is not None else "done",
            sim_cfg=_sim_cfg,
            stop_subgoal=job.stop_subgoal, provenance=job.provenance, max_steps=job.max_steps,
        )
        ok = log.success if job.stop_subgoal is None else _segment_ok(log, job.stop_subgoal)
        samples = None
        if job.want_samples and (ok or not job.only_success):
            samples = episode_samples(log, max_samples=job.max_samples, seed=job.seed,
                                      observation_mode=job.observation_mode)
        if not job.keep_snapshots:
            log.snapshots = {}
        return JobResult(job if job.start is None else _light(job), log if job.keep_log else None, samples, ok,
                         log.n_valid(), log.sim_steps, log.wall)
    except Exception as e:  # keep the pool alive; failures are reported to the caller
        import traceback
        return JobResult(_light(job), None, None, False, 0, 0, 0.0, error=f"{e}\n{traceback.format_exc()[-800:]}")


def _light(job: Job) -> Job:
    j = Job(**{**job.__dict__})
    j.start = None
    return j


def _segment_ok(log: EpisodeLog, stop_subgoal: int) -> bool:
    """A regenerated segment is kept if every subgoal up to ``stop_subgoal`` holds at the end."""
    from ..taskgraph.schema import TaskGraph

    order = TaskGraph.from_dict(log.graph).order()
    final = np.asarray(log.meta["final_sat"], dtype=bool)
    upto = order[: order.index(stop_subgoal) + 1]
    return bool(final[upto].all()) and log.meta["final_held"] < 0


class Runner:
    """Thin wrapper around a process pool; reuse it across calls with the same policy."""

    def __init__(self, workers: int | None = None, policy: dict | None = None):
        import multiprocessing as mp

        self.workers = workers or max(1, (os.cpu_count() or 2) - 2)
        self.ctx = mp.get_context("spawn")
        # single-threaded BLAS in the workers: spawned children import numpy before ``_init`` runs, so the
        # limit must be in the environment they inherit (otherwise every worker reserves per-core buffers
        # and many parallel pools exhaust the Windows commit limit)
        saved = {k: os.environ.get(k) for k in THREAD_VARS}
        os.environ.update({k: "1" for k in THREAD_VARS})
        try:
            self.pool = self.ctx.Pool(self.workers, initializer=_init, initargs=(policy,))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def map(self, jobs: list[Job], chunksize: int = 1) -> list[JobResult]:
        if not jobs:
            return []
        return list(self.pool.imap(_run, jobs, chunksize=chunksize))

    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_serial(jobs: list[Job], policy: dict | None = None) -> list[JobResult]:
    _init(policy)
    return [_run(j) for j in jobs]
