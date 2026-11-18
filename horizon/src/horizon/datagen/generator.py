"""Compositional data generator (paper Sec. 5.3) and the ARSTAG-style open-loop recipe.

Demonstrations are produced by executing the state-based expert subgoal by subgoal: each
subgoal starts from the state the previous one actually ended in, so stitching is
physically consistent, and the whole trajectory is validated against every subgoal
predicate before it is kept. Task-consistent randomisation comes from the task-family
samplers; recovery branches come from perturbations injected at subgoal boundaries,
with the perturbed steps excluded from supervision (ARSTAG Sec. III-B).

Budgets are counted in *supervised transitions*, the unit used by the matched-budget
protocol (paper Sec. 6.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.env import EnvConfig
from .parallel import Job, JobResult, Runner
from .rollout import PerturbationConfig


@dataclass
class TaskDistribution:
    families: tuple[str, ...] = ("pack", "sort", "stack")
    H_values: tuple[int, ...] = (2, 4, 6, 8, 10)
    H_weights: tuple[float, ...] | None = None
    seed_base: int = 0  # training tasks use seed_base + [0, 1e6); held-out tests use a disjoint range

    def sample(self, rng: np.random.Generator) -> tuple[str, int, int]:
        fam = str(rng.choice(self.families))
        w = None
        if self.H_weights is not None:
            w = np.asarray(self.H_weights, dtype=float)
            w = w / w.sum()
        H = int(rng.choice(self.H_values, p=w))
        return fam, H, int(self.seed_base + rng.integers(0, 1_000_000))


@dataclass
class OpenLoopConfig:
    """ARSTAG recipe: nominal demos + recovery demos from injected execution perturbations."""

    recovery_frac: float = 0.3
    recovery_pert: PerturbationConfig = field(default_factory=lambda: PerturbationConfig(
        slip=0.15, displace=0.15, knock=0.1, traj=0.25, target=0.25))
    latency_choices: tuple[int, ...] = (0, 0, 1)
    expert_noise: float = 0.0


@dataclass
class CollectStats:
    attempted: int = 0
    kept: int = 0
    transitions: int = 0
    sim_steps: int = 0
    wall: float = 0.0
    shortfall: int = 0  # budget left unfilled after a stop-loss (refilled with nominal demos by the caller)
    errors: list = field(default_factory=list)

    def merge(self, other: "CollectStats") -> None:
        self.attempted += other.attempted
        self.kept += other.kept
        self.transitions += other.transitions
        self.sim_steps += other.sim_steps
        self.wall += other.wall
        self.errors += other.errors[:5]


def open_loop_job(dist: TaskDistribution, cfg: OpenLoopConfig, rng: np.random.Generator,
                  provenance: str = "expert") -> Job:
    fam, H, ts = dist.sample(rng)
    recovery = rng.random() < cfg.recovery_frac
    return Job(
        family=fam, H=H, task_seed=ts, seed=int(rng.integers(0, 2**31 - 1)), controller="expert",
        pert=cfg.recovery_pert if recovery else PerturbationConfig(),
        env_cfg=EnvConfig(latency=int(rng.choice(cfg.latency_choices))), expert_noise=cfg.expert_noise,
        provenance="recovery" if recovery else provenance, only_success=True,
    )


def collect(runner: Runner, make_job, budget: int, batch: int | None = None, max_batches: int = 200,
            min_yield: float = 0.15, require_success: bool = True) -> tuple[list, CollectStats]:
    """Run jobs from ``make_job()`` until ``budget`` supervised transitions are collected.

    Only successful (validated) rollouts contribute; the last block is truncated so the
    returned dataset matches the budget exactly. Requests whose starting states the expert
    cannot recover from (yield below ``min_yield`` after two batches) stop early and report
    the unfilled amount as ``shortfall``.
    """
    batch = batch or max(8, runner.workers * 2)
    blocks: list = []
    st = CollectStats()
    total = 0
    for _ in range(max_batches):
        if total >= budget:
            break
        res: list[JobResult] = runner.map([make_job() for _ in range(batch)])
        for r in res:
            st.attempted += 1
            st.sim_steps += r.sim_steps
            st.wall += r.wall
            if r.error:
                st.errors.append(r.error)
                continue
            if (require_success and not r.success) or r.samples is None or r.samples.n == 0:
                continue
            if total >= budget:
                break
            b = r.samples
            if total + b.n > budget:
                b = b.take(budget - total)
            blocks.append(b)
            total += b.n
            st.kept += 1
        if total < budget and st.attempted >= 2 * batch and st.kept / st.attempted < min_yield:
            break
    st.transitions = total
    st.shortfall = max(0, budget - total)
    return blocks, st


def generate_open_loop(runner: Runner, dist: TaskDistribution, budget: int, cfg: OpenLoopConfig | None = None,
                       seed: int = 0) -> tuple[list, CollectStats]:
    cfg = cfg or OpenLoopConfig()
    rng = np.random.default_rng(seed)
    return collect(runner, lambda: open_loop_job(dist, cfg, rng), budget)
