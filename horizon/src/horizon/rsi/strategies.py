"""Data-allocation strategies for one self-improvement round (paper Sec. 5.5, 6.4).

Every strategy receives the same evaluation rollouts of the current policy and the same
transition budget, and returns generation requests. They differ only in *what* they ask
the expert to produce:

* ``ours``      - subgoal-level attribution -> failure-type-specific operators
                  (regenerate subgoal / recover from failure state / boundary recovery
                  branches), budget proportional to attributed failure mass.
* ``oracle``    - same, with privileged (ground-truth) attribution: the upper bound.
* ``uniform``   - same operators, but source states and operators drawn uniformly over
                  all visited states (closed loop, no attribution).
* ``dagger``    - relabel uniformly sampled visited states with expert actions.
* ``embodirsi`` - EmbodiRSI-style: corrective continuation from the episode's error state
                  to task completion (CEC) + extra nominal data for task types in
                  proportion to their failure rate (ADC). No subgoal typing.
* ``hybrid``    - half of the round budget allocated by attribution (dominant failures),
                  half uniformly (keeps the hand-offs between subgoals covered).
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..datagen.generator import OpenLoopConfig, TaskDistribution, open_loop_job
from ..datagen.parallel import Job
from ..datagen.rollout import EpisodeLog, PerturbationConfig
from ..monitor.attribution import Attribution
from ..sim.env import EnvConfig
from ..taskgraph.schema import TaskGraph

# failure type -> [(operator, share)]; operators are defined in ``_op_job`` below
ROUTING: dict[str, list[tuple[str, float]]] = {
    "grasp": [("recover", 0.5), ("regen", 0.3), ("boundary", 0.2)],
    "drop": [("recover", 0.5), ("boundary", 0.5)],
    "placement": [("recover", 0.6), ("regen", 0.4)],
    "regression": [("recover", 0.6), ("regen", 0.4)],
    "wrong_object": [("regen", 0.6), ("recover", 0.4)],
    "transition": [("regen", 0.6), ("boundary", 0.4)],
    "stall": [("recover", 0.7), ("regen", 0.3)],
    "verifier": [("regen", 1.0)],
}
BOUNDARY_PERT = PerturbationConfig(slip=0.3, displace=0.3, knock=0.15, traj=0.3, target=0.3)
SEGMENT_MAX_STEPS = 500  # a regenerated segment covers one or two subgoals (~90 steps each) plus recovery
OFF_TABLE_Z = -0.05  # an object observed below the table plane has fallen off: unrecoverable


@dataclass
class GenRequest:
    label: str
    budget: int
    make_job: Callable[[], Job] | None = None
    samples: list | None = None  # pre-computed samples (DAgger relabelling)
    require_success: bool = True  # False: keep failed rollouts (DAgger collection)


@dataclass
class RoundInputs:
    logs: list[EpisodeLog]  # evaluation rollouts of the current policy (with snapshots)
    auto: list[Attribution | None]
    oracle: list[Attribution | None]
    budget: int
    rng: np.random.Generator
    dist: TaskDistribution
    open_cfg: OpenLoopConfig
    nominal_frac: float = 0.2
    ops_enabled: tuple[str, ...] = ("regen", "recover", "boundary")
    meta: dict = field(default_factory=dict)
    policy: str = "hier"
    round: int = 1


# ---------------------------------------------------------------------- job builders
def _graph(log: EpisodeLog) -> TaskGraph:
    return TaskGraph.from_dict(log.graph)


def _next_in_order(log: EpisodeLog, j: int) -> int:
    order = _graph(log).order()
    p = order.index(j)
    return order[min(p + 1, len(order) - 1)]


def _snap_at_subgoal_start(log: EpisodeLog, j: int, last: bool = True):
    times = sorted(log.snapshots)
    hits = [t for t in times if t < log.T and int(log.k_true[t]) == j and (t == 0 or int(log.k_true[t - 1]) != j)]
    if not hits:
        hits = [t for t in times if t < log.T and int(log.k_true[t]) == j]
    if not hits:
        return None
    return log.snapshots[hits[-1] if last else hits[0]]


def _snap_before(log: EpisodeLog, t_fail: int, back: int = 6):
    times = [t for t in sorted(log.snapshots) if t <= max(0, t_fail - back)]
    if not times:
        times = sorted(log.snapshots)[:1]
    return log.snapshots[times[-1]] if times else None


def _base_job(log: EpisodeLog, rng: np.random.Generator, provenance: str) -> Job:
    return Job(family=log.family, H=log.H, task_seed=log.seed, seed=int(rng.integers(0, 2**31 - 1)),
               controller="expert", env_cfg=EnvConfig(), provenance=provenance, only_success=True)


def _op_job(op: str, log: EpisodeLog, a: Attribution | None, j: int, rng: np.random.Generator) -> Job | None:
    """Build one generation job for operator ``op`` targeting subgoal ``j`` of ``log``'s task."""
    job = _base_job(log, rng, op)
    if op == "regen":
        snap = _snap_at_subgoal_start(log, j, last=True)
        job.stop_subgoal = _next_in_order(log, j)
    elif op == "recover":
        t_fail = a.t_fail if a is not None else int(rng.integers(0, log.T))
        snap = _snap_before(log, t_fail)
        tail = a.victim if (a is not None and a.victim is not None) else j
        order = _graph(log).order()
        job.stop_subgoal = order[min(max(order.index(j), order.index(tail)) + 1, len(order) - 1)]
    elif op == "boundary":
        snap = _snap_at_subgoal_start(log, j, last=False)
        job.pert = BOUNDARY_PERT
        job.stop_subgoal = _next_in_order(log, j)
    elif op == "cec":
        t_fail = a.t_fail if a is not None else log.T - 1
        snap = _snap_before(log, t_fail)
        job.stop_subgoal = None
    else:
        raise ValueError(op)
    if snap is None:
        return None
    job.start = snap
    if job.stop_subgoal is not None:
        job.max_steps = SEGMENT_MAX_STEPS
    return job


def _factory(op: str, items: list[tuple[EpisodeLog, Attribution | None, int]], rng: np.random.Generator,
             fallback: Callable[[], Job]) -> Callable[[], Job]:
    def make() -> Job:
        for _ in range(6):
            log, a, j = items[int(rng.integers(len(items)))]
            job = _op_job(op, log, a, j, rng)
            if job is not None:
                return job
        return fallback()
    return make


# ---------------------------------------------------------------------- strategies
def plan_attributed(inp: RoundInputs, use_oracle: bool = False) -> list[GenRequest]:
    rng = inp.rng
    attrs = inp.oracle if use_oracle else inp.auto
    nominal = lambda: open_loop_job(inp.dist, inp.open_cfg, rng, provenance="nominal")  # noqa: E731
    fails = [(log, a) for log, a in zip(inp.logs, attrs) if a is not None]
    reqs: list[GenRequest] = []
    n_nom = int(inp.budget * inp.nominal_frac) if fails else inp.budget
    if fails:
        groups: dict[tuple, list] = defaultdict(list)
        for log, a in fails:
            groups[(log.family, a.ftype)].append((log, a, a.subgoal))
        targeted = inp.budget - n_nom
        total = sum(len(v) for v in groups.values())
        for (fam, ftype), items in sorted(groups.items()):
            share = targeted * len(items) / total
            ops = [(op, w) for op, w in ROUTING[ftype] if op in inp.ops_enabled] or [("regen", 1.0)]
            wsum = sum(w for _, w in ops)
            for op, w in ops:
                b = int(round(share * w / wsum))
                if b > 0:
                    reqs.append(GenRequest(f"{op}:{fam}:{ftype}", b, _factory(op, items, rng, nominal)))
    used = sum(r.budget for r in reqs)
    reqs.append(GenRequest("nominal", inp.budget - used, nominal))
    return reqs


def plan_uniform(inp: RoundInputs) -> list[GenRequest]:
    rng = inp.rng
    nominal = lambda: open_loop_job(inp.dist, inp.open_cfg, rng, provenance="nominal")  # noqa: E731
    n_nom = int(inp.budget * inp.nominal_frac)
    items = []
    for log in inp.logs:
        visited = sorted({int(k) for k in log.k_true if k >= 0})
        for j in visited:
            items.append((log, None, j))
    reqs: list[GenRequest] = []
    if items:
        ops = list(inp.ops_enabled)
        for op in ops:
            reqs.append(GenRequest(f"{op}:uniform", (inp.budget - n_nom) // len(ops), _factory(op, items, rng, nominal)))
    used = sum(r.budget for r in reqs)
    reqs.append(GenRequest("nominal", inp.budget - used, nominal))
    return reqs


def plan_dagger(inp: RoundInputs) -> list[GenRequest]:
    from ..policy.dataset import episode_samples

    rng = inp.rng
    blocks = []
    pool = []
    for li, log in enumerate(inp.logs):
        keep = log.valid & (log.k_true >= 0)
        # data hygiene: once an object has left the table the task is unrecoverable and the expert's
        # labels (reaching for an object below the table) are meaningless; drop those states
        fallen = np.nonzero((log.obj_obs[:, :, 2] < OFF_TABLE_Z).any(axis=1))[0]
        if len(fallen):
            keep[fallen[0]:] = False
        ok = np.nonzero(keep)[0]
        pool += [(li, int(t)) for t in ok]
    if not pool:
        return [GenRequest("nominal", inp.budget, lambda: open_loop_job(inp.dist, inp.open_cfg, rng))]
    take = min(inp.budget, len(pool))
    chosen = rng.choice(len(pool), size=take, replace=False)
    per_log: dict[int, list[int]] = defaultdict(list)
    for c in chosen:
        li, t = pool[int(c)]
        per_log[li].append(t)
    for li, ts in per_log.items():
        log = inp.logs[li]
        mask = np.zeros(log.T, dtype=bool)
        mask[ts] = True
        b = episode_samples(log, t_mask=mask)
        b.provenance = "dagger"
        blocks.append(b)
    reqs = [GenRequest("dagger", sum(b.n for b in blocks), samples=blocks)]
    if take < inp.budget:
        reqs.append(GenRequest("nominal", inp.budget - take, lambda: open_loop_job(inp.dist, inp.open_cfg, rng)))
    return reqs


def plan_embodirsi(inp: RoundInputs) -> list[GenRequest]:
    rng = inp.rng
    fails = [(log, a) for log, a in zip(inp.logs, inp.auto) if a is not None]
    reqs: list[GenRequest] = []
    nominal = lambda: open_loop_job(inp.dist, inp.open_cfg, rng, provenance="nominal")  # noqa: E731
    b_cec = inp.budget // 2 if fails else 0
    if fails:
        items = [(log, a, a.subgoal) for log, a in fails]
        reqs.append(GenRequest("cec", b_cec, _factory("cec", items, rng, nominal)))
    # ADC: nominal demos for task types weighted by observed failure rate
    rate: dict[tuple[str, int], list[int]] = defaultdict(list)
    for log in inp.logs:
        rate[(log.family, log.H)].append(0 if log.success else 1)
    types = sorted(rate)
    w = np.array([np.mean(rate[t]) + 0.05 for t in types])
    w = w / w.sum()

    def adc() -> Job:
        fam, H = types[int(rng.choice(len(types), p=w))]
        d = TaskDistribution(families=(fam,), H_values=(H,), seed_base=inp.dist.seed_base)
        return open_loop_job(d, inp.open_cfg, rng, provenance="adc")

    reqs.append(GenRequest("adc", inp.budget - b_cec, adc if types else nominal))
    return reqs


MAX_PER_EPISODE = 300


def plan_dagger_beta(inp: RoundInputs) -> list[GenRequest]:
    """Standard DAgger: beta-mixed collection rollouts on fresh training tasks, expert labels."""
    rng = inp.rng
    beta = 0.5 ** inp.round

    def make() -> Job:
        fam, H, ts = inp.dist.sample(rng)
        return Job(family=fam, H=H, task_seed=ts, seed=int(rng.integers(0, 2**31 - 1)), controller="mixed",
                   learner=inp.policy, beta=beta, pert=PerturbationConfig.suite(0.1), label_expert=True,
                   max_samples=MAX_PER_EPISODE, provenance="dagger", only_success=False)

    return [GenRequest(f"dagger_beta{beta:g}", inp.budget, make, require_success=False)]


ATTR_DAGGER_FRAC = 0.6  # share of collection episodes started at attributed failing subgoals


def plan_dagger_attr(inp: RoundInputs) -> list[GenRequest]:
    """Attribution-guided DAgger: beta-mixed collection, but most episodes start from a snapshot taken
    at the beginning of the subgoal that automatic attribution blamed for a failure (the rest start
    from fresh tasks, as in standard DAgger). Labels and the per-episode cap are unchanged."""
    rng = inp.rng
    beta = 0.5 ** inp.round
    fails = [(log, a) for log, a in zip(inp.logs, inp.auto) if a is not None]

    def fresh() -> Job:
        fam, H, ts = inp.dist.sample(rng)
        return Job(family=fam, H=H, task_seed=ts, seed=int(rng.integers(0, 2**31 - 1)), controller="mixed",
                   learner=inp.policy, beta=beta, pert=PerturbationConfig.suite(0.1), label_expert=True,
                   max_samples=MAX_PER_EPISODE, provenance="dagger", only_success=False)

    def make() -> Job:
        if fails and rng.random() < ATTR_DAGGER_FRAC:
            for _ in range(6):
                log, a = fails[int(rng.integers(len(fails)))]
                snap = _snap_at_subgoal_start(log, a.subgoal, last=True)
                if snap is not None:
                    job = fresh()
                    job.family, job.H, job.task_seed = log.family, log.H, log.seed
                    job.start, job.provenance = snap, "dagger_attr"
                    return job
        return fresh()

    return [GenRequest(f"dagger_attr_beta{beta:g}", inp.budget, make, require_success=False)]


HYBRID_ATTR_SHARE = 0.5  # hybrid: share of the round budget allocated by attribution, the rest uniformly


def plan_hybrid(inp: RoundInputs) -> list[GenRequest]:
    """Attribution targets the dominant failures; uniform sampling keeps every hand-off covered
    (the failure tables show the two are complementary: v3 report Sec. 4.4)."""
    b_attr = int(inp.budget * HYBRID_ATTR_SHARE)
    return (plan_attributed(dataclasses.replace(inp, budget=b_attr))
            + plan_uniform(dataclasses.replace(inp, budget=inp.budget - b_attr)))


STRATEGIES = {
    "ours": lambda inp: plan_attributed(inp, use_oracle=False),
    "oracle": lambda inp: plan_attributed(inp, use_oracle=True),
    "uniform": plan_uniform,
    "dagger": plan_dagger,
    "dagger_beta": plan_dagger_beta,
    "dagger_attr": plan_dagger_attr,
    "embodirsi": plan_embodirsi,
    "hybrid": plan_hybrid,
}
