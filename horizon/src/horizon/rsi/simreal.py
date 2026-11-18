"""Sim-real proxy experiment: attribution routes failures to the right repair (v3 report, Sec. 7).

The robot trains only in its own simulator and is deployed on the "real" scene. Besides its
policy, three things it believes can be wrong (``Belief``):

* **scene**: support heights are reconstructed too low (``ReconModel``), so placements press into
  the tray / board and the believed geometry never confirms them - simulated data cannot fix this;
* **judge**: the deployed success detector's tolerance (``judge_scale``). A strict judge rejects
  correct placements and System 2 keeps re-doing finished subgoals; a lenient one turned out to be
  harmless in this testbed (``scripts/diag/judge_scale.py``);
* **plan**: the instruction compiler mis-grounds the instruction (``GroundingError``, e.g. green
  objects -> blue bin); the robot executes and verifies the wrong plan perfectly, in its simulator
  and on the real system alike.

Each round the system deploys on the real scene, collects evidence, may run ONE repair (except
route "all"), and always collects DAgger data in its current simulator:

* twin test (scene evidence): re-run the same policy on the same task seeds in the simulator and
  compare judged progress per family (the fraction of planned subgoals the robot's own judge
  confirms); a large sim-over-real gap flags the family's scene model. Progress rather than full
  success: a strict judge or a weak policy keeps full success low in both worlds, which caps the gap;
* audits (judge and plan evidence) on a small human-audited subset: the auditor labels the outcome
  and, at keyframes, each subgoal of the robot's own plan. A failed episode is a *plan* failure if
  a planned subgoal was achieved but the instruction's was not, and a *judge* failure if the
  judge's final verdicts disagree with the auditor's on the robot's own plan. Families flagged by
  the twin test are explained by the scene first: a wrong scene also corrupts the judge's geometry.

Repairs: ``geometry`` re-estimates support heights from real observations (objects resting on a
support; a held object whose descent stalls over a support); ``verifier`` refits the judge's
tolerance to the auditor's keyframe labels; ``graph`` recompiles with the auditor's corrections;
``dynamics`` is a decoy with nothing to fix. Routing variants: none (sim-only DAgger), routed
(ours: one repair per round, the largest estimated failure share), gated (ours without the repair
budget: every repair with evidence), random, all (every repair every round), perfect (correct
beliefs from the start).
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from ..datagen.generator import OpenLoopConfig, TaskDistribution, collect, open_loop_job
from ..datagen.parallel import Job, Runner
from ..datagen.rollout import EpisodeLog, PerturbationConfig
from ..eval.evaluate import TEST_SEED_BASE, summarize
from ..eval.protocol import evaluation_manifest, grid_records
from ..monitor.verifier import Verifier, VerifierConfig
from ..policy.dataset import Dataset
from ..policy.features import state_from_log
from ..policy.model import TrainConfig
from ..real2sim.recon_error import DEPLOY_WORLD, ReconModel, single_view_bias
from ..sim.scene import SceneSpec
from ..taskgraph.grounding import GroundingError
from ..taskgraph.schema import TaskGraph
from .loop import train_policy

REPAIRS = ("geometry", "verifier", "graph", "dynamics")
SCALE_GRID = tuple(round(x, 2) for x in np.arange(0.4, 2.81, 0.2))
HARMFUL_JUDGE = 0.7  # judge scales below this cost success (scripts/diag/judge_scale.py); report only


@dataclass
class SimRealConfig:
    name: str
    route: str = "routed"  # none | routed | gated | random | all | perfect
    seed: int = 0
    policy: str = "hier"
    init_budget: int = 80000
    round_budget: int = 40000
    rounds: int = 3
    real_episodes: int = 60
    audit_frac: float = 0.25
    twin_episodes: int = 40
    judge_scale: float = 2.5  # deployed judge's tolerance scale before calibration
    grounding: dict = field(default_factory=dict)  # GroundingError dict of the robot's compiler ({} = correct)
    eval_pert: float = 0.1
    min_share: float = 0.05  # routed: do nothing when no source explains at least this share of episodes
    min_evidence: int = 2  # audited failures needed before a judge / plan repair is considered
    explain_away: bool = True  # audits from families flagged by the twin test are not judge / plan evidence
    keyframe_stride: int = 10  # the auditor labels every k-th frame (judge recalibration)
    repair_costs: dict = field(default_factory=lambda: {k: 1.0 for k in REPAIRS})
    repair_budget: float | None = None  # cumulative cost, including rejected attempts
    cascade_max_repairs: int = 3
    cascade_replay: bool = True
    geometry_min_episodes: int = 3
    validation_margin: float = 0.005
    paired_feedback: bool = False  # independent round RNGs and common rollout seeds
    eval_seed_base: int = TEST_SEED_BASE
    curve_seed_base: int | None = None
    feedback_seed_base: int | None = None
    confirmation_seed_base: int = 40_000_000
    evaluation_split: str = "legacy"  # legacy | development | confirmation
    repair_order: tuple[str, ...] = ("geometry", "verifier", "graph")
    fixed_repairs_per_round: int = 1
    random_include_decoy: bool = True
    audit_memory: bool = False
    reject_verifier_candidate: bool = True
    world: dict = field(default_factory=DEPLOY_WORLD.to_dict)
    bias: dict = field(default_factory=lambda: single_view_bias().to_dict())


    def __post_init__(self):
        routes = {"none", "routed", "gated", "random", "all", "perfect", "cost_gated", "cascade", "fixed", "greedy_share", "greedy_ratio", "memory"}
        if self.route not in routes:
            raise ValueError(f"Unknown repair route: {self.route}")
        if self.init_budget <= 0 or self.round_budget <= 0 or self.rounds < 0:
            raise ValueError("Training budgets must be positive and rounds nonnegative")
        if self.real_episodes <= 0 or self.twin_episodes < 0 or not 0 <= self.audit_frac <= 1:
            raise ValueError("Invalid deployment or audit budget")
        if self.evaluation_split not in ("legacy", "development", "confirmation"):
            raise ValueError("Unknown evaluation split")
        self.repair_order = tuple(self.repair_order)
        if len(self.repair_order) != 3 or set(self.repair_order) != {"geometry", "verifier", "graph"}:
            raise ValueError("Fixed repair order must contain each real operator once")
        if not 1 <= self.fixed_repairs_per_round <= 3:
            raise ValueError("fixed_repairs_per_round must be 1..3")
        costs = {k: float(self.repair_costs.get(k, 1.0)) for k in REPAIRS}
        if set(self.repair_costs) - set(REPAIRS) or any(not math.isfinite(v) or v <= 0 for v in costs.values()):
            raise ValueError("Repair costs must be finite, positive, and use known operator names")
        self.repair_costs = costs
        if self.repair_budget is not None and (not math.isfinite(self.repair_budget) or self.repair_budget < 0):
            raise ValueError("Repair budget must be finite and nonnegative")
        if self.cascade_max_repairs < 1 or self.geometry_min_episodes < 1 or self.keyframe_stride < 1:
            raise ValueError("Repair and keyframe counts must be positive")
        if not math.isfinite(self.validation_margin) or self.validation_margin < 0:
            raise ValueError("Validation margin must be finite and nonnegative")


@dataclass
class Belief:
    """Everything the robot believes that can be wrong: its scene model, its judge, its compiler."""

    recon: ReconModel
    judge_scale: float
    grounding: GroundingError

    @staticmethod
    def initial(cfg: SimRealConfig) -> "Belief":
        if cfg.route == "perfect":
            return Belief(ReconModel(), 1.0, GroundingError())
        return Belief(ReconModel.from_dict(cfg.bias), cfg.judge_scale, GroundingError.from_dict(cfg.grounding))

    def to_dict(self) -> dict:
        return {"recon": self.recon.to_dict(), "judge_scale": self.judge_scale, "grounding": self.grounding.to_dict()}


# ---------------------------------------------------------------------- jobs
def _job(fam: str, H: int, ts: int, seed: int, cfg: SimRealConfig, b: Belief, *, real: bool,
         controller: str = "hier", pert: float = 0.0, **kw) -> Job:
    return Job(family=fam, H=H, task_seed=ts, seed=seed, controller=controller, world=dict(cfg.world),
               recon=b.recon.to_dict(), real=real, verifier_cfg=VerifierConfig(tol_scale=b.judge_scale),
               grounding=None if b.grounding.is_identity() else b.grounding.to_dict(),
               pert=PerturbationConfig.suite(pert), **kw)


def test_jobs(cfg: SimRealConfig, b: Belief, n_per: int, pert: float, real: bool,
              families=("pack", "sort", "stack"), H_values=(2, 4, 6, 8, 10),
              seed_base: int | None = None) -> list[Job]:
    base = cfg.eval_seed_base if seed_base is None else seed_base
    records = grid_records(base, families, H_values, n_per, stable_ids=cfg.evaluation_split != "legacy")
    return [_job(e["family"], e["H"], e["task_seed"], e["rollout_seed"], cfg, b, real=real,
                 controller=cfg.policy, pert=pert, want_samples=False, keep_log=True,
                 provenance="test") for e in records]


def evaluate_sr(runner: Runner, cfg: SimRealConfig, b: Belief, n_per: int, pert: float, real: bool,
                families=("pack", "sort", "stack"), H_values=(2, 4, 6, 8, 10),
                seed_base: int | None = None) -> dict:
    res = runner.map(test_jobs(cfg, b, n_per, pert, real, families, H_values, seed_base))
    errors = [r.error for r in res if r.error]
    if errors:
        raise RuntimeError(f"Evaluation rollout failed: {errors[0]}")
    out = summarize(res, H_values)
    for e, r in zip(out["episodes"], [x for x in res if x.log is not None]):
        e["judged_done"] = judged_done(r.log)
    return out


# ---------------------------------------------------------------------- evidence
def judged_done(log: EpisodeLog) -> bool:
    """The robot's own verdict: System 2 reported completion before the time limit."""
    return bool(len(log.k_sel) and int(log.k_sel[-1]) < 0)


def _planned(log: EpisodeLog) -> TaskGraph:
    return TaskGraph.from_dict(log.meta.get("belief_graph", log.graph))


def _plan_final(log: EpisodeLog) -> np.ndarray:
    return np.asarray(log.meta.get("plan_final_sat", log.meta["final_sat"]), dtype=bool)


def judged_progress(log: EpisodeLog) -> float:
    """Fraction of the robot's planned subgoals that its own judge confirms at the end of the episode."""
    return float(np.asarray(log.sat_ver[-1], dtype=bool).mean()) if len(log.sat_ver) else 0.0


def twin_test(real_logs: list[EpisodeLog], twin_logs: list[EpisodeLog]) -> dict:
    """Per family: judged progress in the twin minus judged progress on the real system (judged
    success is kept for the report)."""
    by = defaultdict(lambda: {"real": [], "sim": [], "real_done": [], "sim_done": []})
    for side, logs in (("real", real_logs), ("sim", twin_logs)):
        for l in logs:
            by[l.family][side].append(judged_progress(l))
            by[l.family][side + "_done"].append(judged_done(l))
    out = {}
    for fam, v in by.items():
        if v["real"] and v["sim"]:
            out[fam] = {"real": float(np.mean(v["real"])), "sim": float(np.mean(v["sim"])),
                        "gap": float(np.mean(v["sim"]) - np.mean(v["real"])), "n": len(v["real"]),
                        "real_done": float(np.mean(v["real_done"])), "sim_done": float(np.mean(v["sim_done"]))}
    return out


def flagged_families(twin: dict, gap_threshold: float = 0.2, min_family: int = 6) -> list[str]:
    """Families whose twin reproduces real outcomes badly: their scene model is suspect."""
    return sorted(f for f, v in twin.items() if v["gap"] > gap_threshold and v["n"] >= min_family)


def audit_test(audited: list[EpisodeLog], flagged=()) -> dict:
    """Judge and plan evidence from human-audited episodes (families in ``flagged`` are skipped)."""
    out = {"audited": len(audited), "judged_done": 0, "false_success": 0, "used": 0, "judge": 0, "plan": 0}
    for log in audited:
        done = judged_done(log)
        out["judged_done"] += int(done)
        out["false_success"] += int(done and not log.success)
        if log.family in flagged:
            continue
        out["used"] += 1
        if log.success:
            continue
        plan, truth = _plan_final(log), np.asarray(log.meta["final_sat"], dtype=bool)
        if (plan & ~truth).any():  # the robot did what it planned; the plan was wrong
            out["plan"] += 1
        elif (np.asarray(log.sat_ver[-1], dtype=bool) != plan).any():  # the judge misread its own plan
            out["judge"] += 1
    out["fsr"] = out["false_success"] / max(1, out["judged_done"])
    return out


def route(cfg: SimRealConfig, twin: dict, audit: dict, flagged: list[str], rng: np.random.Generator,
          remaining_budget: float = math.inf, round_index: int = 1) -> tuple[list[str], dict]:
    """Estimated share of episodes each repair would recover; pick the repair(s) for this round."""
    n_twin = max(1, sum(v["n"] for v in twin.values()))
    n_aud = max(1, audit["audited"])
    shares = {
        "geometry": float(sum(twin[f]["gap"] * twin[f]["n"] for f in flagged) / n_twin),
        "verifier": audit["judge"] / n_aud if audit["judge"] >= cfg.min_evidence else 0.0,
        "graph": audit["plan"] / n_aud if audit["plan"] >= cfg.min_evidence else 0.0,
    }
    if cfg.route == "cost_gated":
        from .repair_cascade import select_cost_subset

        return select_cost_subset(shares, cfg.repair_costs, remaining_budget, cfg.min_share), shares
    if cfg.route in ("greedy_share", "greedy_ratio"):
        from .repair_memory import select_feasible_greedy

        return select_feasible_greedy(shares, cfg.repair_costs, remaining_budget, cfg.min_share,
                                      ratio=cfg.route == "greedy_ratio"), shares
    if cfg.route in ("cascade", "memory"):
        return [], shares  # sequential decisions follow evidence replay
    if cfg.route == "routed":
        best = max(shares, key=shares.get)
        return ([best] if shares[best] >= cfg.min_share else []), shares
    if cfg.route == "gated":  # no repair budget: every repair that has evidence, scene first
        return [k for k in ("geometry", "graph", "verifier") if shares[k] >= cfg.min_share], shares
    if cfg.route == "fixed":
        start = (round_index - 1) * cfg.fixed_repairs_per_round
        return list(cfg.repair_order[start:start + cfg.fixed_repairs_per_round]), shares
    if cfg.route == "random":
        operators = REPAIRS if cfg.random_include_decoy else REPAIRS[:3]
        return [str(rng.choice(operators))], shares
    if cfg.route == "all":
        return ["geometry", "verifier", "graph"], shares
    return [], shares


def active_errors(b: Belief, logs: list[EpisodeLog]) -> list[str]:
    """Which injected errors are still present (privileged; used for the report only)."""
    out = []
    for log in logs:
        true = {f.name: f.top_z for f in SceneSpec.from_dict(log.scene).fixtures if f.kind == "plate"}
        believed = {f.name: f.top_z for f in _believed(log).fixtures if f.kind == "plate"}
        if any(abs(z - believed.get(n, z)) > 0.003 for n, z in true.items()):
            out.append("geometry")
            break
    if b.judge_scale < HARMFUL_JUDGE:
        out.append("verifier")
    if not b.grounding.is_identity():
        out.append("graph")
    return out


# ---------------------------------------------------------------------- repairs
def _believed(log: EpisodeLog) -> SceneSpec:
    return SceneSpec.from_dict(log.meta["belief"]) if "belief" in log.meta else SceneSpec.from_dict(log.scene)


def support_observations(logs: list[EpisodeLog]) -> dict[str, list[float]]:
    """Observed top heights of support fixtures (kind "plate": board, tray) from real episodes.

    Two observable signals: (1) an object resting on a support, not in hand, with the gripper well
    above it -> support top = object bottom; (2) a held object whose commanded descent stalls over
    a support with nothing beneath it -> the support is where the object's bottom stopped. (Without
    the "nothing beneath" test, descents onto stack layers or onto other cans dominate the
    observations once the support heights are right, and a repeated repair corrupts the model.)
    """
    obs: dict[str, list[float]] = defaultdict(list)
    for log in logs:
        sc = _believed(log)
        supports = [f for f in sc.fixtures if f.kind == "plate"]
        if not supports:
            continue
        half = np.array([o.half_height for o in sc.objects])
        rad = np.array([o.footprint_radius for o in sc.objects])
        P = log.obj_obs[:, :, :3].astype(float)
        G = log.grip.astype(float)
        T = len(G)
        for f in supports:
            cx, cy = f.pos[0], f.pos[1]
            hx, hy = f.size[0], f.size[1]
            inside = (np.abs(P[:, :, 0] - cx) < hx - 0.01) & (np.abs(P[:, :, 1] - cy) < hy - 0.01)
            # (1) resting at the end of the episode: the lowest object inside the footprint
            t = T - 1
            idx = [i for i in range(P.shape[1]) if inside[t, i] and int(log.held_est[t]) != i
                   and G[t, 2] > P[t, i, 2] + half[i] + 0.03]
            if idx:
                i = min(idx, key=lambda j: P[t, j, 2] - half[j])
                obs[f.name].append(float(P[t, i, 2] - half[i]))
            # (2) stalled descents while holding an object over the support
            for t in range(0, T - 4, 2):
                i = int(log.held_est[t])
                if i < 0 or not inside[t, i] or log.action[t, 2] > -0.3:
                    continue
                if abs(G[t + 3, 2] - G[t, 2]) < 0.001 and G[t, 2] < 0.2:
                    d = np.linalg.norm(P[t, :, :2] - P[t, i, :2], axis=1)
                    beneath = (d < rad + rad[i]) & (P[t, :, 2] < P[t, i, 2]) & (np.arange(len(d)) != i)
                    if not beneath.any():
                        obs[f.name].append(float(P[t, i, 2] - half[i]))
    return obs


def support_observations_per_episode(logs: list[EpisodeLog]) -> dict[str, list[float]]:
    """One value per (episode, fixture): the episode's own median of its observations.

    Frame-level observations are not pooled across episodes. Pooling lets a single episode with an
    oscillating descent contribute dozens of windows and dominate the median, which is how a
    repeated repair moved a 19.84 mm belief to 93.54 mm. Here each episode casts one vote.
    """
    out: dict[str, list[float]] = defaultdict(list)
    for log in logs:
        for name, vals in support_observations([log]).items():
            if vals:
                out[name].append(float(np.median(vals)))
    return out


def geometry_repair(recon: ReconModel, logs: list[EpisodeLog], min_obs: int = 5, tol: float = 0.003,
                    robust: bool = False, min_episodes: int = 2, max_spread: float = 0.010,
                    max_dev: float = 0.005, step_cap: float = 2.0) -> tuple[ReconModel, dict]:
    """Re-estimate support heights; update the reconstruction's correction factors.

    ``robust=True`` switches to the repaired estimator: one vote per episode (see
    :func:`support_observations_per_episode`), episodes that disagree with the majority by more
    than ``max_dev`` are dropped before the median, and the per-round correction is bounded by
    ``step_cap``. The default keeps the historical behaviour so frozen results stay reproducible;
    only new experiments should opt in.
    """
    if not logs:
        return recon, {}
    if robust:
        keep = {n: v for n, v in support_observations_per_episode(logs).items() if len(v) >= min_episodes}
    else:
        keep = support_observations(logs)
    believed_top: dict[str, float] = {}  # believed tops can differ by family (board vs tray)
    for log in logs:
        for f in _believed(log).fixtures:
            if f.kind == "plate":
                believed_top[f.name] = f.top_z
    new = ReconModel(dict(recon.bias), dict(recon.corrections))
    report = {}
    for name, vals in keep.items():
        if name not in believed_top:
            continue
        if not robust and len(vals) < min_obs:
            continue
        arr = np.asarray(vals, float)
        est = float(np.median(arr))
        cur = believed_top[name]
        spread = float(np.median(np.abs(arr - est))) if arr.size > 1 else 0.0
        dev = float(np.max(np.abs(arr - est))) if arr.size > 1 else 0.0
        report[name] = {"believed": cur, "estimated": est, "n": int(arr.size),
                        "spread": spread, "max_dev": dev}
        if robust and arr.size > 1:
            inliers = arr[np.abs(arr - est) <= max_dev]
            report[name]["dropped"] = int(arr.size - inliers.size)
            if spread > max_spread or inliers.size < min_episodes:
                report[name]["rejected"] = "inconsistent"
                continue
            est = float(np.median(inliers))
            report[name]["estimated"] = est
            report[name]["n_used"] = int(inliers.size)
        if abs(est - cur) > tol and cur > 1e-4:
            ratio = est / cur
            if robust:
                ratio = float(np.clip(ratio, 1.0 / step_cap, step_cap))
                report[name]["applied_ratio"] = ratio
            new.corrections[name] = new.corrections.get(name, 1.0) * ratio
    return new, report


def verifier_recal(audited: list[EpisodeLog], current: float, stride: int = 10, tie: float = 0.005) -> tuple[float, dict]:
    """Refit the judge's tolerance scale to the auditor's keyframe labels of the robot's own plan.

    Balanced accuracy over (keyframe, subgoal) pairs. If the current scale is within ``tie`` of the
    best it is kept (no evidence, no change); otherwise the geometric centre of the near-optimal
    scales is taken (the largest margin to both kinds of disagreement).
    """
    if not audited:
        return current, {}
    per_log = []
    for log in audited:
        truth = log.sat_plan if log.sat_plan is not None else log.sat_true
        ts = sorted(set(range(0, log.T, stride)) | {log.T - 1})
        states = [state_from_log(log.grip[t], log.obj_obs[t], t) for t in ts]
        per_log.append((_believed(log), _planned(log), states, np.asarray(truth[ts], dtype=bool)))
    scores = {}
    for k in sorted(set(SCALE_GRID) | {round(float(current), 2)}):
        tp = tn = fp = fn = 0
        for sc, graph, states, truth in per_log:
            ver = Verifier(sc, VerifierConfig(tol_scale=k))
            pred = np.array([ver.satisfied(s, graph) for s in states], dtype=bool).reshape(truth.shape)
            tp += int((pred & truth).sum())
            tn += int((~pred & ~truth).sum())
            fp += int((pred & ~truth).sum())
            fn += int((~pred & truth).sum())
        scores[k] = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
    top = max(scores.values())
    near = [k for k in scores if scores[k] >= top - tie]
    cur = round(float(current), 2)
    if cur in near:
        best = cur
    else:
        centre = np.sqrt(min(near) * max(near))
        best = min(near, key=lambda k: (abs(np.log(k / centre)), k))
    return float(best), {"balanced_accuracy": {float(k): v for k, v in scores.items()}, "chosen": float(best),
                         "frames": sum(len(x[2]) for x in per_log)}


def graph_repair(g: GroundingError, audited: list[EpisodeLog]) -> tuple[GroundingError, dict]:
    """Recompile with the auditor's corrections: an object placed where the robot's plan said but
    not where the instruction says reveals a grounding confusion for the object's colour."""
    seen: Counter = Counter()
    for log in audited:
        if "belief_graph" not in log.meta:
            continue
        sc = SceneSpec.from_dict(log.scene)
        plan, inst = TaskGraph.from_dict(log.meta["belief_graph"]), TaskGraph.from_dict(log.graph)
        ps, ts = _plan_final(log), np.asarray(log.meta["final_sat"], dtype=bool)
        for k, (gp, gi) in enumerate(zip(plan.subgoals, inst.subgoals)):
            if ps[k] and not ts[k] and gp.target != gi.target:
                seen[sc.objects[sc.object_index(gp.obj)].color] += 1
    new = GroundingError({c: v for c, v in g.confuse.items() if c not in seen})
    return new, {"corrected": dict(seen), "remaining": dict(new.confuse)}


# ---------------------------------------------------------------------- one deployment round
def deploy_and_attribute(runner: Runner, cfg: SimRealConfig, b: Belief, dist: TaskDistribution,
                         rng: np.random.Generator, remaining_budget: float = math.inf, round_index: int = 1) -> dict:
    """Deploy on the real scene, re-run a subset in the twin, audit a subset; evidence -> repair choice."""
    feedback_dist = dist if cfg.feedback_seed_base is None else replace(dist, seed_base=cfg.feedback_seed_base)
    tasks = [feedback_dist.sample(rng) for _ in range(cfg.real_episodes)]
    real_jobs = [_job(f, H, ts, int(rng.integers(1 << 30)), cfg, b, real=True, controller=cfg.policy,
                      pert=cfg.eval_pert, want_samples=False, keep_log=True, provenance="real") for f, H, ts in tasks]
    real_results = runner.map(real_jobs)
    errors = [x.error for x in real_results if x.error]
    if errors or any(x.log is None for x in real_results):
        raise RuntimeError(f"Deployment rollout failed: {errors[0] if errors else 'missing log'}")
    real_logs = [x.log for x in real_results]
    twin_idx = rng.choice(len(real_logs), size=min(cfg.twin_episodes, len(real_logs)), replace=False)
    twin_jobs = [_job(real_logs[i].family, real_logs[i].H, real_logs[i].seed, real_jobs[i].seed if cfg.paired_feedback else int(rng.integers(1 << 30)), cfg, b,
                      real=False, controller=cfg.policy, pert=cfg.eval_pert, want_samples=False, keep_log=True,
                      provenance="twin") for i in twin_idx]
    twin_results = runner.map(twin_jobs)
    errors = [x.error for x in twin_results if x.error]
    if errors or any(x.log is None for x in twin_results):
        raise RuntimeError(f"Twin rollout failed: {errors[0] if errors else 'missing log'}")
    twin_logs = [x.log for x in twin_results]
    n_aud = int(round(cfg.audit_frac * len(real_logs)))
    audit_idx = rng.choice(len(real_logs), size=n_aud, replace=False)
    audited = [real_logs[i] for i in audit_idx]
    twin = twin_test([real_logs[i] for i in twin_idx], twin_logs)
    flagged = flagged_families(twin) if cfg.explain_away else []
    audit = audit_test(audited, flagged)
    chosen, shares = route(cfg, twin, audit, flagged, rng, remaining_budget, round_index)
    return {"real_logs": real_logs, "audited": audited, "usable": [l for l in audited if l.family not in flagged],
            "twin": twin, "flagged": flagged, "audit": audit, "shares": shares, "chosen": chosen,
            "feedback_tasks": [{"family": j.family, "H": j.H, "task_seed": j.task_seed,
                                "rollout_seed": j.seed, "audited": i in audit_idx, "twin": i in twin_idx}
                               for i, j in enumerate(real_jobs)],
            "audit_keyframes": sum(len(set(range(0, l.T, cfg.keyframe_stride)) | {l.T - 1}) for l in audited),
            "audit_predicate_labels": sum(len(set(range(0, l.T, cfg.keyframe_stride)) | {l.T - 1}) * l.H for l in audited),
            "twin_replays": len(twin_logs),
            "active": active_errors(b, real_logs),
            "real_judged_done": float(np.mean([judged_done(l) for l in real_logs])),
            "real_success": float(np.mean([l.success for l in real_logs]))}


def apply_repairs(b: Belief, chosen: list[str], ev: dict, cfg: SimRealConfig) -> dict:
    """Run the chosen repairs on the round's evidence; updates ``b`` in place, returns the reports."""
    reports = {}
    for rep in chosen:
        if rep == "geometry":
            b.recon, reports["geometry_report"] = geometry_repair(b.recon, ev["real_logs"])
        elif rep == "verifier":
            b.judge_scale, rpt = verifier_recal(ev["usable"], b.judge_scale, cfg.keyframe_stride)
            reports["verifier_report"] = {"chosen": rpt.get("chosen"), "frames": rpt.get("frames")}
        elif rep == "graph":
            b.grounding, reports["graph_report"] = graph_repair(b.grounding, ev["usable"])
    return reports



def budgeted_repairs(b: Belief, chosen: list[str], ev: dict, cfg: SimRealConfig, budget: float) -> dict:
    """Charge every executed attempt, including no-ops; separately record state changes."""
    attempts, committed, reports = [], [], {}
    spent = 0.0
    for repair in chosen:
        cost = cfg.repair_costs[repair]
        if spent + cost > budget + 1e-9:
            continue
        before = b.to_dict()
        reports.update(apply_repairs(b, [repair], ev, cfg))
        attempts.append(repair)
        spent += cost
        if b.to_dict() != before:
            committed.append(repair)
    return {**reports, "chosen": attempts, "repair_attempts": attempts,
            "repair_committed": committed, "repair_cost": spent}

# ---------------------------------------------------------------------- loop
def run_simreal(cfg: SimRealConfig, dist: TaskDistribution, open_cfg: OpenLoopConfig, train_cfg: TrainConfig,
                test_n_per: int = 20, curve_n_per: int = 8, workers: int | None = None,
                out_dir: str | Path | None = None, verbose: bool = True) -> dict:
    manifest = evaluation_manifest(cfg, dist, test_n_per, curve_n_per)
    t_start = time.time()
    say = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(cfg.seed)
    b = Belief.initial(cfg)
    tcfg = TrainConfig(**{**asdict(train_cfg), "seed": train_cfg.seed + cfg.seed})

    def nominal_job() -> Job:
        j = open_loop_job(dist, open_cfg, rng)
        j.world, j.recon = dict(cfg.world), b.recon.to_dict()
        j.grounding = None if b.grounding.is_identity() else b.grounding.to_dict()
        return j

    with Runner(workers) as runner:
        blocks, st = collect(runner, nominal_job, cfg.init_budget)
    data = Dataset(blocks)
    say(f"[{cfg.name}] base: {data.n} transitions in the robot's simulator")
    history: list[dict] = []
    repairs_done: list[str] = []
    from .repair_memory import AuditMemory, memory_repairs
    audit_memory = AuditMemory()
    real_used = audits_used = 0
    audit_frames = audit_labels = twins = curve_episodes = 0
    repair_cost_used = 0.0
    eval_grid = {"families": dist.families, "H_values": dist.H_values}
    for r in range(1, cfg.rounds + 1):
        params, _ = train_policy(data, cfg.policy, tcfg)
        h: dict = {"round": r, "n": data.n, "belief": b.to_dict(), "verifier_scale": b.judge_scale}
        with Runner(workers, policy={"params": params}) as runner:
            curve = evaluate_sr(runner, cfg, b, curve_n_per, 0.0, real=True, seed_base=cfg.curve_seed_base, **eval_grid)
            h["real_test"] = curve["overall"]
            h["real_test_episode_count"] = len(curve["episodes"])
            curve_episodes += len(curve["episodes"])
            h["real_test_by_family"] = curve["by_family_H"]
            if cfg.route not in ("none", "perfect"):
                feedback_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, r, 937])) if cfg.paired_feedback else rng
                remaining = math.inf if cfg.repair_budget is None else max(0.0, cfg.repair_budget - repair_cost_used)
                ev = deploy_and_attribute(runner, cfg, b, dist, feedback_rng, remaining, round_index=r)
                real_used += len(ev["real_logs"])
                audits_used += len(ev["audited"])
                audit_frames += ev["audit_keyframes"]
                audit_labels += ev["audit_predicate_labels"]
                twins += ev["twin_replays"]
                h.update({k: ev[k] for k in ("twin", "flagged", "audit", "shares", "chosen", "active",
                                             "real_judged_done", "real_success", "feedback_tasks", "audit_keyframes",
                                             "audit_predicate_labels", "twin_replays")})
                if cfg.route == "memory":
                    h.update(memory_repairs(b, ev, cfg, remaining, audit_memory))
                elif cfg.route == "cascade":
                    from .repair_cascade import cascade_repairs

                    h.update(cascade_repairs(b, ev, cfg, remaining))
                else:
                    h.update(budgeted_repairs(b, ev["chosen"], ev, cfg, remaining))
                repair_cost_used += h["repair_cost"]
                h["repair_cost_used"] = repair_cost_used
                repairs_done += h["chosen"]
            beta = 0.5 ** r
            data_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, r, 251])) if cfg.paired_feedback else rng

            def dagger_job() -> Job:
                f, H, ts = dist.sample(data_rng)
                return _job(f, H, ts, int(data_rng.integers(1 << 30)), cfg, b, real=False, controller="mixed",
                            pert=cfg.eval_pert, learner=cfg.policy, beta=beta, label_expert=True, max_samples=300,
                            provenance="dagger", only_success=False)

            new, st = collect(runner, dagger_job, cfg.round_budget, require_success=False)
        data.add(new)
        h["added"] = st.transitions
        say(f"[{cfg.name}] round {r}: real test {h['real_test']:.3f}; active {h.get('active', '-')}; "
            f"flagged {h.get('flagged', '-')}; shares {({k: round(v, 3) for k, v in h.get('shares', {}).items()})}; "
            f"chosen {h.get('chosen', [])} -> corrections {({k: round(v, 2) for k, v in b.recon.corrections.items()})}, "
            f"judge {b.judge_scale}, grounding {b.grounding.confuse}", flush=True)
        history.append(h)
    params, _ = train_policy(data, cfg.policy, tcfg)
    with Runner(workers, policy={"params": params}) as runner:
        final = {"real_nominal": evaluate_sr(runner, cfg, b, test_n_per, 0.0, real=True, **eval_grid),
                 "real_pert0.15": evaluate_sr(runner, cfg, b, test_n_per, 0.15, real=True, **eval_grid),
                 "sim_nominal": evaluate_sr(runner, cfg, b, test_n_per, 0.0, real=False, **eval_grid)}
    say(f"[{cfg.name}] final: real {final['real_nominal']['overall']:.3f} / pert {final['real_pert0.15']['overall']:.3f}; "
        f"sim {final['sim_nominal']['overall']:.3f}; repairs {repairs_done}", flush=True)
    out = {"config": asdict(cfg), "history": history, "final": final, "repairs": repairs_done,
           "final_belief": b.to_dict(), "real_episodes_used": real_used, "audits_used": audits_used, "n": data.n,
           "repair_cost_used": repair_cost_used, "train_steps": tcfg.steps * (cfg.rounds + 1),
           "wall": time.time() - t_start, "evaluation_manifest": manifest,
           "audit_keyframes_used": audit_frames, "audit_predicate_labels_used": audit_labels,
           "twin_replays_used": twins, "curve_evaluation_episodes": curve_episodes}
    if out_dir is not None:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{cfg.name}_s{cfg.seed}.json").write_text(json.dumps(out, indent=1, default=str))
        meta = {"arch": np.array(str(params.get("arch", "mlp")))}
        if "n_layers" in params:
            meta["n_layers"] = np.array(params["n_layers"])
        np.savez_compressed(p / f"{cfg.name}_s{cfg.seed}_policy.npz",
                            **{k: v for k, v in params.items() if isinstance(v, np.ndarray)}, **meta)
    return out
