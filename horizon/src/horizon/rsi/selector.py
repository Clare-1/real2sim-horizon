"""Failure-conditioned selection of the next learning operation.

Minimal-validation pieces 2 and 3 (see ``最小验证规格_2026-11-12.md`` §4②③).

The selection input is a *leak-proof* view of the deployment feedback: a failure profile, the
budget, and the history of past decisions. It deliberately has no field for the injected
condition, the true twin parameters, or development scores, so a selector cannot cheat by
reading the answer. ``Observation.from_dict`` rejects those keys explicitly rather than ignoring
them, so a fixture that tries to leak fails loudly instead of silently passing a weaker test.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Protocol

ACTIONS = ("calibrate_geometry", "recalibrate_verifier", "regenerate_uniform",
           "regenerate_targeted", "add_recovery_data", "randomize_physics", "dagger_round",
           "stop")

# Keys that must never reach a selector. Kept explicit so the guard is self-documenting.
FORBIDDEN_KEYS = frozenset({
    "condition", "cause", "true_cause", "true_height", "bias", "recon",
    "dev_sr", "development_scores", "oracle",
})

# Cause -> the ordered operations that help. Measured by the R1 matrix (results/r1_matrix_20261112,
# 3 seeds, 95% bootstrap CI), not assumed:
#   geometry_bias : calibrate +38.3pp [+35.0,+40.0]; uniform alone -13.3pp; targeted -8.3pp;
#                   calibrate-then-uniform +46.7pp [+35.0,+55.0] -> order matters, so the sequence
#                   keeps the second step only after the twin is fixed.
#   coverage_deficit: targeted +51.7pp [+40.0,+70.0]; calibrate and uniform both +0.0pp.
# No operation helps both causes, which is the premise the whole line rests on.
ORACLE_SEQUENCE: dict[str, tuple[str, ...]] = {
    "geometry_bias": ("calibrate_geometry", "regenerate_uniform"),
    "coverage_deficit": ("regenerate_targeted",),
    "recovery_deficit": ("add_recovery_data",),
    # A1 fails here on purpose: two causes coexist, so no single operation is optimal and the
    # advisor's suggestion is to treat the mixture as its own class with a two-step sequence.
    "mixed_cause": ("calibrate_geometry", "regenerate_targeted"),
    # Mixture grid (2026-11-18): the union of the component operations, ordered so the twin is
    # fixed before data are collected. Used by configs/minimal_validation_mixed_grid.yaml.
    "mixed_gc": ("calibrate_geometry", "regenerate_targeted"),
    "mixed_gr": ("calibrate_geometry", "add_recovery_data"),
    "mixed_cr": ("regenerate_targeted", "add_recovery_data"),
    "mixed_gcr": ("calibrate_geometry", "regenerate_targeted", "add_recovery_data"),
    # Naturalistic deployment distribution: no discrete cause, so there is no "correct" first
    # operation. The oracle abstains; the other selectors are scored on SR gain, not first-step.
    "natural": (),
    # NOTE: no "dynamics_bias" entry. Friction (1.0->0.01), finger friction (1.5->0.1) and mass
    # (1x->8x) do not change SR on this platform and latency is binary, so there is no gradable
    # dynamics cause to label as correct. Evidence: 重跑风险审计_2026-11-18.md section 8.
    "healthy": (),
}

# First move implied by each sequence; used as the "correct first step" label.
CAUSE_ACTION = {cause: seq[0] for cause, seq in ORACLE_SEQUENCE.items() if seq}
CAUSE_ACTION["healthy"] = "stop"


@dataclass(frozen=True)
class FailureProfile:
    """What the deployment feedback actually shows."""

    n_episodes: int = 0
    per_subgoal_failure: dict = field(default_factory=dict)  # subgoal name -> failure count
    calibration_residual: float | None = None   # |estimated - believed| support height, metres
    calibration_spread: float | None = None     # dispersion across episodes, metres
    recovery_triggers: int = 0
    perturbed_failure_rate: float = 0.0
    sr: float = 0.0
    # Attributed failure types (grasp / drop / placement / stall / transition / verifier).
    # Needed to separate a dynamics gap (drops, failed grasps) from a coverage gap.
    failure_types: dict = field(default_factory=dict)
    # V2 recovery signal, measured from the trajectory log instead of an unset metadata flag.
    # `perturbation_events` counts injected slip/displace/knock/traj/target events;
    # `perturbation_failure_rate` is P(fail | the episode had at least one such event).
    # The two fields above are kept unchanged because the 2026-11-18 screening batch used them.
    perturbation_events: int = 0
    perturbation_failure_rate: float = 0.0

    @property
    def concentration(self) -> float:
        """Fraction of failures carried by the single worst subgoal."""
        total = sum(self.per_subgoal_failure.values())
        return max(self.per_subgoal_failure.values()) / total if total else 0.0


_FAILURE_TYPES = ("grasp", "drop", "placement", "regression", "wrong_object",
                  "transition", "stall", "verifier")


def profile_vector(p: FailureProfile) -> tuple[float, ...]:
    """Fixed-length context for learned selectors; ends with a bias term.

    Only observable fields enter: calibration residual, concentration, SR, perturbation rate,
    subgoal counts and failure-type shares. The injected cause never enters.
    """
    per = p.per_subgoal_failure or {}
    ft = p.failure_types or {}
    total = sum(ft.values()) or 1
    return ((1000.0 * (p.calibration_residual or 0.0), p.concentration, p.sr,
             p.perturbed_failure_rate, float(len(per)), float(sum(per.values())))
            + tuple(ft.get(t, 0) / total for t in _FAILURE_TYPES) + (1.0,))


def diagnosis_features(p: FailureProfile) -> tuple[float, ...]:
    """Feature vector used by the learned diagnosers (no bias term; the model adds it).

    Kept separate from ``profile_vector`` because the diagnoser was trained on this exact order and
    the selector must reproduce it. Both are observable-only.
    """
    per = p.per_subgoal_failure or {}
    ft = p.failure_types or {}
    total = sum(ft.values()) or 1
    return ((1000.0 * (p.calibration_residual or 0.0), p.concentration, float(len(per)),
             float(sum(per.values())), p.sr)
            + tuple(ft.get(t, 0) / total for t in _FAILURE_TYPES))


@dataclass(frozen=True)
class Budget:
    deploy_spent: int = 0
    deploy_total: int = 0
    labels_spent: int = 0
    labels_total: int = 0
    rounds_done: int = 0
    rounds_total: int = 3

    @property
    def exhausted(self) -> bool:
        return (self.deploy_spent >= self.deploy_total and self.deploy_total > 0) or \
               (self.labels_spent >= self.labels_total and self.labels_total > 0)


@dataclass(frozen=True)
class HistoryEntry:
    action: str
    sr_before: float
    sr_after: float
    target: str | None = None
    cost: int = 0
    # Observable context at decision time, recorded by the runner so a learned selector can
    # attribute the reward to the context it actually saw (no injected cause enters).
    context: tuple = ()


@dataclass(frozen=True)
class Observation:
    profile: FailureProfile
    budget: Budget
    history: tuple[HistoryEntry, ...] = ()

    @staticmethod
    def from_dict(payload: dict) -> "Observation":
        bad = FORBIDDEN_KEYS & set(payload)
        if bad:
            raise ValueError(f"selector input leaks forbidden keys: {sorted(bad)}")
        prof = payload.get("profile", {})
        bud = payload.get("budget", {})
        hist = tuple(HistoryEntry(**h) for h in payload.get("history", ()))
        return Observation(FailureProfile(**prof), Budget(**bud), hist)


@dataclass(frozen=True)
class Decision:
    action: str
    target: str | None
    reason: str
    est_cost: int

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action: {self.action}")


class Selector(Protocol):
    name: str

    def decide(self, obs: Observation) -> Decision: ...


def _guard(obs: Observation) -> Decision | None:
    """Shared stopping rule: no budget left, or nothing left to improve."""
    if obs.budget.exhausted:
        return Decision("stop", None, "budget exhausted", 0)
    if obs.profile.sr >= 1.0:
        return Decision("stop", None, "already solved", 0)
    return None


@dataclass
class FixedRecipeSelector:
    """The strong fixed recipe: calibrate once, then keep doing DAgger. No diagnosis."""

    name: str = "fixed"
    calibrate_rounds: int = 1

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        if obs.budget.rounds_done < self.calibrate_rounds:
            return Decision("calibrate_geometry", None, "fixed recipe: calibrate first round", 1)
        return Decision("dagger_round", None, "fixed recipe: always DAgger", 4)


@dataclass
class RulePlannerSelector:
    """Hand-written thresholds; the strongest non-LLM opponent."""

    name: str = "rule"
    tau_geo: float = 0.004          # metres of support-height disagreement
    tau_concentration: float = 0.5
    tau_labels_low: int = 20_000    # subgoal under-covered
    tau_perturbed: float = 0.5
    patience: int = 2
    # "v1" keeps the legacy (dead) signal used by the 2026-11-18 screening batch.
    # "v2" uses the measured perturbation pressure; enable only after the probe shows separation.
    recovery_signal: str = "v1"
    # Stop only when the twin already looks healthy. Without this floor the rule stopped on a
    # recovery twin at SR 0.42 (concentration 0.36) and lost 0.75 SR; healthy sits at 0.92 with
    # concentration exactly 0.50, so the concentration test must include the boundary.
    healthy_sr: float = 0.85

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        p, b = obs.profile, obs.budget
        residual = p.calibration_residual
        v2 = self.recovery_signal == "v2"
        pressure = p.perturbation_failure_rate if v2 else p.perturbed_failure_rate
        # 1 geometry: a real disagreement between estimated and believed support height
        if residual is not None and residual > self.tau_geo:
            return Decision("calibrate_geometry", None,
                            f"residual {residual*1000:.1f}mm > {self.tau_geo*1000:.0f}mm", 1)
        # 2 recovery: perturbation pressure with no recovery behaviour observed
        if v2:
            if p.perturbation_events > 0 and pressure > self.tau_perturbed:
                return Decision("add_recovery_data", None,
                                f"{p.perturbation_failure_rate:.0%} failures under "
                                f"{p.perturbation_events} perturbations", 4)
        elif pressure > self.tau_perturbed and p.recovery_triggers == 0:
            return Decision("add_recovery_data", None, "perturbed failures without recovery", 4)
        # 3 dynamics: deliberately NOT a branch. Friction (1.0->0.01), finger friction (1.5->0.1)
        #   and mass (1x->8x) do not change SR on this platform, and latency is binary, so there is
        #   no gradable dynamics cause to detect. Keeping the branch only produced misfires on
        #   diffuse failures; see 重跑风险审计_2026-11-18.md section 8.
        # 4 nothing localised and already healthy: no update can buy anything
        if ((residual is None or residual <= self.tau_geo)
                and p.concentration <= self.tau_concentration
                and pressure < self.tau_perturbed
                and p.sr >= self.healthy_sr):
            return Decision("stop", None,
                            f"nothing localised and SR {p.sr:.2f} >= {self.healthy_sr:.2f}", 0)
        # 5 coverage: failures concentrated on one subgoal while its labels are thin
        if p.concentration > self.tau_concentration and b.labels_spent < self.tau_labels_low:
            worst = max(p.per_subgoal_failure, key=p.per_subgoal_failure.get)
            return Decision("regenerate_targeted", worst,
                            f"{p.concentration:.0%} of failures in {worst}", 4)
        # 6 give up if two rounds bought nothing
        if len(obs.history) >= self.patience:
            recent = obs.history[-self.patience:]
            if all(h.sr_after <= h.sr_before + 1e-9 for h in recent):
                return Decision("stop", None, f"no gain in {self.patience} rounds", 0)
        return Decision("dagger_round", None, "default: one DAgger round", 4)


@dataclass
class OracleSelector:
    """Knows the true cause and follows that cause's operation *sequence*; upper bound, never a baseline.

    A single-action oracle is not an upper bound: on a wrong twin it calibrates forever and never
    trains, which scores below the fixed recipe. This walks the sequence once, then keeps applying
    the cause's best operation.
    """

    true_cause: str
    name: str = "oracle"

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        seq = ORACLE_SEQUENCE[self.true_cause]
        if not seq:
            return Decision("stop", None, f"oracle: nothing to fix for {self.true_cause}", 0)
        done = {h.action for h in obs.history}
        for action in seq:
            if action not in done:
                return Decision(action, None, f"oracle: {self.true_cause} step {len(done) + 1}", 0)
        return Decision(seq[-1], None, "oracle: repeat the cause's best operation", 4)


def first_step_correct(decision: Decision, true_cause: str) -> bool:
    """The minimal-validation primary metric: did round one pick the operation that helps?"""
    return decision.action == CAUSE_ACTION.get(true_cause)

@dataclass
class RandomSelector:
    """Uniformly random admissible operation. The "did you just pick a soft baseline" control."""

    name: str = "random"
    seed: int = 0

    POOL = ("calibrate_geometry", "regenerate_uniform", "regenerate_targeted",
            "add_recovery_data", "dagger_round")

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        import numpy as np
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, len(obs.history)]))
        action = self.POOL[int(rng.integers(len(self.POOL)))]
        return Decision(action, None, "random draw", 4)


@dataclass
class RoundRobinSelector:
    """Cycles through the operations in a fixed order; a strong "no feedback" control."""

    name: str = "round_robin"
    order: tuple = ("calibrate_geometry", "regenerate_targeted", "dagger_round")

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        k = len(obs.history) % len(self.order)
        return Decision(self.order[k], None, f"round-robin step {k}", 4)


@dataclass
class UCBBanditSelector:
    """UCB1 over operations; reward is the SR change each operation produced."""

    name: str = "bandit"
    c: float = 1.0
    arms: tuple = ("calibrate_geometry", "regenerate_targeted", "add_recovery_data",
                   "dagger_round")

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        import math
        counts: dict = {}
        reward: dict = {}
        for h in obs.history:
            counts[h.action] = counts.get(h.action, 0) + 1
            reward[h.action] = reward.get(h.action, 0.0) + (h.sr_after - h.sr_before)
        for a in self.arms:  # try every arm once
            if counts.get(a, 0) == 0:
                return Decision(a, None, "ucb: untried arm", 4)
        n = sum(counts.values())
        best, best_val = self.arms[0], -1e9
        for a in self.arms:
            mean = reward[a] / counts[a]
            val = mean + self.c * math.sqrt(2.0 * math.log(max(n, 1)) / counts[a])
            if val > best_val:
                best, best_val = a, val
        return Decision(best, None, f"ucb value {best_val:.3f}", 4)


@dataclass
class LinUCBSelector:
    """Contextual bandit over operations; context = the observable failure profile.

    This is the learned selector baseline: it needs no extra simulator calls, only the rounds it
    already runs. Rewards are the SR changes the runner records together with the context that was
    visible when the action was taken, so the update is not confounded by later profiles.
    """

    name: str = "linucb"
    alpha: float = 0.5
    arms: tuple = ("calibrate_geometry", "regenerate_uniform", "regenerate_targeted",
                   "add_recovery_data", "dagger_round")
    A: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    seen: int = 0

    def _sync(self, obs: Observation) -> None:
        """Apply every history entry not seen before, with its own recorded context."""
        for h in obs.history[self.seen:]:
            if not h.context:
                continue
            x = np.asarray(h.context, dtype=float)
            A = self.A.get(h.action, np.eye(x.size))
            b = self.b.get(h.action, np.zeros(x.size))
            self.A[h.action] = A + np.outer(x, x)
            self.b[h.action] = b + (h.sr_after - h.sr_before) * x
        self.seen = len(obs.history)

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        self._sync(obs)
        x = np.asarray(profile_vector(obs.profile), dtype=float)
        best, best_val = self.arms[0], -1e18
        for a in self.arms:
            A = self.A.get(a, np.eye(x.size))
            b = self.b.get(a, np.zeros(x.size))
            theta = np.linalg.solve(A, b)
            bonus = self.alpha * float(np.sqrt(max(x @ np.linalg.solve(A, x), 0.0)))
            val = float(theta @ x) + bonus
            if val > best_val:
                best, best_val = a, val
        return Decision(best, None, f"linucb value {best_val:.3f}", 4)


@dataclass
class LearnedValueSelector:
    """Learned operation-value selector: argmax predicted ΔSR from the observable profile.

    The model is trained offline on counterfactual sweeps (scripts/train_value_model.py) and loaded
    from ``cfg['value_model']``. It never reads the injected cause.
    """

    name: str = "learned_value"
    model_path: str = ""
    actions: tuple = ("calibrate_geometry", "regenerate_uniform", "regenerate_targeted",
                      "add_recovery_data", "dagger_round")
    model: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.model_path:
            import json
            from pathlib import Path
            self.model = json.loads(Path(self.model_path).read_text(encoding="utf-8"))

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        if not self.model:
            raise RuntimeError("learned_value selector needs cfg['value_model']")
        base = list(profile_vector(obs.profile))
        mu = np.asarray(self.model["mu"], float)
        sd = np.asarray(self.model["sd"], float)
        w = np.asarray(self.model["w"], float)
        best, best_val = self.actions[0], -1e18
        for a in self.actions:
            z = np.asarray(base + [1.0 if a == b else 0.0 for b in self.actions] + [0.0], float)
            val = float(np.append((z - mu) / sd, 1.0) @ w)
            if val > best_val:
                best, best_val = a, val
        return Decision(best, None, f"learned value {best_val:+.3f}", 4)


_MULTILABEL_CAUSE = {
    frozenset({"geometry_bias"}): "geometry_bias",
    frozenset({"coverage_deficit"}): "coverage_deficit",
    frozenset({"recovery_deficit"}): "recovery_deficit",
    frozenset({"geometry_bias", "coverage_deficit"}): "mixed_gc",
    frozenset({"geometry_bias", "recovery_deficit"}): "mixed_gr",
    frozenset({"coverage_deficit", "recovery_deficit"}): "mixed_cr",
    frozenset({"geometry_bias", "coverage_deficit", "recovery_deficit"}): "mixed_gcr",
}


@dataclass
class MultiLabelSelector:
    """Learned multi-label diagnosis -> an operation sequence.

    A single-label classifier collapses every mixture to its dominant component. This selector
    predicts each component independently and maps the predicted set to the corresponding sequence:
    {geometry, coverage} -> calibrate then targeted, and so on. An empty set with high SR stops.
    """

    name: str = "multilabel"
    model_path: str = ""
    threshold: float = 0.5
    model: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.model_path:
            import json
            from pathlib import Path
            self.model = json.loads(Path(self.model_path).read_text(encoding="utf-8"))
            self.threshold = float(self.model.get("threshold", self.threshold))

    def _labels(self, p: FailureProfile) -> list[str]:
        if not self.model:
            raise RuntimeError("multilabel selector needs cfg['multilabel_model']")
        x = np.asarray(diagnosis_features(p), float)
        on = []
        for lab in self.model["labels"]:
            m = self.model["models"][lab]
            z = np.append((x - np.asarray(m["mu"], float)) / np.asarray(m["sd"], float), 1.0)
            prob = 1.0 / (1.0 + np.exp(-float(z @ np.asarray(m["w"], float))))
            if prob >= self.threshold:
                on.append(lab)
        return on

    def decide(self, obs: Observation) -> Decision:
        stop = _guard(obs)
        if stop:
            return stop
        on = self._labels(obs.profile)
        cause = _MULTILABEL_CAUSE.get(frozenset(on))
        if cause is None:
            if not on and obs.profile.sr >= 0.85:
                return Decision("stop", None, "multilabel: no component and already healthy", 0)
            return Decision("dagger_round", None, f"multilabel: unknown set {sorted(on)}", 4)
        seq = ORACLE_SEQUENCE[cause]
        done = {h.action for h in obs.history}
        for action in seq:
            if action not in done:
                return Decision(action, None, f"multilabel {cause} step {len(done) + 1}", 4)
        return Decision(seq[-1], None, "multilabel: repeat the best operation", 4)
