"""Policy-free diagnosis and the selector built on it.

Why this exists: the learned policy is optimised inside the twin, so "success measured in the twin"
is circular.  The probe (``scripts/probe_policy_free_signals.py``) measures three quantities that do
not depend on the policy being diagnosed:

    twin_cost      = expert_SR(privileged) - expert_SR(with twin belief)   -> the twin is wrong
    knowledge_gap  = expert_SR(with twin belief) - policy_SR(deployment)   -> data/policy shortage
    perturbed_gap  = knowledge_gap(perturbed) - knowledge_gap(nominal)     -> recovery-type failure
    selfcheck_bias = policy_SR(twin) - policy_SR(deployment)               -> the circular signal

``diagnose`` turns them into an action; ``PolicyFreeSelector`` calls it every round, so a mixed
cause is handled as "fix the dominant one, then re-diagnose".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .selector import Decision, Observation


@dataclass(frozen=True)
class ProbeSignals:
    twin_cost: float | None = None
    knowledge_gap: float | None = None
    perturbed_gap: float | None = None
    selfcheck_bias: float | None = None
    # policy_SR(nominal) - policy_SR(perturbed): how much *the policy* loses under perturbation.
    # Measured 0.00 for recovery (already 1.00 nominal), 1.00 for the recovery-deficit scenario,
    # 0.30 for healthy -- a wider margin than `perturbed_gap`, which lands on the threshold.
    perturbation_drop: float | None = None

    def as_dict(self) -> dict:
        return {"twin_cost": self.twin_cost, "knowledge_gap": self.knowledge_gap,
                "perturbed_gap": self.perturbed_gap, "selfcheck_bias": self.selfcheck_bias,
                "perturbation_drop": self.perturbation_drop}


@dataclass(frozen=True)
class ProbeThresholds:
    twin: float = 0.3
    recovery: float = 0.5
    knowledge: float = 0.5
    healthy: float = 0.35


def diagnose(signals: ProbeSignals, tau: ProbeThresholds | None = None) -> tuple[str, str]:
    """Map the policy-free signals to an operation.  Returns (action, reason)."""
    tau = tau or ProbeThresholds()
    if signals.twin_cost is None and signals.knowledge_gap is None:
        raise RuntimeError("no probe signals: refusing to fall back to a hand-written rule")
    if signals.twin_cost is not None and signals.twin_cost > tau.twin:
        return "calibrate_geometry", f"twin_cost={signals.twin_cost:+.2f}: the twin itself is wrong"
    if signals.perturbation_drop is not None and signals.perturbation_drop > tau.recovery:
        return "add_recovery_data", (f"perturbation_drop={signals.perturbation_drop:+.2f}: "
                                     "the policy only fails under perturbation")
    if signals.knowledge_gap is not None and signals.knowledge_gap > tau.knowledge:
        return "regenerate_targeted", f"knowledge_gap={signals.knowledge_gap:+.2f}: twin fine, data missing"
    if signals.knowledge_gap is not None and signals.knowledge_gap <= tau.healthy:
        return "stop", f"knowledge_gap={signals.knowledge_gap:+.2f}: nothing left to fix"
    # between tau.healthy and tau.knowledge: not decisive, so take one DAgger round
    return "dagger_round", (f"knowledge_gap={signals.knowledge_gap:+.2f}: "
                            "between the stop and targeted thresholds")


@dataclass
class PolicyFreeSelector:
    """Calls ``probe(obs)`` (policy-free measurements) then acts on the signals.

    ``probe`` must return :class:`ProbeSignals`.  When it is None the selector fails loudly: a
    "policy-free" row whose signals were replaced by a rule would be a misreported experiment.
    """

    probe: Callable[[Observation], ProbeSignals] | None = None
    thresholds: ProbeThresholds = field(default_factory=ProbeThresholds)
    name: str = "policy_free"

    def decide(self, obs: Observation) -> Decision:
        if self.probe is None:
            raise RuntimeError(
                "policy_free needs a probe (expert/privileged rollouts); refusing to substitute a "
                "hand-written rule. Configure cfg['policy_free_probe'] or remove the selector.")
        action, reason = diagnose(self.probe(obs), self.thresholds)
        # A mixed cause is handled by re-diagnosing next round: the probe is recomputed after the
        # operation ran, so the second-largest cause can become the dominant one.
        return Decision(action, None, reason, 0)
