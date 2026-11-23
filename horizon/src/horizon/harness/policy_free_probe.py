"""Runs the policy-free probe for the live selector (expert vs policy, privileged vs twin).

Called from ``_run_one`` before each decision when the selector is ``policy_free``.  Six groups of
``n`` episodes each: privileged expert, expert-with-twin, policy-in-deployment, policy-in-twin, and
the two perturbed groups needed for the recovery signal.  Nothing here looks at the cause label.
"""
from __future__ import annotations

import numpy as np

from ..datagen.parallel import Runner
from ..datagen.rollout import PerturbationConfig
from ..real2sim.recon_error import ReconModel
from ..rsi.policy_free import ProbeSignals
from .experiment import check_results, job_for


def build_probe(store, cfg, recon, bank, params, workers, n_episodes: int = 3, pert: float = 0.7):
    family = cfg["families"][0]
    H = int(cfg["H_values"][0])

    def _sr(controller, physics=None, believe=True, p=0.0):
        jobs = []
        for i in range(n_episodes):
            ts = bank + i
            belief = recon if believe else ReconModel()   # identity belief == privileged
            jobs.append(job_for(store, family, H, ts, ts + 17, cfg, belief, real=True,
                                controller=controller, pert=PerturbationConfig.suite(p),
                                physics=physics, want_samples=False, keep_log=True,
                                provenance="policy-free-probe"))
        with Runner(workers, policy={"params": params}) as runner:
            res = runner.map(jobs)
            check_results(res)
        return float(np.mean([bool(r.log and r.log.success) for r in res]))

    def probe(_obs):
        a2 = _sr("expert", None, False, 0.0)
        a = _sr("expert", None, True, 0.0)
        c = _sr("hier", None, True, 0.0)
        d = _sr("hier", recon.to_dict(), True, 0.0)
        a_p = _sr("expert", None, True, pert)
        c_p = _sr("hier", None, True, pert)
        gap = a - c
        return ProbeSignals(twin_cost=a2 - a, knowledge_gap=gap,
                            perturbed_gap=(a_p - c_p) - gap, selfcheck_bias=d - c)

    return probe
