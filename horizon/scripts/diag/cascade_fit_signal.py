"""Inspect whether recorded audits support paying for a verifier fit; no test set used."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from horizon.datagen.generator import TaskDistribution
from horizon.datagen.parallel import Runner
from horizon.rsi.loop import load_params
from horizon.rsi.simreal import SimRealConfig, Belief, deploy_and_attribute, audit_test, verifier_recal
from horizon.rsi.repair_cascade import robust_geometry_candidate, replay_audits, split_audits, audit_accuracy


def main():
    seed = 0
    cfg = SimRealConfig(name="diagnostic", seed=seed, judge_scale=.4,
                        grounding={"confuse": {"green": "blue", "blue": "green"}}, paired_feedback=True)
    b = Belief.initial(cfg)
    params = load_params(f"results/main_v1/dagger_beta_s{seed}_policy.npz")
    with Runner(4, policy={"params": params}) as runner:
        ev = deploy_and_attribute(runner, cfg, b, TaskDistribution(),
                                  np.random.default_rng(np.random.SeedSequence([seed, 1, 937])))
    candidate, geo = robust_geometry_candidate(b.recon, ev["real_logs"])
    audited = replay_audits(ev["audited"], b.recon, candidate, .4)
    fit, check = split_audits(audited)
    scale, report = verifier_recal(fit, .4)
    out = {"geometry": geo, "scale": scale, "fit": audit_test(fit), "check": audit_test(check),
           "all": audit_test(audited), "accuracy_before": audit_accuracy(check,.4,10),
           "accuracy_after": audit_accuracy(check,scale,10)}
    path = Path("results/optimization_20261025/calibration_signal.json")
    path.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2),flush=True)


if __name__ == "__main__":
    main()
