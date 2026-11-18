"""Independent checks that a reused measurement cannot hide different work/state."""
import copy
import importlib.util
from pathlib import Path

import pytest


def validator_module():
    path = Path(__file__).parents[1] / "scripts/validate_harness_artifacts.py"
    spec = importlib.util.spec_from_file_location("reuse_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case():
    validator = validator_module()
    belief = {"bias": {}, "corrections": {}}
    score = {"nominal": {"overall": 0.5, "episodes": [{"success": True}, {"success": False}]}}
    key = lambda p: validator.canonical({"experiment": "run", "policy": p, "recon": belief, "evaluation": "curve"})
    previous = {"round": 1, "policy_after": "p1", "recon_after": belief,
                "post_train_evaluation_key": key("p1"), "post_train": score}
    current = {"round": 2, "policy_before": "p1", "policy_after": "p2",
               "recon_before": belief, "recon_after": belief, "post_train_evaluation_key": key("p2"),
               "reused_equal_state_evaluation": {"pre": True}, "pre_evaluation_source_round": 1,
               "pre": copy.deepcopy(score),
               "ledger": {"pre_evaluation": {"episodes": 0, "sim_steps": 0, "worker_seconds": 0.0}}}
    return validator, previous, current


def test_same_system_reuse_is_accepted_and_cost_is_zero():
    validator, previous, current = case()
    assert validator.validate_pre_evaluation({"reuse_pre_evaluation": True}, "run", current, [previous])


@pytest.mark.parametrize("mutation", ["policy", "belief", "evaluation", "cost", "source_round", "fingerprint"])
def test_independent_validator_rejects_forged_reuse(mutation):
    validator, previous, current = case()
    if mutation == "policy": current["policy_before"] = "another-policy"
    if mutation == "belief": current["recon_before"] = {"bias": {"tray": 0.3}, "corrections": {}}
    if mutation == "evaluation": current["pre"]["nominal"]["overall"] = 1.0
    if mutation == "cost": current["ledger"]["pre_evaluation"]["sim_steps"] = 100
    if mutation == "source_round": current["pre_evaluation_source_round"] = 0
    fingerprint = "another-run" if mutation == "fingerprint" else "run"
    with pytest.raises(ValueError):
        validator.validate_pre_evaluation({"reuse_pre_evaluation": True}, fingerprint, current, [previous])


def test_old_record_without_cache_fields_remains_valid():
    validator, _, current = case()
    current["reused_equal_state_evaluation"] = {"post_cal": False, "post_train": False}
    current.pop("post_train_evaluation_key")
    current.pop("pre_evaluation_source_round")
    assert validator.validate_pre_evaluation({}, "run", current, []) is False
