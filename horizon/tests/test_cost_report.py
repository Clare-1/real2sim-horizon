"""Reporting must reject experiments that cannot be compared fairly."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("make_cost_report", Path(__file__).parents[1] / "scripts/make_cost_report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def minimal_run(name="routed", seed=0):
    return {"config": {"name": name, "seed": seed, "repair_budget": 3},
            "repair_cost_used": 1, "history": [{"round": 1, "repair_cost": 1}],
            "final": {"real_nominal": {"by_family_H": {"pack:H2": {"n": 4}}}}}


@pytest.mark.parametrize("fault,pattern", [("budget", "budget exceeded"),
                                         ("ledger", "ledger"),
                                         ("grid", "evaluation grids"),
                                         ("checkpoint", "starting policies"),
                                         ("protocol", "fixed-policy")])
def test_unfair_or_inconsistent_results_are_rejected(tmp_path, fault, pattern):
    a, b = minimal_run(), minimal_run("cascade")
    if fault == "budget":
        b["repair_cost_used"] = 4
    elif fault == "ledger":
        b["history"][0]["repair_cost"] = 0
    elif fault == "grid":
        b["final"]["real_nominal"]["by_family_H"]["pack:H2"]["n"] = 5
    elif fault == "checkpoint":
        b["provenance"] = {"checkpoint_sha256": "different"}
    else:
        b["protocol"] = "fixed_policy_probe"
    for r in (a, b):
        (tmp_path / (r["config"]["name"] + "_s0.json")).write_text(json.dumps(r))
    with pytest.raises(ValueError, match=pattern):
        report.load_runs(tmp_path)


def test_previous_summary_is_not_loaded_as_an_experiment(tmp_path):
    (tmp_path / "cost_summary.json").write_text('{}')
    (tmp_path / "routed_s0.json").write_text(json.dumps(minimal_run()))
    assert list(report.load_runs(tmp_path)) == ["routed"]
