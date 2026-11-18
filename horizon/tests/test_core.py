"""Core invariants: task graphs, predicates, snapshots, expert, attribution, metrics."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from horizon.datagen.rollout import ExpertController, PerturbationConfig, rollout
from horizon.eval.metrics import fit_horizon_decay, wilson
from horizon.monitor.attribution import attribute_auto, attribute_oracle
from horizon.policy.dataset import episode_samples
from horizon.policy.features import FLAT_DIM, HIER_DIM
from horizon.sim.env import TabletopEnv
from horizon.taskgraph.compiler import CompileError, TemplateCompiler
from horizon.taskgraph.feasibility import check
from horizon.taskgraph.repair import repair
from horizon.taskgraph.schema import Pred, TaskGraph
from horizon.tasks.factory import make_task


@pytest.mark.parametrize("family", ["pack", "sort", "stack"])
def test_compile_and_feasible(family):
    task = make_task(family, 5, seed=3)
    assert task.graph.H == 5
    assert task.report.ok
    assert task.graph.order() == list(range(5))
    rt = TaskGraph.from_dict(task.graph.to_dict())
    assert rt.to_dict() == task.graph.to_dict()


def test_compiler_rejects_unknown_instruction():
    task = make_task("pack", 2, seed=0)
    with pytest.raises(CompileError):
        TemplateCompiler().compile(task.scene, "dance a little")


def test_symbolic_check_and_reorder_repair():
    task = make_task("stack", 4, seed=1)
    g = copy.deepcopy(task.graph)
    # swap two layers so a layer is stacked onto one that has not been placed yet
    g.subgoals[1], g.subgoals[2] = g.subgoals[2], g.subgoals[1]
    for i, s in enumerate(g.subgoals):
        s.deps = [g.subgoals[i - 1].sid] if i else []
    rep = check(task.scene, g)
    assert "precondition" in rep.kinds()
    fixed = repair(task.scene, g, rep)
    assert fixed.report.ok, fixed.report.issues
    assert fixed.log


def test_pred_parse_roundtrip():
    p = Pred.parse("on(obj1, plate)")
    assert p == Pred("on", ("obj1", "plate")) and str(p) == "on(obj1, plate)"


def test_snapshot_restore_is_exact():
    task = make_task("pack", 3, seed=2)
    env = TabletopEnv(task.scene, seed=0)
    env.reset()
    a = np.array([0.5, -0.3, -0.2, 0.1, 1.0])
    for _ in range(10):
        env.step(a)
    snap = env.get_snapshot()
    ref = [env.step(a).obj_pos.copy() for _ in range(5)]
    env.set_snapshot(snap)
    again = [env.step(a).obj_pos.copy() for _ in range(5)]
    for x, y in zip(ref, again):
        np.testing.assert_allclose(x, y, atol=1e-9)


@pytest.mark.parametrize("family", ["pack", "sort", "stack"])
def test_expert_solves_and_logs(family):
    task = make_task(family, 3, seed=5)
    log = rollout(task, ExpertController(), seed=0, snapshot_events=True)
    assert log.success
    assert log.sat_true[-1].all()
    assert set(np.unique(log.k_true)) >= {0, 1, 2}
    assert attribute_auto(log) is None and attribute_oracle(log) is None
    b = episode_samples(log)
    assert b.x_hier.shape[1] == HIER_DIM and b.x_flat.shape[1] == FLAT_DIM
    assert np.isfinite(b.x_hier).all() and np.isfinite(b.x_flat).all()


def test_expert_recovers_from_slip():
    task = make_task("pack", 2, seed=7)
    log = rollout(task, ExpertController(), seed=1, pert=PerturbationConfig(slip=1.0))
    assert any(kind == "slip" for _, kind, _ in log.events)
    assert log.success
    assert not log.valid.all()  # perturbed steps are excluded from supervision


def test_segment_from_snapshot():
    task = make_task("sort", 3, seed=4)
    full = rollout(task, ExpertController(), seed=0, snapshot_events=True)
    t1 = min(t for t in full.snapshots if full.k_true[t] == 1)
    seg = rollout(task, ExpertController(), seed=0, start=full.snapshots[t1], stop_when="subgoal", stop_subgoal=1)
    assert seg.k_true[0] == 1
    assert seg.meta["final_sat"][0] and seg.meta["final_sat"][1]


def test_wilson_and_decay_fit():
    p, lo, hi = wilson(8, 10)
    assert lo < p < hi
    H = np.array([2, 4, 6, 8, 10])
    n = np.full(5, 400)
    k = np.round(n * 0.9 * np.exp(-0.15 * H)).astype(int)
    fit = fit_horizon_decay(H, k, n)
    assert abs(fit["alpha"] - 0.15) < 0.02 and abs(fit["S0"] - 0.9) < 0.05


def test_mjcf_import_roundtrip(tmp_path):
    from horizon.real2sim.mjcf_import import import_mjcf
    from horizon.sim.mjcf import build_mjcf

    task = make_task("sort", 4, seed=2)
    p = tmp_path / "scene.xml"
    p.write_text(build_mjcf(task.scene))
    sc = import_mjcf(p)
    assert len(sc.objects) == 4
    for a, b in zip(sorted(task.scene.objects, key=lambda o: o.name), sorted(sc.objects, key=lambda o: o.name)):
        assert a.name == b.name
        assert abs(a.half_height - b.half_height) < 0.006
        assert np.linalg.norm(np.array(a.pos[:2]) - np.array(b.pos[:2])) < 0.02
    assert sum(f.kind == "bin" for f in sc.fixtures) == 3


def test_llm_graph_json_goes_through_feasibility():
    from horizon.taskgraph.llm_compiler import graph_from_json

    task = make_task("stack", 3, seed=0)
    names = [g.obj for g in task.graph.subgoals]
    data = {"subgoals": [
        {"obj": names[0], "relation": "on", "target": "plate", "after": []},
        {"obj": names[2], "relation": "on", "target": names[1], "after": [0]},  # wrong order on purpose
        {"obj": names[1], "relation": "on", "target": names[0], "after": [0]},
    ], "unresolved": []}
    g = graph_from_json(data, task.scene, "make a sandwich", "t")
    rep = check(task.scene, g)
    assert "precondition" in rep.kinds()
    fixed = repair(task.scene, g, rep)
    assert fixed.report.ok
    assert [s.obj for s in fixed.graph.subgoals] == names


def test_agentic_rsr_scene_handoff():
    """Agentic RSR Stage-3 MJCF -> SceneSpec -> task graph -> our expert solves it."""
    from pathlib import Path

    from horizon.datagen.rollout import rollout as _rollout
    from horizon.real2sim.mjcf_import import import_mjcf
    from horizon.tasks.factory import TaskInstance

    fx = Path(__file__).resolve().parents[2] / "baselines/agentic_rsr/tests/fixtures/stage3/code_as_policy_tabletop.xml"
    if not fx.exists():
        pytest.skip("Agentic RSR checkout not present")
    sc = import_mjcf(fx)
    assert {o.name for o in sc.objects} == {"blue_block", "green_block", "red_block"}
    assert {f.color for f in sc.fixtures if f.kind == "bin"} == {"yellow", "blue", "green"}
    with pytest.raises(CompileError):  # no red bin: needs clarification, as ARSTAG asks the user
        TemplateCompiler().compile(sc, "Sort the objects into the bins matching their colours, in this order: red_block.")
    g = TemplateCompiler().compile(sc, "Sort the objects into the bins matching their colours, in this order: blue_block, green_block.")
    rep = check(sc, g)
    assert rep.ok
    log = _rollout(TaskInstance("sort", g.H, 0, sc, g, g.instruction, rep), ExpertController(), seed=0)
    assert log.success


def test_geometry_repair_recovers_support_heights():
    """Support heights reconstructed too low are re-estimated from real observations."""
    import dataclasses

    from horizon.real2sim.recon_error import DEPLOY_WORLD, ReconModel, single_view_bias
    from horizon.rsi.simreal import geometry_repair

    recon = single_view_bias()
    logs = []
    for fam in ("pack", "stack"):
        for seed in range(3):
            task = make_task(fam, 3, seed)
            real = DEPLOY_WORLD.apply(task.scene)
            task = dataclasses.replace(task, scene=real, belief=recon.apply(real))
            logs.append(rollout(task, ExpertController(), seed=seed))
    repaired, report = geometry_repair(recon, logs)
    assert set(report) == {"plate", "tray"}
    for name in ("plate", "tray"):
        assert abs(report[name]["estimated"] - 0.02) < 0.004, report
    believed = repaired.apply(DEPLOY_WORLD.apply(make_task("pack", 3, 0).scene))
    assert abs(next(f.top_z for f in believed.fixtures if f.name == "tray") - 0.02) < 0.004
    assert isinstance(ReconModel.from_dict(repaired.to_dict()), ReconModel)


def test_grounding_error_is_found_by_audit_and_recompiled():
    """A swapped colour grounding is executed perfectly, judged done, and corrected from audits."""
    import dataclasses

    from horizon.rsi.simreal import audit_test, graph_repair, judged_done
    from horizon.taskgraph.grounding import GroundingError

    gerr = GroundingError({"green": "blue", "blue": "green"})
    logs = []
    for seed in range(2):
        task = make_task("sort", 4, seed)
        planned = gerr.apply(task.scene, task.graph)
        assert [g.target for g in planned.subgoals] != [g.target for g in task.graph.subgoals]
        logs.append(rollout(dataclasses.replace(task, belief_graph=planned), ExpertController(), seed=seed))
    for log in logs:
        assert judged_done(log) and not log.success  # the robot's plan is done; the instruction is not
        assert log.sat_plan is not None and all(log.meta["plan_final_sat"])
    audit = audit_test(logs)
    assert audit["plan"] == len(logs) and audit["judge"] == 0
    fixed, report = graph_repair(gerr, logs)
    assert fixed.is_identity(), report
    assert GroundingError.from_dict(gerr.to_dict()) == gerr


def test_verifier_recal_loosens_a_strict_judge():
    """Keyframe audit labels move a too-strict judge to a tolerance that confirms correct placements."""
    from horizon.rsi.simreal import verifier_recal

    logs = [rollout(make_task("pack", 3, seed), ExpertController(), seed=seed) for seed in range(3)]
    scale, report = verifier_recal(logs, current=0.4)
    acc = report["balanced_accuracy"]
    assert scale >= 0.8 and acc[scale] >= max(acc.values()) - 0.005, report
    assert acc[scale] > acc[0.4]
    assert verifier_recal(logs, current=1.0)[0] == 1.0  # already optimal: no change without evidence


def test_twin_physics_keeps_the_robots_belief():
    """physics= replaces only the world the robot acts in; the robot keeps its (biased) belief."""
    from horizon.datagen.parallel import Job, run_serial
    from horizon.real2sim.recon_error import DEPLOY_WORLD, single_view_bias

    bias = single_view_bias().to_dict()

    def job(physics):
        return Job(family="pack", H=2, task_seed=3, seed=4, controller="expert", world=DEPLOY_WORLD.to_dict(),
                   recon=bias, real=True, physics=physics, want_samples=False, keep_log=True)

    real, twin_true, twin_belief = run_serial([job(None), job({}), job(bias)])
    assert real.error is None and twin_true.error is None and twin_belief.error is None
    assert real.success == twin_true.success  # a perfect twin reproduces the real outcome
    assert not real.success and twin_belief.success  # a twin built from the wrong belief hides the gap
