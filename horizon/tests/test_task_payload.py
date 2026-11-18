"""External tasks retain their identity, controller belief, and spawn compatibility."""
from __future__ import annotations

import copy
import json

import pytest

from horizon.datagen.parallel import Job, Runner, run_serial
from horizon.real2sim.recon_error import RealWorld, ReconModel
from horizon.sim.scene import FixtureSpec, ObjectSpec, SceneSpec
from horizon.taskgraph.feasibility import check
from horizon.taskgraph.schema import Pred, Subgoal, TaskGraph
from horizon.tasks.factory import TaskInstance
from horizon.tasks.serialization import task_from_payload, task_to_payload


def external_task() -> TaskInstance:
    scene = SceneSpec(
        scene_id="external-scene", provenance="external-json-test",
        objects=[ObjectSpec("external_block", "box", (0.02, 0.02, 0.02), (-0.12, 0.30, 0.02))],
        fixtures=[FixtureSpec("external_slot", "slot", (0.08, 0.0, 0.0), (0.05, 0.05))],
    )
    instruction = "Move the imported block into its named slot."
    graph = TaskGraph("external-plan", "pack", instruction, [
        Subgoal("external-step", "pick_place", "external_block", "external_slot",
                Pred("at_slot", ("external_block", "external_slot"))),
    ])
    return TaskInstance("pack", 1, 91, scene, graph, instruction, check(scene, graph),
                        attempts=2, log=["imported from a JSON scene"])


def test_json_roundtrip_preserves_distinct_beliefs_and_controller_view():
    task = external_task()
    task.belief = copy.deepcopy(task.scene)
    task.belief.provenance = "estimated-scene"
    task.belief.fixtures[0].pos = (0.08, 0.0, 0.006)
    task.belief_graph = copy.deepcopy(task.graph)
    task.belief_graph.version = 7
    task.belief_graph.repair_log = ["controller plan"]
    payload = json.loads(json.dumps(task_to_payload(task)))
    loaded = task_from_payload(payload)
    assert task_to_payload(loaded) == payload
    assert loaded.report.ok and loaded.attempts == 2 and loaded.log == task.log
    assert loaded.scene.fixtures[0].pos[2] == 0.0
    assert loaded.belief.fixtures[0].pos[2] == 0.006
    view = loaded.controller_view()
    assert view.scene.provenance == "estimated-scene"
    assert view.graph.version == 7
    assert view.belief is None and view.belief_graph is None
    payload["scene"]["objects"][0]["pos"][0] = 99
    assert loaded.scene.objects[0].pos[0] == -0.12


@pytest.mark.parametrize("field,value", [("family", "sort"), ("H", 2), ("task_seed", 92)])
def test_job_metadata_mismatch_is_an_explicit_error(field, value):
    task = external_task()
    args = {"family": task.family, "H": task.H, "task_seed": task.seed, "seed": 0,
            "task_payload": task_to_payload(task)}
    args[field] = value
    result = run_serial([Job(**args)])[0]
    assert result.log is None and result.samples is None
    assert "mismatch" in result.error
    assert ("seed" if field == "task_seed" else field) in result.error


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p["graph"]["subgoals"][0].update(deps=["external-step"]), "cycle"),
    (lambda p: p["graph"]["subgoals"][0].update(obj="missing"), "unknown object"),
    (lambda p: p["graph"]["subgoals"][0].update(effect="at_slot(external_block)"), "arity"),
    (lambda p: p["graph"].update(family="sort"), "family/H"),
    (lambda p: p.update(version=999), "version"),
])
def test_invalid_graph_or_schema_is_rejected(mutation, error):
    payload = task_to_payload(external_task())
    mutation(payload)
    with pytest.raises(ValueError, match=error):
        task_from_payload(payload)


def test_invalid_belief_graph_is_validated_too():
    payload = task_to_payload(external_task())
    payload["belief_graph"] = copy.deepcopy(payload["graph"])
    payload["belief_graph"]["subgoals"][0]["target"] = "missing"
    with pytest.raises(ValueError, match="controller graph.*unknown target"):
        task_from_payload(payload)


@pytest.mark.parametrize("value", [object(), float("nan")])
def test_python_objects_and_nonfinite_values_are_not_payloads(value):
    payload = task_to_payload(external_task())
    payload["scene"]["physics"]["invalid"] = value
    with pytest.raises(ValueError, match="finite JSON values"):
        task_from_payload(payload)


def test_serial_external_scene_executes_without_calling_template(monkeypatch):
    import horizon.tasks.factory as factory

    def forbidden(*args, **kwargs):
        raise AssertionError("an explicit task must never call make_task")

    monkeypatch.setattr(factory, "make_task", forbidden)
    task = external_task()
    result = run_serial([Job(task.family, task.H, task.seed, 0,
                             task_payload=task_to_payload(task), keep_log=True)])[0]
    assert result.error is None, result.error
    assert result.success and result.samples.n > 0
    assert result.log.scene["scene_id"] == task.scene.scene_id
    assert result.log.graph["task_id"] == task.graph.task_id
    assert result.log.scene["objects"][0]["name"] == "external_block"


def test_spawn_external_expert_matches_serial_execution():
    task = external_task()
    job = Job(task.family, task.H, task.seed, 0, task_payload=task_to_payload(task), keep_log=True)
    serial = run_serial([job])[0]
    with Runner(workers=1) as runner:
        spawned = runner.map([job])[0]
    assert serial.error is None and spawned.error is None, (serial.error, spawned.error)
    assert serial.success and spawned.success
    assert spawned.log.scene["scene_id"] == "external-scene"
    assert spawned.log.graph == serial.log.graph
    assert spawned.n_valid == serial.n_valid
    assert (spawned.log.action == serial.log.action).all()


def test_world_and_reconstruction_are_applied_to_external_base():
    task = external_task()
    payload = task_to_payload(task)
    result = run_serial([Job(task.family, task.H, task.seed, 0, task_payload=payload,
                             world=RealWorld(tray_height=0.02).to_dict(),
                             recon=ReconModel(bias={"tray": 0.3}).to_dict(), real=True,
                             keep_log=True, want_samples=False, max_steps=1)])[0]
    assert result.error is None, result.error
    assert result.log.scene["scene_id"] == "external-scene"
    assert result.log.scene["fixtures"][0]["pos"][2] == 0.02
    assert result.log.meta["belief"]["fixtures"][0]["pos"][2] == pytest.approx(0.006)
    assert payload["scene"]["fixtures"][0]["pos"][2] == 0.0
