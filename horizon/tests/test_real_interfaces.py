"""Real-robot scaffold: scripted robot, deployment log, twin calibration (no hardware)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from horizon.real import (load_jsonl_robot, make_mock_robot, median_support, placements_from_log,
                          rows_to_episode, run_episodes, update_recon_bias)


def test_scripted_robot_reaches_success_and_logs(tmp_path):
    robot = make_mock_robot()
    log = tmp_path / "deploy.jsonl"
    rows = run_episodes(robot, lambda s: [0, 0, 0, 0, 1], episodes=1, max_steps=50, log_path=log)
    assert rows and rows[-1]["success"] is True
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[0])["episode"] == 0


def test_csv_replay_round_trip(tmp_path):
    robot = make_mock_robot()
    sample = robot.reset()
    row = {"t": sample.t, "obj_pos": sample.obj_pos.tolist(), "obj_quat": sample.obj_quat.tolist(),
           "grip_pos": sample.grip_pos.tolist(), "grip_yaw": sample.grip_yaw,
           "grip_open": sample.grip_open, "held": sample.held, "success": True}
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    replay = load_jsonl_robot(path)
    assert replay.reset().success is True


def test_twin_calibration_does_not_mutate_input():
    recon = {"bias": {"plate": 0.1, "tray": 0.2}}
    out = update_recon_bias(recon, 0.05, -0.02)
    assert recon["bias"] == {"plate": 0.1, "tray": 0.2}
    assert out["bias"] == {"plate": 0.15000000000000002, "tray": 0.18000000000000002}
    assert median_support([0.02, 0.03, 0.025]) == 0.025


def test_placements_from_log_uses_successful_rows():
    rows = [{"success": True, "obj_pos": [[0, 0, 0.03]]},
            {"success": False, "obj_pos": [[0, 0, 0.09]]}]
    assert placements_from_log(rows) == [0.03]


def test_success_fn_can_override_the_robot_flag(tmp_path):
    robot = make_mock_robot()
    rows = run_episodes(robot, lambda s: [0, 0, 0, 0, 1], episodes=1, max_steps=30,
                        log_path=tmp_path / "d.jsonl", success_fn=lambda s: s.t >= 3)
    assert rows[-1]["t"] == 3 and rows[-1]["success"] is True


def test_rows_to_episode_builds_the_expected_arrays():
    task = SimpleNamespace(key="pack_H4_s0", family="pack", H=4, seed=0,
                           scene=SimpleNamespace(to_dict=lambda: {"objects": []}),
                           graph=SimpleNamespace(to_dict=lambda: {"subgoals": []}))
    base = {"grip_pos": [0.0, 0.0, 0.1], "grip_yaw": 0.0, "grip_open": 0.08,
            "grip_open_vel": 0.0, "held": -1, "action": [0, 0, 0, 0, 1],
            "obj_pos": [[0.0, 0.0, 0.02]], "obj_quat": [[1.0, 0.0, 0.0, 0.0]]}
    rows = [{**base, "t": 0, "success": False}, {**base, "t": 1, "success": True}]
    log = rows_to_episode(rows, task)
    assert log.T == 2 and log.success is True
    assert log.grip.shape == (2, 6) and log.obj_obs.shape == (2, 1, 7)
