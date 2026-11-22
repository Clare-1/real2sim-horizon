"""Piper (松灵) adapter tests: unit conversion, bounded steps, gripper convention, no-SDK error.

No hardware and no CAN bus: ``FakePiper`` records the integer commands the SDK would receive.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from horizon.real.apriltag import StaticPoseSource
from horizon.real.piper import DEG, POS_UNIT_M, PiperArm, PiperConfig, PiperRobot


class _Pose:
    def __init__(self, xyz, rpy):
        (self.X_axis, self.Y_axis, self.Z_axis) = [int(round(v / POS_UNIT_M)) for v in xyz]
        (self.RX_axis, self.RY_axis, self.RZ_axis) = [int(round(v / DEG / 1e-3)) for v in rpy]


class _EndPoseMsg:
    def __init__(self, pose):
        self.end_pose = pose


class _GripMsg:
    def __init__(self, opening_m):
        self.gripper_state = type("G", (), {"grippers_angle": int(round(opening_m / POS_UNIT_M))})()


class _Sdk:
    """Minimal stand-in for ``piper_sdk.C_PiperInterface``."""

    def __init__(self, port):
        self.port = port
        self.connected = False
        self.enabled = False
        self.xyz = (0.25, 0.0, 0.20)
        self.rpy = (0.0, 60 * DEG, 0.0)
        self.gripper_angle = 70000
        self.commands = []

    def ConnectPort(self):
        self.connected = True

    def EnablePiper(self):
        self.enabled = True
        return True

    def MotionCtrl_2(self, *args):
        self.commands.append(("motion",) + args)

    def EndPoseCtrl(self, x, y, z, rx, ry, rz):
        self.commands.append(("pose", x, y, z, rx, ry, rz))
        self.xyz = (x * POS_UNIT_M, y * POS_UNIT_M, z * POS_UNIT_M)
        self.rpy = (rx * 1e-3 * DEG, ry * 1e-3 * DEG, rz * 1e-3 * DEG)

    def GripperCtrl(self, angle, effort, code, zero):
        self.commands.append(("gripper", angle, effort, code, zero))
        self.gripper_angle = angle

    def GetArmEndPoseMsgs(self):
        return _EndPoseMsg(_Pose(self.xyz, self.rpy))

    def GetArmGripperMsgs(self):
        return _GripMsg(self.gripper_angle * POS_UNIT_M)

    def DisableArm(self):
        self.enabled = False


def _sdk_module(fake):
    return type("M", (), {"C_PiperInterface": staticmethod(lambda port: fake)})


def _robot(max_step=0.015):
    cfg = PiperConfig(max_step=max_step, settle_s=0.0, gripper_open_m=0.070)
    fake = _Sdk(cfg.can_port)
    arm = PiperArm(cfg, sdk=_sdk_module(fake))
    source = StaticPoseSource(obj_pos=np.zeros((2, 3)), obj_quat=np.tile([1.0, 0, 0, 0], (2, 1)))
    return PiperRobot(arm=arm, pose_source=source, cfg=cfg), fake


def test_reset_connects_enables_and_returns_the_home_pose():
    robot, fake = _robot()
    sample = robot.reset()
    assert fake.connected and fake.enabled
    assert sample.grip_pos == pytest.approx(np.asarray(PiperConfig().home_xyz), abs=1e-6)
    assert any(c[0] == "motion" for c in fake.commands)
    assert any(c[0] == "pose" for c in fake.commands)


def test_one_step_moves_at_most_max_step_metres():
    robot, fake = _robot(max_step=0.015)
    robot.reset()
    before = float(fake.xyz[0])
    sample = robot.step(np.array([1.0, 0.0, 0.0, 0.0, 1.0]))   # push +x as hard as allowed
    assert sample.grip_pos[0] - before == pytest.approx(0.015, abs=1e-6)


def test_step_is_clipped_to_the_workspace():
    robot, fake = _robot(max_step=10.0)
    robot.reset()
    sample = robot.step(np.array([1.0, 1.0, 1.0, 0.0, 1.0]))
    assert sample.grip_pos[2] <= PiperConfig().workspace_hi[2] + 1e-9


def test_gripper_convention_matches_the_simulator():
    robot, fake = _robot()
    robot.reset()
    robot.step(np.array([0.0, 0.0, 0.0, 0.0, 1.0]))    # +1 closes
    closed = [c for c in fake.commands if c[0] == "gripper"][-1][1]
    robot.step(np.array([0.0, 0.0, 0.0, 0.0, -1.0]))   # -1 opens
    opened = [c for c in fake.commands if c[0] == "gripper"][-1][1]
    assert closed == 0
    assert opened == pytest.approx(PiperConfig().gripper_open_m / POS_UNIT_M, rel=1e-6)


def test_missing_sdk_raises_a_setup_error_not_a_crash():
    arm = PiperArm(PiperConfig(), sdk=type("M", (), {})())
    with pytest.raises(ImportError, match="no C_PiperInterface"):
        arm.connect()


def test_yaw_step_conversion_round_trips_through_the_sdk_units():
    robot, fake = _robot()
    robot.reset()
    robot.step(np.array([0.0, 0.0, 0.0, 1.0, 0.0]))
    pose_cmd = [c for c in fake.commands if c[0] == "pose"][-1]
    assert pose_cmd[6] == int(round(PiperConfig().max_yaw_step / DEG / 1e-3)), pose_cmd


def test_gripper_feedback_unit_is_configurable():
    """SDK builds disagree on the gripper unit; the config must be able to fix it."""
    cfg = PiperConfig(settle_s=0.0, gripper_feedback_unit_m=1e-3)
    fake = _Sdk(cfg.can_port)
    fake.gripper_angle = 35          # 35 units * 0.001 m = 35 mm
    arm = PiperArm(cfg, sdk=_sdk_module(fake))
    arm.connect()
    assert arm.gripper_feedback_m() == pytest.approx(0.035, abs=1e-9)


def test_integrate_from_measured_uses_the_measured_pose():
    """With measured integration each step is relative to where the arm really is, not to the
    last command, so a blocked arm cannot accumulate a phantom target."""
    cfg = PiperConfig(max_step=0.015, settle_s=0.0, integrate_from_measured=True)
    fake = _Sdk(cfg.can_port)
    arm = PiperArm(cfg, sdk=_sdk_module(fake))
    source = StaticPoseSource(obj_pos=np.zeros((2, 3)), obj_quat=np.tile([1.0, 0, 0, 0], (2, 1)))
    robot = PiperRobot(arm=arm, pose_source=source, cfg=cfg)
    robot.reset()
    # Pretend the arm is blocked: the commanded pose never moves, so the measured pose stays put.
    fake.EndPoseCtrl = lambda *a: fake.commands.append(("pose",) + a)
    robot.step(np.array([1.0, 0.0, 0.0, 0.0, 1.0]))
    first_cmd = [c for c in fake.commands if c[0] == "pose"][-1][1]
    robot.step(np.array([1.0, 0.0, 0.0, 0.0, 1.0]))
    second_cmd = [c for c in fake.commands if c[0] == "pose"][-1][1]
    # Both commands are ``measured + 15 mm``, so they are identical: no phantom accumulation.
    assert second_cmd == first_cmd, (first_cmd, second_cmd)
