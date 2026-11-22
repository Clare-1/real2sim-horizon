"""AgileX Piper (松灵 Piper) adapter for the real-robot track.

Two pieces, both SDK-free at import time:

* :class:`PiperArm` wraps ``piper_sdk`` (CAN interface). It converts the SDK's integer units
  (0.001 mm for position, 0.001 deg for angles, 0.001 mm for the gripper) into SI units.
* :class:`PiperRobot` implements the :class:`~horizon.real.interfaces.Robot` protocol with the same
  action semantics as the simulator: ``[dx, dy, dz, dyaw, grip] in [-1, 1]``, where one control step
  moves at most ``max_step`` metres, at most ``max_yaw_step`` radians, and the gripper interpolates
  between ``gripper_open_m`` (action -1) and 0 (action +1).

Object poses never come from the arm: they come from a :class:`PoseSource` (AprilTag board, depth +
detector, or ROS). That keeps the state policy identical to the simulator.

Setup on the robot machine (needs the arm):

    sudo ip link set up can0 type can bitrate 1000000
    pip install piper_sdk
    python scripts/run_real_deploy.py --robot piper --can can0

Nothing here touches the network or the CAN bus until ``connect()`` is called, so the adapter is
unit-testable with a fake arm (``tests/test_piper_adapter.py``).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .interfaces import PoseSample

POS_UNIT_M = 1e-6      # SDK position unit: 0.001 mm
ANG_MDEG_RAD = 1e-3    # SDK angle unit: 0.001 deg
DEG = math.pi / 180.0


@dataclass
class PiperConfig:
    """Hardware constants. Defaults are stock Piper numbers; measure yours before trusting them."""

    can_port: str = "can0"
    home_xyz: tuple = (0.25, 0.0, 0.20)          # metres, tool centre point at reset
    home_rpy_deg: tuple = (0.0, 60.0, 0.0)       # Piper vertical grasp pose (deg)
    max_step: float = 0.015                      # metres per control step for a unit action
    max_yaw_step: float = 0.15                   # radians per control step for a unit action
    gripper_open_m: float = 0.070                # fully open finger separation (Piper: 70 mm)
    gripper_effort: int = 1000                   # SDK effort unit; vendor default
    gripper_hold_effort: int = 500               # effort while closing on an object
    speed_pct: int = 30                          # motion speed percentage
    move_timeout_s: float = 8.0
    settle_s: float = 0.25                       # wait after a command before reading the state
    feedback_timeout_s: float = 2.0
    workspace_lo: tuple = (-0.4, -0.4, 0.02)
    workspace_hi: tuple = (0.5, 0.4, 0.45)


class PiperArm:
    """Thin, testable wrapper over ``piper_sdk.C_PiperInterface``.

    The wrapper exists so that unit conversion lives in one place, the integration can be tested
    without hardware, and an SDK rename cannot leak into the deployment loop.
    """

    def __init__(self, cfg: PiperConfig | None = None, sdk=None):
        self.cfg = cfg or PiperConfig()
        self._sdk = sdk            # injected for tests; ``None`` means import piper_sdk
        self._piper = None

    # ------------------------------------------------------------------ lifecycle
    def connect(self):
        if self._piper is not None:
            return self
        sdk = self._sdk
        if sdk is None:
            try:
                import piper_sdk
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "piper_sdk is not installed. On the robot machine run `pip install piper_sdk` "
                    "and bring up CAN first:\n"
                    "  sudo ip link set up can0 type can bitrate 1000000"
                ) from exc
            sdk = piper_sdk
        interface = (getattr(sdk, "C_PiperInterface", None)
                     or getattr(sdk, "C_PiperInterface_V2", None))
        if interface is None:
            raise ImportError("piper_sdk has no C_PiperInterface class")
        self._piper = interface(self.cfg.can_port)
        self._piper.ConnectPort()
        return self

    def enable(self, timeout_s: float | None = None):
        """Enable the arm; every motion command is ignored until this succeeds."""
        self._require()
        deadline = time.time() + (timeout_s or self.cfg.move_timeout_s)
        while True:
            if self._call_first(("EnablePiper", "EnableArm"), 7):
                return True
            if time.time() > deadline:
                raise TimeoutError("Piper did not report enabled; check CAN and the E-stop")
            time.sleep(0.05)

    def close(self) -> None:
        if self._piper is not None:
            for name in ("DisableArm", "DisconnectPort"):
                fn = getattr(self._piper, name, None)
                if fn is not None:
                    try:
                        fn()
                    except Exception:
                        pass
        self._piper = None

    # ------------------------------------------------------------------ state
    def end_pose(self) -> tuple[float, float, float, float, float, float]:
        """Current tool pose in SI units: (x, y, z) metres, (rx, ry, rz) radians."""
        self._require()
        getter = getattr(self._piper, "GetArmEndPoseMsgs", None)
        if getter is None:
            raise RuntimeError("piper_sdk build has no GetArmEndPoseMsgs; cannot read the pose")
        deadline = time.time() + self.cfg.feedback_timeout_s
        while True:
            msg = getter()
            pose = getattr(msg, "end_pose", None) or getattr(msg, "EndPose", None)
            if pose is not None:
                return (float(getattr(pose, "X_axis", 0.0)) * POS_UNIT_M,
                        float(getattr(pose, "Y_axis", 0.0)) * POS_UNIT_M,
                        float(getattr(pose, "Z_axis", 0.0)) * POS_UNIT_M,
                        float(getattr(pose, "RX_axis", 0.0)) * ANG_MDEG_RAD * DEG,
                        float(getattr(pose, "RY_axis", 0.0)) * ANG_MDEG_RAD * DEG,
                        float(getattr(pose, "RZ_axis", 0.0)) * ANG_MDEG_RAD * DEG)
            if time.time() > deadline:
                raise TimeoutError("Piper end-pose feedback timed out")
            time.sleep(0.02)

    def gripper_feedback_m(self) -> float:
        """Measured finger separation in metres (Piper unit: 0.001 mm)."""
        self._require()
        getter = getattr(self._piper, "GetArmGripperMsgs", None)
        if getter is None:
            return 0.0
        msg = getter()
        grip = getattr(msg, "gripper_state", None) or getattr(msg, "GripperState", None)
        if grip is None:
            return 0.0
        return float(getattr(grip, "grippers_angle", 0.0)) * POS_UNIT_M

    # ------------------------------------------------------------------ commands
    def command_pose(self, xyz, rpy) -> None:
        """Move the tool to an absolute pose (metres, radians)."""
        self._require()
        x, y, z = [float(v) for v in xyz]
        rx, ry, rz = [float(v) for v in rpy]
        motion = getattr(self._piper, "MotionCtrl_2", None)
        if motion is not None:
            # ctrl_mode=CAN, move_mode=MOVE J, speed pct, is_mit_mode=0
            motion(0x01, 0x01, int(self.cfg.speed_pct), 0x00)
        ctrl = getattr(self._piper, "EndPoseCtrl", None)
        if ctrl is None:
            raise RuntimeError("piper_sdk build has no EndPoseCtrl; cannot command a pose")
        ctrl(int(round(x / POS_UNIT_M)), int(round(y / POS_UNIT_M)), int(round(z / POS_UNIT_M)),
             int(round(rx / DEG / ANG_MDEG_RAD)), int(round(ry / DEG / ANG_MDEG_RAD)),
             int(round(rz / DEG / ANG_MDEG_RAD)))

    def command_gripper(self, opening_m: float, effort: int | None = None) -> None:
        """Command the finger separation in metres (0 = closed)."""
        self._require()
        angle = float(np.clip(opening_m, 0.0, self.cfg.gripper_open_m))
        ctrl = getattr(self._piper, "GripperCtrl", None)
        if ctrl is None:
            raise RuntimeError("piper_sdk build has no GripperCtrl; cannot command the gripper")
        ctrl(int(round(angle / POS_UNIT_M)),
             int(effort if effort is not None else self.cfg.gripper_effort), 0x01, 0)

    # ------------------------------------------------------------------ helpers
    def _require(self) -> None:
        if self._piper is None:
            raise RuntimeError("PiperArm.connect() must be called before use")

    def _call_first(self, names: tuple, *args) -> bool:
        for name in names:
            fn = getattr(self._piper, name, None)
            if fn is None:
                continue
            try:
                out = fn(*args)
            except TypeError:      # older SDKs take no argument
                out = fn()
            if out is None or out is True or out == 1:
                return True
        return False


@dataclass
class PiperRobot:
    """Deployment-loop robot: the simulator action space on top of a Piper arm.

    ``pose_source`` supplies the object poses (AprilTag board); the arm supplies the tool pose. Both
    are merged into one :class:`PoseSample`, so the state policy sees the features it was trained on.
    """

    arm: PiperArm
    pose_source: object            # PoseSource: read() -> PoseSample with obj_pos/obj_quat filled
    cfg: PiperConfig = field(default_factory=PiperConfig)
    target: np.ndarray | None = None      # absolute TCP target (x, y, z, yaw)
    _t: int = 0

    def reset(self) -> PoseSample:
        self.arm.connect()
        self.arm.enable()
        self.arm.command_gripper(self.cfg.gripper_open_m)
        self.arm.command_pose(self.cfg.home_xyz, [v * DEG for v in self.cfg.home_rpy_deg])
        time.sleep(self.cfg.settle_s)
        x, y, z, _, _, rz = self.arm.end_pose()
        self.target = np.array([x, y, z, rz], dtype=float)
        self._t = 0
        return self._sample()

    def step(self, action: np.ndarray) -> PoseSample:
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        if a.size < 5:
            raise ValueError(f"PiperRobot.step wants 5 values [dx,dy,dz,dyaw,grip], got {a.size}")
        x, y, z, _, ry, rz = self.arm.end_pose()
        xyz, yaw = self._integrate(np.array([x, y, z]), rz, a)
        self.arm.command_pose(xyz, [0.0, ry, yaw])
        opening = self.cfg.gripper_open_m * (1.0 - float(a[4])) / 2.0   # +1 closes, -1 opens
        effort = self.cfg.gripper_effort if a[4] < 0.5 else self.cfg.gripper_hold_effort
        self.arm.command_gripper(opening, effort)
        time.sleep(self.cfg.settle_s)
        self._t += 1
        return self._sample()

    def close(self) -> None:
        self.arm.close()

    def _integrate(self, pos: np.ndarray, yaw: float, action: np.ndarray):
        """One bounded step, clipped to the workspace: the same limits as ``TabletopEnv.step``."""
        target = self.target if self.target is not None else np.array([*pos, yaw])
        nxt = target[:3] + action[:3] * self.cfg.max_step
        nxt = np.clip(nxt, np.asarray(self.cfg.workspace_lo), np.asarray(self.cfg.workspace_hi))
        nxt_yaw = float(target[3] + action[3] * self.cfg.max_yaw_step)
        self.target = np.array([nxt[0], nxt[1], nxt[2], nxt_yaw])
        return nxt, nxt_yaw

    def _sample(self) -> PoseSample:
        perceived = self.pose_source.read()
        x, y, z, _, _, rz = self.arm.end_pose()
        return PoseSample(
            t=self._t,
            obj_pos=np.asarray(perceived.obj_pos, dtype=float),
            obj_quat=np.asarray(perceived.obj_quat, dtype=float),
            grip_pos=np.array([x, y, z], dtype=float),
            grip_yaw=float(rz),
            grip_open=float(self.arm.gripper_feedback_m() or perceived.grip_open),
            grip_open_vel=0.0,
            held=int(getattr(perceived, "held", -1)),
            success=bool(getattr(perceived, "success", False)),
        )
