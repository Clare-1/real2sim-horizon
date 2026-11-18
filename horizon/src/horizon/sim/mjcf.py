"""MJCF builder: SceneSpec -> MuJoCo XML with a floating parallel-jaw gripper.

The gripper is a 4-DoF Cartesian stage (x, y, z slides + yaw hinge) carrying two coupled
fingers, all gravity-compensated and driven by PD position actuators. It stands in for an
arm + IK controller: the policy acts in end-effector space exactly as ARSTAG's expert
programs do, and reachability is checked separately by the feasibility module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .scene import BIN_FLOOR, BIN_WALL, FixtureSpec, ObjectSpec, SceneSpec

FINGER_HALF = (0.012, 0.005, 0.022)  # finger box half extents (x, y-thickness, z-length)
PALM_Z = 0.035  # palm centre above the tool centre point (TCP)
PALM_HALF = (0.02, 0.065, 0.01)
MAX_FINGER = 0.05  # per-finger travel -> max opening 0.10 m
TCP_Z_MIN = 0.024  # fingertips stay 2 mm above the table


@dataclass
class SimConfig:
    timestep: float = 0.002
    control_hz: float = 20.0
    kp_xyz: float = 2000.0
    kv_xyz: float = 100.0
    kp_yaw: float = 30.0
    kv_yaw: float = 0.4
    kp_finger: float = 500.0
    kv_finger: float = 10.0
    finger_force: float = 25.0
    finger_friction: float = 1.5
    offscreen_width: int = 256
    offscreen_height: int = 256

    @property
    def n_substeps(self) -> int:
        return int(round(1.0 / (self.control_hz * self.timestep)))


def _fmt(*xs: float) -> str:
    return " ".join(f"{x:.6g}" for x in xs)


def _object_xml(o: ObjectSpec, friction_scale: float, mass_scale: float) -> str:
    z = o.pos[2]
    geom_type = "box" if o.shape == "box" else "cylinder"
    size = _fmt(*o.size)
    mu = o.friction * friction_scale
    return (
        f'    <body name="{o.name}" pos="{_fmt(o.pos[0], o.pos[1], z)}" euler="0 0 {o.yaw:.6g}">\n'
        f'      <freejoint name="{o.name}_jnt"/>\n'
        f'      <geom name="{o.name}" type="{geom_type}" size="{size}" rgba="{_fmt(*o.rgba)}" '
        f'mass="{o.mass * mass_scale:.6g}" friction="{mu:.4g} 0.01 0.001" condim="4"/>\n'
        f"    </body>\n"
    )


def _fixture_xml(f: FixtureSpec) -> str:
    x, y, z = f.pos
    if f.kind == "slot":
        hx, hy = f.size[:2]
        return (
            f'    <geom name="{f.name}" type="box" size="{_fmt(hx, hy, 0.0005)}" pos="{_fmt(x, y, z + 0.0005)}" '
            f'euler="0 0 {f.yaw:.6g}" rgba="{_fmt(*f.rgba)}" contype="0" conaffinity="0" group="2"/>\n'
        )
    if f.kind == "plate":
        hx, hy, hz = f.size
        return (
            f'    <geom name="{f.name}" type="box" size="{_fmt(hx, hy, hz)}" pos="{_fmt(x, y, z + hz)}" '
            f'euler="0 0 {f.yaw:.6g}" rgba="{_fmt(*f.rgba)}" friction="1 0.01 0.001" condim="4"/>\n'
        )
    if f.kind == "bin":
        ihx, ihy, wall_h = f.size
        t = BIN_WALL
        rgba = _fmt(*f.rgba)
        hz = wall_h / 2.0
        parts = [f'    <body name="{f.name}" pos="{_fmt(x, y, z)}" euler="0 0 {f.yaw:.6g}">\n']
        parts.append(
            f'      <geom name="{f.name}_floor" type="box" size="{_fmt(ihx + t, ihy + t, BIN_FLOOR / 2)}" '
            f'pos="0 0 {BIN_FLOOR / 2:.6g}" rgba="{rgba}"/>\n'
        )
        for k, (px, py, sx, sy) in enumerate(
            [(ihx + t / 2, 0, t / 2, ihy + t), (-ihx - t / 2, 0, t / 2, ihy + t),
             (0, ihy + t / 2, ihx, t / 2), (0, -ihy - t / 2, ihx, t / 2)]
        ):
            parts.append(
                f'      <geom name="{f.name}_wall{k}" type="box" size="{_fmt(sx, sy, hz)}" '
                f'pos="{_fmt(px, py, hz)}" rgba="{rgba}"/>\n'
            )
        parts.append("    </body>\n")
        return "".join(parts)
    raise ValueError(f"unknown fixture kind {f.kind}")


def _gripper_xml(cfg: SimConfig, home: tuple[float, float, float, float]) -> str:
    fx, fy, fz = FINGER_HALF
    px, py, pz = PALM_HALF
    mu = cfg.finger_friction
    return f"""    <body name="g_x" pos="0 0 0">
      <joint name="gx" type="slide" axis="1 0 0" range="-0.45 0.45" damping="2" armature="0.01"/>
      <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
      <body name="g_y" gravcomp="1">
        <joint name="gy" type="slide" axis="0 1 0" range="-0.6 0.6" damping="2" armature="0.01"/>
        <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
        <body name="g_z" gravcomp="1">
          <joint name="gz" type="slide" axis="0 0 1" range="{TCP_Z_MIN} 0.5" damping="2" armature="0.01"/>
          <inertial pos="0 0 0" mass="0.2" diaginertia="1e-4 1e-4 1e-4"/>
          <body name="hand" gravcomp="1">
            <joint name="gyaw" type="hinge" axis="0 0 1" range="-3.3 3.3" damping="0.05" armature="0.001"/>
            <inertial pos="0 0 {PALM_Z}" mass="0.3" diaginertia="6e-4 6e-4 6e-4"/>
            <geom name="palm" type="box" size="{_fmt(px, py, pz)}" pos="0 0 {PALM_Z}" rgba="0.15 0.15 0.15 1"/>
            <site name="tcp" pos="0 0 0" size="0.004" rgba="1 0 0 1" group="3"/>
            <camera name="wrist" pos="0.07 0 0.08" xyaxes="0 -1 0 0.6 0 0.8" fovy="70"/>
            <body name="finger_l_body" gravcomp="1">
              <joint name="finger_l" type="slide" axis="0 1 0" range="0 {MAX_FINGER}" damping="1" armature="0.001"/>
              <geom name="finger_l" type="box" size="{_fmt(fx, fy, fz)}" pos="0 {fy:.6g} 0" mass="0.03"
                    friction="{mu} 0.02 0.002" condim="4" rgba="0.5 0.5 0.55 1"/>
            </body>
            <body name="finger_r_body" gravcomp="1">
              <joint name="finger_r" type="slide" axis="0 -1 0" range="0 {MAX_FINGER}" damping="1" armature="0.001"/>
              <geom name="finger_r" type="box" size="{_fmt(fx, fy, fz)}" pos="0 {-fy:.6g} 0" mass="0.03"
                    friction="{mu} 0.02 0.002" condim="4" rgba="0.5 0.5 0.55 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
"""


def build_mjcf(scene: SceneSpec, cfg: SimConfig | None = None) -> str:
    cfg = cfg or SimConfig()
    tx, ty = scene.table_half
    fs = float(scene.physics.get("friction_scale", 1.0))
    ms = float(scene.physics.get("mass_scale", 1.0))
    objects = "".join(_object_xml(o, fs, ms) for o in scene.objects)
    fixtures = "".join(_fixture_xml(f) for f in scene.fixtures)
    gripper = _gripper_xml(cfg, scene.gripper_home)
    home = scene.gripper_home
    return f"""<mujoco model="{scene.scene_id}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{cfg.timestep}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="{cfg.offscreen_width}" offheight="{cfg.offscreen_height}"/>
    <quality shadowsize="0"/>
  </visual>
  <default>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".86 .86 .84" rgb2=".78 .78 .76" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.02"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <light pos="0.8 0.6 1.2" dir="-0.6 -0.4 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 -0.76" rgba="0.35 0.35 0.38 1"/>
    <geom name="table" type="box" size="{_fmt(tx, ty, 0.02)}" pos="0 0 -0.02" material="grid" friction="0.9 0.01 0.001" condim="4"/>
    <camera name="front" pos="0.95 0 0.75" xyaxes="0 1 0 -0.62 0 0.78" fovy="55"/>
    <camera name="top" pos="0 0 1.25" xyaxes="0 -1 0 1 0 0" fovy="50"/>
{fixtures}{objects}{gripper}  </worldbody>
  <contact>
    <exclude body1="finger_l_body" body2="finger_r_body"/>
  </contact>
  <equality>
    <joint joint1="finger_r" joint2="finger_l" polycoef="0 1 0 0 0" solref="0.004 1"/>
  </equality>
  <actuator>
    <position name="act_x" joint="gx" kp="{cfg.kp_xyz}" kv="{cfg.kv_xyz}" ctrlrange="-0.45 0.45"/>
    <position name="act_y" joint="gy" kp="{cfg.kp_xyz}" kv="{cfg.kv_xyz}" ctrlrange="-0.6 0.6"/>
    <position name="act_z" joint="gz" kp="{cfg.kp_xyz}" kv="{cfg.kv_xyz}" ctrlrange="{TCP_Z_MIN} 0.5"/>
    <position name="act_yaw" joint="gyaw" kp="{cfg.kp_yaw}" kv="{cfg.kv_yaw}" ctrlrange="-3.3 3.3"/>
    <position name="act_finger" joint="finger_l" kp="{cfg.kp_finger}" kv="{cfg.kv_finger}" ctrlrange="0 {MAX_FINGER}"
              forcelimited="true" forcerange="{-cfg.finger_force} {cfg.finger_force}"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="{_home_qpos(scene, home)}"/>
  </keyframe>
</mujoco>
"""


def _home_qpos(scene: SceneSpec, home: tuple[float, float, float, float]) -> str:
    # qpos layout follows body order in the XML: objects (7 each) then gripper (4 + 2)
    vals: list[float] = []
    for o in scene.objects:
        q = _yaw_quat(o.yaw)
        vals += [o.pos[0], o.pos[1], o.pos[2], *q]
    vals += [home[0], home[1], home[2], home[3], MAX_FINGER, MAX_FINGER]
    return _fmt(*vals)


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
