"""MuJoCo tabletop environment with end-effector delta control, perception noise,
control latency, perturbation hooks and full-state snapshots.

Snapshots are what make subgoal-attributed regeneration possible: the RSI controller
restores a state the *policy* reached (e.g. right before a missed grasp at stage 7) and
asks the state-based expert to continue from there.
"""

from __future__ import annotations

import collections
import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from .mjcf import MAX_FINGER, TCP_Z_MIN, SimConfig, build_mjcf
from .scene import SceneSpec
from .state import Geometry, WorldState, quat_yaw, yaw_quat

GRIP_JOINTS = ("gx", "gy", "gz", "gyaw", "finger_l", "finger_r")
ACTUATORS = ("act_x", "act_y", "act_z", "act_yaw", "act_finger")
WORKSPACE_LO = np.array([-0.42, -0.57, TCP_Z_MIN])
WORKSPACE_HI = np.array([0.42, 0.57, 0.45])


@dataclass
class EnvConfig:
    max_step: float = 0.015  # m per control step for a unit action
    max_yaw_step: float = 0.15  # rad per control step for a unit action
    lead: float = 0.02  # anti-windup: target may lead the TCP by at most this much
    lead_yaw: float = 0.3
    # perception noise: per-episode bias (calibration / reconstruction offset) + per-step jitter
    bias_xy: float = 0.003
    bias_z: float = 0.001
    jitter_xy: float = 0.0015
    jitter_z: float = 0.001
    jitter_yaw: float = 0.03
    proprio_jitter: float = 0.0005
    latency: int = 0  # control steps of actuation delay (ARSTAG-style latency randomisation)
    settle_steps: int = 150  # physics steps before the first observation


@dataclass
class Snapshot:
    mj_state: np.ndarray
    target: np.ndarray
    finger_target: float
    t: int
    obs_bias: np.ndarray
    latency_queue: list
    forced_open: int
    rng_state: dict


class TabletopEnv:
    def __init__(self, scene: SceneSpec, sim_cfg: SimConfig | None = None,
                 env_cfg: EnvConfig | None = None, seed: int = 0):
        self.scene = scene
        self.sim_cfg = sim_cfg or SimConfig()
        self.cfg = env_cfg or EnvConfig()
        self.geom = Geometry(scene)
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(scene, self.sim_cfg))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_obj = len(scene.objects)
        self.obj_geom = np.array([m.geom(o.name).id for o in scene.objects], dtype=int)
        self.obj_qadr = np.array([m.jnt_qposadr[m.joint(f"{o.name}_jnt").id] for o in scene.objects], dtype=int)
        self.obj_vadr = np.array([m.jnt_dofadr[m.joint(f"{o.name}_jnt").id] for o in scene.objects], dtype=int)
        self.g_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in GRIP_JOINTS], dtype=int)
        self.g_vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in GRIP_JOINTS], dtype=int)
        self.act_id = np.array([m.actuator(a).id for a in ACTUATORS], dtype=int)
        self.finger_geom = (m.geom("finger_l").id, m.geom("finger_r").id)
        self._geom_to_obj = {int(g): i for i, g in enumerate(self.obj_geom)}
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_size = mujoco.mj_stateSize(m, self._state_spec)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.target = np.zeros(4)
        self.finger_target = MAX_FINGER
        self.obs_bias = np.zeros((self.n_obj, 3))
        self.latency_queue: collections.deque = collections.deque()
        self.forced_open = 0
        self.sim_steps = 0  # physics steps executed (compute accounting)

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> WorldState:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for i, o in enumerate(self.scene.objects):
            a = self.obj_qadr[i]
            d.qpos[a:a + 3] = (o.pos[0], o.pos[1], o.pos[2] + 0.001)
            d.qpos[a + 3:a + 7] = yaw_quat(o.yaw)
        hx, hy, hz, hyaw = self.scene.gripper_home
        d.qpos[self.g_qadr] = (hx, hy, hz, hyaw, MAX_FINGER, MAX_FINGER)
        self.target = np.array([hx, hy, hz, hyaw])
        self.finger_target = MAX_FINGER
        self._apply_ctrl()
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d, nstep=self.cfg.settle_steps)
        self.sim_steps += self.cfg.settle_steps
        self.t = 0
        c = self.cfg
        self.obs_bias = np.concatenate([
            self.rng.normal(0, c.bias_xy, (self.n_obj, 2)),
            self.rng.normal(0, c.bias_z, (self.n_obj, 1)),
        ], axis=1)
        self.latency_queue = collections.deque(np.zeros(5) for _ in range(c.latency))
        self.forced_open = 0
        return self.observe()

    def step(self, action: np.ndarray) -> WorldState:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if self.cfg.latency > 0:
            self.latency_queue.append(a)
            a = self.latency_queue.popleft()
        c = self.cfg
        q = self.data.qpos[self.g_qadr]
        pos = q[:3]
        tgt = self.target[:3] + a[:3] * c.max_step
        tgt = np.clip(tgt, pos - c.lead, pos + c.lead)
        self.target[:3] = np.clip(tgt, WORKSPACE_LO, WORKSPACE_HI)
        yaw_t = self.target[3] + a[3] * c.max_yaw_step
        self.target[3] = float(np.clip(yaw_t, q[3] - c.lead_yaw, q[3] + c.lead_yaw))
        grip = a[4]
        if self.forced_open > 0:
            grip = -1.0
            self.forced_open -= 1
        self.finger_target = MAX_FINGER * (1.0 - float(np.clip(grip, -1, 1))) / 2.0
        self._apply_ctrl()
        n = self.sim_cfg.n_substeps
        mujoco.mj_step(self.model, self.data, nstep=n)
        self.sim_steps += n
        self.t += 1
        return self.observe()

    def _apply_ctrl(self) -> None:
        d = self.data
        d.ctrl[self.act_id[:4]] = self.target
        d.ctrl[self.act_id[4]] = self.finger_target

    # ---------------------------------------------------------------- state
    def held_object(self) -> int:
        """Object touched by both fingers (privileged contact check), else -1."""
        d = self.data
        left: set[int] = set()
        right: set[int] = set()
        fl, fr = self.finger_geom
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = int(con.geom1), int(con.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if b in self._geom_to_obj:
                    if a == fl:
                        left.add(self._geom_to_obj[b])
                    elif a == fr:
                        right.add(self._geom_to_obj[b])
        both = left & right
        if not both:
            return -1
        if len(both) == 1:
            return next(iter(both))
        tcp = self.data.qpos[self.g_qadr[:3]]
        return min(both, key=lambda i: np.linalg.norm(self._obj_pos(i) - tcp))

    def _obj_pos(self, i: int) -> np.ndarray:
        a = self.obj_qadr[i]
        return self.data.qpos[a:a + 3].copy()

    def true_state(self) -> WorldState:
        d = self.data
        pos = np.stack([d.qpos[a:a + 3] for a in self.obj_qadr]) if self.n_obj else np.zeros((0, 3))
        quat = np.stack([d.qpos[a + 3:a + 7] for a in self.obj_qadr]) if self.n_obj else np.zeros((0, 4))
        vel = np.stack([d.qvel[v:v + 3] for v in self.obj_vadr]) if self.n_obj else np.zeros((0, 3))
        q = d.qpos[self.g_qadr]
        qd = d.qvel[self.g_vadr]
        return WorldState(
            obj_pos=pos.copy(), obj_quat=quat.copy(), grip_pos=q[:3].copy(), grip_yaw=float(q[3]),
            grip_open=float(q[4] + q[5]), grip_open_vel=float(qd[4] + qd[5]),
            held=self.held_object(), t=self.t, noisy=False, obj_vel=vel.copy(),
        )

    def observe(self) -> WorldState:
        """Perceived state: biased + jittered object poses, slightly noisy proprioception.

        ``held`` is left at -1; perception-side estimates are produced by the verifier.
        """
        s = self.true_state()
        c = self.cfg
        n = self.n_obj
        jit = np.concatenate([self.rng.normal(0, c.jitter_xy, (n, 2)), self.rng.normal(0, c.jitter_z, (n, 1))], axis=1)
        obj_pos = s.obj_pos + self.obs_bias + jit
        obj_quat = s.obj_quat.copy()
        if n:
            dyaw = self.rng.normal(0, c.jitter_yaw, n)
            for i in range(n):
                obj_quat[i] = _yaw_rotate(obj_quat[i], dyaw[i])
        return WorldState(
            obj_pos=obj_pos, obj_quat=obj_quat,
            grip_pos=s.grip_pos + self.rng.normal(0, c.proprio_jitter, 3), grip_yaw=s.grip_yaw,
            grip_open=s.grip_open, grip_open_vel=s.grip_open_vel, held=-1, t=s.t, noisy=True,
        )

    def obj_speed(self) -> np.ndarray:
        d = self.data
        return np.array([np.linalg.norm(d.qvel[v:v + 3]) for v in self.obj_vadr])

    # ---------------------------------------------------------------- perturbations
    def force_release(self, n_steps: int) -> None:
        """Slip: fingers open for ``n_steps`` regardless of the commanded gripper action."""
        self.forced_open = max(self.forced_open, int(n_steps))

    def free_spot(self, i: int, rng: np.random.Generator, dmin: float, dmax: float, tries: int = 60) -> np.ndarray | None:
        """A table location near object i that overlaps no object, bin, plate or slot in use."""
        d = self.data
        here = d.qpos[self.obj_qadr[i]:self.obj_qadr[i] + 2].copy()
        r_i = self.geom.radius[i]
        others = [(d.qpos[self.obj_qadr[j]:self.obj_qadr[j] + 2], self.geom.radius[j]) for j in range(self.n_obj) if j != i]
        blocks = []
        for f in self.scene.fixtures:
            if f.kind == "bin":
                blocks.append((np.array(f.pos[:2]), max(f.size[0], f.size[1]) * 1.42 + 0.01))
            elif f.kind == "plate":
                blocks.append((np.array(f.pos[:2]), max(f.size[0], f.size[1]) * 1.42))
        for _ in range(tries):
            ang = rng.uniform(0, 2 * np.pi)
            p = here + rng.uniform(dmin, dmax) * np.array([np.cos(ang), np.sin(ang)])
            if np.any(p < WORKSPACE_LO[:2] + 0.06) or np.any(p > WORKSPACE_HI[:2] - 0.15):
                continue
            if any(np.linalg.norm(p - q) < r_i + rq + 0.015 for q, rq in others):
                continue
            if any(np.linalg.norm(p - q) < rb + r_i for q, rb in blocks):
                continue
            return p
        return None

    def displace_object(self, i: int, dxy: np.ndarray, dyaw: float = 0.0, settle: int = 50,
                        to_xy: np.ndarray | None = None) -> None:
        """Move object i to rest on the table at ``to_xy`` (or shifted by ``dxy``), e.g. a knock or a nudge."""
        d = self.data
        a = self.obj_qadr[i]
        if to_xy is not None:
            d.qpos[a:a + 2] = np.asarray(to_xy)
        else:
            d.qpos[a:a + 2] += np.asarray(dxy)
        d.qpos[a + 3:a + 7] = _yaw_rotate(_upright(d.qpos[a + 3:a + 7].copy()), dyaw)
        d.qpos[a + 2] = self.geom.half_z_extent(i, d.qpos[a + 3:a + 7]) + 0.002
        d.qvel[self.obj_vadr[i]:self.obj_vadr[i] + 6] = 0.0
        mujoco.mj_forward(self.model, d)
        if settle:
            mujoco.mj_step(self.model, d, nstep=settle)
            self.sim_steps += settle

    def nudge_gripper(self, dpos: np.ndarray) -> None:
        """Trajectory perturbation: displace the end effector (ARSTAG Sec. III-B)."""
        d = self.data
        new = np.clip(d.qpos[self.g_qadr[:3]] + np.asarray(dpos), WORKSPACE_LO, WORKSPACE_HI)
        d.qpos[self.g_qadr[:3]] = new
        d.qvel[self.g_vadr[:3]] = 0.0
        self.target[:3] = new
        self._apply_ctrl()
        mujoco.mj_forward(self.model, d)

    # ---------------------------------------------------------------- snapshots
    def get_snapshot(self) -> Snapshot:
        st = np.empty(self._state_size)
        mujoco.mj_getState(self.model, self.data, st, self._state_spec)
        return Snapshot(
            mj_state=st, target=self.target.copy(), finger_target=float(self.finger_target), t=self.t,
            obs_bias=self.obs_bias.copy(), latency_queue=[q.copy() for q in self.latency_queue],
            forced_open=self.forced_open, rng_state=copy.deepcopy(self.rng.bit_generator.state),
        )

    def set_snapshot(self, snap: Snapshot, reseed: int | None = None) -> WorldState:
        mujoco.mj_setState(self.model, self.data, snap.mj_state, self._state_spec)
        mujoco.mj_forward(self.model, self.data)
        self.target = snap.target.copy()
        self.finger_target = snap.finger_target
        self.t = snap.t
        self.obs_bias = snap.obs_bias.copy()
        self.latency_queue = collections.deque(q.copy() for q in snap.latency_queue)
        self.forced_open = snap.forced_open
        if reseed is not None:
            self.rng = np.random.default_rng(reseed)
            return self.observe()
        self.rng.bit_generator.state = copy.deepcopy(snap.rng_state)
        obs = self.observe()
        # restoring must not advance the noise stream, so replays stay bit-identical
        self.rng.bit_generator.state = copy.deepcopy(snap.rng_state)
        return obs

    # ---------------------------------------------------------------- rendering
    def render(self, camera: str = "front", width: int = 256, height: int = 256) -> np.ndarray:
        if not hasattr(self, "_renderer") or self._renderer.width != width:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()


def _upright(q: np.ndarray) -> np.ndarray:
    """Keep only the heading of q (the object is set down flat)."""
    yaw = quat_yaw(q)
    return yaw_quat(yaw)


def _yaw_rotate(q: np.ndarray, dyaw: float) -> np.ndarray:
    """Pre-multiply quaternion q by a rotation of dyaw about world z."""
    if dyaw == 0.0:
        return q
    c, s = np.cos(dyaw / 2), np.sin(dyaw / 2)
    w, x, y, z = q
    out = np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])
    return out / np.linalg.norm(out)


def obj_yaw(s: WorldState, i: int) -> float:
    return quat_yaw(s.obj_quat[i])
