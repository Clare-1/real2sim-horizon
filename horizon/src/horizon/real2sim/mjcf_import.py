"""Import a reconstructed MuJoCo scene (e.g. Agentic RSR Stage 3, Real2Gym) as a SceneSpec.

Open real2sim pipelines export MJCF with one free-jointed body per reconstructed object.
We keep only the geometry (robot-referencing equality/actuator/contact sections are
stripped, as Agentic RSR's ``prepare_sim_scene.py`` does before inserting a robot),
compile, settle under gravity, and approximate each movable body by the box or cylinder
bounding its collision geoms - the primitive level our skill library, predicates and
grasp planner operate on. Static bodies whose names mention a container keyword become
fixtures. The scene is re-expressed in our frame: support surface at z = 0 and, if a
robot mount body is given, the mount at (-0.55, 0).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from ..sim.scene import FixtureSpec, ObjectSpec, SceneSpec
from ..sim.state import quat_yaw

BIN_WORDS = ("bin", "basket", "box", "container", "bowl", "tray")
PLATE_WORDS = ("plate", "dish", "board")
STRIP = ("equality", "actuator", "contact", "sensor", "tendon", "keyframe", "custom")
OUR_MOUNT = np.array([-0.55, 0.0])


def _geometry_only_xml(path: Path) -> str:
    root = ET.parse(path).getroot()
    for tag in STRIP:
        for el in root.findall(tag):
            root.remove(el)
    base = path.parent.resolve()
    comp = root.find("compiler")
    if comp is None:
        comp = ET.Element("compiler")
        root.insert(0, comp)
    for attr in ("meshdir", "texturedir", "assetdir"):
        v = comp.get(attr)
        if v is not None and not Path(v).is_absolute():
            comp.set(attr, str((base / v).resolve()))
    if comp.get("meshdir") is None and comp.get("assetdir") is None:
        comp.set("assetdir", str(base))
    return ET.tostring(root, encoding="unicode")


def _body_aabb(m: mujoco.MjModel, d: mujoco.MjData, body: int) -> tuple[np.ndarray, np.ndarray] | None:
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    found = False
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or (m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0):
            continue
        found = True
        c = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        t = m.geom_type[g]
        s = m.geom_size[g]
        if t in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            ext = np.abs(R) @ s
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            ext = np.abs(R) @ np.array([s[0], s[0], s[1]])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ext = np.abs(R) @ np.array([s[0], s[0], s[1] + s[0]])
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = np.full(3, s[0])
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            verts = m.mesh_vert[v0:v0 + nv] @ R.T + c
            lo, hi = np.minimum(lo, verts.min(0)), np.maximum(hi, verts.max(0))
            continue
        else:
            ext = np.full(3, m.geom_rbound[g])
        lo, hi = np.minimum(lo, c - ext), np.maximum(hi, c + ext)
    return (lo, hi) if found else None


def import_mjcf(path: str | Path, scene_id: str | None = None, settle_steps: int = 500,
                robot_mount: str | None = None) -> SceneSpec:
    path = Path(path)
    m = mujoco.MjModel.from_xml_string(_geometry_only_xml(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mujoco.mj_step(m, d, nstep=settle_steps)
    free_bodies = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    boxes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for b in range(1, m.nbody):
        bb = _body_aabb(m, d, b)
        if bb is not None and np.isfinite(bb[0]).all():
            boxes[b] = bb
    movable = [b for b in free_bodies if b in boxes]
    if not movable:
        raise ValueError(f"{path}: no free-jointed bodies with collision geometry")
    table_z = min(boxes[b][0][2] for b in movable)
    shift = np.zeros(2)
    if robot_mount is not None:
        shift = OUR_MOUNT - d.xpos[m.body(robot_mount).id][:2]
    objs, fixes = [], []
    for b in sorted(movable):
        lo, hi = boxes[b]
        name = m.body(b).name or f"body{b}"
        size = (hi - lo) / 2
        centre = (hi + lo) / 2
        is_cyl = any(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER for g in range(m.ngeom) if m.geom_bodyid[g] == b)
        shape = "cylinder" if is_cyl else "box"
        sz = (float(max(size[0], size[1])), float(size[2])) if is_cyl else tuple(float(s) for s in size)
        rgba = tuple(float(v) for v in m.geom_rgba[next(g for g in range(m.ngeom) if m.geom_bodyid[g] == b)])
        objs.append(ObjectSpec(
            name=name, shape=shape, size=sz,
            pos=(float(centre[0] + shift[0]), float(centre[1] + shift[1]), float(centre[2] - table_z)),
            yaw=0.0 if is_cyl else float(quat_yaw(d.xquat[b])), rgba=rgba,
            mass=float(m.body_subtreemass[b]) or 0.08, category=name.split("_")[-1], color=name.split("_")[0],
        ))
    for b, (lo, hi) in boxes.items():
        if b in free_bodies:
            continue
        name = (m.body(b).name or "").lower()
        size = (hi - lo) / 2
        cx, cy = (hi[:2] + lo[:2]) / 2 + shift
        if any(w in name for w in PLATE_WORDS):
            fixes.append(FixtureSpec(name=name, kind="plate", pos=(float(cx), float(cy), 0.0),
                                     size=(float(size[0]), float(size[1]), float(size[2]))))
        elif any(w in name for w in BIN_WORDS):
            fixes.append(FixtureSpec(name=name, kind="bin", pos=(float(cx), float(cy), 0.0),
                                     size=(float(size[0]) - 0.006, float(size[1]) - 0.006, float(hi[2] - table_z)),
                                     color=name.split("_")[0]))
    return SceneSpec(scene_id=scene_id or path.stem, objects=objs, fixtures=fixes, provenance=f"mjcf:{path.name}")
