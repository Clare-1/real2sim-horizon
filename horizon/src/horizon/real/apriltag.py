"""AprilTag object-pose source for the real-robot track.

The advisor's plan asks for tag-based object poses rather than real-world visual sim2real, so the
state policy trained in the twin can run unchanged. This module converts a tag detection stream into
the :class:`~horizon.real.interfaces.PoseSample` the deployment loop expects.

Two backends:

* :class:`AprilTagPoseSource` — needs a camera frame and the ``pupil_apriltags`` package;
  ``frame_fn`` is injected so the same code works with cv2, a ROS subscriber or a video file.
* :class:`StaticPoseSource` — a fixed layout, for plumbing tests and for the "measure once" setup.

Tag convention: ``tag_ids`` maps a tag id to an object index in the task scene, so a missing tag
becomes ``-1`` in the mask rather than silently shifting indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interfaces import PoseSample


@dataclass
class StaticPoseSource:
    """Fixed object layout (metres / wxyz), optionally with a scripted gripper pose."""

    obj_pos: np.ndarray
    obj_quat: np.ndarray
    grip_pos: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.0, 0.20]))
    grip_yaw: float = 0.0
    grip_open: float = 0.07
    held: int = -1
    success: bool = False
    _t: int = 0

    def read(self) -> PoseSample:
        self._t += 1
        return PoseSample(t=self._t, obj_pos=np.asarray(self.obj_pos, float),
                          obj_quat=np.asarray(self.obj_quat, float),
                          grip_pos=np.asarray(self.grip_pos, float), grip_yaw=self.grip_yaw,
                          grip_open=self.grip_open, held=self.held, success=self.success)


@dataclass
class TagBoardConfig:
    """Camera and tag geometry: measure these once per cell and keep them in the config."""

    tag_size_m: float = 0.03
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    family: str = "tag36h11"
    # camera -> robot base transform: 3x3 rotation and 3-vector translation (metres)
    cam_to_base_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    cam_to_base_t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # tag id -> object index in the task scene
    tag_ids: dict = field(default_factory=dict)
    yaw_from_tag: bool = True


CLASS_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


class AprilTagPoseSource:
    """Detect tags in a frame and return object poses in the robot base frame.

    ``frame_fn`` must return a grayscale ``np.ndarray`` (H, W) or (H, W, 3). Detections are matched to
    object indices through ``cfg.tag_ids``; every object keeps its previous pose when its tag is not
    visible, which makes a dropped detection a stale reading rather than a jump to the origin.
    """

    def __init__(self, frame_fn, cfg: TagBoardConfig, num_objects: int):
        self.frame_fn = frame_fn
        self.cfg = cfg
        self.num_objects = int(num_objects)
        self._detector = None
        self._pos = np.zeros((self.num_objects, 3))
        self._quat = np.tile(CLASS_QUAT_WXYZ, (self.num_objects, 1))
        self._t = 0

    def _detect(self, frame: np.ndarray):
        if self._detector is None:
            try:
                from pupil_apriltags import Detector
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "pupil_apriltags is required for tag-based object poses: "
                    "`pip install pupil-apriltags`") from exc
            self._detector = Detector(families=self.cfg.family)
        if frame.ndim == 3:
            frame = frame[..., :3].mean(axis=2)
        return self._detector.detect(
            np.ascontiguousarray(frame.astype(np.uint8)),
            estimate_tag_pose=True, camera_params=(self.cfg.fx, self.cfg.fy, self.cfg.cx, self.cfg.cy),
            tag_size=float(self.cfg.tag_size_m))

    def read(self) -> PoseSample:
        detections = self._detect(self.frame_fn())
        R_cb, t_cb = np.asarray(self.cfg.cam_to_base_R, float), np.asarray(self.cfg.cam_to_base_t, float)
        for det in detections:
            idx = self.cfg.tag_ids.get(int(det.tag_id))
            if idx is None or idx >= self.num_objects:
                continue
            p = R_cb @ np.asarray(det.pose_t, float).reshape(3) + t_cb
            self._pos[idx] = p
            if self.cfg.yaw_from_tag:
                # tag +z faces the camera; yaw in the base frame from the tag normal
                n = R_cb @ np.asarray(det.pose_R, float)[:, 2]
                yaw = float(np.arctan2(n[1], n[0]))
                self._quat[idx] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self._t += 1
        return PoseSample(t=self._t, obj_pos=self._pos.copy(), obj_quat=self._quat.copy(),
                          grip_pos=np.zeros(3), grip_yaw=0.0, grip_open=0.0)
