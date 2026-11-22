"""Real-robot track: PoseSource/Robot protocols, mock/CSV replay, deployment and twin calibration."""
from .apriltag import AprilTagPoseSource, StaticPoseSource, TagBoardConfig
from .calibrate import median_support, placements_from_log, update_recon_bias
from .deploy import rows_to_episode, run_episodes
from .interfaces import PoseSample, PoseSource, Robot
from .mock import ScriptedRobot, load_jsonl_robot, make_mock_robot, make_mock_robot_for_task
from .piper import PiperArm, PiperConfig, PiperRobot

__all__ = [
    "PoseSample", "PoseSource", "Robot", "ScriptedRobot", "load_jsonl_robot", "make_mock_robot",
    "make_mock_robot_for_task", "run_episodes", "rows_to_episode", "median_support",
    "placements_from_log", "update_recon_bias",
    # hardware backends (AgileX Piper): imported without touching CAN or a camera
    "PiperArm", "PiperConfig", "PiperRobot", "AprilTagPoseSource", "StaticPoseSource",
    "TagBoardConfig",
]
