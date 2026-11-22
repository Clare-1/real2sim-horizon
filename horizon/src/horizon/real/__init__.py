"""Real-robot track: PoseSource/Robot protocols, mock/CSV replay, deployment and twin calibration."""
from .calibrate import median_support, placements_from_log, update_recon_bias
from .deploy import run_episodes
from .interfaces import PoseSample, PoseSource, Robot
from .mock import ScriptedRobot, load_jsonl_robot, make_mock_robot

__all__ = [
    "PoseSample", "PoseSource", "Robot", "ScriptedRobot", "load_jsonl_robot", "make_mock_robot",
    "run_episodes", "median_support", "placements_from_log", "update_recon_bias",
]
