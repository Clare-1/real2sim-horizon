"""Evaluate a LIBERO image-BC checkpoint (mock env by default; real LIBERO needs the package).

    python scripts/eval_libero.py --policy weights/libero/bc.pt --env mock --episodes 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon.libero import ImageBC, MockLiberoEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--env", choices=("mock", "libero"), default="mock")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    if args.env == "libero":
        try:
            import libero  # noqa: F401
        except ImportError as exc:
            raise SystemExit("real LIBERO env is not installed; use --env mock for the smoke test") from exc
        raise SystemExit("wire the official LIBERO env here: reset(task_id) / step(action)")

    env = MockLiberoEnv(horizon=args.horizon)
    model = ImageBC.load(args.policy)
    wins = 0
    for ep in range(args.episodes):
        obs = env.reset(args.task_id)
        done = success = False
        for _ in range(args.horizon):
            obs, _, done, info = env.step(model.act(obs))
            if info.get("task_success"):
                success = True
                break
            if done:
                break
        wins += int(success)
        print(f"episode {ep}: success={success}")
    print(f"SR={wins}/{args.episodes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
