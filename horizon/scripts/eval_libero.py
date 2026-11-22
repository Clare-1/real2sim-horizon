"""Evaluate a LIBERO image-BC checkpoint (mock env by default; real LIBERO needs the package).

    python scripts/eval_libero.py --policy weights/libero/bc.pt --env mock --episodes 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon.libero import ImageBC, LiberoEnvAdapter, MockLiberoEnv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--env", choices=("mock", "libero"), default="mock")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--bddl", default="", help="BDDL file for --env libero")
    ap.add_argument("--init-state", default="", help="optional .npz with an init_state array")
    ap.add_argument("--camera", type=int, default=128)
    ap.add_argument("--settle", type=int, default=10)
    args = ap.parse_args()

    if args.env == "libero":
        if not args.bddl:
            raise SystemExit("--env libero needs --bddl (official LIBERO BDDL file)")
        init_state = None
        if args.init_state:
            init_state = np.load(args.init_state)["init_state"]
        env = LiberoEnvAdapter(args.bddl, args.camera, args.camera,
                               init_state=init_state, settle_steps=args.settle)
    else:
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
