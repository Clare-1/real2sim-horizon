"""Run a real-robot deployment loop with the mock or a recorded CSV/JSONL replay.

The hardware adapter is intentionally out of scope: implement ``Robot``/``PoseSource`` in your
driver, then replace ``policy_fn`` with the state-policy wrapper. This script proves the loop,
logging and success handling before any robot is connected.

    python scripts/run_real_deploy.py --robot mock --episodes 2 --out /tmp/deploy.jsonl
    python scripts/run_real_deploy.py --robot csv --csv logs/episode.jsonl --episodes 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon.real import (load_jsonl_robot, make_mock_robot, make_mock_robot_for_task,
                          run_episodes)  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=("mock", "csv"), default="mock")
    ap.add_argument("--csv", default="")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", default="results/real_deploy_20261122.jsonl")
    ap.add_argument("--task", default="", help="task payload JSON (enables the learned controller)")
    ap.add_argument("--policy", default="", help="policy weights .npz")
    args = ap.parse_args()

    if args.robot == "csv":
        if not args.csv:
            ap.error("--robot csv needs --csv")
        robot = load_jsonl_robot(args.csv)
    else:
        robot = make_mock_robot()

    if args.task:
        if not args.policy:
            ap.error("--task needs --policy")
        from horizon.monitor.verifier import VerifierConfig
        from horizon.policy.controllers import HierarchicalController
        from horizon.rsi.loop import load_params
        from horizon.tasks.serialization import task_from_payload

        task = task_from_payload(json.loads(Path(args.task).read_text(encoding="utf-8"))).controller_view()
        params = load_params(args.policy)
        controller = HierarchicalController(params, VerifierConfig(tol_scale=1.0))
        controller.reset(task)
        if args.robot == "mock":
            robot = make_mock_robot_for_task(task, frames=args.max_steps)

        def policy_fn(sample):
            action, _ = controller.act(sample.to_world_state(), sample.to_world_state())
            return action

        print(f"task={task.key}  objects={len(task.scene.objects)}  "
              f"policy_dim={params['x_mu'].shape[0]}")
    else:
        if args.policy:
            from horizon.policy.model import NumpyMLP
            from horizon.rsi.loop import load_params
            params = load_params(args.policy)
            print(f"policy input dim = {params['x_mu'].shape[0]} (TaskFeaturizer.hier output)")
            NumpyMLP(params)  # validates the checkpoint shape

        # Placeholder policy when no task payload is given.
        def policy_fn(sample):
            return [0.0, 0.0, 0.0, 0.0, 1.0]

    rows = run_episodes(robot, policy_fn, args.episodes, args.max_steps, args.out)
    by_ep: dict[int, bool] = {}
    for r in rows:
        by_ep[r["episode"]] = by_ep.get(r["episode"], False) or r["success"]
    print(f"episodes={args.episodes}  steps={len(rows)}  success={sum(by_ep.values())}/{len(by_ep)}")
    print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
