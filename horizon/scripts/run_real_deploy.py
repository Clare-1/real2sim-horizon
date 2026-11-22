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

from horizon.real import make_mock_robot, load_jsonl_robot, run_episodes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=("mock", "csv"), default="mock")
    ap.add_argument("--csv", default="")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--out", default="results/real_deploy_20261122.jsonl")
    ap.add_argument("--policy", default="", help="optional weights path; only the input dim is read")
    args = ap.parse_args()

    if args.robot == "csv":
        if not args.csv:
            ap.error("--robot csv needs --csv")
        robot = load_jsonl_robot(args.csv)
    else:
        robot = make_mock_robot()

    if args.policy:
        from horizon.policy.model import NumpyMLP
        from horizon.rsi.loop import load_params
        params = load_params(args.policy)
        print(f"policy input dim = {params['x_mu'].shape[0]} (TaskFeaturizer.hier output)")
        NumpyMLP(params)  # validates the checkpoint shape

    # Placeholder policy: zero action. Replace with mlp.act(feat.hier(sample.to_world_state(), k)).
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
