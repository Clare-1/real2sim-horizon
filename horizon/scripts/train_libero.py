"""Train the small LIBERO image-BC baseline on HDF5 demonstrations.

    python scripts/train_libero.py --dataset /path/libero_10.hdf5 --out weights/libero/bc.pt

The official LIBERO release is required; no weights are shipped because training data are
external (disk/checkout blocked on the current machine).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from horizon.libero import ImageBC, concat_demos, load_hdf5_demos  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="weights/libero/bc.pt")
    ap.add_argument("--image-key", default="obs/agentview_rgb")
    ap.add_argument("--state-key", default="obs/ee_states")
    ap.add_argument("--action-key", default="actions")
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    demos = load_hdf5_demos(args.dataset, args.image_key, args.state_key, args.action_key,
                            args.max_demos)
    images, state, actions = concat_demos(demos)
    print(f"demos={len(demos)}  frames={len(actions)}  image={images.shape}  "
          f"state={state.shape}  action={actions.shape}")
    model = ImageBC(action_dim=actions.shape[1])
    losses = model.fit(images, actions, epochs=args.epochs, batch=args.batch)
    model.save(args.out)
    print("losses:", [round(x, 5) for x in losses])
    print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
