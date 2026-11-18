"""Run the R1 matrix (cause x operation); no model API calls."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from horizon.harness.r1_matrix import run_r1_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_r1_matrix(config, args.out, args.project_root, args.workers)


if __name__ == "__main__":
    main()
