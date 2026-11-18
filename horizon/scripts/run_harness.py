"""Run the artifact-backed E0/E1 learning experiment; no model API calls."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from horizon.harness.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_experiment(config, args.out, args.project_root, args.workers)


if __name__ == "__main__":
    main()
