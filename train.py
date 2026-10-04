#!/usr/bin/env python3
"""Training entry point for PIAS-SSM.

Usage:
    python train.py --config configs/cwru.json
    python train.py --config configs/ottawa.json --seed 42 --device cuda
"""

from __future__ import annotations

import argparse

from pias_ssm.config import ExperimentConfig
from pias_ssm.training.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-path")
    parser.add_argument("--device")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.data_path:
        config.data_path = args.data_path
    if args.device:
        config.training.device = args.device
    seeds = [args.seed] if args.seed is not None else config.training.seeds
    for seed in seeds:
        print(run_training(config, seed, args.output_root))


if __name__ == "__main__":
    main()
