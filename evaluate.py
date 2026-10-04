#!/usr/bin/env python3
"""Evaluation entry point for PIAS-SSM checkpoints.

Usage:
    python evaluate.py --config configs/cwru.json \
        --checkpoint outputs/CWRU/main/seed_42/checkpoint_last.pt --snr -5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pias_ssm.config import ExperimentConfig
from pias_ssm.data.dataset import BearingDataset
from pias_ssm.training.engine import build_model, load_checkpoint, resolve_device
from pias_ssm.training.evaluator import evaluate_model
from pias_ssm.utils import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--snr", type=float, default=-5.0)
    parser.add_argument("--noise", default="gaussian")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    if args.device:
        config.training.device = args.device
    seed = args.seed if args.seed is not None else config.training.seeds[0]
    dataset = BearingDataset(
        config.resolved_data_path,
        config.episode.split_ratio,
        config.episode.split_strategy,
        seed,
    )
    device = resolve_device(config.training.device)
    model = build_model(config, dataset).to(device)
    load_checkpoint(model, args.checkpoint, device)
    result = evaluate_model(
        model,
        dataset,
        config,
        seed,
        args.snr,
        args.noise,
        args.episodes,
        device,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).parent / "evaluation_custom.json"
    )
    atomic_write_json(result.to_dict(), output)
    print(result.summary)


if __name__ == "__main__":
    main()
