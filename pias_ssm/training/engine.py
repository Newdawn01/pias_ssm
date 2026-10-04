"""Reproducible episodic training and checkpoint management."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.nn import functional as F

from ..config import ExperimentConfig, REVISION_ROOT
from ..data.dataset import BearingDataset
from ..data.noise import add_noise
from ..models.model import PIASSSM, count_parameters
from ..utils import atomic_write_json, ensure_directory, set_seed
from .evaluator import evaluate_model


EXPECTED_SAMPLING_RATES = {"CWRU": 12_000.0, "Ottawa": 2_000.0}


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def build_model(config: ExperimentConfig, dataset: BearingDataset) -> torch.nn.Module:
    expected_rate = EXPECTED_SAMPLING_RATES.get(config.dataset)
    if expected_rate is not None and dataset.sampling_rate_hz != expected_rate:
        raise ValueError(
            f"{config.dataset} must use target_fs={expected_rate:g} Hz, "
            f"but the data file reports {dataset.sampling_rate_hz:g} Hz"
        )
    return PIASSSM(
        config=config.model,
        dataset_name=config.dataset,
        sampling_rate_hz=dataset.sampling_rate_hz,
        sequence_length=dataset.sequence_length,
        input_channels=int(dataset.samples.shape[1]),
    )


def curriculum_snr_range(
    epoch: int,
    total_epochs: int,
    low: float,
    high: float,
    fraction: float,
) -> tuple[float, float]:
    transition_epochs = max(1.0, total_epochs * fraction)
    progress = min(epoch / transition_epochs, 1.0)
    current_low = 15.0 + progress * (low - 15.0)
    current_high = 20.0 + progress * (high - 20.0)
    return current_low, current_high


def _noisy_episode(
    dataset: BearingDataset,
    config: ExperimentConfig,
    epoch: int,
    episode_index: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    episode_seed = seed * 1_000_000 + epoch * config.training.episodes_per_epoch + episode_index
    episode = dataset.sample_episode(
        "train",
        config.episode.n_way,
        config.episode.n_support,
        config.episode.n_query,
        episode_seed,
    )
    snr_low, snr_high = curriculum_snr_range(
        epoch,
        config.training.epochs,
        config.training.train_snr_low,
        config.training.train_snr_high,
        config.training.curriculum_fraction,
    )
    generator = torch.Generator().manual_seed(episode_seed + 31)
    query_snrs = torch.empty(episode.query_x.shape[0]).uniform_(
        snr_low,
        snr_high,
        generator=generator,
    )
    query_x = add_noise(
        episode.query_x,
        query_snrs,
        config.noise_type,
        dataset.sampling_rate_hz,
        episode.query_prior_hz,
        generator,
    )
    support_x = episode.support_x
    if config.training.noise_support_during_training:
        support_snrs = torch.empty(episode.support_x.shape[0]).uniform_(
            snr_low,
            snr_high,
            generator=generator,
        )
        support_x = add_noise(
            support_x,
            support_snrs,
            config.noise_type,
            dataset.sampling_rate_hz,
            episode.support_prior_hz,
            generator,
        )
    return (
        support_x,
        episode.support_y,
        episode.support_prior_hz,
        query_x,
        episode.query_y,
        episode.query_prior_hz,
    )


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    dataset: BearingDataset,
    epoch: int,
    path: Path,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "dataset_summary": dataset.summary(),
            "epoch": epoch,
            "parameter_count": count_parameters(model),
        },
        path,
    )


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["model_state"] if "model_state" in payload else payload
    model.load_state_dict(state)
    return payload


def run_training(
    config: ExperimentConfig,
    seed: int,
    output_root: str | Path | None = None,
) -> Path:
    """Train one seed, save best/last checkpoints, and run the full SNR sweep."""

    set_seed(seed)
    dataset = BearingDataset(
        config.resolved_data_path,
        config.episode.split_ratio,
        config.episode.split_strategy,
        seed,
    )
    device = resolve_device(config.training.device)
    model = build_model(config, dataset).to(device)
    optimizer_class = (
        torch.optim.Adam
        if config.training.optimizer == "adam"
        else torch.optim.AdamW
    )
    optimizer = optimizer_class(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    if config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.scheduler_step_size,
            gamma=config.training.scheduler_gamma,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
        )
    root = Path(output_root) if output_root else REVISION_ROOT / "outputs"
    run_directory = ensure_directory(
        root / config.dataset / config.experiment_name / f"seed_{seed}"
    )
    config.save(run_directory / "config.json")
    atomic_write_json(dataset.summary(), run_directory / "dataset_summary.json")

    best_monitor_accuracy = -1.0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(config.training.epochs):
        model.train()
        losses: list[float] = []
        accuracies: list[float] = []
        for episode_index in range(config.training.episodes_per_epoch):
            support_x, support_y, support_prior, query_x, query_y, query_prior = _noisy_episode(
                dataset,
                config,
                epoch,
                episode_index,
                seed,
            )
            logits = model(
                support_x.to(device),
                support_y.to(device),
                query_x.to(device),
                support_prior.to(device),
                query_prior.to(device),
            )
            loss = F.cross_entropy(logits, query_y.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.training.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            accuracies.append(float((logits.argmax(1).cpu() == query_y).float().mean()))
        scheduler.step()

        monitor = evaluate_model(
            model,
            dataset,
            config,
            seed=seed + 10_000 + epoch,
            snr_db=config.training.train_snr_low,
            noise_type=config.noise_type,
            episodes=config.training.validation_episodes,
            device=device,
            split="train",
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": sum(losses) / len(losses),
            "train_accuracy": sum(accuracies) / len(accuracies),
            "train_pool_monitor_accuracy": float(monitor.summary["mean"]),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        if record["train_pool_monitor_accuracy"] > best_monitor_accuracy:
            best_monitor_accuracy = float(record["train_pool_monitor_accuracy"])
            _save_checkpoint(
                model,
                optimizer,
                config,
                dataset,
                epoch + 1,
                run_directory / "checkpoint_best_train_monitor.pt",
            )
        atomic_write_json({"epochs": history}, run_directory / "history.json")

    _save_checkpoint(
        model,
        optimizer,
        config,
        dataset,
        config.training.epochs,
        run_directory / "checkpoint_last.pt",
    )
    if config.training.checkpoint_selection == "best_train_monitor":
        load_checkpoint(
            model,
            run_directory / "checkpoint_best_train_monitor.pt",
            device,
        )
    evaluations = {}
    for snr in config.evaluation_snrs:
        result = evaluate_model(
            model,
            dataset,
            config,
            seed=seed + 100_000,
            snr_db=float(snr),
            noise_type=config.noise_type,
            device=device,
        )
        evaluations[f"{snr:g}dB"] = result.to_dict()
    atomic_write_json(
        {
            "seed": seed,
            "parameter_count": count_parameters(model),
            "elapsed_seconds": time.perf_counter() - started,
            "evaluations": evaluations,
        },
        run_directory / "evaluation.json",
    )
    return run_directory
