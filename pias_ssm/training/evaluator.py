"""Few-shot evaluation with raw episode outcomes and confidence intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from ..config import ExperimentConfig
from ..data.dataset import BearingDataset
from ..data.noise import add_noise
from .statistics import summarize


@dataclass(slots=True)
class EvaluationResult:
    dataset: str
    noise_type: str
    snr_db: float | None
    seed: int
    episodes: int
    episode_accuracies: list[float]
    summary: dict[str, float | int]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _device_from_name(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset: BearingDataset,
    config: ExperimentConfig,
    seed: int,
    snr_db: float | None,
    noise_type: str | None = None,
    episodes: int | None = None,
    device: str | torch.device | None = None,
    split: str = "test",
) -> EvaluationResult:
    resolved_device = (
        _device_from_name(config.training.device)
        if device is None
        else torch.device(device)
    )
    model.to(resolved_device).eval()
    noise_name = noise_type or config.noise_type
    episode_count = episodes or config.training.test_episodes
    accuracies: list[float] = []
    confusion = torch.zeros(
        config.episode.n_way,
        config.episode.n_way,
        dtype=torch.long,
    )

    for episode_index in range(episode_count):
        episode_seed = seed * 1_000_000 + episode_index
        episode = dataset.sample_episode(
            split,
            config.episode.n_way,
            config.episode.n_support,
            config.episode.n_query,
            episode_seed,
        )
        generator = torch.Generator().manual_seed(episode_seed + 97)
        support_x = episode.support_x
        if config.training.noise_support_during_evaluation:
            support_x = add_noise(
                support_x,
                snr_db,
                noise_name,
                dataset.sampling_rate_hz,
                episode.support_prior_hz,
                generator,
            )
        query_x = add_noise(
            episode.query_x,
            snr_db,
            noise_name,
            dataset.sampling_rate_hz,
            episode.query_prior_hz,
            generator,
        )
        logits = model(
            support_x.to(resolved_device),
            episode.support_y.to(resolved_device),
            query_x.to(resolved_device),
            episode.support_prior_hz.to(resolved_device),
            episode.query_prior_hz.to(resolved_device),
        )
        predictions = logits.argmax(dim=1).cpu()
        accuracy = (predictions == episode.query_y).float().mean().item()
        accuracies.append(accuracy)
        flattened = episode.query_y * config.episode.n_way + predictions
        confusion += torch.bincount(
            flattened,
            minlength=config.episode.n_way**2,
        ).reshape(config.episode.n_way, config.episode.n_way)

    return EvaluationResult(
        dataset=config.dataset,
        noise_type=noise_name,
        snr_db=snr_db,
        seed=seed,
        episodes=episode_count,
        episode_accuracies=accuracies,
        summary=summarize(accuracies, seed=seed),
        confusion_matrix=confusion.tolist(),
    )
