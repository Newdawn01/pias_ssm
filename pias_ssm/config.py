"""Typed, serializable experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REVISION_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ModelConfig:
    """PIAS-SSM architecture and reviewer-controlled hyperparameters."""

    architecture: str = "pias_ssm"
    embedding_dim: int = 128
    # Paper Sec. 3.2.2: uniform wavelet-centre init (not fault-harmonic).
    # "physics" stays available as an ablation but must not be the default, or
    # the R2-9 "centres converge to BPFI/BPFO" check would be circular.
    wavelet_mode: str = "learnable"
    use_speed_channel: bool = False
    use_aso: bool = True
    use_physics_prior: bool = True
    aso_gate: str = "open"
    fundamental_bandwidth: float = 0.02
    harmonic_bandwidth: float = 0.03
    harmonic_weight: float = 0.5
    harmonic_orders: tuple[int, ...] = (2, 3)
    fusion_gate_init: float = 0.5
    freeze_fusion_gate: bool = False
    num_segments: int = 4
    ssm_layers: int = 1
    state_size: int = 64
    prior_scale: float = 1.0
    capacity_adapter_hidden: int = 0

    def validate(self) -> None:
        if self.architecture != "pias_ssm":
            raise ValueError("architecture must be pias_ssm")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.num_segments <= 0:
            raise ValueError("num_segments must be positive")
        if self.embedding_dim % self.num_segments:
            raise ValueError("embedding_dim must be divisible by num_segments")
        if self.wavelet_mode not in {"physics", "learnable", "conv"}:
            raise ValueError("wavelet_mode must be physics, learnable, or conv")
        if self.aso_gate not in {"open", "fixed", "conditional"}:
            raise ValueError("aso_gate must be open, fixed, or conditional")
        for name, value in (
            ("fundamental_bandwidth", self.fundamental_bandwidth),
            ("harmonic_bandwidth", self.harmonic_bandwidth),
        ):
            if not 0.0 < value <= 0.5:
                raise ValueError(f"{name} must be in normalized cycles/sample (0, 0.5]")
        if not 0.0 <= self.fusion_gate_init <= 1.0:
            raise ValueError("fusion_gate_init must be in [0, 1]")
        if self.prior_scale < 0.0:
            raise ValueError("prior_scale must be non-negative")
        if self.ssm_layers < 0 or self.capacity_adapter_hidden < 0:
            raise ValueError("layer counts and adapter width must be non-negative")
        if any(order < 2 for order in self.harmonic_orders):
            raise ValueError("harmonic_orders must contain integers of order two or greater")


@dataclass(slots=True)
class EpisodeConfig:
    """N-way K-shot protocol and leakage-safe split settings."""

    n_way: int = 5
    n_support: int = 5
    n_query: int = 15
    split_ratio: float = 0.7
    split_strategy: str = "file"

    def validate(self) -> None:
        if min(self.n_way, self.n_support, self.n_query) <= 0:
            raise ValueError("n_way, n_support, and n_query must be positive")
        if not 0.0 < self.split_ratio < 1.0:
            raise ValueError("split_ratio must be in (0, 1)")
        if self.split_strategy not in {"file", "sample"}:
            raise ValueError("split_strategy must be file or sample")


@dataclass(slots=True)
class TrainingConfig:
    """Optimization, evaluation, and reproducibility settings."""

    epochs: int = 200
    episodes_per_epoch: int = 100
    validation_episodes: int = 100
    test_episodes: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.5
    checkpoint_selection: str = "last_epoch"
    train_snr_low: float = -5.0
    train_snr_high: float = 20.0
    curriculum_fraction: float = 0.6
    seeds: list[int] = field(default_factory=lambda: [42])
    device: str = "cuda"
    gradient_clip_norm: float | None = None
    noise_support_during_training: bool = True
    noise_support_during_evaluation: bool = True

    def validate(self) -> None:
        if min(self.epochs, self.episodes_per_epoch, self.test_episodes) <= 0:
            raise ValueError("training counts must be positive")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be adam or adamw")
        if self.scheduler not in {"step", "cosine"}:
            raise ValueError("scheduler must be step or cosine")
        if self.scheduler_step_size <= 0:
            raise ValueError("scheduler_step_size must be positive")
        if self.scheduler_gamma <= 0.0:
            raise ValueError("scheduler_gamma must be positive")
        if self.checkpoint_selection not in {"last_epoch", "best_train_monitor"}:
            raise ValueError(
                "checkpoint_selection must be last_epoch or best_train_monitor"
            )
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive or null")
        if self.train_snr_low > self.train_snr_high:
            raise ValueError("train_snr_low cannot exceed train_snr_high")
        if not 0.0 < self.curriculum_fraction <= 1.0:
            raise ValueError("curriculum_fraction must be in (0, 1]")
        if not self.seeds:
            raise ValueError("at least one random seed is required")


@dataclass(slots=True)
class ExperimentConfig:
    """Complete experiment configuration loaded from JSON."""

    dataset: str
    data_path: str
    model: ModelConfig = field(default_factory=ModelConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation_snrs: list[float] = field(
        default_factory=lambda: [5, 3, 0, -3, -5]
    )
    experiment_name: str = "main"
    noise_type: str = "gaussian"

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["model"] = ModelConfig(**payload.get("model", {}))
        payload["episode"] = EpisodeConfig(**payload.get("episode", {}))
        payload["training"] = TrainingConfig(**payload.get("training", {}))
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.dataset not in {"CWRU", "Ottawa"}:
            raise ValueError("dataset must be CWRU or Ottawa")
        self.model.validate()
        self.episode.validate()
        self.training.validate()
        if self.noise_type not in {
            "none",
            "gaussian",
            "pink",
            "brown",
            "impulsive",
            "nonstationary",
            "narrowband",
        }:
            raise ValueError(f"unsupported noise_type: {self.noise_type}")

    @property
    def resolved_data_path(self) -> Path:
        path = Path(self.data_path)
        return path.resolve() if path.is_absolute() else (REVISION_ROOT / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clone(self, **updates: Any) -> "ExperimentConfig":
        payload = self.to_dict()
        payload.update(updates)
        payload["model"] = ModelConfig(**payload["model"])
        payload["episode"] = EpisodeConfig(**payload["episode"])
        payload["training"] = TrainingConfig(**payload["training"])
        clone = ExperimentConfig(**payload)
        clone.validate()
        return clone
