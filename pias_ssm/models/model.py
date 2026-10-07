"""Composed PIAS-SSM few-shot model."""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .classifier import SegmentedPrototypeClassifier
from .spectral import AdaptiveSpectralOperator, NoiseEstimator, SimpleSpectralOperator
from .state_space import SSMAttentionBlock
from .wavelet import ConvolutionFrontend, MorletFrontend


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


class CapacityAdapter(nn.Module):
    """Trainable residual MLP used only for parameter-matched controls."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings + torch.tanh(self.scale) * self.network(embeddings)


class PIASSSMEncoder(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        dataset_name: str,
        sampling_rate_hz: float,
        sequence_length: int,
        input_channels: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.input_channels = input_channels if config.use_speed_channel else 1
        if config.wavelet_mode == "conv":
            self.frontend = ConvolutionFrontend(self.input_channels, config.embedding_dim)
        else:
            self.frontend = MorletFrontend(
                in_channels=self.input_channels,
                out_channels=config.embedding_dim,
                sequence_length=sequence_length,
                sampling_rate_hz=sampling_rate_hz,
                dataset_name=dataset_name,
                physics_initialization=config.wavelet_mode == "physics",
                use_speed_channel=config.use_speed_channel,
            )
        # Paper Fig. 1 / Fig. 2: the ASO carries no learned noise gate, so the
        # noise estimator is only allocated for the "conditional" ablation.
        self.noise_estimator = (
            NoiseEstimator(self.input_channels)
            if config.use_aso and config.aso_gate == "conditional"
            else None
        )
        # Paper Fig. 1 topology, illustrated for the 2048-point sequence:
        # frontend [B,E,L] -> AvgPool(2) [B,E,L/2] -> ASO -> AvgPool(2)
        # [B,E,L/4] -> SSM. Both CWRU and Ottawa use L=2048.
        self.pre_pool = nn.AvgPool1d(2, stride=2)
        if config.use_aso:
            # The ASO analyses the pre-pooled sequence, so physical frequencies
            # must be normalised by the pooled rate (fs / pre-pool factor),
            # not by the raw dataset rate. Otherwise every Gaussian prior kernel
            # lands at half its correct frequency and the modes merge.
            self.spectral_operator = AdaptiveSpectralOperator(
                channels=config.embedding_dim,
                sampling_rate_hz=sampling_rate_hz / float(self.pre_pool.kernel_size[0]),
                use_physics_prior=config.use_physics_prior,
                gate_type=config.aso_gate,
                fundamental_bandwidth=config.fundamental_bandwidth,
                harmonic_bandwidth=config.harmonic_bandwidth,
                harmonic_orders=config.harmonic_orders,
                harmonic_weight=config.harmonic_weight,
            )
        else:
            self.spectral_operator = SimpleSpectralOperator(config.embedding_dim)
        self.post_pool = nn.AvgPool1d(2, stride=2)
        self.sequence_blocks = nn.ModuleList(
            [
                SSMAttentionBlock(
                    embedding_dim=config.embedding_dim,
                    state_size=config.state_size,
                    fusion_gate_init=config.fusion_gate_init,
                    freeze_fusion_gate=config.freeze_fusion_gate,
                )
                for _ in range(config.ssm_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.capacity_adapter = (
            CapacityAdapter(config.embedding_dim, config.capacity_adapter_hidden)
            if config.capacity_adapter_hidden > 0
            else nn.Identity()
        )

    def forward(
        self,
        signals: torch.Tensor,
        prior_hz: torch.Tensor | None,
    ) -> torch.Tensor:
        signals = signals[:, : self.input_channels, :]
        noise_level = (
            self.noise_estimator(signals) if self.noise_estimator is not None else None
        )
        features = self.frontend(signals)
        features = self.pre_pool(features)
        features = self.spectral_operator(features, prior_hz=prior_hz, noise_level=noise_level)
        features = self.post_pool(features)
        for block in self.sequence_blocks:
            features = block(features)
        embeddings = self.pool(features).squeeze(-1)
        return self.capacity_adapter(embeddings)


class PIASSSM(nn.Module):
    """Physics-informed adaptive spectral state-space few-shot classifier."""

    def __init__(
        self,
        config: ModelConfig,
        dataset_name: str,
        sampling_rate_hz: float,
        sequence_length: int,
        input_channels: int = 2,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.dataset_name = dataset_name
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.encoder = PIASSSMEncoder(
            config,
            dataset_name,
            sampling_rate_hz,
            sequence_length,
            input_channels,
        )
        self.classifier = SegmentedPrototypeClassifier(
            config.embedding_dim,
            config.num_segments,
        )

    def _prepare_prior(self, prior_hz: torch.Tensor | None) -> torch.Tensor | None:
        if not self.config.use_physics_prior or prior_hz is None:
            return None
        return prior_hz * self.config.prior_scale

    def encode(self, signals: torch.Tensor, prior_hz: torch.Tensor | None) -> torch.Tensor:
        return self.encoder(signals, self._prepare_prior(prior_hz))

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        support_prior_hz: torch.Tensor | None = None,
        query_prior_hz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        support_embeddings = self.encode(support_x, support_prior_hz)
        query_embeddings = self.encode(query_x, query_prior_hz)
        return self.classifier(support_embeddings, support_y, query_embeddings)
