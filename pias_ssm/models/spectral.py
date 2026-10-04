"""Adaptive spectral operator with dataset-correct physical frequencies."""

from __future__ import annotations

import torch
from torch import nn


class NoiseEstimator(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 8, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Conv1d(8, 16, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.network(signals)


class ConditionalGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        return self.network(noise_level)


class SimpleSpectralOperator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(channels, channels, 1, bias=False)
        self.normalization = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()

    def forward(
        self,
        features: torch.Tensor,
        prior_hz: torch.Tensor | None = None,
        noise_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del prior_hz, noise_level
        spectrum = torch.fft.rfft(features, dim=-1, norm="ortho")
        projected = torch.complex(
            self.projection(spectrum.real),
            self.projection(spectrum.imag),
        )
        output = torch.fft.irfft(projected, n=features.shape[-1], dim=-1, norm="ortho")
        return self.activation(self.normalization(output + features))


class AdaptiveSpectralOperator(nn.Module):
    """Amplitude-guided spectral mask plus soft physical Gaussian priors.

    Bandwidths are expressed in normalized cycles/sample. Physical frequencies
    are divided by the dataset-specific sampling rate. Harmonics above Nyquist
    are discarded instead of being clamped onto an artificial edge peak.
    """

    def __init__(
        self,
        channels: int,
        sampling_rate_hz: float,
        use_physics_prior: bool,
        gate_type: str,
        fundamental_bandwidth: float,
        harmonic_bandwidth: float,
        harmonic_orders: tuple[int, ...],
        harmonic_weight: float,
    ) -> None:
        super().__init__()
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.use_physics_prior = use_physics_prior
        self.gate_type = gate_type
        self.fundamental_bandwidth = float(fundamental_bandwidth)
        self.harmonic_bandwidth = float(harmonic_bandwidth)
        self.harmonic_orders = tuple(harmonic_orders)
        self.harmonic_weight = float(harmonic_weight)
        hidden = max(8, channels // 4)
        self.mask_generator = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 5, padding=2, bias=False),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Conv1d(channels, channels, 1, bias=False)
        self.normalization = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        if use_physics_prior:
            self.prior_projection = nn.Conv1d(1, channels, 1, bias=False)
            if gate_type == "fixed":
                self.fixed_gate_logit = nn.Parameter(torch.tensor(0.0))
            elif gate_type == "conditional":
                self.conditional_gate = ConditionalGate()

    def physical_boost(
        self,
        prior_hz: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        batch = prior_hz.shape[0]
        frequencies = torch.fft.rfftfreq(
            sequence_length,
            d=1.0 / self.sampling_rate_hz,
            device=prior_hz.device,
        )
        normalized_axis = frequencies / self.sampling_rate_hz
        fundamental = prior_hz.mean(dim=-1) / self.sampling_rate_hz
        boost = torch.zeros(
            batch,
            1,
            normalized_axis.numel(),
            dtype=prior_hz.dtype,
            device=prior_hz.device,
        )
        for mode in range(fundamental.shape[1]):
            center = fundamental[:, mode : mode + 1]
            offset = (normalized_axis.unsqueeze(0) - center) / self.fundamental_bandwidth
            boost[:, 0, :] += torch.exp(-0.5 * offset.square())
        for order in self.harmonic_orders:
            for mode in range(fundamental.shape[1]):
                center = fundamental[:, mode : mode + 1] * order
                valid = (center > 0.0) & (center < 0.5)
                kernel = torch.exp(
                    -0.5
                    * ((normalized_axis.unsqueeze(0) - center) / self.harmonic_bandwidth).square()
                )
                boost[:, 0, :] += self.harmonic_weight * kernel * valid
        return boost

    def forward(
        self,
        features: torch.Tensor,
        prior_hz: torch.Tensor | None = None,
        noise_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        spectrum = torch.fft.rfft(features, dim=-1, norm="ortho")
        _, _, mask = self.compute_masks(features, prior_hz, noise_level)
        output = torch.fft.irfft(
            spectrum * mask.to(spectrum.dtype),
            n=features.shape[-1],
            dim=-1,
            norm="ortho",
        )
        output = self.output_projection(output) + features
        return self.activation(self.normalization(output))

    def compute_masks(
        self,
        features: torch.Tensor,
        prior_hz: torch.Tensor | None,
        noise_level: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Return data, physical, and combined masks for mechanism diagnostics."""

        spectrum = torch.fft.rfft(features, dim=-1, norm="ortho")
        data_mask = self.mask_generator(spectrum.abs().clamp_min(1e-8))
        physical_mask = None
        if self.use_physics_prior and prior_hz is not None:
            boost = self.physical_boost(prior_hz, features.shape[-1])
            physical_mask = torch.sigmoid(self.prior_projection(boost))
            if self.gate_type == "fixed":
                physical_mask = torch.sigmoid(self.fixed_gate_logit) * physical_mask
            elif self.gate_type == "conditional":
                if noise_level is None:
                    noise_level = torch.zeros(features.shape[0], 1, device=features.device)
                physical_mask = self.conditional_gate(noise_level).unsqueeze(-1) * physical_mask
        combined = data_mask if physical_mask is None else data_mask + physical_mask
        return data_mask, physical_mask, combined
