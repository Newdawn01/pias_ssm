"""Learnable complex Morlet frontends."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


BEARING_COEFFICIENTS = {
    "CWRU": {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 2.3576, "FTF": 0.3983},
    "Ottawa": {"BPFI": 5.4276, "BPFO": 3.5724, "BSF": 2.3254, "FTF": 0.3970},
}

RPM_RANGES = {
    "CWRU": (1730.0, 1750.0, 1772.0, 1797.0),
    "Ottawa": (900.0, 1100.0, 1300.0, 1500.0),
}


class ConvolutionFrontend(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.network(signals)


class MorletFrontend(nn.Module):
    """Complex Morlet filter bank with trainable centers and log-scales."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sequence_length: int,
        sampling_rate_hz: float,
        dataset_name: str,
        physics_initialization: bool,
        use_speed_channel: bool,
    ) -> None:
        super().__init__()
        if out_channels % 2:
            raise ValueError("out_channels must be even for complex Morlet pairs")
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.num_wavelets = out_channels // 2
        self.kernel_size = min(129, sequence_length)
        if self.kernel_size % 2 == 0:
            self.kernel_size -= 1
        self.use_speed_channel = use_speed_channel and in_channels > 1
        centers = self._initial_centers(dataset_name, physics_initialization)
        normalized_centers = (centers / 0.5).clamp(1e-5, 1.0 - 1e-5)
        self.center_logits = nn.Parameter(torch.logit(normalized_centers))
        self.log_scale = nn.Parameter(torch.full((self.num_wavelets,), math.log(6.0)))
        self.vibration_projection = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        if self.use_speed_channel:
            speed_channels = max(8, out_channels // 4)
            self.speed_projection = nn.Sequential(
                nn.Conv1d(1, speed_channels, 15, padding=7, bias=False),
                nn.BatchNorm1d(speed_channels),
                nn.GELU(),
            )
            self.fusion = nn.Conv1d(out_channels + speed_channels, out_channels, 1, bias=False)
        self.normalization = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def _initial_centers(self, dataset_name: str, physics: bool) -> torch.Tensor:
        if not physics:
            return torch.linspace(0.005, 0.49, self.num_wavelets)
        coefficients = BEARING_COEFFICIENTS[dataset_name]
        candidates: list[float] = []
        for harmonic in range(1, 9):
            for rpm in RPM_RANGES[dataset_name]:
                shaft_hz = rpm / 60.0
                for coefficient in coefficients.values():
                    normalized = coefficient * shaft_hz * harmonic / self.sampling_rate_hz
                    if 0.0 < normalized < 0.5:
                        candidates.append(normalized)
        unique = torch.tensor(sorted(set(round(value, 6) for value in candidates)))
        if unique.numel() >= self.num_wavelets:
            positions = torch.linspace(0, unique.numel() - 1, self.num_wavelets).long()
            return unique[positions].float()
        supplement = torch.linspace(0.005, 0.49, self.num_wavelets - unique.numel())
        return torch.cat((unique.float(), supplement))[: self.num_wavelets]

    def kernels(self) -> torch.Tensor:
        time = torch.arange(self.kernel_size, device=self.center_logits.device).float()
        time = time - self.kernel_size // 2
        time = time.view(1, 1, -1)
        centers = self.center_frequency.view(-1, 1, 1)
        scales = self.log_scale.exp().clamp(3.0, 20.0).view(-1, 1, 1)
        envelope = torch.exp(-0.5 * (time / scales).square())
        real = envelope * torch.cos(2.0 * math.pi * centers * time)
        imaginary = envelope * torch.sin(2.0 * math.pi * centers * time)
        kernels = torch.cat((real, imaginary), dim=0)
        return kernels / kernels.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)

    def learned_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "center_normalized": self.center_frequency.detach(),
            "center_hz": self.center_frequency.detach() * self.sampling_rate_hz,
            "scale": self.log_scale.detach().exp().clamp(3.0, 20.0),
        }

    @property
    def center_frequency(self) -> torch.Tensor:
        """Smoothly constrained center frequencies in cycles/sample."""

        return 0.5 * torch.sigmoid(self.center_logits)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        vibration = signals[:, :1, :]
        kernels = self.kernels().to(dtype=signals.dtype)
        features = F.conv1d(vibration, kernels, padding=self.kernel_size // 2)
        features = self.vibration_projection(features)
        if self.use_speed_channel:
            speed = self.speed_projection(signals[:, 1:2, :])
            features = self.fusion(torch.cat((features, speed), dim=1))
        return self.activation(self.normalization(features))
