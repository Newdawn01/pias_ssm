"""Noise models used in the reviewer robustness experiments."""

from __future__ import annotations

import math

import torch


SUPPORTED_NOISES = {
    "none",
    "gaussian",
    "pink",
    "brown",
    "impulsive",
    "nonstationary",
    "narrowband",
}


def _normal(
    shape: torch.Size,
    reference: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(shape, dtype=reference.dtype, device=reference.device, generator=generator)


def _colored_noise(
    reference: torch.Tensor,
    beta: float,
    generator: torch.Generator | None,
) -> torch.Tensor:
    batch, _, length = reference.shape
    white = _normal(torch.Size((batch, 1, length)), reference, generator)
    spectrum = torch.fft.rfft(white, dim=-1)
    frequencies = torch.fft.rfftfreq(length, device=reference.device)
    weights = frequencies.clamp_min(1.0 / length).pow(-beta / 2.0)
    weights[0] = 0.0
    return torch.fft.irfft(spectrum * weights, n=length, dim=-1)


def _narrowband_noise(
    reference: torch.Tensor,
    sampling_rate_hz: float,
    prior_hz: torch.Tensor | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if prior_hz is None:
        raise ValueError("narrowband noise requires fault-frequency priors")
    batch, _, length = reference.shape
    frequencies = prior_hz.mean(dim=-1)
    if frequencies.ndim == 1:
        frequencies = frequencies.unsqueeze(1)
    frequencies = frequencies[:, : min(4, frequencies.shape[1])]
    time = torch.arange(length, device=reference.device, dtype=reference.dtype)
    time = time / float(sampling_rate_hz)
    phase = 2.0 * math.pi * torch.rand(
        (batch, frequencies.shape[1], 1),
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )
    waves = torch.sin(
        2.0 * math.pi * frequencies.to(reference.dtype).unsqueeze(-1) * time + phase
    )
    return waves.mean(dim=1, keepdim=True)


def _raw_noise(
    reference: torch.Tensor,
    noise_type: str,
    sampling_rate_hz: float,
    prior_hz: torch.Tensor | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    batch, _, length = reference.shape
    shape = torch.Size((batch, 1, length))
    if noise_type == "gaussian":
        return _normal(shape, reference, generator)
    if noise_type == "pink":
        return _colored_noise(reference, beta=1.0, generator=generator)
    if noise_type == "brown":
        return _colored_noise(reference, beta=2.0, generator=generator)
    if noise_type == "impulsive":
        background = 0.15 * _normal(shape, reference, generator)
        mask = torch.rand(shape, device=reference.device, generator=generator) < 0.01
        impulses = 8.0 * _normal(shape, reference, generator) * mask
        return background + impulses
    if noise_type == "nonstationary":
        base = _normal(shape, reference, generator)
        time = torch.linspace(0.0, 1.0, length, device=reference.device)
        envelope = 0.15 + 1.85 * (time * (0.5 + 0.5 * torch.sin(6.0 * math.pi * time)))
        return base * envelope.view(1, 1, -1)
    if noise_type == "narrowband":
        return _narrowband_noise(reference, sampling_rate_hz, prior_hz, generator)
    raise ValueError(f"unsupported noise type: {noise_type}")


def add_noise(
    signals: torch.Tensor,
    snr_db: float | torch.Tensor | None,
    noise_type: str = "gaussian",
    sampling_rate_hz: float = 2000.0,
    prior_hz: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add calibrated noise to vibration channel 0 while preserving speed channels.

    Args:
        signals: Tensor shaped ``[batch, channels, length]``.
        snr_db: Scalar or per-sample SNR in decibels. ``None`` leaves input unchanged.
        noise_type: One of :data:`SUPPORTED_NOISES`.
        sampling_rate_hz: Dataset-specific rate used by narrowband interference.
        prior_hz: Physical frequencies shaped ``[batch, modes, time]``.
    """

    if snr_db is None or noise_type == "none":
        return signals
    if noise_type not in SUPPORTED_NOISES:
        raise ValueError(f"unsupported noise type: {noise_type}")
    if signals.ndim != 3:
        raise ValueError("signals must have shape [batch, channels, length]")

    vibration = signals[:, :1, :]
    noise = _raw_noise(vibration, noise_type, sampling_rate_hz, prior_hz, generator)
    signal_power = vibration.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    noise_power = noise.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

    snr = torch.as_tensor(snr_db, dtype=vibration.dtype, device=vibration.device)
    if snr.ndim == 0:
        snr = snr.repeat(vibration.shape[0])
    if snr.numel() != vibration.shape[0]:
        raise ValueError("per-sample snr_db must match batch size")
    target_noise_power = signal_power / torch.pow(10.0, snr.view(-1, 1, 1) / 10.0)
    scaled_noise = noise * torch.sqrt(target_noise_power / noise_power)

    output = signals.clone()
    output[:, :1, :] = vibration + scaled_noise
    return output
