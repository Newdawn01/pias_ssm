"""Diagonal state-space and attention fusion blocks."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _probability_to_logit(value: float) -> float:
    clipped = min(max(value, 1e-5), 1.0 - 1e-5)
    return math.log(clipped / (1.0 - clipped))


class S4DKernel(nn.Module):
    def __init__(self, channels: int, state_size: int = 64) -> None:
        super().__init__()
        if state_size % 2:
            raise ValueError("state_size must be even")
        half = state_size // 2
        self.log_a_real = nn.Parameter(torch.full((channels, half), math.log(0.5)))
        imaginary = math.pi * torch.arange(half).float()
        self.a_imaginary = nn.Parameter(imaginary.unsqueeze(0).expand(channels, -1).clone())
        self.b = nn.Parameter(torch.randn(channels, half, 2) * 0.5)
        self.c = nn.Parameter(torch.randn(channels, half, 2) * 0.5)
        self.log_dt = nn.Parameter(torch.empty(channels).uniform_(math.log(0.001), math.log(0.1)))

    def forward(self, length: int) -> torch.Tensor:
        a = -self.log_a_real.exp() + 1j * self.a_imaginary
        dt = self.log_dt.exp().unsqueeze(-1)
        a_bar = torch.exp(a * dt)
        b = torch.view_as_complex(self.b.contiguous())
        c = torch.view_as_complex(self.c.contiguous())
        b_bar = (a_bar - 1.0) / a * b
        steps = torch.arange(length, device=a.device).view(1, -1, 1)
        powers = a_bar.unsqueeze(1).pow(steps)
        return (b_bar.unsqueeze(1) * powers * c.unsqueeze(1)).sum(dim=-1).real


class S4D(nn.Module):
    def __init__(self, channels: int, state_size: int = 64) -> None:
        super().__init__()
        self.kernel = S4DKernel(channels, state_size)
        self.skip = nn.Parameter(torch.randn(channels))
        self.normalization = nn.LayerNorm(channels)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        length = sequence.shape[1]
        kernel = self.kernel(length)
        kernel_fft = torch.fft.rfft(kernel, n=2 * length)
        sequence_fft = torch.fft.rfft(sequence.transpose(1, 2), n=2 * length)
        output = torch.fft.irfft(kernel_fft.unsqueeze(0) * sequence_fft, n=2 * length)
        output = output[..., :length].transpose(1, 2)
        return self.normalization(output + sequence * self.skip)


class SSMAttentionBlock(nn.Module):
    """Parallel S4D and multi-head attention with an inspectable fusion gate."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int = 64,
        num_heads: int = 2,
        state_size: int = 64,
        fusion_gate_init: float = 0.5,
        freeze_fusion_gate: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.ssm_projection = nn.Linear(embedding_dim, hidden_size * 2)
        self.ssm = S4D(hidden_size * 2, state_size)
        self.attention_qkv = nn.Linear(embedding_dim, hidden_size * 3)
        self.attention_projection = nn.Linear(hidden_size, hidden_size)
        self.fusion_gate_logit = nn.Parameter(
            torch.tensor(_probability_to_logit(fusion_gate_init), dtype=torch.float32)
        )
        self.fusion_gate_logit.requires_grad_(not freeze_fusion_gate)
        self.output_projection = nn.Linear(hidden_size, embedding_dim)
        self.normalization = nn.LayerNorm(embedding_dim)

    @property
    def fusion_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.fusion_gate_logit)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features.transpose(1, 2)
        ssm_projected = self.ssm(self.ssm_projection(residual))
        ssm_value, ssm_gate = ssm_projected.chunk(2, dim=-1)
        ssm_output = F.silu(ssm_value) * F.silu(ssm_gate)

        query, key, value = self.attention_qkv(residual).chunk(3, dim=-1)
        shape = (features.shape[0], features.shape[-1], self.num_heads, self.head_dim)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        attention = F.scaled_dot_product_attention(query, key, value)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(features.shape[0], features.shape[-1], -1)
        attention = self.attention_projection(attention)

        fused = self.fusion_gate * ssm_output + (1.0 - self.fusion_gate) * attention
        output = self.normalization(self.output_projection(fused) + residual)
        return output.transpose(1, 2)
