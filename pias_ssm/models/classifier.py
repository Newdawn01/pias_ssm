"""Segmented prototype classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SegmentedPrototypeClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_segments: int) -> None:
        super().__init__()
        if embedding_dim % num_segments:
            raise ValueError("embedding_dim must be divisible by num_segments")
        self.num_segments = num_segments
        self.segment_logits = nn.Parameter(torch.zeros(num_segments))
        self.log_temperature = nn.Parameter(torch.tensor(10.0).log())

    @property
    def segment_weights(self) -> torch.Tensor:
        return self.segment_logits.softmax(dim=0)

    def forward(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        classes = torch.unique(support_labels, sorted=True)
        prototypes = torch.stack(
            [support_embeddings[support_labels == label].mean(dim=0) for label in classes]
        )
        query_segments = query_embeddings.chunk(self.num_segments, dim=-1)
        prototype_segments = prototypes.chunk(self.num_segments, dim=-1)
        similarities = []
        for query, prototype in zip(query_segments, prototype_segments, strict=True):
            similarities.append(
                F.normalize(query, dim=-1) @ F.normalize(prototype, dim=-1).transpose(0, 1)
            )
        stacked = torch.stack(similarities, dim=-1)
        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        return (stacked * self.segment_weights).sum(dim=-1) * temperature


def segment_cosine_distortion(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Per-sample, per-segment cosine distance between clean and noisy embeddings."""

    if clean.shape != noisy.shape:
        raise ValueError("clean and noisy embeddings must have identical shapes")
    clean_segments = clean.chunk(num_segments, dim=-1)
    noisy_segments = noisy.chunk(num_segments, dim=-1)
    distances = [
        1.0 - F.cosine_similarity(clean_part, noisy_part, dim=-1)
        for clean_part, noisy_part in zip(clean_segments, noisy_segments, strict=True)
    ]
    return torch.stack(distances, dim=-1)
