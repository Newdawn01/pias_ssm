"""Validated PIAS-SSM data access and reproducible few-shot episodes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REQUIRED_KEYS = {
    "samples",
    "fault_labels",
    "condition_labels",
    "instantaneous_rpm",
    "fault_freq_prior",
    "target_fs",
}


@dataclass(slots=True)
class Episode:
    support_x: torch.Tensor
    support_y: torch.Tensor
    support_prior_hz: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    query_prior_hz: torch.Tensor
    support_indices: torch.Tensor
    query_indices: torch.Tensor
    condition: int


class BearingDataset:
    """Load a processed bearing dataset and enforce its physical metadata."""

    def __init__(
        self,
        path: str | Path,
        split_ratio: float = 0.7,
        split_strategy: str = "file",
        seed: int = 42,
    ) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"dataset does not exist: {self.path}")
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        missing = REQUIRED_KEYS.difference(payload)
        if missing:
            raise KeyError(f"dataset is missing required keys: {sorted(missing)}")

        self.samples = payload["samples"].float().contiguous()
        self.labels = payload["fault_labels"].long().contiguous()
        self.conditions = payload["condition_labels"].long().contiguous()
        self.rpm = payload["instantaneous_rpm"].float().contiguous()
        self.priors_hz = payload["fault_freq_prior"].float().contiguous()
        self.sampling_rate_hz = float(payload["target_fs"])
        self.file_ids = payload.get("file_ids")
        if self.file_ids is not None:
            self.file_ids = self.file_ids.long().contiguous()
        self.sequence_length = int(self.samples.shape[-1])
        self.num_classes = int(torch.unique(self.labels).numel())
        self.num_conditions = int(torch.unique(self.conditions).numel())
        self.split_ratio = split_ratio
        self.split_strategy = split_strategy
        self.seed = seed
        self._validate_shapes()
        self._groups = self._build_groups()

    def _validate_shapes(self) -> None:
        count = self.samples.shape[0]
        if self.samples.ndim != 3 or self.samples.shape[1] < 1:
            raise ValueError("samples must have shape [N, channels, length]")
        metadata = (self.labels, self.conditions, self.rpm, self.priors_hz)
        if any(tensor.shape[0] != count for tensor in metadata):
            raise ValueError("all dataset tensors must contain the same number of samples")
        if self.rpm.shape[-1] != self.sequence_length:
            raise ValueError("instantaneous_rpm length does not match samples")
        if self.priors_hz.ndim != 3 or self.priors_hz.shape[-1] != self.sequence_length:
            raise ValueError("fault_freq_prior must have shape [N, modes, length]")
        if self.sampling_rate_hz <= 0:
            raise ValueError("target_fs must be positive")
        if self.split_strategy == "file" and self.file_ids is None:
            raise ValueError("file split requested but dataset has no file_ids")

    def summary(self) -> dict[str, object]:
        summary: dict[str, object] = {
            "path": str(self.path),
            "samples": int(self.samples.shape[0]),
            "shape": list(self.samples.shape),
            "classes": self.num_classes,
            "conditions": self.num_conditions,
            "sampling_rate_hz": self.sampling_rate_hz,
            "sequence_length": self.sequence_length,
            "has_file_ids": self.file_ids is not None,
            "unique_files": (
                int(torch.unique(self.file_ids).numel()) if self.file_ids is not None else 0
            ),
            "prior_modes": int(self.priors_hz.shape[1]),
        }
        summary["split_role_samples"] = {
            role: sum(int(indices.numel()) for indices in groups.values())
            for role, groups in self._groups.items()
        }
        if self.file_ids is not None:
            summary["split_role_files"] = {
                role: len(
                    {
                        int(file_id)
                        for indices in groups.values()
                        for file_id in self.file_ids[indices].tolist()
                    }
                )
                for role, groups in self._groups.items()
            }
        return summary

    def _build_groups(self) -> dict[str, dict[tuple[int, int], torch.Tensor]]:
        groups: dict[str, dict[tuple[int, int], torch.Tensor]] = {
            "train": {},
            "test_support": {},
            "test_query": {},
        }
        rng = np.random.default_rng(self.seed)
        for condition in torch.unique(self.conditions).tolist():
            for label in torch.unique(self.labels).tolist():
                indices = torch.where((self.conditions == condition) & (self.labels == label))[0]
                if indices.numel() == 0:
                    continue
                train, test_support, test_query = self._split_indices(indices, rng)
                groups["train"][(int(condition), int(label))] = train
                groups["test_support"][(int(condition), int(label))] = test_support
                groups["test_query"][(int(condition), int(label))] = test_query
        return groups

    def _split_indices(
        self,
        indices: torch.Tensor,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if indices.numel() < 3:
            raise ValueError("each condition/class group needs at least three samples")
        if self.split_strategy == "file":
            assert self.file_ids is not None
            files = torch.unique(self.file_ids[indices]).cpu().numpy()
            rng.shuffle(files)
            if len(files) >= 3:
                train_count = min(
                    max(1, int(round(len(files) * self.split_ratio))),
                    len(files) - 2,
                )
                remaining = files[train_count:]
                support_count = max(1, len(remaining) // 2)
                train_files = torch.as_tensor(files[:train_count], dtype=torch.long)
                support_files = torch.as_tensor(
                    remaining[:support_count],
                    dtype=torch.long,
                )
                train_mask = torch.isin(self.file_ids[indices], train_files)
                support_mask = torch.isin(self.file_ids[indices], support_files)
                query_mask = ~(train_mask | support_mask)
                return indices[train_mask], indices[support_mask], indices[query_mask]
            warnings.warn(
                "A condition/class group has fewer than three files; using sample-level roles.",
                stacklevel=2,
            )
        permutation = torch.as_tensor(rng.permutation(indices.numel()), dtype=torch.long)
        train_count = min(
            max(1, int(round(indices.numel() * self.split_ratio))),
            indices.numel() - 2,
        )
        remaining = indices.numel() - train_count
        support_count = max(1, remaining // 2)
        train = indices[permutation[:train_count]]
        test_support = indices[permutation[train_count : train_count + support_count]]
        test_query = indices[permutation[train_count + support_count :]]
        return train, test_support, test_query

    def valid_conditions(
        self,
        split: str,
        n_way: int,
        n_support: int,
        n_query: int,
    ) -> list[int]:
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        valid: list[int] = []
        for condition in range(self.num_conditions):
            if split == "train":
                per_class = n_support + n_query
                eligible = [
                    label
                    for (group_condition, label), indices in self._groups["train"].items()
                    if group_condition == condition and indices.numel() >= per_class
                ]
            else:
                eligible = [
                    label
                    for group_condition, label in self._groups["test_support"]
                    if group_condition == condition
                    and self._groups["test_support"][(group_condition, label)].numel()
                    >= n_support
                    and self._groups["test_query"][(group_condition, label)].numel()
                    >= n_query
                ]
            if len(eligible) >= n_way:
                valid.append(condition)
        return valid

    def sample_episode(
        self,
        split: str,
        n_way: int,
        n_support: int,
        n_query: int,
        seed: int,
        condition: int | None = None,
    ) -> Episode:
        """Sample a deterministic episode with disjoint support/query examples."""

        per_class = n_support + n_query
        valid_conditions = self.valid_conditions(split, n_way, n_support, n_query)
        if not valid_conditions:
            raise RuntimeError(
                f"no {split} condition can provide {n_way}-way {n_support}-shot "
                f"with {n_query} queries without replacement"
            )
        generator = torch.Generator().manual_seed(seed)
        if condition is None:
            position = int(torch.randint(len(valid_conditions), (1,), generator=generator).item())
            condition = valid_conditions[position]
        if condition not in valid_conditions:
            raise ValueError(f"condition {condition} is not valid for this episode protocol")

        if split == "train":
            labels = sorted(
                label
                for group_condition, label in self._groups["train"]
                if group_condition == condition
                and self._groups["train"][(group_condition, label)].numel() >= per_class
            )
        else:
            labels = sorted(
                label
                for group_condition, label in self._groups["test_support"]
                if group_condition == condition
                and self._groups["test_support"][(group_condition, label)].numel() >= n_support
                and self._groups["test_query"][(group_condition, label)].numel() >= n_query
            )
        class_order = torch.randperm(len(labels), generator=generator)[:n_way]
        selected = [labels[index] for index in class_order.tolist()]
        support_indices: list[torch.Tensor] = []
        query_indices: list[torch.Tensor] = []
        support_labels: list[torch.Tensor] = []
        query_labels: list[torch.Tensor] = []
        for episode_label, source_label in enumerate(selected):
            if split == "train":
                pool = self._groups["train"][(condition, source_label)]
                chosen = pool[torch.randperm(pool.numel(), generator=generator)[:per_class]]
                support_indices.append(chosen[:n_support])
                query_indices.append(chosen[n_support:])
            else:
                support_pool = self._groups["test_support"][(condition, source_label)]
                query_pool = self._groups["test_query"][(condition, source_label)]
                support_order = torch.randperm(support_pool.numel(), generator=generator)
                query_order = torch.randperm(query_pool.numel(), generator=generator)
                support_indices.append(support_pool[support_order[:n_support]])
                query_indices.append(query_pool[query_order[:n_query]])
            support_labels.append(torch.full((n_support,), episode_label, dtype=torch.long))
            query_labels.append(torch.full((n_query,), episode_label, dtype=torch.long))

        support_idx = torch.cat(support_indices)
        query_idx = torch.cat(query_indices)
        return Episode(
            support_x=self.samples[support_idx].clone(),
            support_y=torch.cat(support_labels),
            support_prior_hz=self.priors_hz[support_idx].clone(),
            query_x=self.samples[query_idx].clone(),
            query_y=torch.cat(query_labels),
            query_prior_hz=self.priors_hz[query_idx].clone(),
            support_indices=support_idx,
            query_indices=query_idx,
            condition=condition,
        )
