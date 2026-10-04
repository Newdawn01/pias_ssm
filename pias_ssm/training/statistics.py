"""Dependency-light statistics for repeated few-shot episodes."""

from __future__ import annotations

import math

import numpy as np


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 2026,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def summarize(values: list[float], seed: int = 2026) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sample")
    lower, upper = bootstrap_mean_ci(array, seed=seed)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "sem": float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0,
        "ci95_low": lower,
        "ci95_high": upper,
    }


def paired_sign_flip_test(
    first: list[float],
    second: list[float],
    permutations: int = 20000,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Two-sided paired randomization test for identical episode seeds."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("paired samples must be non-empty and have equal shape")
    differences = left - right
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=differences.size)
        exceedances += abs(float((differences * signs).mean())) >= observed
    return {
        "n_pairs": int(differences.size),
        "mean_difference": float(differences.mean()),
        "p_value": float((exceedances + 1) / (permutations + 1)),
    }
