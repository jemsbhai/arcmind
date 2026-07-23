"""Small, dependency-light statistical utilities for benchmark aggregation."""

from collections.abc import Callable, Sequence

import numpy as np


def interquartile_mean(values: Sequence[float]) -> float:
    """Return the mean of the central 50% using fractional boundary weights."""
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")

    lower = 0.25 * array.size
    upper = 0.75 * array.size
    weighted_sum = 0.0
    total_weight = 0.0

    for index, value in enumerate(array):
        left = max(float(index), lower)
        right = min(float(index + 1), upper)
        weight = max(0.0, right - left)
        weighted_sum += weight * float(value)
        total_weight += weight

    return weighted_sum / total_weight


def bootstrap_interval(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    num_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute a deterministic percentile bootstrap confidence interval."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive")

    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(num_resamples, array.size), replace=True)
    estimates = np.asarray([statistic(sample) for sample in samples])
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def summarize(values: Sequence[float], *, seed: int = 0) -> dict[str, float]:
    """Summarize per-seed values with robust aggregates and uncertainty."""
    array = np.asarray(values, dtype=np.float64)
    lower, upper = bootstrap_interval(array, seed=seed)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "iqm": interquartile_mean(array),
        "ci95_low": lower,
        "ci95_high": upper,
        "num_seeds": int(array.size),
    }
