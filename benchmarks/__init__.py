"""Local-first, reproducible benchmarks for ArcMind research."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.delayed_recall import DelayedRecallConfig, DelayedRecallDataset
    from benchmarks.statistics import bootstrap_interval, interquartile_mean

__all__ = [
    "DelayedRecallConfig",
    "DelayedRecallDataset",
    "bootstrap_interval",
    "interquartile_mean",
]


def __getattr__(name: str) -> Any:
    """Load framework-specific helpers only when they are requested."""
    if name in {"DelayedRecallConfig", "DelayedRecallDataset"}:
        from benchmarks import delayed_recall

        return getattr(delayed_recall, name)
    if name in {"bootstrap_interval", "interquartile_mean"}:
        from benchmarks import statistics

        return getattr(statistics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
