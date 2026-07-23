"""Pure metadata registry for POBAX upper-reference environments."""

from __future__ import annotations

from typing import Final

UPPER_REFERENCE_SPECS: Final = {
    "tmaze_10-perfect-memory": {
        "environment_source": {
            "source_environment": "tmaze_10",
            "perfect_memory": True,
        },
        "environment_reference": {
            "primary_environment": "tmaze_10",
            "reference_class": "persistent_cue_upper_reference",
        },
    },
    "rocksample_11_11-fully-observable": {
        "environment_source": {
            "source_environment": "rocksample_11_11",
            "perfect_memory": True,
        },
        "environment_reference": {
            "primary_environment": "rocksample_11_11",
            "reference_class": "full_markov_observation",
        },
    },
    "battleship_10-perfect-recall": {
        "environment_source": {
            "source_environment": "battleship_10",
            "perfect_memory": True,
            "observation_adapter": "BattleshipPerfectRecallObservationWrapper",
        },
        "environment_reference": {
            "primary_environment": "battleship_10",
            "reference_class": "perfect_recall_history",
        },
    },
    "Navix-DMLab-Maze-01-fully-observable": {
        "environment_source": {
            "source_environment": "Navix-DMLab-Maze-F-01-v0",
            "perfect_memory": False,
        },
        "environment_reference": {
            "primary_environment": "Navix-DMLab-Maze-01-v0",
            "reference_class": "full_markov_observation",
        },
    },
    "Walker-F-v0": {
        "environment_source": {
            "source_environment": "Walker-F-v0",
            "perfect_memory": False,
        },
        "environment_reference": {
            "primary_environment": "Walker-V-v0",
            "reference_class": "full_markov_observation",
        },
    },
    "HalfCheetah-F-v0": {
        "environment_source": {
            "source_environment": "HalfCheetah-F-v0",
            "perfect_memory": False,
        },
        "environment_reference": {
            "primary_environment": "HalfCheetah-V-v0",
            "reference_class": "full_markov_observation",
        },
    },
}

UPPER_REFERENCE_ENVIRONMENTS: Final = frozenset(UPPER_REFERENCE_SPECS)
UPPER_TO_PRIMARY_ENVIRONMENT: Final = {
    environment: specification["environment_reference"]["primary_environment"]
    for environment, specification in UPPER_REFERENCE_SPECS.items()
}


def expected_environment_source(environment: str) -> dict[str, object]:
    """Return the exact source invocation that a frozen cell must record."""
    specification = UPPER_REFERENCE_SPECS.get(environment)
    if specification is None:
        return {
            "source_environment": environment,
            "perfect_memory": False,
        }
    return dict(specification["environment_source"])


def expected_environment_reference(environment: str) -> dict[str, str] | None:
    """Return the exact upper-reference classification for an environment."""
    specification = UPPER_REFERENCE_SPECS.get(environment)
    if specification is None:
        return None
    return dict(specification["environment_reference"])


__all__ = [
    "UPPER_REFERENCE_ENVIRONMENTS",
    "UPPER_REFERENCE_SPECS",
    "UPPER_TO_PRIMARY_ENVIRONMENT",
    "expected_environment_reference",
    "expected_environment_source",
]
