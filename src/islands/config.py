"""Configuration for independent evolutionary islands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class IslandConfig:
    """Search strategy and population settings for one island."""

    island_id: str
    strategy: str
    population_size: int = 4
    candidates_per_cycle: int = 2
    migration_size: int = 1
    operation: str = "mutation"
    top_p: float = 0.1
    temperature: float = 0.2
    apply_quality_control: bool = False
    inference_submission: bool = True
    workspace: Path | None = None

    def __post_init__(self) -> None:
        if self.operation not in {"mutation", "crossover"}:
            raise ValueError("operation must be 'mutation' or 'crossover'")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")


def default_island_configs() -> list[IslandConfig]:
    """Return diverse strategies solving the same objective."""

    return [
        IslandConfig("accuracy", "accuracy-focused", operation="mutation"),
        IslandConfig("efficiency", "parameter-efficient", operation="mutation"),
        IslandConfig("stability", "conservative-stability", operation="mutation"),
        IslandConfig(
            "novelty",
            "architecture-novelty",
            operation="crossover",
            candidates_per_cycle=1,
        ),
    ]
