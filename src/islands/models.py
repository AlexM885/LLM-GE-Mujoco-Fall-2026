"""Shared data models for island candidates and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Fitness:
    """Objectives shared by every island: maximize accuracy, minimize parameters."""

    accuracy: float
    parameters: int

    def dominates(self, other: "Fitness") -> bool:
        return (
            self.accuracy >= other.accuracy
            and self.parameters <= other.parameters
            and (self.accuracy > other.accuracy or self.parameters < other.parameters)
        )


@dataclass
class ValidationResult:
    """Deterministic validation outcome for generated code."""

    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    """Model candidate tracked across islands and the global archive."""

    gene_id: str
    island_id: str
    generation: int
    strategy: str
    code: str
    parents: list[str] = field(default_factory=list)
    fitness: Fitness | None = None
    validation: ValidationResult | None = None
    status: str = "generated"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.fitness is not None:
            data["fitness"] = asdict(self.fitness)
        return data
