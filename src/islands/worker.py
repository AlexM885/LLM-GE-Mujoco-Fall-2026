"""One independent island worker."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Union

from .config import IslandConfig
from .generation import GeneratedCode
from .models import Candidate, Fitness
from .validation import CandidateValidator

CodeGenerator = Callable[
    [IslandConfig, int, list[Candidate]], Union[str, GeneratedCode]
]
Evaluator = Callable[[Candidate], Fitness]


class IslandWorker:
    """Generate, validate, evaluate, and retain candidates for one strategy."""

    def __init__(
        self,
        config: IslandConfig,
        generator: CodeGenerator,
        evaluator: Evaluator,
        validator: CandidateValidator | None = None,
    ) -> None:
        self.config = config
        self.generator = generator
        self.evaluator = evaluator
        self.validator = validator or CandidateValidator()
        self.population: list[Candidate] = []

    def run_cycle(self, generation: int) -> list[Candidate]:
        results: list[Candidate] = []
        for index in range(self.config.candidates_per_cycle):
            gene_id = f"{self.config.island_id}-g{generation}-{index}"
            generated = self.generator(self.config, generation, self.population)
            if isinstance(generated, GeneratedCode):
                code = generated.code
                metadata = {
                    **generated.metadata,
                    "source_path": str(generated.source_path),
                }
            else:
                code = generated
                metadata = {}
            candidate = Candidate(
                gene_id=gene_id,
                island_id=self.config.island_id,
                generation=generation,
                strategy=self.config.strategy,
                code=code,
                parents=[self.population[-1].gene_id] if self.population else [],
                metadata=metadata,
            )
            candidate.validation = self.validator.validate(candidate)
            if not candidate.validation.passed:
                candidate.status = "rejected"
            else:
                candidate.fitness = self.evaluator(candidate)
                candidate.status = "evaluated"
                results.append(candidate)

        self.population.extend(results)
        self.population = self._select_population()
        return results

    def receive(self, migrants: list[Candidate]) -> None:
        for candidate in migrants:
            imported = replace(
                candidate,
                gene_id=f"{self.config.island_id}-import-{candidate.gene_id}",
                island_id=self.config.island_id,
                metadata={**candidate.metadata, "migrated_from": candidate.island_id},
            )
            self.population.append(imported)
        self.population = self._select_population()

    def _select_population(self) -> list[Candidate]:
        ranked = sorted(
            self.population,
            key=lambda item: (
                item.fitness is None,
                -item.fitness.accuracy if item.fitness else 0,
                item.fitness.parameters if item.fitness else 0,
            ),
        )
        return ranked[: self.config.population_size]
