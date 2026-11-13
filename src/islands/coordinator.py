"""Parallel coordinator for multiple independent evolutionary islands."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from .archive import ParetoArchive
from .config import IslandConfig
from .migration import select_migrants
from .models import Candidate
from .validation import CandidateValidator
from .worker import CodeGenerator, IslandWorker


class IslandCoordinator:
    """Run islands in parallel while sharing evaluation and migration results."""

    def __init__(
        self,
        configs: Iterable[IslandConfig],
        generator: CodeGenerator,
        evaluator,
        max_workers: int | None = None,
    ) -> None:
        self.workers = {
            config.island_id: IslandWorker(
                config, generator, evaluator, CandidateValidator()
            )
            for config in configs
        }
        self.archive = ParetoArchive()
        self.max_workers = max_workers or len(self.workers)

    def run(self, generations: int, migrate_every: int = 1) -> list[Candidate]:
        if generations < 0:
            raise ValueError("generations must be non-negative")
        for generation in range(generations):
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batches = list(
                    executor.map(
                        lambda worker: worker.run_cycle(generation),
                        self.workers.values(),
                    )
                )
            evaluated = [candidate for batch in batches for candidate in batch]
            self.archive.update(evaluated)
            if migrate_every > 0 and (generation + 1) % migrate_every == 0:
                self._migrate()
        return self.archive.candidates()

    def _migrate(self) -> None:
        migrants = select_migrants(self.archive.candidates(), 1)
        if not migrants:
            return
        for worker in self.workers.values():
            worker.receive(migrants)
