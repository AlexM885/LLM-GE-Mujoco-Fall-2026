"""Global Pareto archive shared by all islands."""

from .models import Candidate


class ParetoArchive:
    """Keep non-dominated evaluated candidates across all islands."""

    def __init__(self) -> None:
        self._candidates: dict[str, Candidate] = {}

    def update(self, candidates: list[Candidate]) -> None:
        for candidate in candidates:
            if candidate.fitness is not None and candidate.status == "evaluated":
                self._candidates[candidate.gene_id] = candidate
        self._remove_dominated()

    def _remove_dominated(self) -> None:
        candidates = list(self._candidates.values())
        dominated = {
            candidate.gene_id
            for candidate in candidates
            if any(
                other.gene_id != candidate.gene_id
                and other.fitness is not None
                and candidate.fitness is not None
                and other.fitness.dominates(candidate.fitness)
                for other in candidates
            )
        }
        for gene_id in dominated:
            self._candidates.pop(gene_id, None)

    def candidates(self) -> list[Candidate]:
        return sorted(
            self._candidates.values(),
            key=lambda item: (-item.fitness.accuracy, item.fitness.parameters),
        )
