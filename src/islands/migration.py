"""Migration policies between independent islands."""

from .models import Candidate


def select_migrants(candidates: list[Candidate], count: int) -> list[Candidate]:
    """Select strongest candidates from the shared archive."""

    evaluated = [candidate for candidate in candidates if candidate.fitness is not None]
    ranked = sorted(
        evaluated,
        key=lambda item: (-item.fitness.accuracy, item.fitness.parameters),
    )
    return ranked[:count]
