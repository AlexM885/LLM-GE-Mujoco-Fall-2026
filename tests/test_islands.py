import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from islands import IslandCoordinator
from islands.config import IslandConfig
from islands.models import Candidate, Fitness
from islands.validation import CandidateValidator


VALID_CODE = """# --OPTION--
def get_optimizer(model, lr, weight_decay=0, nesterov=True):
    return None

# --OPTION--
class ExquisiteNetV2:
    pass
"""


def test_validator_accepts_required_model_shape():
    candidate = Candidate("gene", "test", 0, "test", VALID_CODE)
    result = CandidateValidator().validate(candidate)
    assert result.passed


def test_validator_rejects_invalid_code():
    candidate = Candidate("gene", "test", 0, "test", "def broken(")
    result = CandidateValidator().validate(candidate)
    assert not result.passed
    assert result.checks["syntax"] is False


def test_coordinator_runs_islands_and_builds_archive():
    configs = [
        IslandConfig("a", "accuracy-focused", candidates_per_cycle=1),
        IslandConfig("b", "parameter-efficient", candidates_per_cycle=1),
    ]

    def generator(config, generation, population):
        return VALID_CODE

    def evaluator(candidate):
        return Fitness(
            accuracy=0.90 if candidate.island_id == "a" else 0.91,
            parameters=500_000 if candidate.island_id == "a" else 400_000,
        )

    coordinator = IslandCoordinator(configs, generator, evaluator)
    archive = coordinator.run(generations=2)
    assert archive
    assert all(candidate.status == "evaluated" for candidate in archive)
