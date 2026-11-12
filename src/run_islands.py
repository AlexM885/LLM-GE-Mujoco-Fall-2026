"""Run a deterministic multi-island simulation.

This verifies orchestration before connecting real LLM generation and training.
"""

from islands import IslandCoordinator, default_island_configs
from islands.models import Fitness

MODEL_SKELETON = """# --OPTION--
def get_optimizer(model, lr, weight_decay=0, nesterov=True):
    return None

# --OPTION--
class ExquisiteNetV2:
    pass
"""


def demo_generator(config, generation, population):
    return MODEL_SKELETON


def demo_evaluator(candidate):
    bonuses = {
        "accuracy-focused": (0.010, 500_000),
        "parameter-efficient": (0.004, 480_000),
        "conservative-stability": (0.007, 490_000),
        "architecture-novelty": (0.008, 510_000),
    }
    accuracy_bonus, parameters = bonuses[candidate.strategy]
    return Fitness(
        accuracy=0.90 + accuracy_bonus + min(candidate.generation, 5) * 0.001,
        parameters=parameters,
    )


if __name__ == "__main__":
    coordinator = IslandCoordinator(
        default_island_configs(),
        demo_generator,
        demo_evaluator,
    )
    for candidate in coordinator.run(generations=3):
        print(candidate.to_dict())
