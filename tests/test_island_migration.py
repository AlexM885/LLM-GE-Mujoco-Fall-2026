import pytest

from src.islands import Fitness, IslandConfig, IslandCoordinator


@pytest.mark.parametrize(
    ("migrate_every", "expected_parent"),
    [(1, "b-import-a-g0-0"), (0, "b-g0-0")],
)
def test_migration_changes_parent_selection(migrate_every, expected_parent):
    parents_used = {}
    evaluated = {}

    def generate(config, gene_id, parents):
        parents_used[gene_id] = [parent.gene_id for parent in parents]
        return "# --OPTION--\nclass ExquisiteNetV2: pass\ndef get_optimizer(): pass\n"

    def evaluate(candidate):
        evaluated[candidate.gene_id] = candidate
        accuracy = 0.9 if candidate.island_id == "a" else 0.8
        return Fitness(accuracy=accuracy, parameters=100)

    coordinator = IslandCoordinator(
        configs=[
            IslandConfig("a", "test", population_size=2, candidates_per_cycle=1),
            IslandConfig("b", "test", population_size=2, candidates_per_cycle=1),
        ],
        generator=generate,
        evaluator=evaluate,
        max_workers=1,
    )
    coordinator.run(generations=2, migrate_every=migrate_every)

    child = evaluated["b-g1-0"]
    assert parents_used[child.gene_id] == [expected_parent]
    assert child.parents == [expected_parent]
    if migrate_every:
        assert {
            "source_island": "a",
            "destination_island": "b",
            "original_gene_id": "a-g0-0",
            "imported_gene_id": "b-import-a-g0-0",
            "retained": True,
            "generation": 1,
        } in coordinator.migration_events
    else:
        assert coordinator.migration_events == []
def test_crossover_records_both_selected_parents():
    from src.islands import Candidate
    from src.islands.worker import IslandWorker

    parents_used = []

    def generate(config, gene_id, parents):
        parents_used.extend(parent.gene_id for parent in parents)
        return "# --OPTION--\nclass ExquisiteNetV2: pass\ndef get_optimizer(): pass\n"

    worker = IslandWorker(
        IslandConfig("b", "test", operation="crossover", candidates_per_cycle=1),
        generate,
        lambda candidate: Fitness(accuracy=0.8, parameters=100),
    )
    worker.population = [
        Candidate(gene_id, "b", 0, "test", "", fitness=Fitness(accuracy, 100))
        for gene_id, accuracy in [("migrant", 0.9), ("local", 0.8), ("weaker", 0.7)]
    ]

    child, = worker.run_cycle(generation=1)

    assert parents_used == ["migrant", "local"]
    assert child.parents == parents_used
def test_receive_records_migrant_dropped_by_population_selection():
    from src.islands import Candidate
    from src.islands.worker import IslandWorker

    worker = IslandWorker(
        IslandConfig("b", "test", population_size=1),
        generator=lambda *_: "",
        evaluator=lambda candidate: Fitness(accuracy=0.9, parameters=100),
    )
    local = Candidate("b-g0-0", "b", 0, "test", "", fitness=Fitness(0.9, 100))
    migrant = Candidate("a-g0-0", "a", 0, "test", "", fitness=Fitness(0.8, 100))
    worker.population = [local]

    events = worker.receive([migrant])

    assert len(events) == 1
    assert events[0]["imported_gene_id"] == "b-import-a-g0-0"
    assert events[0]["retained"] is False
    assert worker.population == [local]
def test_runner_writes_migration_events(tmp_path, monkeypatch):
    import json
    import sys

    import run_AgenticIslands as runner

    event = {
        "source_island": "a",
        "destination_island": "b",
        "original_gene_id": "a-g0-0",
        "imported_gene_id": "b-import-a-g0-0",
        "retained": True,
        "generation": 1,
    }

    class FakeCoordinator:
        def __init__(self, **kwargs):
            self.migration_events = [event]

        def run(self, generations, migrate_every):
            return []

    output = tmp_path / "results.json"
    monkeypatch.setattr(runner, "IslandCoordinator", FakeCoordinator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_AgenticIslands.py",
            "--root", str(tmp_path),
            "--data", str(tmp_path),
            "--output", str(output),
        ],
    )

    runner.main()

    assert json.loads(output.read_text(encoding="utf-8")) == []
    migration_output = tmp_path / "results_migration.json"
    assert json.loads(migration_output.read_text(encoding="utf-8")) == [event]