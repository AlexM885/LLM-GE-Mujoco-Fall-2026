"""Run parallel agentic islands with the repository's legacy LLM operators."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.islands import (
    Candidate,
    Fitness,
    IslandConfig,
    IslandCoordinator,
    LegacyLLMGenerator,
    default_island_configs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--candidates-per-cycle", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=216)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--migrate-every", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("agentic_islands_results.json"))
    parser.add_argument(
        "--inference-submission",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the legacy LLM scripts to use their configured inference provider.",
    )
    return parser.parse_args()


def build_configs(root: Path, args: argparse.Namespace) -> list[IslandConfig]:
    configs = []
    for config in default_island_configs():
        configs.append(
            IslandConfig(
                island_id=config.island_id,
                strategy=config.strategy,
                population_size=args.population_size,
                candidates_per_cycle=args.candidates_per_cycle,
                migration_size=config.migration_size,
                operation=config.operation,
                top_p=config.top_p,
                temperature=config.temperature,
                apply_quality_control=config.apply_quality_control,
                inference_submission=args.inference_submission,
                workspace=root / "islands" / config.island_id,
            )
        )
    return configs


class TrainingEvaluator:
    """Evaluate one validated source file with the existing training entry point."""

    def __init__(
        self,
        root: Path,
        data: Path,
        epochs: int,
        batch_size: int,
    ) -> None:
        self.root = root
        self.training_root = root / "sota" / "ExquisiteNetV2"
        self.data = data.resolve()
        self.epochs = epochs
        self.batch_size = batch_size

    def __call__(self, candidate: Candidate) -> Fitness:
        source = Path(candidate.metadata["source_path"])
        model_name = f"network_{candidate.gene_id}"
        destination = self.training_root / "models" / f"{model_name}.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        result_file = self.training_root / "results" / f"{candidate.gene_id}_results.txt"
        result_file.unlink(missing_ok=True)
        command = [
            sys.executable,
            "train.py",
            "-data",
            str(self.data),
            "-epoch",
            str(self.epochs),
            "-bs",
            str(self.batch_size),
            "-val_r",
            "0.2",
            "-seed",
            "21",
            "-network",
            f"models.{model_name}",
        ]
        result = subprocess.run(
            command,
            cwd=self.training_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"training failed for {candidate.gene_id}:\n{result.stderr[-4000:]}"
            )
        if not result_file.is_file():
            raise RuntimeError(f"training did not create {result_file}")
        values = result_file.read_text().strip().split(",")
        if len(values) < 2:
            raise RuntimeError(f"invalid training result in {result_file}: {values!r}")
        return Fitness(accuracy=float(values[0]), parameters=int(values[1]))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.generations < 1:
        raise ValueError("--generations must be positive")
    if not args.data.exists():
        raise FileNotFoundError(f"dataset path does not exist: {args.data}")

    configs = build_configs(root, args)
    generator = LegacyLLMGenerator(
        repository_root=root,
        seed_network=root / "sota" / "ExquisiteNetV2" / "network.py",
        template_root=root / "templates",
    )
    evaluator = TrainingEvaluator(root, args.data, args.epochs, args.batch_size)
    coordinator = IslandCoordinator(
        configs=configs,
        generator=generator,
        evaluator=evaluator,
        max_workers=args.workers,
    )
    archive = coordinator.run(args.generations, migrate_every=args.migrate_every)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([candidate.to_dict() for candidate in archive], indent=2))
    print(f"Saved {len(archive)} archived candidates to {output}")


if __name__ == "__main__":
    main()
