"""Adapters for the repository's existing LLM mutation and crossover scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import IslandConfig
from .models import Candidate


@dataclass(frozen=True)
class GeneratedCode:
    """Generated model source plus provenance needed by the island system."""

    code: str
    source_path: Path
    metadata: dict[str, str]


class LegacyLLMGenerator:
    """Run the existing mutation/crossover entry points without modifying them."""

    def __init__(
        self,
        repository_root: Path,
        seed_network: Path,
        template_root: Path,
    ) -> None:
        self.repository_root = repository_root
        self.seed_network = seed_network
        self.template_root = template_root
        self.script_root = repository_root / "src"

    def __call__(
        self,
        config: IslandConfig,
        generation: int,
        population: list[Candidate],
    ) -> GeneratedCode:
        workspace = config.workspace or self.repository_root / "islands" / config.island_id
        model_dir = workspace / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        gene_id = f"{config.island_id}-g{generation}-{len(population)}"
        output_path = model_dir / f"network_{gene_id}.py"

        bootstrap = config.operation == "crossover" and len(population) < 2
        operation = "mutation" if bootstrap else config.operation
        if operation == "mutation":
            parent_path = self._parent_path(population)
            template_path = self._mutation_template(config)
            command = [
                sys.executable,
                str(self.script_root / "llm_mutation.py"),
                str(parent_path),
                str(output_path),
                str(template_path),
            ]
        else:
            parent_paths = self._crossover_parents(population)
            command = [
                sys.executable,
                str(self.script_root / "llm_crossover.py"),
                str(parent_paths[0]),
                str(parent_paths[1]),
                str(output_path),
            ]

        command.extend(
            [
                "--top_p",
                str(config.top_p),
                "--temperature",
                str(config.temperature),
                "--apply_quality_control",
                str(config.apply_quality_control),
                "--inference_submission",
                str(config.inference_submission),
            ]
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.script_root)
        result = subprocess.run(
            command,
            cwd=self.repository_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{operation} failed for {gene_id}: {result.stderr.strip()}"
            )
        if not output_path.is_file():
            raise RuntimeError(f"{operation} did not create {output_path}")

        return GeneratedCode(
            code=output_path.read_text(),
            source_path=output_path,
            metadata={
                "operation": "mutation-bootstrap" if bootstrap else operation,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            },
        )

    def _parent_path(self, population: list[Candidate]) -> Path:
        if not population:
            return self.seed_network
        path = population[-1].metadata.get("source_path")
        return Path(path) if path else self.seed_network

    def _crossover_parents(self, population: list[Candidate]) -> tuple[Path, Path]:
        if len(population) < 2:
            raise ValueError("crossover islands require at least two parent candidates")
        paths = [
            Path(candidate.metadata["source_path"])
            for candidate in population[-2:]
            if candidate.metadata.get("source_path")
        ]
        if len(paths) != 2:
            raise ValueError("crossover parents must have source_path metadata")
        return paths[0], paths[1]

    def _mutation_template(self, config: IslandConfig) -> Path:
        templates = sorted((self.template_root / "FixedPrompts").glob("*/*.txt"))
        if not templates:
            raise FileNotFoundError("no mutation templates found")
        index = sum(ord(char) for char in config.strategy) % len(templates)
        return templates[index]
