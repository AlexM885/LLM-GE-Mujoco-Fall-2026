"""Driver-side LLM variation for the MESB runner.

Mirrors the operator logic of ``run_improved.py``:

* gene ids are ``"xXx" + 24`` random alphanumerics (``generate_random_string``),
* initial individuals are LLM mutations of the seed network
  (``create_individual``; temperature U(0.05, 0.4)),
* crossover of (x, y) produces two children, (x <- y) and (y <- x)
  (``customCrossover``; temperature U(0.05, 0.1)),
* mutation produces one child (``customMutation``; temperature U(0.02, 0.35)),
* a mutation prompt is a MuJoCo template + the MuJoCo constant rules
  (``generate_template``; EoT is disabled for MuJoCo, PROB_EOT = 0.0),
* if an operator fails, the parent gene is kept (``update_individual``).

Parallelism
-----------
Every random decision (coins, gene ids, temperatures, templates, operator
seeds) is drawn up front, in exactly the order the sequential version drew
them. Only then are the operator subprocesses run, ``workers`` at a time:
first all crossovers of a generation, then all mutations (a mutation may take
a crossover child as its input). Results therefore do not depend on the
number of workers, and a resumed run makes the same decisions.

All randomness comes from Python's global ``random`` module, exactly as in
``run_improved.py``; the MESB archive never touches it.
"""

from __future__ import annotations

import json
import random
import shlex
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.cfg import constants_mesb as C

#: Exit code of src/mesb_llm_operator.py for an unusable child.
EXIT_INVALID_CHILD = 3


def generate_gene_id(length: int = 24) -> str:
    chars = string.ascii_letters + string.digits
    return "xXx" + "".join(random.choice(chars) for _ in range(length))


@dataclass
class VariationResult:
    gene_id: str
    operator: str               # "create" | "crossover" | "mutation"
    parents: list[str]
    ok: bool
    temperature: float
    op_seed: int
    prompt_template: str | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)
    seconds: float | None = None


@dataclass
class OperatorJob:
    """One planned operator call (decisions already drawn, not yet run)."""
    result: VariationResult
    args: list[str]
    log_name: str


class Variation:
    """Creates child gene files by calling ``src/mesb_llm_operator.py``."""

    def __init__(self, genes_dir: Path, prompts_dir: Path, log_dir: Path, backend: str = "llm",
                 python_cmd: str = sys.executable, timeout_sec: float = C.LLM_TIMEOUT_SEC,
                 top_p: float = C.TOP_P, llm_model: str = C.LLM_MODEL,
                 crossover_impl: str = "fixed", workers: int = 1) -> None:
        self.genes_dir = Path(genes_dir)
        self.prompts_dir = Path(prompts_dir)
        self.log_dir = Path(log_dir)
        self.backend = backend
        self.python_cmd = python_cmd
        self.timeout_sec = timeout_sec
        self.top_p = top_p
        self.llm_model = llm_model
        self.crossover_impl = crossover_impl
        self.workers = max(1, int(workers))
        self.templates = sorted((C.ROOT_DIR).glob(C.MUTATION_PROMPT_GLOB))
        if not self.templates:
            raise FileNotFoundError(f"no mutation prompts match {C.MUTATION_PROMPT_GLOB}")
        self.rules = (C.ROOT_DIR / C.CONSTANT_RULES_PATH).read_text(encoding="utf-8")
        for d in (self.genes_dir, self.prompts_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

    def gene_file(self, gene_id: str) -> Path:
        return self.genes_dir / f"network_{gene_id}.py"

    def parent_file(self, gene_id: str) -> Path:
        return C.SEED_NETWORK if gene_id == "seed" else self.gene_file(gene_id)

    # ----------------------------------------------------------- planning
    def plan_mutation(self, parent_id: str, temp_range=C.MUTATION_TEMPERATURE,
                      operator: str = "mutation") -> OperatorJob:
        child = generate_gene_id()
        temperature = round(random.uniform(*temp_range), 2)
        template = random.choice(self.templates)
        op_seed = random.randrange(2 ** 31)
        prompt_path = self.prompts_dir / f"{child}_prompt.txt"
        prompt_path.write_text(f"{template.read_text(encoding='utf-8')}\n{self.rules}",
                               encoding="utf-8")
        result = VariationResult(child, operator, [parent_id], False, temperature, op_seed,
                                 str(template.relative_to(C.ROOT_DIR)))
        args = ["mutate", "--input", str(self.parent_file(parent_id)),
                "--output", str(self.gene_file(child)), "--prompt-file", str(prompt_path),
                "--temperature", str(temperature), "--seed", str(op_seed)]
        return OperatorJob(result, args, f"{child}_{operator}.log")

    def plan_crossover(self, x_id: str, y_id: str) -> list[OperatorJob]:
        jobs = []
        for a, b in ((x_id, y_id), (y_id, x_id)):
            child = generate_gene_id()
            temperature = round(random.uniform(*C.CROSSOVER_TEMPERATURE), 2)
            op_seed = random.randrange(2 ** 31)
            result = VariationResult(child, "crossover", [a, b], False, temperature, op_seed)
            args = ["crossover", "--input", str(self.parent_file(a)),
                    "--input-y", str(self.parent_file(b)), "--output", str(self.gene_file(child)),
                    "--temperature", str(temperature), "--seed", str(op_seed),
                    "--crossover-impl", self.crossover_impl]
            jobs.append(OperatorJob(result, args, f"{child}_crossover.log"))
        return jobs

    # ----------------------------------------------------------- execution
    def _run_one(self, job: OperatorJob) -> None:
        cmd = shlex.split(self.python_cmd) + [str(C.LLM_OPERATOR_SCRIPT)] + job.args + [
            "--backend", self.backend, "--top-p", str(self.top_p),
            "--llm-model", self.llm_model]
        log_path = self.log_dir / job.log_name
        start = time.time()
        err = None
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=self.timeout_sec, cwd=C.ROOT_DIR)
            if proc.returncode == EXIT_INVALID_CHILD:
                err = _invalid_reason(log_path) or "invalid child"
            elif proc.returncode != 0:
                err = f"LLM operator exit code {proc.returncode} (log {log_path})"
        except subprocess.TimeoutExpired:
            err = f"LLM operator timed out after {self.timeout_sec:.0f}s (log {log_path})"
        except OSError as exc:
            err = f"could not start LLM operator: {exc}"
        job.result.ok = err is None
        job.result.error = err
        job.result.seconds = round(time.time() - start, 1)

    def run(self, jobs: list[OperatorJob]) -> list[VariationResult]:
        """Run planned jobs, ``workers`` at a time; results keep plan order."""
        if self.workers == 1 or len(jobs) <= 1:
            for job in jobs:
                self._run_one(job)
        else:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(jobs))) as pool:
                list(pool.map(self._run_one, jobs))
        return [j.result for j in jobs]

    # ----------------------------------------------------------- convenience
    def create_many(self, n: int) -> list[VariationResult]:
        jobs = [self.plan_mutation("seed", temp_range=C.CREATE_TEMPERATURE, operator="create")
                for _ in range(n)]
        return self.run(jobs)

    def create(self) -> VariationResult:
        return self.create_many(1)[0]

    def mutate(self, parent_id: str, temp_range=C.MUTATION_TEMPERATURE,
               operator: str = "mutation") -> VariationResult:
        return self.run([self.plan_mutation(parent_id, temp_range, operator)])[0]

    def crossover(self, x_id: str, y_id: str) -> tuple[VariationResult, VariationResult]:
        r1, r2 = self.run(self.plan_crossover(x_id, y_id))
        return r1, r2


def _invalid_reason(log_path: Path) -> str | None:
    try:
        for line in reversed(Path(log_path).read_text(errors="replace").splitlines()):
            if line.startswith("INVALID CHILD:"):
                return line
    except OSError:
        pass
    return None


def vary(parent_ids: list[str], variation: Variation, crossover_probability: float,
         mutation_probability: float) -> tuple[list[str], list[VariationResult]]:
    """Apply the legacy crossover-then-mutation scheme to a list of parents.

    Same order of random draws as ``run_improved.py``: one crossover coin per
    consecutive pair, then one mutation coin per slot. A slot whose operator
    fails keeps its previous gene. Returns (final slot genes, all operator
    results in plan order).
    """
    slots = list(parent_ids)
    log: list[VariationResult] = []

    # Phase 1: crossovers (planned in order, then run in parallel).
    planned: list[tuple[int, list[OperatorJob]]] = []
    for i in range(0, len(slots) - 1, 2):
        if random.random() < crossover_probability:
            planned.append((i, variation.plan_crossover(slots[i], slots[i + 1])))
    variation.run([job for _, pair in planned for job in pair])
    for i, (j1, j2) in planned:
        log += [j1.result, j2.result]
        if j1.result.ok:
            slots[i] = j1.result.gene_id
        if j2.result.ok:
            slots[i + 1] = j2.result.gene_id

    # Phase 2: mutations of the (possibly crossed) slots.
    mutations: list[tuple[int, OperatorJob]] = []
    for i in range(len(slots)):
        if random.random() < mutation_probability:
            mutations.append((i, variation.plan_mutation(slots[i])))
    variation.run([job for _, job in mutations])
    for i, job in mutations:
        log.append(job.result)
        if job.result.ok:
            slots[i] = job.result.gene_id
    return slots, log


def result_to_json(r: VariationResult) -> str:
    return json.dumps(asdict(r))
