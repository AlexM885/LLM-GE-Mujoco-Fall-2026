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

All randomness here comes from Python's global ``random`` module, exactly as
in ``run_improved.py``; the MESB archive never touches it.
"""

from __future__ import annotations

import json
import random
import shlex
import string
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.cfg import constants_mesb as C


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


class Variation:
    """Creates child gene files by calling ``src/mesb_llm_operator.py``."""

    def __init__(self, genes_dir: Path, prompts_dir: Path, log_dir: Path, backend: str = "llm",
                 python_cmd: str = sys.executable, timeout_sec: float = C.LLM_TIMEOUT_SEC,
                 top_p: float = C.TOP_P, inference_submission: bool = True,
                 crossover_impl: str = "fixed") -> None:
        self.genes_dir = Path(genes_dir)
        self.prompts_dir = Path(prompts_dir)
        self.log_dir = Path(log_dir)
        self.backend = backend
        self.python_cmd = python_cmd
        self.timeout_sec = timeout_sec
        self.top_p = top_p
        self.inference_submission = inference_submission
        self.crossover_impl = crossover_impl
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

    # ----------------------------------------------------------- operators
    def _call(self, args: list[str], log_name: str) -> str | None:
        cmd = shlex.split(self.python_cmd) + [str(C.LLM_OPERATOR_SCRIPT)] + args + [
            "--backend", self.backend, "--top-p", str(self.top_p),
            "--inference-submission", str(int(self.inference_submission))]
        log_path = self.log_dir / log_name
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=self.timeout_sec, cwd=C.ROOT_DIR)
        except subprocess.TimeoutExpired:
            return f"LLM operator timed out after {self.timeout_sec:.0f}s (log {log_path})"
        except OSError as exc:
            return f"could not start LLM operator: {exc}"
        if proc.returncode != 0:
            return f"LLM operator exit code {proc.returncode} (log {log_path})"
        return None

    def mutate(self, parent_id: str, temp_range=C.MUTATION_TEMPERATURE,
               operator: str = "mutation") -> VariationResult:
        child = generate_gene_id()
        temperature = round(random.uniform(*temp_range), 2)
        template = random.choice(self.templates)
        op_seed = random.randrange(2 ** 31)
        prompt_path = self.prompts_dir / f"{child}_prompt.txt"
        prompt_path.write_text(f"{template.read_text(encoding='utf-8')}\n{self.rules}",
                               encoding="utf-8")
        err = self._call(["mutate", "--input", str(self.parent_file(parent_id)),
                          "--output", str(self.gene_file(child)),
                          "--prompt-file", str(prompt_path), "--temperature", str(temperature),
                          "--seed", str(op_seed)], f"{child}_{operator}.log")
        return VariationResult(child, operator, [parent_id], err is None, temperature, op_seed,
                               str(template.relative_to(C.ROOT_DIR)), err)

    def create(self) -> VariationResult:
        return self.mutate("seed", temp_range=C.CREATE_TEMPERATURE, operator="create")

    def crossover(self, x_id: str, y_id: str) -> tuple[VariationResult, VariationResult]:
        out = []
        for a, b in ((x_id, y_id), (y_id, x_id)):
            child = generate_gene_id()
            temperature = round(random.uniform(*C.CROSSOVER_TEMPERATURE), 2)
            op_seed = random.randrange(2 ** 31)
            err = self._call(["crossover", "--input", str(self.parent_file(a)),
                              "--input-y", str(self.parent_file(b)),
                              "--output", str(self.gene_file(child)),
                              "--temperature", str(temperature), "--seed", str(op_seed),
                              "--crossover-impl", self.crossover_impl],
                             f"{child}_crossover.log")
            out.append(VariationResult(child, "crossover", [a, b], err is None, temperature,
                                       op_seed, None, err))
        return out[0], out[1]


def vary(parent_ids: list[str], variation: Variation, crossover_probability: float,
         mutation_probability: float) -> tuple[list[str], list[VariationResult]]:
    """Apply the legacy crossover-then-mutation scheme to a list of parents.

    Same order of random draws as ``run_improved.py``: one crossover coin per
    consecutive pair, then one mutation coin per slot. A slot whose operator
    fails keeps its previous gene. Returns (final slot genes, all operator
    results in call order).
    """
    slots = list(parent_ids)
    log: list[VariationResult] = []
    for i in range(0, len(slots) - 1, 2):
        if random.random() < crossover_probability:
            r1, r2 = variation.crossover(slots[i], slots[i + 1])
            log += [r1, r2]
            if r1.ok:
                slots[i] = r1.gene_id
            if r2.ok:
                slots[i + 1] = r2.gene_id
    for i in range(len(slots)):
        if random.random() < mutation_probability:
            r = variation.mutate(slots[i])
            log.append(r)
            if r.ok:
                slots[i] = r.gene_id
    return slots, log


def result_to_json(r: VariationResult) -> str:
    return json.dumps(asdict(r))
