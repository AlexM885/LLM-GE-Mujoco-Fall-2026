"""Run one LLM variation operator for the MESB driver (subprocess entry point).

``run_mesb.py`` calls this script once per mutation / crossover, in the same
way ``run_improved.py`` shells out to ``src/llm_mutation.py`` and
``src/llm_crossover.py``.

Backends
--------
``llm`` (default)
    Re-uses the project's existing operators *unchanged*:
    ``llm_mutation.augment_network`` and ``llm_crossover.augment_network``
    (and therefore ``llm_utils.generate_augmented_code`` with the model set
    by ``LLM_MODEL`` in ``src/cfg/constants.py``). Two things are adapted
    here at runtime, without editing those files:

    * ``ROOT_DIR`` in ``src/cfg/constants.py`` is a hard-coded cluster path;
      it is overridden in the imported modules with the real repository root.
    * The existing ``llm_mutation`` fills its prompt with ``str.format``.
      The MuJoCo prompts contain literal braces (e.g. ``{"policy_class": ...}``),
      so all braces except the single ``{}`` code placeholder are escaped
      before the prompt is handed over.

``mock``
    Deterministic, LLM-free edits of the seed-network hyper-parameters. For
    pipeline smoke tests and CI only; it has no scientific meaning.

Examples::

    python src/mesb_llm_operator.py mutate --input P.py --output C.py \
        --prompt-file prompt.txt --temperature 0.2 --seed 7
    python src/mesb_llm_operator.py crossover --input P1.py --input-y P2.py \
        --output C.py --temperature 0.07 --seed 8
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SPLIT_MARKER = "# --OPTION--"


# ------------------------------------------------------------------ helpers
def escape_for_format(prompt: str) -> str:
    """Escape braces so ``prompt.format(code)`` only fills the ``{}`` slot."""
    if "{}" not in prompt:
        raise ValueError("prompt must contain exactly one literal {} code placeholder")
    head, tail = prompt.split("{}", 1)
    esc = lambda s: s.replace("{", "{{").replace("}", "}}")  # noqa: E731
    return esc(head) + "{}" + esc(tail)


def _import_existing_operators():
    """Import the project's LLM modules and point them at the real repo root."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import cfg.constants as constants  # noqa: E402  (existing module, unchanged)
    constants.ROOT_DIR = str(REPO_ROOT)
    import llm_crossover  # noqa: E402
    import llm_mutation  # noqa: E402
    import llm_utils  # noqa: E402
    for module in (llm_mutation, llm_crossover, llm_utils):
        module.ROOT_DIR = str(REPO_ROOT)
    return llm_mutation, llm_crossover, constants


def _seed(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


# ------------------------------------------------------------------ LLM backend
def llm_mutate(args: argparse.Namespace) -> None:
    llm_mutation, _, constants = _import_existing_operators()
    print(f"LLM backend: LLM_MODEL={getattr(constants, 'LLM_MODEL', '?')} "
          f"inference_submission={args.inference_submission}", flush=True)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write(escape_for_format(prompt))
        escaped_path = fh.name
    _seed(args.seed)
    llm_mutation.augment_network(
        input_filename=args.input, output_filename=args.output, template_txt=escaped_path,
        top_p=args.top_p, temperature=args.temperature, apply_quality_control=False,
        inference_submission=args.inference_submission)


def llm_crossover(args: argparse.Namespace) -> None:
    """LLM crossover using the existing templates and helpers.

    ``--crossover-impl original`` calls ``llm_crossover.augment_network``
    unchanged. The default, ``fixed``, performs the same steps in the same
    random-draw order but writes the LLM output back to the chunk that was
    actually shown to the LLM. The original enumerates ``parts_x[1:]`` from 0
    and then writes ``parts_x[augment_idx]``, i.e. one chunk too early, which
    overwrites the import block (or the policy class) of MuJoCo genes.
    """
    _, xmod, constants = _import_existing_operators()
    print(f"LLM backend: LLM_MODEL={getattr(constants, 'LLM_MODEL', '?')} "
          f"inference_submission={args.inference_submission} "
          f"crossover_impl={args.crossover_impl}", flush=True)
    _seed(args.seed)
    if args.crossover_impl == "original":
        xmod.augment_network(
            input_filename_x=args.input, input_filename_y=args.input_y,
            output_filename=args.output, top_p=args.top_p, temperature=args.temperature,
            apply_quality_control=False, inference_submission=args.inference_submission)
        return
    parts_x = xmod.split_file(args.input)
    parts_y = xmod.split_file(args.input_y)
    candidates = [(x, y, idx) for idx, (x, y) in enumerate(zip(parts_x[1:], parts_y[1:]), start=1)]
    random.shuffle(candidates)
    for x, y, augment_idx in candidates:
        if x.strip() != y.strip():
            break
    template_fname = random.choice(["crossover.txt", "crossover_s.txt"])
    template_txt = (REPO_ROOT / "templates" / "CrossOver" / template_fname).read_text()
    txt2llm = template_txt.format(x.strip(), y.strip())
    code_from_llm = xmod.generate_augmented_code(
        txt2llm, augment_idx - 1, False, args.top_p, args.temperature,
        inference_submission=args.inference_submission)
    note_txt = xmod.extract_note(parts_x[augment_idx])
    parts_x[augment_idx] = f"\n{note_txt}{code_from_llm}\n"
    xmod.write_augmented_code(args.output, parts_x, parts_y)
    print("Job done")


# ------------------------------------------------------------------ mock backend
_LIST_RE = re.compile(r"^(HIDDEN_(?:PI|VF)\s*=\s*)\[([^\]]*)\]", re.M)
_LR_RE = re.compile(r"(learning_rate\s*=\s*)([0-9.eE+-]+)")


def mock_mutate_text(code: str, rng: random.Random) -> str:
    """Perturb layer widths / learning rate; always yields valid seed-style code."""
    def widths(match: re.Match) -> str:
        sizes = [int(s) for s in match.group(2).split(",") if s.strip()]
        sizes = [max(8, int(s * rng.choice([0.5, 1.0, 2.0]))) for s in sizes]
        if rng.random() < 0.3 and len(sizes) < 4:
            sizes.append(rng.choice([32, 64, 128]))
        return f"{match.group(1)}[{', '.join(str(s) for s in sizes)}]"

    def lr(match: re.Match) -> str:
        return f"{match.group(1)}{float(match.group(2)) * rng.choice([0.5, 1.0, 2.0]):.2e}"

    return _LR_RE.sub(lr, _LIST_RE.sub(widths, code))


def mock_mutate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    code = Path(args.input).read_text(encoding="utf-8")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(mock_mutate_text(code, rng), encoding="utf-8")
    print("Job Done")


def mock_crossover(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    x = Path(args.input).read_text(encoding="utf-8").split(SPLIT_MARKER)
    y = Path(args.input_y).read_text(encoding="utf-8").split(SPLIT_MARKER)
    diff = [i for i in range(1, min(len(x), len(y))) if x[i].strip() != y[i].strip()]
    if diff:
        i = rng.choice(diff)
        x[i] = y[i]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(SPLIT_MARKER.join(x), encoding="utf-8")
    print("Job Done")


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MESB wrapper around the LLM variation operators")
    p.add_argument("operator", choices=["mutate", "crossover"])
    p.add_argument("--backend", choices=["llm", "mock"], default="llm")
    p.add_argument("--input", required=True, help="Parent network file (x)")
    p.add_argument("--input-y", help="Second parent (crossover only)")
    p.add_argument("--output", required=True, help="Child network file to write")
    p.add_argument("--prompt-file", help="Mutation prompt with one literal {} placeholder")
    p.add_argument("--top-p", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inference-submission", type=int, choices=[0, 1], default=1,
                   help="1: remote/API LLM (existing default); 0: local transformers model")
    p.add_argument("--crossover-impl", choices=["fixed", "original"], default="fixed",
                   help="'original' calls src/llm_crossover.py unchanged (has an off-by-one "
                        "chunk index, see docs/MESB.md)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.inference_submission = bool(args.inference_submission)
    if args.operator == "crossover" and not args.input_y:
        raise SystemExit("crossover needs --input-y")
    if args.operator == "mutate" and args.backend == "llm" and not args.prompt_file:
        raise SystemExit("mutate with the llm backend needs --prompt-file")
    fn = {("mutate", "llm"): llm_mutate, ("crossover", "llm"): llm_crossover,
          ("mutate", "mock"): mock_mutate, ("crossover", "mock"): mock_crossover}
    fn[(args.operator, args.backend)](args)
    if not Path(args.output).exists():
        print(f"operator finished but {args.output} was not written", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
