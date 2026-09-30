"""Run one LLM variation operator for the MESB driver (subprocess entry point).

``run_mesb.py`` calls this script once per mutation / crossover, the same way
``run_improved.py`` shells out to ``src/llm_mutation.py`` and
``src/llm_crossover.py``.

Backends
--------
``llm`` (default)
    Re-uses the project's existing operators *unchanged*:
    ``llm_mutation.augment_network`` (prompt filling, FORBIDDEN_PATTERNS
    validation, parent-chunk fallback) and ``llm_utils.generate_augmented_code``
    with the model chosen by ``--llm-model``. With ``LOCAL_LLM = True`` in
    ``src/cfg/constants_Mujoco.py`` this talks to the team's local LLM server
    (``server.py``), located through ``hostname.log`` and ``PORT``; the
    operator waits for that server exactly like ``run_improved.py`` does.

    Two runtime guards are installed around the (unmodified) team code:

    * ``llm_utils.clean_code_from_llm`` is replaced by
      ``mesb_llm_guard.extract_code``, which skips placeholder blocks such as
      ``[modified code]`` instead of always taking the first fenced block
      (``--extraction original`` keeps the team's function).
    * the code generator returned by ``llm_utils.get_llm_code_generator`` is
      wrapped so that an empty reply (server restarted, 8-hour job limit)
      waits for the replacement server and retries.

    Every finished child - any backend - is then checked with
    ``mesb_llm_guard.validate_child``. An invalid child is renamed to
    ``*.rejected.py`` and the operator exits with code 3, so the driver keeps
    the parent, exactly as run_improved.py does for a failed LLM job.

    Crossover: ``--crossover-impl original`` calls
    ``llm_crossover.augment_network`` unchanged. The default ``fixed`` performs
    the same steps with the same helpers and random-draw order, but writes the
    LLM output back to the chunk that was shown to the LLM (the original is
    off by one; see ``llm_crossover`` below and docs/MESB.md).

``mock``
    Deterministic, LLM-free edits of the seed-network hyper-parameters. For
    pipeline smoke tests and CI only; it has no scientific meaning.

Examples::

    python src/mesb_llm_operator.py mutate --input P.py --output C.py \
        --prompt-file prompt.txt --temperature 0.2 --seed 7 --llm-model llama3
    python src/mesb_llm_operator.py crossover --input P1.py --input-y P2.py \
        --output C.py --temperature 0.07 --seed 8 --llm-model llama3
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SPLIT_MARKER = "# --OPTION--"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mesb_llm_guard import extract_code, validate_child  # noqa: E402

#: Exit code for "LLM answered, but the child network is unusable".
EXIT_INVALID_CHILD = 3


# ------------------------------------------------------------------ helpers
def _import_existing_operators():
    """Import the project's (unmodified) LLM modules."""
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import cfg.constants as constants  # noqa: E402
    import llm_crossover  # noqa: E402
    import llm_mutation  # noqa: E402
    return llm_mutation, llm_crossover, constants


def wait_for_llm_server(constants, timeout: float, interval: float = 10.0) -> None:
    """Block until the local LLM server answers, mirroring run_improved.py.

    No-op when ``LOCAL_LLM`` is False (remote/API models).
    """
    if not getattr(constants, "LOCAL_LLM", False):
        return
    host_file = Path(constants.HOSTNAME_DIR)
    start, last = time.time(), None
    while time.time() - start <= timeout:
        host = host_file.read_text().strip() if host_file.exists() else ""
        if host:
            url = f"http://{host}:{constants.PORT}/"
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        print(f"LLM server ready at {url}", flush=True)
                        return
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                last = err
        print(f"Waiting for LLM server ({round(time.time() - start)}s/{timeout:.0f}s, "
              f"host file {host_file})", flush=True)
        time.sleep(interval)
    raise TimeoutError(f"LLM server not ready after {timeout:.0f}s (last error: {last}). "
                       "Start it with scripts/mesb/llm_server.sbatch")


def _install_guards(constants, args: argparse.Namespace) -> None:
    """Patch llm_utils at runtime (see module docstring); the file is untouched."""
    import llm_utils  # noqa: E402  (already imported by llm_mutation/llm_crossover)
    if args.extraction == "robust":
        llm_utils.clean_code_from_llm = extract_code
    original_factory = llm_utils.get_llm_code_generator

    def factory(llm_model):
        generator, qc_func = original_factory(llm_model)

        def generate_with_retry(prompt, **kwargs):
            out = None
            for attempt in range(1, args.llm_retries + 1):
                out = generator(prompt, **kwargs)
                text = out[0] if isinstance(out, tuple) else out
                if isinstance(text, str) and text.strip():
                    return out
                print(f"LLM returned no text (attempt {attempt}/{args.llm_retries}); "
                      "waiting for the LLM server before retrying", flush=True)
                if attempt < args.llm_retries:
                    wait_for_llm_server(constants, args.server_timeout)
            return out
        return generate_with_retry, qc_func

    llm_utils.get_llm_code_generator = factory


def _seed(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def _banner(constants, args) -> None:
    print(f"LLM backend: llm_model={args.llm_model} LOCAL_LLM="
          f"{getattr(constants, 'LOCAL_LLM', '?')} MODEL_PATH={getattr(constants, 'MODEL_PATH', '?')}",
          flush=True)


# ------------------------------------------------------------------ LLM backend
def llm_mutate(args: argparse.Namespace) -> None:
    llm_mutation, _, constants = _import_existing_operators()
    _banner(constants, args)
    _install_guards(constants, args)
    wait_for_llm_server(constants, args.server_timeout)
    _seed(args.seed)
    # llm_mutation fills the literal "{}" with str.replace, so the prompt file
    # is passed through as-is (absolute path; os.path.join keeps it absolute).
    llm_mutation.augment_network(
        input_filename=args.input, output_filename=args.output,
        template_txt=str(Path(args.prompt_file).resolve()), top_p=args.top_p,
        llm_model=args.llm_model, temperature=args.temperature, apply_quality_control=False)


def llm_crossover(args: argparse.Namespace) -> None:
    """LLM crossover using the existing templates and helpers.

    The original enumerates ``parts_x[1:]`` from 0 and then writes
    ``parts_x[augment_idx]``, i.e. one chunk before the one it showed the
    LLM, overwriting the import block (or policy class) of MuJoCo genes.
    """
    _, xmod, constants = _import_existing_operators()
    _banner(constants, args)
    _install_guards(constants, args)
    wait_for_llm_server(constants, args.server_timeout)
    _seed(args.seed)
    if args.crossover_impl == "original":
        xmod.augment_network(
            input_filename_x=args.input, input_filename_y=args.input_y,
            output_filename=args.output, top_p=args.top_p, llm_model=args.llm_model,
            temperature=args.temperature, apply_quality_control=False)
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
        txt2llm, augment_idx - 1, False, args.top_p, args.llm_model, args.temperature)
    if not code_from_llm:  # same fallback as the original
        code_from_llm = txt2llm
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
    child = mock_mutate_text(code, rng)
    if "MOCK_INVALID_CHILD" in code:  # test hook: imitate an LLM placeholder reply
        child = child.replace("def get_ppo_kwargs", "ERROR\ndef _gone")
    Path(args.output).write_text(child, encoding="utf-8")
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
    p.add_argument("--llm-model", default="llama3",
                   help="Passed to llm_utils.get_llm_code_generator (as run_improved.py --llm_model)")
    p.add_argument("--server-timeout", type=float,
                   default=float(os.getenv("LLM_SERVER_READY_TIMEOUT", str(4 * 3600))),
                   help="Seconds to wait for the local LLM server (covers a server restart)")
    p.add_argument("--llm-retries", type=int, default=3,
                   help="Attempts per request when the server returns nothing")
    p.add_argument("--extraction", choices=["robust", "original"], default="robust",
                   help="'original' keeps llm_utils.clean_code_from_llm (first code block)")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the child check (not recommended)")
    p.add_argument("--crossover-impl", choices=["fixed", "original"], default="fixed",
                   help="'original' calls src/llm_crossover.py unchanged (off-by-one chunk index)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operator == "crossover" and not args.input_y:
        raise SystemExit("crossover needs --input-y")
    if args.operator == "mutate" and args.backend == "llm" and not args.prompt_file:
        raise SystemExit("mutate with the llm backend needs --prompt-file")
    fn = {("mutate", "llm"): llm_mutate, ("crossover", "llm"): llm_crossover,
          ("mutate", "mock"): mock_mutate, ("crossover", "mock"): mock_crossover}
    fn[(args.operator, args.backend)](args)
    out = Path(args.output)
    if not out.exists():
        print(f"operator finished but {args.output} was not written", file=sys.stderr)
        return 2
    if args.no_validate:
        return 0
    ok, reason, info = validate_child(out)
    if not ok:
        rejected = out.with_name(out.stem + ".rejected.py")
        out.replace(rejected)
        print(f"INVALID CHILD: {reason} (kept as {rejected.name})", flush=True)
        return EXIT_INVALID_CHILD
    print(f"CHILD OK duplicate_definitions={info.get('duplicate_definitions', [])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
