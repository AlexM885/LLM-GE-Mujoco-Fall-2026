"""Per-generation VLM-authored auxiliary training reward emitter.

Reads the cached rollouts of a small, contrastive set of parents from the
current MAP-Elites archive (top-performers and bottom-performers), hands
them to a vision-language model as a single multi-image prompt, and asks
for a short Python snippet that would shape TD3 training reward in a way
orthogonal to Walker2d-v5's base reward.

Design decisions:

- One VLM call per generation (not per parent). Downstream TD3 training
  for every child in the generation uses the same snippet. This matches
  Eureka's reward-code-per-generation granularity and keeps VLM cost
  O(generations) rather than O(population).
- The parent set deliberately mixes top and bottom performers so Gemma
  sees a failure-mode CONTRAST. Pure top-parent observation gave zero
  signal in Neel's prior runs (see docs/neel_monday_standup_2026-10-12.md).
- The snippet is validated by the aux_reward sandbox before being written
  to the cache; a single repair-prompt retry is attempted if the first
  snippet is rejected.
- Zero-aux fallback: if the VLM output is still invalid after one repair,
  we save a cache record with ``aux_reward_code = <zero return>`` and
  ``fallback_reason`` set, so the training arm degrades to the control
  rather than failing. This keeps the pipeline resilient.

Companion modules:
- :mod:`behavior_observer` provides the HFObserver backend; we reuse it
  rather than reload a 27B model.
- :mod:`src.visual_feedback` provides the cache-key machinery; we reuse
  :func:`parent_metadata` to pin exactly what the snippet was derived from.
- :mod:`sota.MujocoRL.aux_reward_sandbox` provides the AST sandbox used
  to validate ``aux_reward_code``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Iterable

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_SRC_DIR = os.path.join(REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_MUJOCO_DIR = os.path.join(REPO_ROOT, "sota", "MujocoRL")
if _MUJOCO_DIR not in sys.path:
    sys.path.insert(0, _MUJOCO_DIR)

try:
    from src.visual_feedback import (
        parent_metadata, telemetry_brief, _cfg,
    )
    from src.behavior_observer import (
        HFObserver, load_capture, parse_observation_json, source_excerpt,
    )
except ImportError:  # pragma: no cover - path fallback
    from visual_feedback import parent_metadata, telemetry_brief, _cfg
    from behavior_observer import (
        HFObserver, load_capture, parse_observation_json, source_excerpt,
    )

try:
    from aux_reward_sandbox import AuxRewardSandboxError, validate_aux_reward_code
except ImportError:  # pragma: no cover
    from sota.MujocoRL.aux_reward_sandbox import (  # type: ignore
        AuxRewardSandboxError, validate_aux_reward_code,
    )

AUX_REWARD_SCHEMA = "aux-reward-cache-v1"
AUX_REWARD_PROMPT_REVISION = "v2-bonus"
TEMPLATE_ROOT = os.path.join(REPO_ROOT, "templates", "AuxReward")

ROLE_PREAMBLE = (
    "You are a reward-code co-designer for MuJoCo Walker2d-v5 TD3 training. "
    "Your output is executed inside an AST-whitelisted sandbox and only "
    "shapes training reward; it never affects reported fitness. Be concise "
    "and return strict JSON only."
)

#: Zero-aux fallback snippet installed when the VLM cannot produce a valid
#: one. Returning 0.0 makes the wrapper a passthrough.
ZERO_AUX_CODE = (
    "def aux_reward(obs, action, info):\n"
    "    return 0.0\n"
)


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
def load_prompt_template(name: str = "emit") -> str:
    path = os.path.join(TEMPLATE_ROOT, f"{name}.txt")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _parent_block(gene_id: str, telemetry: dict, fitness: float | None,
                  frame_indices: Iterable[int] | None = None) -> str:
    """Render one parent's rollout evidence as a prompt subsection."""
    fitness_str = f"{fitness:.1f}" if fitness is not None else "unknown"
    header = f"- gene {gene_id} (eval fitness {fitness_str}):"
    body = telemetry_brief(telemetry).rstrip()
    frames_hint = ""
    if frame_indices is not None:
        frames_hint = "\n    frame indices supplied to the VLM: " + \
                      ", ".join(str(i) for i in frame_indices)
    source = source_excerpt(gene_id, max_chars=1500).rstrip()
    source_block = ""
    if source:
        source_block = (
            "\n    source excerpt (gene file, truncated):\n"
            + "\n".join("    | " + line for line in source.splitlines())
        )
    return f"{header}\n{body}{frames_hint}{source_block}"


def build_user_text(
    template: str,
    *,
    top_parents: list[dict],
    bottom_parents: list[dict],
    generation: int | None,
    best_so_far: float | None,
    archive_coverage: int | None,
    reflection: dict | None,
) -> str:
    """Substitute into the aux_reward emit template.

    ``top_parents`` and ``bottom_parents`` are lists of dicts with
    ``gene_id``, ``telemetry``, optional ``fitness``, optional
    ``frame_indices`` (what the VLM will see from this parent).
    """
    gen_lines = []
    if generation is not None:
        gen_lines.append(f"- generation: {generation}")
    if best_so_far is not None:
        gen_lines.append(f"- best fitness so far: {best_so_far:.2f}")
    if archive_coverage is not None:
        gen_lines.append(f"- archive cells occupied: {archive_coverage}")
    gen_lines.append(f"- env: Walker2d-v5, max rollout 1000 steps")
    gen_block = "\n".join(gen_lines) if gen_lines else "  (no generation context)"

    top_block = (
        "\n".join(
            _parent_block(p["gene_id"], p["telemetry"], p.get("fitness"),
                          p.get("frame_indices"))
            for p in top_parents
        ) or "  (no top parents available)"
    )
    bot_block = (
        "\n".join(
            _parent_block(p["gene_id"], p["telemetry"], p.get("fitness"),
                          p.get("frame_indices"))
            for p in bottom_parents
        ) or "  (no bottom parents available)"
    )

    reflection_block = _reflection_block(reflection)

    text = template
    text = text.replace("<<ROLE_PREAMBLE>>", ROLE_PREAMBLE)
    text = text.replace("<<GEN_CONTEXT>>", gen_block)
    text = text.replace("<<TOP_PARENTS>>", top_block)
    text = text.replace("<<BOTTOM_PARENTS>>", bot_block)
    text = text.replace("<<REFLECTION_BLOCK>>", reflection_block)
    return text


def _reflection_block(reflection: dict | None) -> str:
    """Render Eureka-style reflection on the previous generation's aux reward.

    When ``reflection`` is ``None`` (first generation, or no prior cache),
    we emit a placeholder so the template stays legible.
    """
    if not reflection:
        return "PRIOR AUX REWARD\n  (none; this is the first aux reward for this run)"
    lines = ["PRIOR AUX REWARD (from the previous generation):"]
    code = reflection.get("aux_reward_code", "")
    if code:
        lines.append("    code:")
        for line in code.splitlines():
            lines.append("      " + line)
    lam = reflection.get("lam")
    if lam is not None:
        lines.append(f"    lam used: {lam}")
    rationale = reflection.get("rationale")
    if rationale:
        lines.append(f"    rationale: {rationale}")
    impact = reflection.get("impact")
    if impact:
        lines.append(
            f"    measured impact on children: "
            f"mean child reward {impact.get('mean_child_reward', 'n/a')}, "
            f"delta vs control baseline "
            f"{impact.get('delta_vs_control', 'n/a')}")
    lines.append(
        "  If the measured impact above was negative or small, choose a "
        "substantially different signal this time; do not repeat the same "
        "snippet verbatim.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Output parsing + validation
# --------------------------------------------------------------------------- #
AUX_RESPONSE_REQUIRED = ("aux_reward_code", "lam", "rationale")


def parse_vlm_aux_response(text: str) -> dict:
    """Parse the VLM's response into our aux-reward record.

    Reuses behavior_observer.parse_observation_json's forgiving JSON
    extractor (it tolerates ``\\u0060\\u0060\\u0060json`` fences and prose
    around the object), then does our own shape checks.
    """
    try:
        obj = parse_observation_json(text)
    except ValueError as exc:
        raise ValueError(f"could not parse JSON from VLM response: {exc}") from exc
    for key in AUX_RESPONSE_REQUIRED:
        if key not in obj:
            raise ValueError(f"VLM response missing required key: {key}")
    code = obj.get("aux_reward_code")
    if not isinstance(code, str):
        raise ValueError("aux_reward_code must be a string")
    try:
        lam = float(obj.get("lam"))
    except (TypeError, ValueError) as exc:
        raise ValueError("lam must be a float") from exc
    anneal = obj.get("anneal_steps", 0)
    try:
        anneal = int(anneal) if anneal is not None else 0
    except (TypeError, ValueError):
        anneal = 0
    return {
        "aux_reward_code": _normalize_code(code),
        "lam": max(0.0, min(1.0, lam)),
        "anneal_steps": max(0, anneal),
        "rationale": str(obj.get("rationale") or "").strip(),
    }


_CODE_FENCE_RE = re.compile(r"^```(?:python)?\s*|\s*```$", re.MULTILINE)


def _normalize_code(code: str) -> str:
    """Strip surrounding whitespace, fences, and ensure a trailing newline.

    Some VLMs wrap the code in a ```python fence when asked for a JSON
    string; others don't. Normalise either way.
    """
    text = _CODE_FENCE_RE.sub("", code).strip()
    if not text.endswith("\n"):
        text += "\n"
    return text


# --------------------------------------------------------------------------- #
# Emission pipeline
# --------------------------------------------------------------------------- #
def _frame_subset(frame_files: list[str], max_frames: int) -> list[str]:
    """Pick ``max_frames`` evenly spaced frames from the capture."""
    if not frame_files:
        return []
    if len(frame_files) <= max_frames:
        return list(frame_files)
    step = len(frame_files) / max_frames
    return [frame_files[int(i * step)] for i in range(max_frames)]


def _parent_evidence(
    gene_id: str, max_frames_per_parent: int,
    capture_root: str | None = None,
) -> dict | None:
    """Return the dict shape ``build_user_text`` expects for one parent.

    ``None`` means the parent has no cached rollout and should be skipped.
    ``capture_root`` lets the caller override the FEEDBACK_CAPTURE_DIR
    default so the emitter can pick up captures written anywhere on disk
    (e.g. the pilot's shared scratch root).
    """
    _, _, bundle = load_capture(gene_id, capture_root=capture_root)
    if bundle is None:
        return None
    frame_files = _frame_subset(bundle["frame_files"], max_frames_per_parent)
    frame_indices = list(range(len(frame_files)))
    return {
        "gene_id": gene_id,
        "telemetry": bundle["telemetry"],
        "frame_files": frame_files,
        "frame_indices": frame_indices,
    }


def _attach_fitness(parent_dicts: list[dict], fitness_map: dict | None) -> None:
    if not fitness_map:
        return
    for p in parent_dicts:
        gid = p["gene_id"]
        if gid in fitness_map:
            p["fitness"] = float(fitness_map[gid])


def _collect_frame_messages(parent_dicts: list[dict]) -> list[str]:
    """Flatten frames across parents for the HFObserver backend."""
    frames: list[str] = []
    for p in parent_dicts:
        frames.extend(p.get("frame_files", []))
    return frames


def _prompt_hash(user_text: str) -> str:
    return hashlib.sha256(user_text.encode("utf-8")).hexdigest()


def _parent_hashes(parent_dicts: list[dict]) -> list[str]:
    out = []
    for p in parent_dicts:
        meta = parent_metadata(p["gene_id"])
        if meta is None:
            continue
        out.append(f"{p['gene_id']}:{meta['genome_sha256'][:12]}")
    return out


def emit_aux_reward(
    *,
    top_gene_ids: list[str],
    bottom_gene_ids: list[str],
    generation: int | None = None,
    best_so_far: float | None = None,
    archive_coverage: int | None = None,
    fitness_map: dict | None = None,
    reflection: dict | None = None,
    observer: HFObserver | None = None,
    backend: str = "hf",
    model_id: str | None = None,
    model_revision: str | None = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 1024,
    max_frames_per_parent: int = 8,
    prompt_revision: str = AUX_REWARD_PROMPT_REVISION,
    allow_repair_retry: bool = True,
    capture_root: str | None = None,
) -> dict:
    """Produce one aux_reward cache record for the current generation.

    Returns a dict ready to be serialised to the cache path that
    :mod:`train_rl_vlm` reads. The record always contains a valid
    ``aux_reward_code``, even if that is the zero-aux fallback.
    """
    template = load_prompt_template("emit")

    top_parents = [p for gid in top_gene_ids
                   if (p := _parent_evidence(
                       gid, max_frames_per_parent, capture_root=capture_root))]
    bottom_parents = [p for gid in bottom_gene_ids
                      if (p := _parent_evidence(
                          gid, max_frames_per_parent,
                          capture_root=capture_root))]
    _attach_fitness(top_parents, fitness_map)
    _attach_fitness(bottom_parents, fitness_map)

    if not top_parents and not bottom_parents:
        return _fallback_record(
            reason="no parent captures available",
            generation=generation, best_so_far=best_so_far,
            archive_coverage=archive_coverage, model_id=model_id,
            prompt_revision=prompt_revision,
            parent_set={"top": top_gene_ids, "bottom": bottom_gene_ids},
        )

    user_text = build_user_text(
        template,
        top_parents=top_parents, bottom_parents=bottom_parents,
        generation=generation, best_so_far=best_so_far,
        archive_coverage=archive_coverage, reflection=reflection,
    )
    prompt_hash = _prompt_hash(user_text)
    frames = _collect_frame_messages(top_parents + bottom_parents)

    if observer is None and backend == "hf":
        resolved_model = model_id or _cfg("OBSERVER_MODEL_ID",
                                          "google/gemma-3-27b-it")
        resolved_revision = model_revision or _cfg("OBSERVER_MODEL_REVISION",
                                                   "main")
        observer = HFObserver(
            model_id=resolved_model,
            revision=resolved_revision,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )

    attempt = 1
    validation = "unknown"
    fallback_reason: str | None = None
    parsed: dict | None = None
    raw_response: str = ""

    for attempt in (1, 2):
        try:
            raw_response = _call_observer(
                observer, backend, user_text, frames)
            parsed = parse_vlm_aux_response(raw_response)
            ok, reason = validate_aux_reward_code(parsed["aux_reward_code"])
            if ok:
                validation = "ok" if attempt == 1 else "repaired"
                break
            if not allow_repair_retry or attempt == 2:
                fallback_reason = f"sandbox rejected after {attempt} attempts: {reason}"
                parsed = None
                break
            repair_tpl = load_prompt_template("repair")
            user_text = (
                repair_tpl
                .replace("<<ERROR>>", reason)
                .replace("<<PREVIOUS_RESPONSE>>", raw_response[:2000])
            )
            # Second call retains the same frames; context already includes
            # parent evidence.
        except ValueError as exc:
            if not allow_repair_retry or attempt == 2:
                fallback_reason = f"could not parse VLM response: {exc}"
                parsed = None
                break
            repair_tpl = load_prompt_template("repair")
            user_text = (
                repair_tpl
                .replace("<<ERROR>>", str(exc))
                .replace("<<PREVIOUS_RESPONSE>>", raw_response[:2000])
            )
        except Exception as exc:  # pylint: disable=broad-except
            fallback_reason = f"VLM call failed: {exc}"
            parsed = None
            break

    if parsed is None:
        return _fallback_record(
            reason=fallback_reason or "unknown emission failure",
            generation=generation, best_so_far=best_so_far,
            archive_coverage=archive_coverage, model_id=model_id,
            prompt_revision=prompt_revision,
            parent_set={"top": top_gene_ids, "bottom": bottom_gene_ids},
            prompt_hash=prompt_hash,
            parent_hashes=_parent_hashes(top_parents + bottom_parents),
            attempts=attempt,
            raw_response=raw_response,
        )

    return {
        "schema": AUX_REWARD_SCHEMA,
        "aux_reward_code": parsed["aux_reward_code"],
        "lam": parsed["lam"],
        "anneal_steps": parsed["anneal_steps"],
        "rationale": parsed["rationale"],
        "generation": generation,
        "best_so_far": best_so_far,
        "archive_coverage": archive_coverage,
        "parent_set": {"top": top_gene_ids, "bottom": bottom_gene_ids},
        "parent_hashes": _parent_hashes(top_parents + bottom_parents),
        "model_id": model_id or _cfg("OBSERVER_MODEL_ID",
                                     "google/gemma-3-27b-it"),
        "model_revision": model_revision or _cfg("OBSERVER_MODEL_REVISION",
                                                 "main"),
        "prompt_revision": prompt_revision,
        "prompt_hash": prompt_hash,
        "attempts": attempt,
        "validation": validation,
        "fallback_reason": None,
        "raw_response": raw_response[:4000],
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def _call_observer(
    observer: HFObserver | None,
    backend: str,
    user_text: str,
    frames: list[str],
) -> str:
    """Invoke the chosen backend and return the raw text response."""
    if backend == "mock":
        return _mock_vlm_response(user_text, frames)
    if observer is None:
        raise RuntimeError("hf backend requires an initialised HFObserver")
    return observer.observe(
        system_prompt=ROLE_PREAMBLE,
        user_text=user_text,
        frame_files=frames,
        mode="visual" if frames else "text",
    )


def _mock_vlm_response(_user_text: str, _frames: list[str]) -> str:
    """Deterministic response for the mock backend; used by tests and smoke runs.

    Returns a safe, sandbox-valid aux_reward snippet that penalises pitch
    wobble and asymmetry. Not meant for real experiments.
    """
    return json.dumps({
        "aux_reward_code": (
            "def aux_reward(obs, action, info):\n"
            "    symmetry = -float(np.sum(np.abs(obs[2:5] - obs[5:8])))\n"
            "    wobble = -float(np.square(obs[10]))\n"
            "    return symmetry + 0.5 * wobble\n"
        ),
        "lam": 0.1,
        "anneal_steps": 0,
        "rationale": "mock heuristic: punish asymmetric gaits and pitch wobble",
    })


def _fallback_record(
    *,
    reason: str,
    generation: int | None,
    best_so_far: float | None,
    archive_coverage: int | None,
    model_id: str | None,
    prompt_revision: str,
    parent_set: dict,
    prompt_hash: str | None = None,
    parent_hashes: list[str] | None = None,
    attempts: int = 0,
    raw_response: str = "",
) -> dict:
    """Record emitted when we could not get a valid snippet from the VLM.

    The ``aux_reward_code`` is the zero-aux fallback so ``train_rl_vlm``
    treats the arm identically to the no-feedback control. The reason is
    captured so a reviewer can tell this run should not count toward
    "VLM arm improved X".
    """
    return {
        "schema": AUX_REWARD_SCHEMA,
        "aux_reward_code": ZERO_AUX_CODE,
        "lam": 0.0,
        "anneal_steps": 0,
        "rationale": "fallback: zero aux_reward (no shaping applied)",
        "generation": generation,
        "best_so_far": best_so_far,
        "archive_coverage": archive_coverage,
        "parent_set": parent_set,
        "parent_hashes": parent_hashes or [],
        "model_id": model_id or _cfg("OBSERVER_MODEL_ID",
                                     "google/gemma-3-27b-it"),
        "model_revision": _cfg("OBSERVER_MODEL_REVISION", "main"),
        "prompt_revision": prompt_revision,
        "prompt_hash": prompt_hash,
        "attempts": attempts,
        "validation": "fallback",
        "fallback_reason": reason,
        "raw_response": (raw_response or "")[:4000],
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def write_cache_record(record: dict, cache_path: str) -> str:
    """Atomically write a cache record to disk."""
    target_dir = os.path.dirname(os.path.abspath(cache_path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    os.replace(tmp, cache_path)
    return cache_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_gene_list(value: str) -> list[str]:
    return [g.strip() for g in value.split(",") if g.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit one VLM aux_reward cache record for a generation")
    parser.add_argument("--top-gene-ids", type=str, required=True,
                        help="Comma-separated list of top-performing parent gene ids")
    parser.add_argument("--bottom-gene-ids", type=str, default="",
                        help="Comma-separated list of bottom-performing parent gene ids")
    parser.add_argument("--cache-path", type=str, required=True,
                        help="Where to write the aux_reward cache JSON")
    parser.add_argument("--generation", type=int, default=None)
    parser.add_argument("--best-so-far", type=float, default=None)
    parser.add_argument("--archive-coverage", type=int, default=None)
    parser.add_argument("--fitness-map", type=str, default=None,
                        help='Optional JSON map {"gene_id": fitness, ...}')
    parser.add_argument("--reflection-path", type=str, default=None,
                        help="Path to the previous generation's cache record for Eureka reflection")
    parser.add_argument("--backend", choices=("hf", "mock"), default="hf")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--model-revision", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-frames-per-parent", type=int, default=8)
    parser.add_argument("--prompt-revision", type=str,
                        default=AUX_REWARD_PROMPT_REVISION)
    parser.add_argument("--no-repair-retry", action="store_true")
    parser.add_argument("--capture-root", type=str,
                        default=os.environ.get("FEEDBACK_CAPTURE_DIR"),
                        help="Override the FEEDBACK_CAPTURE_DIR default; "
                             "defaults to $FEEDBACK_CAPTURE_DIR if set.")
    args = parser.parse_args()

    fitness_map = None
    if args.fitness_map:
        with open(args.fitness_map, "r", encoding="utf-8") as handle:
            fitness_map = json.load(handle)
    reflection = None
    if args.reflection_path and os.path.isfile(args.reflection_path):
        with open(args.reflection_path, "r", encoding="utf-8") as handle:
            reflection = json.load(handle)

    start = time.time()
    record = emit_aux_reward(
        top_gene_ids=_parse_gene_list(args.top_gene_ids),
        bottom_gene_ids=_parse_gene_list(args.bottom_gene_ids),
        generation=args.generation,
        best_so_far=args.best_so_far,
        archive_coverage=args.archive_coverage,
        fitness_map=fitness_map,
        reflection=reflection,
        backend=args.backend,
        model_id=args.model_id,
        model_revision=args.model_revision,
        max_new_tokens=args.max_new_tokens,
        max_frames_per_parent=args.max_frames_per_parent,
        prompt_revision=args.prompt_revision,
        allow_repair_retry=not args.no_repair_retry,
        capture_root=args.capture_root,
    )
    record["duration_sec"] = round(time.time() - start, 3)
    path = write_cache_record(record, args.cache_path)
    print(f"VLM aux_reward cache written to {path}")
    print(f"  validation: {record['validation']}")
    print(f"  lam: {record['lam']}  anneal_steps: {record['anneal_steps']}")
    if record.get("fallback_reason"):
        print(f"  fallback_reason: {record['fallback_reason']}")


if __name__ == "__main__":
    main()
