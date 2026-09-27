"""Sparse, cached visual feedback for LLM-guided policy mutation.

This module is deliberately dependency-light: ``run_improved.py`` imports it on
the (CPU-only) evolution controller to read cached observations, so it must not
pull in torch/transformers. The heavy work lives in:

    sota/MujocoRL/capture_rollout.py   render ordered frames + telemetry
    src/behavior_observer.py           frozen VLM -> structured observation

Both producer and consumer agree on a *behaviour cache key* so a parent's
observation is computed once and reused by every mutation of that parent. The
key mixes the trained-checkpoint hash, the genome source hash, the rollout
seeds, camera/frame settings, the observer model revision, and the prompt
revision (see RESEARCH_DIRECTION.md).

Cache layout (``FEEDBACK_CACHE_DIR``)::

    <cache>/<cache_key>.json     one observation record per (parent, config)

Capture layout (``FEEDBACK_CAPTURE_DIR``)::

    <captures>/<parent_gene>/manifest.json
    <captures>/<parent_gene>/telemetry.json
    <captures>/<parent_gene>/frames/frame_0000.png ...
"""

import hashlib
import json
import os

try:  # imported as ``src.visual_feedback`` from run_improved.py
    from src.cfg import constants as C
except Exception:  # noqa: BLE001 - allow pure cache logic to import without torch
    try:  # imported as ``visual_feedback`` from src/*.py jobs
        from cfg import constants as C
    except Exception:  # noqa: BLE001
        # Minimal namespace so schema/cache tests run without the full
        # evolution environment (torch, pyyaml). Real runs always have config.
        class _FallbackConfig:
            pass

        C = _FallbackConfig()

SCHEMA_VERSION = "1"
CACHE_SCHEMA = "behavior-cache-v1"

#: Strict, machine-checkable shape of an observer response.
REQUIRED_OBSERVATION_KEYS = (
    "determinable",
    "visible_posture",
    "motion_pattern",
    "instability",
    "supporting_frames",
    "uncertainty",
)
ALLOWED_UNCERTAINTY = ("low", "medium", "high")


def _cfg(name, default=None):
    return getattr(C, name, default)


def _seeds():
    base = int(_cfg("FEEDBACK_SEED_BASE", 1000))
    episodes = int(_cfg("FEEDBACK_EPISODES", 2))
    return [base + i for i in range(episodes)]


def sha256_file(path, chunk_size=1 << 20):
    """Content hash of a file, or None when the path is missing/unreadable."""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def genome_path(parent_gene):
    variant_dir = _cfg("VARIANT_DIR", "sota/MujocoRL/models")
    model = _cfg("MODEL", "network")
    return os.path.join(variant_dir, f"{model}_{parent_gene}.py")


def checkpoint_path(parent_gene):
    model_dir = _cfg("MUJOCO_TRAINED_MODEL_DIR", "sota/MujocoRL/trained_models")
    model = _cfg("MODEL", "network")
    actual_path = os.path.join(model_dir, f"{parent_gene}.zip")
    legacy_path = os.path.join(model_dir, f"{model}_{parent_gene}.zip")
    return actual_path if os.path.isfile(actual_path) else legacy_path


def parent_metadata(parent_gene, mode=None):
    """Configuration identity for one parent, shared by capture/observe/controller.

    Returns None when the parent has no genome file on disk, because without it
    the cache key would not uniquely identify the behaviour's source.
    """
    source = genome_path(parent_gene)
    genome_sha = sha256_file(source)
    if genome_sha is None:
        return None
    return {
        "parent_gene": parent_gene,
        "genome_sha256": genome_sha,
        "checkpoint_sha256": sha256_file(checkpoint_path(parent_gene)),
        "env_id": _cfg("MUJOCO_ENV_ID", "Walker2d-v5"),
        "episodes": int(_cfg("FEEDBACK_EPISODES", 2)),
        "max_steps": int(_cfg("FEEDBACK_MAX_STEPS", 1000)),
        "seeds": _seeds(),
        "frames": int(_cfg("FEEDBACK_FRAMES", 16)),
        "camera": _cfg("FEEDBACK_CAMERA", "track"),
        "image_max_side": int(_cfg("FEEDBACK_IMAGE_MAX_SIDE", 336)),
        "observer_model_id": _cfg("OBSERVER_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct"),
        "observer_model_revision": _cfg("OBSERVER_MODEL_REVISION", "main"),
        "prompt_revision": _cfg("FEEDBACK_PROMPT_REVISION", "v1"),
        "mode": (mode or _cfg("FEEDBACK_MODE", "off")).strip().lower(),
    }


def compute_cache_key(metadata):
    """Deterministic cache key for a capture/observation configuration."""
    payload = {"schema": CACHE_SCHEMA}
    payload.update(metadata)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cache_key_for_parent(parent_gene, mode=None):
    metadata = parent_metadata(parent_gene, mode=mode)
    if metadata is None:
        return None, None
    return compute_cache_key(metadata), metadata


def cache_dir():
    return _cfg("FEEDBACK_CACHE_DIR", "sota/MujocoRL/behavior_cache")


def capture_root():
    return _cfg("FEEDBACK_CAPTURE_DIR", "sota/MujocoRL/behavior_captures")


def observation_cache_path(cache_key, directory=None):
    return os.path.join(directory or cache_dir(), f"{cache_key}.json")


def capture_dir(parent_gene, root=None):
    return os.path.join(root or capture_root(), parent_gene)


# --------------------------------------------------------------------------- #
# Observation schema
# --------------------------------------------------------------------------- #
def validate_observation(record):
    """Return (ok, reason). Rejects malformed observer output, not uncertainty."""
    if not isinstance(record, dict):
        return False, "record is not a dict"
    for key in REQUIRED_OBSERVATION_KEYS:
        if key not in record:
            return False, f"missing key: {key}"
    if not isinstance(record["determinable"], bool):
        return False, "determinable must be a bool"
    for key in ("visible_posture", "motion_pattern", "instability"):
        if not isinstance(record[key], str) or not record[key].strip():
            return False, f"{key} must be a non-empty string"
    frames = record["supporting_frames"]
    if not isinstance(frames, list) or any(
            not isinstance(f, int) or f < 0 for f in frames):
        return False, "supporting_frames must be a list of non-negative ints"
    if record["uncertainty"] not in ALLOWED_UNCERTAINTY:
        return False, f"uncertainty must be one of {ALLOWED_UNCERTAINTY}"
    return True, ""


def normalize_observation(record, **metadata):
    """Fill envelope fields and clamp fields so a record always validates."""
    normalized = dict(record)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized.setdefault("notes", "")
    if normalized.get("uncertainty") not in ALLOWED_UNCERTAINTY:
        normalized["uncertainty"] = "high"
    if not isinstance(normalized.get("determinable"), bool):
        normalized["determinable"] = False
    if not isinstance(normalized.get("supporting_frames"), list):
        normalized["supporting_frames"] = []
    normalized["supporting_frames"] = [
        int(f) for f in normalized["supporting_frames"]
        if isinstance(f, (int, float)) and int(f) >= 0
    ]
    for key, value in metadata.items():
        normalized[key] = value
    return normalized


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #
def _fmt_float(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def telemetry_brief(telemetry):
    """Compact numerical context shared by the text and visual conditions.

    The text control must always receive everything the visual condition used
    to choose its frames, so this is rendered for both modes.
    """
    if not telemetry:
        return "  (no telemetry recorded)"
    lines = []
    episodes = telemetry.get("episodes", [])
    for ep in episodes:
        idx = ep.get("index", "?")
        lines.append(
            f"  episode {idx}: seed={ep.get('seed')} "
            f"return={_fmt_float(ep.get('return'), 2)} "
            f"steps={ep.get('steps')} "
            f"terminated={ep.get('terminated')}")
        lines.append(
            f"    distance={_fmt_float(ep.get('distance'), 2)} "
            f"mean_torso_z={_fmt_float(ep.get('mean_torso_z'))} "
            f"final_torso_z={_fmt_float(ep.get('final_torso_z'))} "
            f"final_pitch={_fmt_float(ep.get('final_pitch'))}")
        lines.append(
            f"    foot_contact=({_fmt_float(ep.get('foot_contact_0'))}, "
            f"{_fmt_float(ep.get('foot_contact_1'))}) "
            f"mean_ctrl_cost={_fmt_float(ep.get('mean_control_cost'), 4)} "
            f"termination_step={ep.get('termination_step')}")
    return "\n".join(lines) if lines else "  (no telemetry recorded)"


def build_feedback_text(record, telemetry):
    """The text block appended to a mutation prompt.

    Ordering keeps the code rules intact: the caller appends this after the
    prompt body (and its ``{}`` code slot), so the code chunk still precedes the
    feedback and the hard rules are unchanged.
    """
    mode = record.get("mode", "visual")
    lines = [
        "",
        "Additional evidence about the parent policy's measured behaviour.",
        "Use it only to choose one bounded code change. Do not treat it as an",
        "explanation of which hyperparameter is wrong; only a trained child can",
        "confirm a change.",
        "",
        "Numerical telemetry from deterministic rollouts:",
        telemetry_brief(telemetry),
    ]
    if mode == "visual":
        if record.get("determinable"):
            lines += [
                "",
                "Visual observation of ordered frames (frame index 0 is earliest):",
                f"  posture: {record.get('visible_posture', '')}",
                f"  motion: {record.get('motion_pattern', '')}",
                f"  instability: {record.get('instability', '')}",
                f"  supporting frames: {record.get('supporting_frames', [])}",
                f"  uncertainty: {record.get('uncertainty', 'high')}",
            ]
        else:
            lines += [
                "",
                "Visual observation: the behaviour could not be determined from",
                "the sampled frames. Rely on the numerical telemetry only.",
            ]
    if record.get("notes"):
        lines += ["", f"Observer notes: {record['notes']}"]
    lines += [
        "",
        "Hard rules above still apply. Return exactly one complete Python code",
        "block replacing only the shown chunk.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cache I/O and controller-side resolution
# --------------------------------------------------------------------------- #
def load_cached_observation(cache_key, directory=None):
    path = observation_cache_path(cache_key, directory)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    ok, _ = validate_observation(record)
    return record if ok else None


def save_observation(record, directory=None):
    """Atomically persist an observation record, keyed by its cache_key."""
    cache_key = record.get("cache_key")
    if not cache_key:
        raise ValueError("observation record is missing cache_key")
    target_dir = directory or cache_dir()
    os.makedirs(target_dir, exist_ok=True)
    path = observation_cache_path(cache_key, target_dir)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def format_and_store(record, telemetry, cache_directory=None, **metadata):
    """Build feedback_text, normalise the record, and save it to the cache."""
    normalized = normalize_observation(record, **metadata)
    normalized["feedback_text"] = build_feedback_text(normalized, telemetry)
    path = save_observation(normalized, directory=cache_directory)
    return normalized, path


def resolve_feedback(parent_gene, mode=None, cache_directory=None):
    """Controller-side lookup: return (feedback_text, cache_key, record) or None.

    Never raises on a missing parent or missing cache: the evolution loop must
    keep running. When the configured mode is "off", feedback is disabled.
    """
    active_mode = (mode or _cfg("FEEDBACK_MODE", "off")).strip().lower()
    if active_mode == "off":
        return None
    cache_key, metadata = cache_key_for_parent(parent_gene, mode=active_mode)
    if cache_key is None:
        return None
    record = load_cached_observation(cache_key, directory=cache_directory)
    if record is None:
        return None
    text = record.get("feedback_text")
    if not text:
        text = build_feedback_text(record, None)
    return text, cache_key, record
