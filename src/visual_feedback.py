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


def structured_planner_enabled():
    value = _cfg("FEEDBACK_STRUCTURED_PLANNER", False)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


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
        "render_stride": int(_cfg("FEEDBACK_RENDER_STRIDE", 5)),
        "frame_sampling": _cfg("FEEDBACK_FRAME_SAMPLING", "bookend"),
        "observer_model_id": _cfg("OBSERVER_MODEL_ID", "google/gemma-3-27b-it"),
        "observer_model_revision": _cfg("OBSERVER_MODEL_REVISION", "main"),
        "prompt_revision": _cfg("FEEDBACK_PROMPT_REVISION", "v1"),
        "structured_planner": structured_planner_enabled(),
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
    plan = record.get("mutation_plan")
    if plan is not None:
        if not isinstance(plan, dict):
            return False, "mutation_plan must be a dict"
        for key in ("failure_mode", "mutation_goal", "priority_metric"):
            if not isinstance(plan.get(key), str) or not plan[key].strip():
                return False, f"mutation_plan.{key} must be a non-empty string"
        for key in ("evidence", "allowed_code_changes", "forbidden_changes"):
            values = plan.get(key)
            if not isinstance(values, list) or not values:
                return False, f"mutation_plan.{key} must be a non-empty list"
            if any(not isinstance(item, str) or not item.strip() for item in values):
                return False, f"mutation_plan.{key} must contain non-empty strings"
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
    if isinstance(normalized.get("mutation_plan"), dict):
        normalized["mutation_plan"] = normalize_mutation_plan(
            normalized["mutation_plan"], normalized, metadata.get("telemetry"))
    for key, value in metadata.items():
        normalized[key] = value
    return normalized


def normalize_mutation_plan(plan, record=None, telemetry=None):
    """Clamp a VLM-generated planner response into a small, safe schema."""
    fallback = deterministic_mutation_plan(record or {}, telemetry or {})
    if not isinstance(plan, dict):
        return fallback

    normalized = {}
    for key in ("failure_mode", "mutation_goal", "priority_metric"):
        value = str(plan.get(key, fallback[key])).strip()
        normalized[key] = value[:240] if value else fallback[key]
    for key in ("evidence", "allowed_code_changes", "forbidden_changes"):
        values = plan.get(key, fallback[key])
        if not isinstance(values, list):
            values = fallback[key]
        cleaned = []
        for item in values[:6]:
            text = str(item).strip()
            if text:
                cleaned.append(text[:220])
        normalized[key] = cleaned or fallback[key]
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


def mutation_focus(record, telemetry):
    """Actionable but bounded target for the code editor.

    The VLM remains an observer; this derived block converts observation and
    deterministic telemetry into one engineering objective. It deliberately
    avoids naming a specific hyperparameter so the edit is still discovered by
    training/evaluation rather than prescribed by the observer.
    """
    episodes = telemetry.get("episodes", []) if telemetry else []
    if not episodes:
        return [
            "  objective: make the smallest plausible robustness improvement",
            "  reason: no rollout telemetry was available to localize failure",
            "  avoid: broad rewrites or changing the environment/reward",
        ]

    mean_return = sum(float(ep.get("return", 0.0) or 0.0) for ep in episodes) / len(episodes)
    mean_steps = sum(float(ep.get("steps", 0.0) or 0.0) for ep in episodes) / len(episodes)
    mean_distance = sum(float(ep.get("distance", 0.0) or 0.0) for ep in episodes) / len(episodes)
    contacts = []
    for ep in episodes:
        c0 = ep.get("foot_contact_0")
        c1 = ep.get("foot_contact_1")
        if c0 is not None and c1 is not None:
            contacts.append(abs(float(c0) - float(c1)))
    contact_gap = sum(contacts) / len(contacts) if contacts else 0.0

    visual_text = " ".join(
        str(record.get(key, "")).lower()
        for key in ("visible_posture", "motion_pattern", "instability", "notes")
    )
    collapse_seen = any(term in visual_text for term in (
        "collapse", "fall", "tips", "tipping", "drops", "loss of balance",
        "unstable", "wobble",
    ))

    if mean_steps < 250 or collapse_seen:
        objective = "increase early-episode balance and healthy-duration stability"
        reason = (
            f"mean rollout length is {_fmt_float(mean_steps, 1)} steps and the "
            "visual/telemetry evidence indicates early instability"
        )
        avoid = "do not chase speed until the walker survives longer"
    elif mean_distance < 3.0 or mean_return < 700:
        objective = "convert a mostly stationary gait into forward locomotion"
        reason = (
            f"mean distance is {_fmt_float(mean_distance, 2)} with mean return "
            f"{_fmt_float(mean_return, 1)}, so survival is not becoming travel"
        )
        avoid = "do not add large networks; prefer one targeted policy/training change"
    elif contact_gap > 0.25:
        objective = "reduce left-right foot-contact imbalance while preserving forward speed"
        reason = f"mean foot-contact asymmetry is {_fmt_float(contact_gap, 3)}"
        avoid = "do not collapse both contacts to a standing/stalling policy"
    else:
        objective = "improve forward speed while preserving the observed stable gait"
        reason = (
            f"rollouts last {_fmt_float(mean_steps, 1)} steps on average and "
            f"travel {_fmt_float(mean_distance, 2)} distance"
        )
        avoid = "do not destabilize posture for a speculative speed gain"

    return [
        f"  objective: {objective}",
        f"  reason: {reason}",
        f"  avoid: {avoid}",
        "  edit scope: one bounded code change; keep reward/env/evaluation unchanged",
    ]


def deterministic_mutation_plan(record, telemetry):
    """Structured fallback used when the VLM does not provide a valid plan."""
    focus_lines = mutation_focus(record or {}, telemetry or {})
    objective = focus_lines[0].split("objective:", 1)[-1].strip()
    reason = focus_lines[1].split("reason:", 1)[-1].strip()
    avoid = focus_lines[2].split("avoid:", 1)[-1].strip()
    return {
        "failure_mode": objective,
        "evidence": [
            reason,
            str((record or {}).get("instability", "observer instability unknown"))[:220],
        ],
        "mutation_goal": objective,
        "allowed_code_changes": [
            "make one bounded change to TD3/PPO training hyperparameters",
            "adjust training-only reward shaping or exploration schedule if already present",
            "preserve the model interface and existing evaluation script",
        ],
        "forbidden_changes": [
            "do not change the MuJoCo environment or final evaluation reward",
            "do not hard-code actions, trajectories, seeds, or evaluation outputs",
            avoid,
        ],
        "priority_metric": "mean evaluation reward after training, with episode length as a stability check",
    }


def build_feedback_text(record, telemetry):
    """The text block appended to a mutation prompt.

    Ordering keeps the code rules intact: the caller appends this after the
    prompt body (and its ``{}`` code slot), so the code chunk still precedes the
    feedback and the hard rules are unchanged.
    """
    mode = record.get("mode", "visual")
    plan = record.get("mutation_plan")
    if not isinstance(plan, dict):
        plan = deterministic_mutation_plan(record, telemetry)

    lines = [
        "",
        "Additional evidence about the parent policy's measured behaviour.",
        "Use it to choose one bounded code change. Treat the mutation plan as",
        "a hypothesis, not proof; only a trained child can confirm a change.",
        "",
        "Numerical telemetry from deterministic rollouts:",
        telemetry_brief(telemetry),
        "",
        "Structured mutation plan:",
        f"  failure_mode: {plan.get('failure_mode', '')}",
        "  evidence:",
        *[f"    - {item}" for item in plan.get("evidence", [])],
        f"  mutation_goal: {plan.get('mutation_goal', '')}",
        "  allowed_code_changes:",
        *[f"    - {item}" for item in plan.get("allowed_code_changes", [])],
        "  forbidden_changes:",
        *[f"    - {item}" for item in plan.get("forbidden_changes", [])],
        f"  priority_metric: {plan.get('priority_metric', '')}",
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
