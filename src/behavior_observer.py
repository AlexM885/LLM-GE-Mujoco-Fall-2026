"""Turn a captured parent rollout into a cached, structured behaviour record.

This is the VLM half of the sparse-visual-feedback method. It is intentionally a
separate process/environment from both the evolution controller (CPU, no torch)
and the text code generator (see ``server.py``), so a VLM dependency conflict
cannot break evolution. Load the model once and process a batch of parents, as
``observer.sh`` does; persistent idle GPU allocation costs more than inference.

Backends:
    hf    Qwen2.5-VL (or any Qwen-VL) through transformers. Needs GPU.
    mock  deterministic heuristic observer, no GPU. Used to validate the
          end-to-end plumbing and by the unit tests.

Standalone usage::

    uv run --project src/observer_env python src/behavior_observer.py \
        --gene-ids <geneA>,<geneB> --mode visual

The controller only reads the JSON this writes, keyed by the same behaviour
cache key (see ``src/visual_feedback.py``).
"""

import argparse
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_SRC_DIR = os.path.join(REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from src.visual_feedback import (
        capture_dir as vf_capture_dir,
        compute_cache_key,
        load_cached_observation,
        parent_metadata,
        telemetry_brief,
        validate_observation,
        format_and_store,
        _cfg,
    )
except ImportError:  # pragma: no cover - path fallback for nested launches
    from visual_feedback import (
        capture_dir as vf_capture_dir,
        compute_cache_key,
        load_cached_observation,
        parent_metadata,
        telemetry_brief,
        validate_observation,
        format_and_store,
        _cfg,
    )

PROMPT_TEMPLATE = os.path.join(REPO_ROOT, "templates", "Observer", "observe.txt")

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful locomotion observer. You describe only what the supplied "
    "evidence supports and say when a behaviour cannot be determined. You never "
    "diagnose hyperparameters or prescribe fixes."
)


def load_prompt_template():
    try:
        with open(PROMPT_TEMPLATE, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return (
            "Observe the parent policy's behaviour using the evidence below.\n"
            "Only the numerical telemetry and the ordered frames are evidence.\n"
            "Return a single JSON object with exactly these keys:\n"
            "determinable (bool), visible_posture (str), motion_pattern (str),\n"
            "instability (str), supporting_frames (list[int]), uncertainty\n"
            "('low'|'medium'|'high'), notes (str).\n\n"
            "MODE: <<MODE>>\nFRAMES: <<FRAMES>>\nTELEMETRY:\n<<TELEMETRY>>\n"
        )


def build_user_text(template, telemetry, frame_files, mode):
    if mode == "visual":
        frame_block = "\n".join(
            f"  frame {i}: {os.path.basename(path)}"
            for i, path in enumerate(frame_files)
        ) or "  (no frames supplied)"
    else:
        frame_block = "  (frames withheld for the text-control condition)"
    text = template
    text = text.replace("<<MODE>>", mode)
    text = text.replace("<<FRAMES>>", frame_block)
    text = text.replace("<<TELEMETRY>>", telemetry_brief(telemetry))
    return text


def parse_observation_json(text):
    """Extract the first JSON object from a model response, tolerating fences."""
    if not text:
        raise ValueError("empty model response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(cleaned[start:end + 1])


def load_capture(gene_id, capture_root=None):
    path = vf_capture_dir(gene_id, root=capture_root)
    manifest_path = os.path.join(path, "manifest.json")
    telemetry_path = os.path.join(path, "telemetry.json")
    if not os.path.isfile(manifest_path) or not os.path.isfile(telemetry_path):
        return None, None, None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(telemetry_path, "r", encoding="utf-8") as handle:
        telemetry = json.load(handle)
    frame_files = [os.path.join(path, entry["file"])
                   for entry in telemetry.get("frames", [])]
    return path, manifest, {"telemetry": telemetry, "frame_files": frame_files}


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def mock_observe(telemetry, frame_files, mode):
    """Deterministic, GPU-free stand-in for the VLM.

    Derives a coarse description from telemetry so the capture -> observe ->
    mutate path can be exercised without a model. It is never used in a real
    comparison; the record is tagged ``backend: mock``.
    """
    episodes = telemetry.get("episodes", [])
    if not episodes:
        return {
            "determinable": False,
            "visible_posture": "unknown",
            "motion_pattern": "unknown",
            "instability": "unknown",
            "supporting_frames": [],
            "uncertainty": "high",
            "notes": "no telemetry available",
        }
    worst = min(episodes, key=lambda ep: ep.get("return", 0.0))
    final_z = worst.get("final_torso_z", 0.0) or 0.0
    steps = worst.get("steps", 0) or 0
    mean_z = worst.get("mean_torso_z", 0.0) or 0.0
    if final_z < 0.8:
        posture = "torso drops out of the healthy height range"
    elif mean_z < 0.9:
        posture = "torso stays low through most of the rollout"
    else:
        posture = "torso remains upright through the rollout"
    if steps < 200:
        motion = "episode terminates early, little forward travel"
        instability = "loss of balance shortly after the start"
    elif worst.get("distance", 0.0) < 1.0:
        motion = "mostly in place, small steps"
        instability = "noisy but sustained stance"
    else:
        motion = "forward progression with periodic foot contacts"
        instability = "transient wobble without early termination"
    support = []
    if mode == "visual" and frame_files:
        support = [max(0, len(frame_files) - 2)]
    return {
        "determinable": True,
        "visible_posture": posture,
        "motion_pattern": motion,
        "instability": instability,
        "supporting_frames": support,
        "uncertainty": "high" if mode == "visual" else "medium",
        "notes": ("heuristic mock observer; replace with a frozen VLM for real "
                  "measurements"),
    }


class HFObserver:
    def __init__(self, model_id, revision, dtype="bfloat16", max_new_tokens=512):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype=torch_dtype, device_map="auto")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)

    def observe(self, system_prompt, user_text, frame_files, mode):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        if mode == "visual":
            for path in frame_files:
                messages[-1]["content"].append({"type": "image", "image": path})
        return self._generate(messages)

    def _generate(self, messages):
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:  # pragma: no cover - depends on observer env
            raise RuntimeError(
                "qwen_vl_utils is required for the hf backend; install "
                "src/observer_env") from exc

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def observe_parent(gene_id, observer=None, backend="hf", mode="visual",
                   capture_root=None, cache_root=None, force=False,
                   system_prompt=None, model_id=None, revision=None):
    """Observe one parent and persist a validated observation record.

    Returns the record, or None when the parent has no capture. When a valid
    cached record already exists and ``force`` is false, it is returned as-is.
    """
    metadata = parent_metadata(gene_id, mode=mode)
    if metadata is None:
        print(f"  skip {gene_id}: no genome file for cache key")
        return None
    if model_id:
        metadata["observer_model_id"] = model_id
    if revision:
        metadata["observer_model_revision"] = revision
    cache_key = compute_cache_key(metadata)

    if not force:
        cached = load_cached_observation(cache_key, directory=cache_root)
        if cached:
            print(f"  cached {gene_id}: {cache_key[:12]}")
            return cached

    capture_path, manifest, bundle = load_capture(gene_id, capture_root=capture_root)
    if bundle is None:
        print(f"  skip {gene_id}: no capture at {capture_path}")
        return None
    if manifest.get("parent_gene") not in (None, gene_id):
        print(f"  WARNING: capture gene mismatch {manifest.get('parent_gene')} != {gene_id}")
    for field in ("episodes", "max_steps", "camera", "image_max_side"):
        config_value = metadata.get(field)
        capture_value = manifest.get(field)
        if capture_value is not None and config_value is not None and capture_value != config_value:
            print(f"  WARNING: {gene_id} capture {field}={capture_value} "
                  f"!= config {config_value}; cache key uses config")

    telemetry = bundle["telemetry"]
    frame_files = bundle["frame_files"]
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    user_text = build_user_text(load_prompt_template(), telemetry, frame_files, mode)

    start = time.time()
    raw = ""
    usage = {}
    if backend == "mock":
        record = mock_observe(telemetry, frame_files, mode)
    elif backend == "hf":
        if observer is None:
            raise RuntimeError("hf backend requires a loaded HFObserver")
        raw = observer.observe(system_prompt, user_text, frame_files, mode)
        try:
            record = parse_observation_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  parse failed for {gene_id}: {exc}; writing undeterminable record")
            record = {
                "determinable": False,
                "visible_posture": "unknown",
                "motion_pattern": "unknown",
                "instability": "unknown",
                "supporting_frames": [],
                "uncertainty": "high",
                "notes": f"unparseable observer output: {str(exc)[:200]}",
            }
        usage = {"raw_output": raw}
    else:
        raise ValueError(f"unknown backend: {backend}")

    ok, reason = validate_observation(record)
    if not ok:
        print(f"  invalid record for {gene_id}: {reason}; coercing")
        record = {
            "determinable": False,
            "visible_posture": "unknown",
            "motion_pattern": "unknown",
            "instability": "unknown",
            "supporting_frames": [],
            "uncertainty": "high",
            "notes": f"invalid observer output: {reason}",
        }
        ok, reason = validate_observation(record)

    elapsed = round(time.time() - start, 3)
    stats = {
        "backend": backend,
        "parent_gene": gene_id,
        "mode": mode,
        "observed_at": time.time(),
        "observe_seconds": elapsed,
        "frame_count": len(frame_files) if mode == "visual" else 0,
        "cost_within_budget": elapsed <= float(_cfg("FEEDBACK_MAX_COST_SECONDS", 120)),
        "usage": usage,
    }
    normalized, path = format_and_store(
        record, telemetry, cache_key=cache_key, stats=stats, **metadata,
        cache_directory=cache_root)
    print(f"  observed {gene_id} via {backend} in {elapsed}s -> {path}")
    return normalized


def _parse_gene_ids(args):
    genes = list(args.gene_ids or [])
    if args.gene_file:
        with open(args.gene_file, "r", encoding="utf-8") as handle:
            genes += [line.strip() for line in handle if line.strip()]
    out = []
    for item in genes:
        out.extend([part.strip() for part in item.split(",") if part.strip()])
    return out


def main():
    parser = argparse.ArgumentParser(description="Observe captured parent rollouts")
    parser.add_argument("--gene-ids", nargs="*", default=[],
                        help="Parent gene ids (comma separated or repeated)")
    parser.add_argument("--gene-file", default=None,
                        help="File with one parent gene id per line")
    parser.add_argument("--mode", default=None, choices=["visual", "telemetry"],
                        help="visual = frames + telemetry; telemetry = text control")
    parser.add_argument("--backend", default=None, choices=["hf", "mock"])
    parser.add_argument("--capture-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if a cached record exists")
    args = parser.parse_args()

    mode = (args.mode or _cfg("FEEDBACK_MODE", "visual")).strip().lower()
    if mode == "off":
        mode = "visual"
    backend = (args.backend or _cfg("OBSERVER_BACKEND", "hf")).strip().lower()
    model_id = args.model_id or _cfg("OBSERVER_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    revision = args.revision or _cfg("OBSERVER_MODEL_REVISION", "main")
    max_new_tokens = args.max_new_tokens or int(_cfg("OBSERVER_MAX_NEW_TOKENS", 512))
    dtype = args.dtype or _cfg("OBSERVER_TORCH_DTYPE", "bfloat16")

    genes = _parse_gene_ids(args)
    if not genes:
        parser.error("no gene ids supplied")

    observer = None
    if backend == "hf":
        print(f"Loading observer {model_id}@{revision} ({dtype})")
        observer = HFObserver(model_id, revision, dtype=dtype,
                              max_new_tokens=max_new_tokens)

    processed = 0
    for gene_id in genes:
        if observe_parent(gene_id, observer=observer, backend=backend, mode=mode,
                          capture_root=args.capture_root, cache_root=args.cache_root,
                          force=args.force, model_id=model_id, revision=revision):
            processed += 1
    print(f"Observed {processed}/{len(genes)} requested parents")
    print("Job Done")


if __name__ == "__main__":
    main()
