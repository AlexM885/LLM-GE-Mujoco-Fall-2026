"""Named, structured evaluation results for MuJoCo genes.

The canonical per-gene result is a JSON file. Evolutionary code must read
metrics *by name* from it, never by CSV column position (see docs/MESB.md,
"Positional fitness bug").

This module has no gymnasium / torch / SB3 dependency so the evolution
driver and the tests can import it cheaply.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sota.MujocoRL.mesb_archive import ArchiveRecord

SCHEMA_VERSION = 1

#: Names of the two MESB behaviour descriptors, in archive-axis order.
DESCRIPTOR_KEYS: tuple[str, str] = ("mean_distance", "mean_control_cost")
#: Name of the quality measure that decides cell elites.
OBJECTIVE_KEY = "mean_reward"

REQUIRED_SUCCESS_FIELDS = ("gene_id", "status", "mean_reward", "std_reward", "behavior")


class MalformedResultError(ValueError):
    """The result file exists but cannot be interpreted."""


def atomic_write_json(path: str | os.PathLike, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file + rename so readers never see half a file.

    Non-finite floats are written as JSON ``NaN``/``Infinity`` tokens
    (Python's default) so that a bad value is preserved and then rejected
    on read rather than silently coerced.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def failure_payload(gene_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    """Standard payload for a failed evaluation."""
    payload = {"schema_version": SCHEMA_VERSION, "gene_id": gene_id, "status": "failed",
               "error": reason}
    payload.update(extra)
    return payload


@dataclass(frozen=True)
class EvaluationOutcome:
    """Parsed result of one gene evaluation.

    ``ok`` is True only if the file reports success *and* every value MESB
    needs is present and finite.
    """

    gene_id: str
    ok: bool
    reason: str | None
    payload: dict[str, Any]

    @property
    def mean_reward(self) -> float | None:
        return self.payload.get(OBJECTIVE_KEY) if self.ok else None

    @property
    def param_count(self) -> int | None:
        pc = self.payload.get("param_count")
        return None if pc is None else int(pc)

    def measures(self) -> tuple[float, ...]:
        if not self.ok:
            raise ValueError(f"gene {self.gene_id} did not evaluate successfully: {self.reason}")
        behavior = self.payload["behavior"]
        return tuple(float(behavior[k]) for k in DESCRIPTOR_KEYS)

    def to_archive_record(self, generation: int) -> ArchiveRecord:
        if not self.ok:
            raise ValueError(f"gene {self.gene_id} (generation {generation}) did not evaluate "
                             f"successfully and must not enter the archive: {self.reason}")
        return ArchiveRecord(
            gene_id=self.gene_id,
            generation=generation,
            objective=float(self.payload[OBJECTIVE_KEY]),
            measures=self.measures(),
            param_count=self.param_count,
            model_path=self.payload.get("model_path"),
        )


def _finite(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def validate_payload(payload: Any, expected_gene_id: str | None = None) -> tuple[bool, str | None]:
    """Return (ok, reason). ``ok`` requires status=success and finite metrics."""
    if not isinstance(payload, dict):
        return False, "result is not a JSON object"
    gene_id = payload.get("gene_id")
    if expected_gene_id is not None and gene_id != expected_gene_id:
        return False, f"gene_id mismatch: file has {gene_id!r}, expected {expected_gene_id!r}"
    status = payload.get("status")
    if status != "success":
        return False, f"status={status!r}: {payload.get('error', 'no error message')}"
    missing = [k for k in REQUIRED_SUCCESS_FIELDS if k not in payload]
    if missing:
        return False, f"missing fields {missing}"
    for key in (OBJECTIVE_KEY, "std_reward"):
        if not _finite(payload[key]):
            return False, f"non-finite {key}={payload[key]!r}"
    behavior = payload["behavior"]
    if not isinstance(behavior, dict):
        return False, "behavior is not an object"
    for key in DESCRIPTOR_KEYS:
        if key not in behavior:
            return False, f"missing behaviour descriptor {key!r}"
        if not _finite(behavior[key]):
            return False, f"non-finite behaviour descriptor {key}={behavior[key]!r}"
    return True, None


def read_outcome(path: str | os.PathLike, gene_id: str) -> EvaluationOutcome:
    """Load and validate a result JSON. Never raises for bad content."""
    path = Path(path)
    if not path.exists():
        return EvaluationOutcome(gene_id, False, f"result file missing: {path}", {})
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        return EvaluationOutcome(gene_id, False, f"malformed result JSON: {exc}", {})
    ok, reason = validate_payload(payload, expected_gene_id=gene_id)
    if not isinstance(payload, dict):
        payload = {}
    return EvaluationOutcome(gene_id, ok, reason, payload)
