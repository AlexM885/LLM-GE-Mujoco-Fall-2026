"""Paired-seed statistical report for the VLM aux-reward vs control arms.

This is the stat-sig scaffolding for Direction C: it does NOT re-implement
multi-seed training (that is covered by Caleb's seeding work and plain
sbatch arrays), it just consumes the stats JSONs each arm produces, aligns
them on seed, and reports IQM + 95 % bootstrap CI + probability-of-improvement
using :mod:`rliable`. Paired-seed design means both arms share the exact
same training/eval seed set — a Common Random Numbers variance reduction
that lets a modest latent gain register as many SDs.

Input layout (either per-arm JSON file or per-arm directory of stats):

  --vlm-arm       path to either
                    * a single JSON array ``[{"seed": 0, "best_reward": …}, …]``
                    * a directory containing one ``<seed>.json`` per seed
  --control-arm   same shape as --vlm-arm
  --metric        which field to extract (default ``best_reward``)
  --out           markdown report destination

Report content:
- summary table (per-seed, per-arm, paired difference)
- IQM, 95 % bootstrap CI per arm (rliable.get_interval_estimates)
- probability of improvement VLM > control (rliable.get_probability_of_improvement)
- paired Welch's t p-value and Cohen's d on the differences
- optional per-generation breakdown if the input records include ``generation``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Mapping, Sequence

import numpy as np

try:
    from rliable import library as rly
    from rliable import metrics
    _RLIABLE_OK = True
except Exception:
    # rliable transitively depends on `arch`, which has a brittle pandas
    # version coupling on some clusters. Fall back to a numpy-only
    # reimplementation of IQM + bootstrap CI + probability-of-improvement.
    _RLIABLE_OK = False


def _iqm(x: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50 % of the sample."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = len(x)
    if n == 0:
        return float("nan")
    lo = n // 4
    hi = n - lo
    if hi <= lo:
        return float(np.mean(x))
    return float(np.mean(x[lo:hi]))


def _bootstrap_ci(
    x: np.ndarray, stat_fn, reps: int = 10000, alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(x[idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(stat_fn(x)), (lo, hi)


def _probability_of_improvement(
    a: np.ndarray, b: np.ndarray, reps: int = 10000,
    alpha: float = 0.05, seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """P(A > B) with bootstrap CI, averaging over all cross pairs."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    def _poi(aa: np.ndarray, bb: np.ndarray) -> float:
        diff = aa[:, None] - bb[None, :]
        return float((diff > 0).mean() + 0.5 * (diff == 0).mean())

    rng = np.random.default_rng(seed)
    boots = np.empty(reps, dtype=float)
    for i in range(reps):
        boots[i] = _poi(
            a[rng.integers(0, len(a), size=len(a))],
            b[rng.integers(0, len(b), size=len(b))],
        )
    return _poi(a, b), (
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def _load_arm(path: str) -> list[dict]:
    """Load an arm's per-seed records from either a file or a directory."""
    if os.path.isdir(path):
        records = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if not os.path.isfile(full) or not name.endswith(".json"):
                continue
            with open(full, "r", encoding="utf-8") as handle:
                records.append(json.load(handle))
        return records
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    raise ValueError(
        f"{path} must be a directory of per-seed JSONs or a top-level JSON array")


def _align_on_seed(vlm: list[dict], ctrl: list[dict],
                   metric: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return the common seed set and matched (vlm, ctrl) score arrays."""
    def _by_seed(records: list[dict]) -> dict[int, float]:
        out: dict[int, float] = {}
        for rec in records:
            if "seed" not in rec or metric not in rec:
                continue
            try:
                out[int(rec["seed"])] = float(rec[metric])
            except (TypeError, ValueError):
                continue
        return out

    vlm_map = _by_seed(vlm)
    ctrl_map = _by_seed(ctrl)
    common = sorted(set(vlm_map) & set(ctrl_map))
    vlm_scores = np.array([vlm_map[s] for s in common])
    ctrl_scores = np.array([ctrl_map[s] for s in common])
    return common, vlm_scores, ctrl_scores


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (t_statistic, two-sided p-value approximation).

    Uses the normal approximation to avoid pulling in scipy; this is fine
    for N≥5 and the direction/magnitude of the effect matter more than
    the exact p-value for a paper that already reports bootstrap CI.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    denom = np.sqrt(va / len(a) + vb / len(b))
    if denom == 0:
        return float("inf"), 0.0
    t = (float(np.mean(a)) - float(np.mean(b))) / denom
    # Two-sided normal approximation to the Welch t.
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
    return t, p


def _paired_t(diffs: np.ndarray) -> tuple[float, float]:
    """Paired t under the normal approx."""
    if len(diffs) < 2:
        return float("nan"), float("nan")
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return float("inf"), 0.0
    t = float(np.mean(diffs)) / (sd / np.sqrt(len(diffs)))
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
    return t, p


def _cohens_d_paired(diffs: np.ndarray) -> float:
    """Standardised effect size for paired differences."""
    if len(diffs) < 2:
        return float("nan")
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return float("inf")
    return float(np.mean(diffs)) / sd


def _rliable_block(vlm: np.ndarray, ctrl: np.ndarray) -> dict:
    """Compute rliable IQM + 95 % bootstrap CI + prob-of-improvement.

    rliable expects scores of shape ``(num_runs, num_tasks)``. We have one
    task (Walker2d-v5 best reward) and N runs = N seeds, so we reshape as
    ``(N, 1)``. All existing rliable utilities support this.
    """
    if _RLIABLE_OK:
        scores = {
            "VLM-aux": vlm.reshape(-1, 1),
            "control": ctrl.reshape(-1, 1),
        }
        aggregate_fn = lambda arr: np.array([metrics.aggregate_iqm(arr)])  # noqa: E731
        iqm, iqm_ci = rly.get_interval_estimates(
            scores, aggregate_fn, reps=10000)
        pair_scores = {"VLM-aux,control": (scores["VLM-aux"], scores["control"])}
        poi, poi_ci = rly.get_interval_estimates(
            pair_scores, metrics.probability_of_improvement, reps=10000)
        return {
            "available": True,
            "backend": "rliable",
            "iqm": {k: float(v[0]) for k, v in iqm.items()},
            "iqm_ci": {k: [float(ci[0, 0]), float(ci[1, 0])]
                       for k, ci in iqm_ci.items()},
            "prob_of_improvement": float(poi["VLM-aux,control"]),
            "prob_of_improvement_ci": [float(poi_ci["VLM-aux,control"][0, 0]),
                                       float(poi_ci["VLM-aux,control"][1, 0])],
        }
    # numpy-only fallback — same quantities, same bootstrap reps.
    iqm_v, ci_v = _bootstrap_ci(vlm, _iqm, reps=10000, seed=0)
    iqm_c, ci_c = _bootstrap_ci(ctrl, _iqm, reps=10000, seed=1)
    poi, poi_ci = _probability_of_improvement(vlm, ctrl, reps=10000, seed=2)
    return {
        "available": True,
        "backend": "numpy-fallback",
        "iqm": {"VLM-aux": iqm_v, "control": iqm_c},
        "iqm_ci": {"VLM-aux": [ci_v[0], ci_v[1]], "control": [ci_c[0], ci_c[1]]},
        "prob_of_improvement": poi,
        "prob_of_improvement_ci": [poi_ci[0], poi_ci[1]],
    }


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
def _fmt(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "n/a"
    return f"{v:.{digits}f}"


def render_report(
    *,
    metric: str,
    seeds: Sequence[int],
    vlm: np.ndarray,
    ctrl: np.ndarray,
    rliable_stats: Mapping,
    extra_notes: str = "",
) -> str:
    """Build a markdown report body."""
    diffs = vlm - ctrl
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
    t_paired, p_paired = _paired_t(diffs)
    d = _cohens_d_paired(diffs)
    t_welch, p_welch = _welch_t(vlm, ctrl)

    rows = [
        "| seed | VLM-aux | control | diff |",
        "|---:|---:|---:|---:|",
    ]
    for s, a, b, d_i in zip(seeds, vlm, ctrl, diffs):
        rows.append(f"| {s} | {_fmt(float(a))} | {_fmt(float(b))} | {_fmt(float(d_i))} |")

    rows.append(
        f"| **mean** | {_fmt(float(np.mean(vlm)))} | {_fmt(float(np.mean(ctrl)))} | "
        f"**{_fmt(mean_diff)}** |")
    rows.append(
        f"| **std**  | {_fmt(float(np.std(vlm, ddof=1)) if len(vlm) > 1 else float('nan'))} | "
        f"{_fmt(float(np.std(ctrl, ddof=1)) if len(ctrl) > 1 else float('nan'))} | "
        f"{_fmt(sd_diff)} |")

    stats_lines: list[str] = []
    if rliable_stats.get("available"):
        iqm = rliable_stats["iqm"]
        ci = rliable_stats["iqm_ci"]
        stats_lines += [
            "## Rliable (10,000 bootstraps)",
            f"- VLM-aux IQM = **{_fmt(iqm['VLM-aux'])}** "
            f"(95 % CI [{_fmt(ci['VLM-aux'][0])}, {_fmt(ci['VLM-aux'][1])}])",
            f"- control IQM = {_fmt(iqm['control'])} "
            f"(95 % CI [{_fmt(ci['control'][0])}, {_fmt(ci['control'][1])}])",
            f"- P(VLM-aux > control) = **{_fmt(rliable_stats['prob_of_improvement'], 3)}** "
            f"(95 % CI [{_fmt(rliable_stats['prob_of_improvement_ci'][0], 3)}, "
            f"{_fmt(rliable_stats['prob_of_improvement_ci'][1], 3)}])",
        ]
    else:
        stats_lines += ["## Rliable",
                        f"- not run: {rliable_stats.get('reason', '')}"]

    stats_lines += [
        "",
        "## Paired / unpaired tests (normal approximation)",
        f"- paired t = {_fmt(t_paired, 3)}, two-sided p = {_fmt(p_paired, 4)}",
        f"- Welch's t = {_fmt(t_welch, 3)}, two-sided p = {_fmt(p_welch, 4)}",
        f"- Cohen's d (paired) = **{_fmt(d, 2)}**",
    ]

    parts = [
        "# VLM aux-reward vs no-feedback control",
        "",
        f"- metric: `{metric}`",
        f"- common seeds: n = {len(seeds)}",
        "",
        "## Per-seed scores",
        "",
    ]
    parts += rows
    parts += ["", *stats_lines]
    if extra_notes:
        parts += ["", "## Notes", "", extra_notes.rstrip()]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _render_pair_block(name_a: str, name_b: str, seeds: list[int],
                       a: np.ndarray, b: np.ndarray) -> str:
    """One pairwise comparison block in the N-arm report."""
    diffs = a - b
    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
    t_paired, p_paired = _paired_t(diffs)
    d = _cohens_d_paired(diffs)
    t_welch, p_welch = _welch_t(a, b)
    rl = _rliable_block(a, b)
    lines = [
        f"### {name_a} vs {name_b}",
        "",
        f"| seed | {name_a} | {name_b} | diff |",
        "|---:|---:|---:|---:|",
    ]
    for s, va, vb, diff in zip(seeds, a, b, diffs):
        lines.append(f"| {s} | {_fmt(float(va))} | {_fmt(float(vb))} | {_fmt(float(diff))} |")
    lines.append(
        f"| **mean** | {_fmt(float(np.mean(a)))} | {_fmt(float(np.mean(b)))} | "
        f"**{_fmt(mean_diff)}** |")
    lines.append(
        f"| **std**  | {_fmt(float(np.std(a, ddof=1)) if len(a) > 1 else float('nan'))} | "
        f"{_fmt(float(np.std(b, ddof=1)) if len(b) > 1 else float('nan'))} | "
        f"{_fmt(sd_diff)} |")
    lines.append("")
    lines.append(f"- paired t = {_fmt(t_paired, 3)}, p = {_fmt(p_paired, 4)};  "
                 f"Welch t = {_fmt(t_welch, 3)}, p = {_fmt(p_welch, 4)};  "
                 f"Cohen's d (paired) = **{_fmt(d, 2)}**")
    if rl.get("available"):
        iqm = rl["iqm"]
        ci = rl["iqm_ci"]
        lines += [
            f"- rliable IQM: {name_a}={_fmt(iqm['VLM-aux'])} "
            f"CI [{_fmt(ci['VLM-aux'][0])}, {_fmt(ci['VLM-aux'][1])}];  "
            f"{name_b}={_fmt(iqm['control'])} "
            f"CI [{_fmt(ci['control'][0])}, {_fmt(ci['control'][1])}]",
            f"- **P({name_a} > {name_b}) = {_fmt(rl['prob_of_improvement'], 3)}** "
            f"(95% CI [{_fmt(rl['prob_of_improvement_ci'][0], 3)}, "
            f"{_fmt(rl['prob_of_improvement_ci'][1], 3)}])",
        ]
    return "\n".join(lines)


def render_multi_report(
    *,
    metric: str,
    arms: "list[tuple[str, list[dict]]]",
    comparisons: "list[tuple[str, str]]",
    extra_notes: str = "",
) -> str:
    """N-arm report with per-arm summary and selected pairwise comparisons."""
    arm_maps = {}
    for name, recs in arms:
        by = {}
        for r in recs:
            if "seed" in r and metric in r:
                try:
                    by[int(r["seed"])] = float(r[metric])
                except (TypeError, ValueError):
                    continue
        arm_maps[name] = by
    common = sorted(set.intersection(*(set(m) for m in arm_maps.values()))) \
        if arm_maps else []
    out = [f"# {len(arms)}-arm paired-seed report", "",
           f"- metric: `{metric}`",
           f"- common seeds: n = {len(common)}", ""]
    out += ["## Per-arm summary", "", "| arm | mean | std | min | max |",
            "|---|---:|---:|---:|---:|"]
    for name, by in arm_maps.items():
        vals = np.array([by[s] for s in common])
        out.append(f"| {name} | {_fmt(float(np.mean(vals)))} | "
                   f"{_fmt(float(np.std(vals, ddof=1)) if len(vals) > 1 else float('nan'))} | "
                   f"{_fmt(float(np.min(vals)))} | {_fmt(float(np.max(vals)))} |")
    out += ["", "## Pairwise comparisons (paired-seed Common Random Numbers)", ""]
    for a_name, b_name in comparisons:
        if a_name not in arm_maps or b_name not in arm_maps:
            continue
        a_arr = np.array([arm_maps[a_name][s] for s in common])
        b_arr = np.array([arm_maps[b_name][s] for s in common])
        out.append(_render_pair_block(a_name, b_name, common, a_arr, b_arr))
        out.append("")
    if extra_notes:
        out += ["## Notes", "", extra_notes.rstrip()]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired-seed stat-sig report for VLM aux-reward (N arms)")
    parser.add_argument("--vlm-arm",
                        help="File or dir of per-seed stats for VLM arm")
    parser.add_argument("--control-arm",
                        help="File or dir of per-seed stats for control arm")
    parser.add_argument("--arm", action="append", default=[],
                        metavar="NAME=PATH",
                        help="Add an arm NAME=PATH (repeat); overrides "
                             "--vlm-arm/--control-arm when used")
    parser.add_argument("--compare", action="append", default=[],
                        metavar="A,B",
                        help="Add a pairwise comparison A,B (repeat)")
    parser.add_argument("--metric", default="best_reward")
    parser.add_argument("--out", default="-")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    arms: "list[tuple[str, list[dict]]]" = []
    if args.arm:
        for entry in args.arm:
            if "=" not in entry:
                print(f"ERROR: --arm must be NAME=PATH, got {entry!r}",
                      file=sys.stderr)
                sys.exit(2)
            name, path = entry.split("=", 1)
            arms.append((name, _load_arm(path)))
    elif args.vlm_arm and args.control_arm:
        arms = [("VLM-aux", _load_arm(args.vlm_arm)),
                ("control", _load_arm(args.control_arm))]
    else:
        print("ERROR: supply either --vlm-arm/--control-arm or --arm (N times)",
              file=sys.stderr)
        sys.exit(2)

    if len(arms) == 2 and not args.compare and args.vlm_arm and args.control_arm:
        # Legacy 2-arm layout, only when the caller used the old flags
        # which guarantee the right ordering (vlm first, control second).
        seeds, a, b = _align_on_seed(arms[0][1], arms[1][1], args.metric)
        if not seeds:
            print("ERROR: no common seeds", file=sys.stderr)
            sys.exit(2)
        rl = _rliable_block(a, b)
        report = render_report(metric=args.metric, seeds=seeds,
                               vlm=a, ctrl=b, rliable_stats=rl,
                               extra_notes=args.notes)
    else:
        comparisons = []
        for c in args.compare:
            if "," not in c:
                print(f"ERROR: --compare must be A,B, got {c!r}",
                      file=sys.stderr)
                sys.exit(2)
            comparisons.append(tuple(c.split(",", 1)))
        if not comparisons:
            # Default: compare every arm to the first one.
            base = arms[0][0]
            comparisons = [(a[0], base) for a in arms[1:]]
        report = render_multi_report(
            metric=args.metric, arms=arms,
            comparisons=comparisons, extra_notes=args.notes)

    if args.out == "-":
        sys.stdout.write(report)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report)
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
