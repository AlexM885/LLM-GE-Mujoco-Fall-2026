"""Compare the fixed-grid and MESB arms of one matched pair.

1. Runs Shazeb's report (analysis/scripts/xqc_island_report.py, unchanged) on
   each arm, so both are summarised by the same code as his run2.
2. Reads both arms' genes.csv and computes, per generation (cumulative):
   - best-so-far, mean of the top 10 so far, median of that generation's valid genes;
   - valid / invalid / timed-out counts;
   - coverage and QD score on two SHARED grids, so the arms are scored on the
     same cells: (a) his fixed 20x20 grid (his ranges and get_bin rule);
     (b) a sliding-boundaries 20x20 grid built once from the union of both
     arms' valid genes;
   - each arm's own archive size (from his report's generations.csv).
3. Writes compare.csv, compare.md (tables + the pre-agreed verdict) and
   compare.png.

    uv run python scripts/mesb/matched/compare_arms.py --pair xqc_matched_p1
"""

import argparse
import csv
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from sota.MujocoRL.mesb_archive import compute_boundaries_1d  # noqa: E402

ARMS = ("fixed", "mesb")
BINS = 20
# Shazeb's MAP_ELITES_DESCRIPTORS for the XQC island (constants_Mujoco.py).
FIXED_RANGES = (("mean_distance", -500.0, 2000.0), ("mean_control_cost", 0.0, 0.6))


def fixed_cell(d, c):
    out = []
    for value, (_, lo, hi) in zip((d, c), FIXED_RANGES):
        frac = float(np.clip((value - lo) / (hi - lo), 0, 1))
        out.append(int(frac * (BINS - 1)))  # his get_bin rule on this branch
    return tuple(out)


def sliding_cell(d, c, edges):
    out = []
    for value, e in zip((d, c), edges):
        idx = int(np.searchsorted(np.asarray(e[1:BINS]), value, side="right"))
        out.append(min(max(idx, 0), BINS - 1))
    return tuple(out)


def load_genes(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r["generation"] = int(r["generation"])
            r["ok"] = r["status"] == "ok"
            for k in ("mean_reward", "mean_distance", "mean_control_cost"):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    r[k] = float("nan")
            rows.append(r)
    return rows


def archive_stats(genes, cell_fn, offset=0.0):
    """Cells filled and QD score (sum of best reward per cell, minus ``offset``)."""
    best = {}
    for g in genes:
        c = cell_fn(g["mean_distance"], g["mean_control_cost"])
        if c not in best or g["mean_reward"] > best[c]:
            best[c] = g["mean_reward"]
    return len(best), sum(v - offset for v in best.values())


def per_generation(rows, edges, offset):
    out = []
    for gen in sorted({r["generation"] for r in rows}):
        upto = [r for r in rows if r["generation"] <= gen and r["ok"] and math.isfinite(r["mean_reward"])]
        this = [r for r in rows if r["generation"] == gen]
        ok_now = [r["mean_reward"] for r in this if r["ok"]]
        rewards = sorted((r["mean_reward"] for r in upto), reverse=True)
        fc, fq = archive_stats(upto, fixed_cell, offset)
        sc, sq = archive_stats(upto, lambda d, c: sliding_cell(d, c, edges), offset)
        out.append({
            "generation": gen,
            "evaluated": len(this), "ok": len(ok_now),
            "invalid": sum(r["status"] == "invalid" for r in this),
            "no_result": sum(r["status"] == "no_result" for r in this),
            "best_so_far": round(rewards[0], 1) if rewards else None,
            "top10_mean_so_far": round(st.mean(rewards[:10]), 1) if rewards else None,
            "median_ok_this_gen": round(st.median(ok_now), 1) if ok_now else None,
            "shared_fixed_cells": fc, "shared_fixed_qd": round(fq, 1),
            "shared_sliding_cells": sc, "shared_sliding_qd": round(sq, 1),
        })
    return out


def verdict(last):
    f, m = last["fixed"], last["mesb"]
    wins_cov = m["shared_sliding_cells"] > f["shared_sliding_cells"]
    wins_qd = m["shared_sliding_qd"] > f["shared_sliding_qd"]
    fb, mb = f["best_so_far"] or 0.0, m["best_so_far"] or 0.0
    not_worse = mb >= fb - 0.02 * abs(fb)
    return wins_cov, wins_qd, not_worse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pair", required=True)
    p.add_argument("--base", default=str(REPO_ROOT / "mujoco_rl_output" / "matched"))
    p.add_argument("--out", default=None, help="default: <base>/<pair>/compare")
    p.add_argument("--skip-reports", action="store_true", help="reuse existing report outputs")
    args = p.parse_args(argv)
    pair_dir = Path(args.base) / args.pair
    out = Path(args.out) if args.out else pair_dir / "compare"
    out.mkdir(parents=True, exist_ok=True)

    genes = {}
    for arm in ARMS:
        rep = out / arm
        if not args.skip_reports:
            subprocess.run([sys.executable, str(REPO_ROOT / "analysis/scripts/xqc_island_report.py"),
                            str(pair_dir / arm), "--out", str(rep)], check=True, cwd=REPO_ROOT)
        genes[arm] = load_genes(rep / "genes.csv")

    pooled = [g for arm in ARMS for g in genes[arm] if g["ok"] and math.isfinite(g["mean_reward"])]
    edges = [compute_boundaries_1d([g[k] for g in pooled], BINS).tolist()
             for k in ("mean_distance", "mean_control_cost")]
    # QD needs non-negative terms: shift by the worst valid reward of either arm when
    # it is negative (XQC rewards are normally in the thousands, offset 0).
    offset = min(0.0, min(g["mean_reward"] for g in pooled))
    table = {arm: per_generation(genes[arm], edges, offset) for arm in ARMS}

    with open(out / "compare.csv", "w", newline="") as fh:
        cols = ["arm"] + list(table["fixed"][0])
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for arm in ARMS:
            for row in table[arm]:
                w.writerow({"arm": arm, **row})
    (out / "shared_grid_edges.json").write_text(json.dumps(
        {"bins": BINS, "pooled_valid_genes": len(pooled), "qd_offset": offset, "edges": edges},
        indent=2))

    last = {arm: table[arm][-1] for arm in ARMS}
    wins_cov, wins_qd, not_worse = verdict(last)
    lines = [f"# Matched pair {args.pair}: fixed grid vs MESB", "",
             "| metric (final generation, cumulative) | fixed | MESB |", "|---|---|---|"]
    for key in ("best_so_far", "top10_mean_so_far", "shared_sliding_cells", "shared_sliding_qd",
                "shared_fixed_cells", "shared_fixed_qd"):
        lines.append(f"| {key} | {last['fixed'][key]} | {last['mesb'][key]} |")
    totals = {arm: {k: sum(r[k] for r in table[arm]) for k in ("evaluated", "ok", "invalid", "no_result")}
              for arm in ARMS}
    for k in ("evaluated", "ok", "invalid", "no_result"):
        lines.append(f"| genes {k} (all generations) | {totals['fixed'][k]} | {totals['mesb'][k]} |")
    lines += ["", "Verdict rule agreed in advance: MESB helps if it beats the fixed arm on "
              "shared-grid coverage and shared-grid QD score without a lower best-so-far "
              "(within 2%). One pair is evidence, not proof.", "",
              f"- coverage on the shared sliding grid: {'MESB higher' if wins_cov else 'MESB not higher'}",
              f"- QD score on the shared sliding grid: {'MESB higher' if wins_qd else 'MESB not higher'}",
              f"- best-so-far: {'MESB not worse' if not_worse else 'MESB worse by more than 2%'}",
              "- verdict for this pair: "
              + ("MESB helps" if (wins_cov and wins_qd and not_worse) else "not shown")]
    (out / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for arm, style in (("fixed", "o-"), ("mesb", "s-")):
            g = [r["generation"] for r in table[arm]]
            axes[0].plot(g, [r["best_so_far"] for r in table[arm]], style, label=arm)
            axes[1].plot(g, [r["shared_sliding_cells"] for r in table[arm]], style, label=arm)
            axes[2].plot(g, [r["shared_sliding_qd"] for r in table[arm]], style, label=arm)
        for ax, title in zip(axes, ("best reward so far", "cells on shared grid",
                                    "QD score on shared grid")):
            ax.set_title(title)
            ax.set_xlabel("generation")
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(out / "compare.png", dpi=130)
    except ImportError:
        pass
    print(f"\nwrote {out}/compare.csv, compare.md, compare.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
