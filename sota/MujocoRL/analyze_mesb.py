"""Plots and a summary for a MESB run directory (headless; matplotlib Agg).

Usage::

    python sota/MujocoRL/analyze_mesb.py --run-dir mujoco_rl_output/<run_name>

    # Also replay the run's evaluations through a fixed equal-width grid and a
    # fresh sliding-boundary archive (same order, same resolution) to compare:
    python sota/MujocoRL/analyze_mesb.py --run-dir ... --compare-fixed -500 2000 0 500

Writes to ``<run-dir>/mesb/figures/`` (or ``--out-dir``):

    final_archive.png                 elites on the *actual* (unequal) MESB cells
    snapshots/archive_gen_XXXX.png    the same, per generation
    coverage.png, qd_score.png, best_reward.png, mean_elite_reward.png
    boundaries_mean_distance.png      every boundary position over time
    boundaries_mean_control_cost.png
    descriptor_distributions.png      histograms with final boundaries overlaid
    fixed_vs_sliding.png              (only with --compare-fixed)
    summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sota.MujocoRL.mesb_archive import (ArchiveRecord, FixedGridArchive,  # noqa: E402
                                        SlidingBoundariesArchive)

X_KEY, Y_KEY = "mean_distance", "mean_control_cost"
X_LABEL = "mean distance traveled (final x - initial x)"
Y_LABEL = "mean control cost per episode"


# ------------------------------------------------------------------ loading
def _num(v: str) -> float | None:
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_run(run_dir: Path) -> dict[str, Any]:
    mesb = run_dir / "mesb"
    snapshots = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted((mesb / "archive_snapshots").glob("gen_*.json"))]
    return {
        "config": json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if (run_dir / "config.json").exists() else {},
        "metrics": read_csv(mesb / "metrics.csv"),
        "evaluations": read_csv(run_dir / "evaluations.csv"),
        "boundaries": read_jsonl(mesb / "boundaries.jsonl"),
        "snapshots": snapshots,
    }


# ------------------------------------------------------------------ plots
def plot_archive(snapshot: dict, evaluations: list[dict], path: Path, vmin=None, vmax=None,
                 title: str | None = None) -> None:
    """Elites drawn on their true cell rectangles (cells are NOT equal width)."""
    bx = snapshot["boundaries"][X_KEY]
    by = snapshot["boundaries"][Y_KEY]
    elites = snapshot["elites"]
    fig, ax = plt.subplots(figsize=(9, 7))
    pts = [(float(e[X_KEY]), float(e[Y_KEY])) for e in evaluations
           if e.get("status") == "success" and int(e["generation"]) <= snapshot["generation"]]
    if pts:
        ax.scatter(*zip(*pts), s=5, c="white", edgecolors="grey", linewidths=0.4, zorder=2.5,
                   label="all evaluations")
    rewards = [float(e["mean_reward"]) for e in elites]
    if rewards:
        vmin = min(rewards) if vmin is None else vmin
        vmax = max(rewards) if vmax is None else vmax
        rects = [Rectangle((e["distance_lo"], e["control_cost_lo"]),
                           e["distance_hi"] - e["distance_lo"],
                           e["control_cost_hi"] - e["control_cost_lo"]) for e in elites]
        pc = PatchCollection(rects, cmap="viridis", alpha=0.85, edgecolor="none", zorder=2)
        pc.set_array(np.array(rewards))
        pc.set_clim(vmin, vmax if vmax > vmin else vmin + 1e-9)
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, label="elite mean reward")
        ax.scatter([e[X_KEY] for e in elites], [e[Y_KEY] for e in elites], s=10, c="black",
                   zorder=3, label="elite descriptor")
    for x in bx:
        ax.axvline(x, color="k", lw=0.4, alpha=0.5, zorder=0)
    for y in by:
        ax.axhline(y, color="k", lw=0.4, alpha=0.5, zorder=0)
    pad_x = (bx[-1] - bx[0]) * 0.03 or 1.0
    pad_y = (by[-1] - by[0]) * 0.03 or 1.0
    ax.set_xlim(bx[0] - pad_x, bx[-1] + pad_x)
    ax.set_ylim(by[0] - pad_y, by[-1] + pad_y)
    s = snapshot["summary"]
    ax.set_title(title or (f"{s['boundary_mode']} archive, generation {snapshot['generation']}: "
                           f"{s['occupied_cells']}/{s['total_cells']} cells "
                           f"({100 * s['coverage']:.1f}%), raw QD {s['raw_qd_score']:.1f}"))
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_curve(metrics: list[dict], key: str, ylabel: str, path: Path) -> bool:
    pts = [(int(m["generation"]), _num(m.get(key))) for m in metrics]
    pts = [(g, v) for g, v in pts if v is not None]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*zip(*pts), marker="o")
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_boundary_evolution(events: list[dict], key: str, label: str, path: Path) -> bool:
    ev = [e for e in events if e["event"] in ("initialize", "remap")]
    if not ev:
        return False
    x = [e["total_seen"] for e in ev]
    B = np.array([e["boundaries"][key] for e in ev], dtype=float)
    last_seen = max(e["total_seen"] for e in events)
    if last_seen > x[-1]:  # extend the final segment to the end of the run
        x.append(last_seen)
        B = np.vstack([B, B[-1]])
    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(1, B.shape[1] - 1):
        ax.step(x, B[:, j], where="post", lw=1, color=plt.cm.viridis(j / B.shape[1]))
    ax.step(x, B[:, 0], where="post", ls="--", color="grey", label="observed min / max")
    ax.step(x, B[:, -1], where="post", ls="--", color="grey")
    for e in ev:
        ax.axvline(e["total_seen"], color="k", lw=0.3, alpha=0.3)
    ax.set_xlabel("successful evaluations seen by the archive (remap points)")
    ax.set_ylabel(f"{label} boundary value")
    ax.set_title(f"{label}: {B.shape[1] - 2} internal boundaries, {len(ev) - 1} remaps "
                 "(vertical lines = remap points)")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_distributions(evaluations: list[dict], final: dict | None, path: Path) -> bool:
    ok = [e for e in evaluations if e.get("status") == "success"]
    if not ok:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, label in ((axes[0], X_KEY, X_LABEL), (axes[1], Y_KEY, Y_LABEL)):
        vals = [float(e[key]) for e in ok]
        ax.hist(vals, bins=40, color="tab:blue", alpha=0.7)
        if final:
            for b in final["boundaries"][key]:
                ax.axvline(b, color="red", lw=0.6)
        ax.set_xlabel(label)
        ax.set_ylabel("evaluations")
    fig.suptitle("Descriptor distributions (red = final MESB boundaries)")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


# ------------------------------------------------------------------ replay
def replay(evaluations: list[dict], config: dict, fixed_ranges: list[list[float]]) -> dict:
    """Replay the run's successful evaluations (in logged order) through a fixed
    grid and a sliding archive with identical resolution. Only the boundary
    approach differs. Initial population = generation 0."""
    dims = config.get("mesb_dims", [20, 20])
    ok = [e for e in evaluations if e.get("status") == "success"]
    recs = [ArchiveRecord(e["gene_id"], int(e["generation"]), float(e["mean_reward"]),
                          (float(e[X_KEY]), float(e[Y_KEY]))) for e in ok]
    init = [r for r in recs if r.generation == 0]
    rest = [r for r in recs if r.generation != 0]
    out = {}
    for name, archive in (
        ("fixed", FixedGridArchive(dims=dims, ranges=fixed_ranges)),
        ("sliding", SlidingBoundariesArchive(
            dims=dims, remap_frequency=config.get("mesb_remap_frequency", 100),
            buffer_capacity=config.get("mesb_buffer_capacity"),
            remap_at_generation_end=config.get("mesb_remap_at_generation_end", False))),
    ):
        if not init:
            continue
        archive.initialize(init)
        curve = [(archive.total_seen, archive.coverage())]
        gen = 0
        for r in rest:
            if r.generation != gen:
                archive.end_generation()
                gen = r.generation
            archive.add(r)
            curve.append((archive.total_seen, archive.coverage()))
        archive.end_generation()
        out[name] = {"summary": archive.summary(), "curve": curve}
    return out


def plot_replay(result: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in result.items():
        ax.plot(*zip(*r["curve"]), label=name)
    ax.set_xlabel("successful evaluations")
    ax.set_ylabel("coverage")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Replay of the same evaluations: fixed grid vs sliding boundaries")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ summary
def best_model_path(run: dict) -> str | None:
    """Model of the highest-reward successful evaluation in the run."""
    ok = [e for e in run["evaluations"] if e.get("status") == "success" and e.get("model_path")]
    best = max(ok, key=lambda e: float(e["mean_reward"]), default=None)
    return None if best is None else best["model_path"]


def _f(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}" if math.isfinite(v) else str(v)
    return str(v)


def write_summary(run: dict, figures: list[str], replay_result: dict | None, path: Path) -> None:
    cfg, evals = run["config"], run["evaluations"]
    ok = [e for e in evals if e.get("status") == "success"]
    final = run["snapshots"][-1] if run["snapshots"] else None
    lines = [f"# MESB run summary: {cfg.get('run_name', path.parent.parent.name)}", ""]
    lines += [
        f"- selection mode: `{cfg.get('selection_mode')}`, archive boundaries: "
        f"`{cfg.get('archive_boundaries', 'n/a')}`",
        f"- environment: `{cfg.get('env_id')}`, PPO timesteps {cfg.get('eval_timesteps')}, "
        f"eval episodes {cfg.get('eval_episodes')} x {cfg.get('eval_max_steps')} steps, "
        f"seed {cfg.get('seed')}",
        f"- archive dims {cfg.get('mesb_dims')}, remap frequency "
        f"{cfg.get('mesb_remap_frequency')}, buffer capacity "
        f"{cfg.get('mesb_buffer_capacity') or 'unlimited'}, remap at generation end "
        f"{cfg.get('mesb_remap_at_generation_end')}",
        f"- git: `{cfg.get('git_branch')}` @ `{cfg.get('git_commit')}`", "",
        "| metric | value |", "|---|---|",
        f"| total evaluations | {len(evals)} |",
        f"| successful evaluations | {len(ok)} |",
        f"| failed evaluations | {len(evals) - len(ok)} |",
    ]
    if final:
        s = final["summary"]
        best = next((e for e in final["elites"] if e["gene_id"] == s["best_gene_id"]), None)
        lines += [
            f"| final occupied cells | {s['occupied_cells']} |",
            f"| total cells | {s['total_cells']} |",
            f"| final coverage | {100 * s['coverage']:.2f}% |",
            f"| raw QD score | {_f(s['raw_qd_score'])} |",
            f"| shifted QD score | {_f(s['shifted_qd_score'])} |",
            f"| mean elite reward | {_f(s['mean_elite_objective'])} |",
            f"| best reward | {_f(s['best_objective'])} |",
            f"| best gene ID | `{s['best_gene_id']}` |",
            f"| best gene descriptor (distance, control cost) | "
            f"({_f(best[X_KEY] if best else None)}, {_f(best[Y_KEY] if best else None)}) |",
            f"| remaps | {s['num_remaps']} |",
            f"| best model | `{best_model_path(run)}` |", "",
            "## Final boundaries", "",
            f"- distance: {[round(b, 3) for b in final['boundaries'][X_KEY]]}",
            f"- control cost: {[round(b, 3) for b in final['boundaries'][Y_KEY]]}",
        ]
    else:
        best = max(ok, key=lambda e: float(e["mean_reward"]), default=None)
        lines += [
            f"| best reward | {_f(float(best['mean_reward']) if best else None)} |",
            f"| best gene ID | `{best['gene_id'] if best else 'n/a'}` |",
            "", "_No archive in this run (nsga2 mode). Use `--compare-fixed` to replay the "
            "evaluations through fixed and sliding archives._",
        ]
    if replay_result:
        lines += ["", "## Replay comparison (same evaluations, same resolution)", "",
                  "| archive | occupied | coverage | raw QD | mean elite | best |",
                  "|---|---|---|---|---|---|"]
        for name, r in replay_result.items():
            s = r["summary"]
            lines.append(f"| {name} | {s['occupied_cells']}/{s['total_cells']} | "
                         f"{100 * s['coverage']:.2f}% | {_f(s['raw_qd_score'])} | "
                         f"{_f(s['mean_elite_objective'])} | {_f(s['best_objective'])} |")
        lines.append("\n_Replay is offline: in the fixed-grid row, parent selection was not "
                     "driven by that grid, so it is a descriptive comparison, not a controlled "
                     "run of fixed-grid MAP-Elites._")
    lines += ["", "## Figures", ""] + [f"- `{f}`" for f in figures]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyse a MESB run directory")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", default=None, help="Default: <run-dir>/mesb/figures")
    p.add_argument("--compare-fixed", type=float, nargs=4, default=None,
                   metavar=("DIST_LO", "DIST_HI", "CTRL_LO", "CTRL_HI"),
                   help="Replay evaluations through a fixed grid with these ranges")
    p.add_argument("--no-snapshots", action="store_true", help="Skip per-generation plots")
    p.add_argument("--print-best-model", action="store_true",
                   help="Only print the model path of the best controller and exit")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    if args.print_best_model:
        path = best_model_path(load_run(run_dir))
        if path is None:
            print("no successful evaluation with a saved model", file=sys.stderr)
            return 1
        print(path)
        return 0
    out = Path(args.out_dir) if args.out_dir else run_dir / "mesb" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    run = load_run(run_dir)
    made: list[str] = []

    snaps = run["snapshots"]
    if snaps:
        all_r = [float(e["mean_reward"]) for s in snaps for e in s["elites"]]
        vmin, vmax = (min(all_r), max(all_r)) if all_r else (None, None)
        plot_archive(snaps[-1], run["evaluations"], out / "final_archive.png")
        made.append("final_archive.png")
        if not args.no_snapshots:
            (out / "snapshots").mkdir(exist_ok=True)
            for s in snaps:
                plot_archive(s, run["evaluations"],
                             out / "snapshots" / f"archive_gen_{s['generation']:04d}.png",
                             vmin, vmax)
            made.append(f"snapshots/ ({len(snaps)} generations)")
    for key, label, name in (("coverage", "coverage (occupied / total cells)", "coverage.png"),
                             ("raw_qd_score", "raw QD score (sum of elite rewards)",
                              "qd_score.png"),
                             ("best_reward", "best mean reward so far", "best_reward.png"),
                             ("mean_elite_reward", "mean elite reward", "mean_elite_reward.png")):
        if plot_curve(run["metrics"], key, label, out / name):
            made.append(name)
    for key, label in ((X_KEY, "distance"), (Y_KEY, "control cost")):
        name = f"boundaries_{key}.png"
        if plot_boundary_evolution(run["boundaries"], key, label, out / name):
            made.append(name)
    if plot_distributions(run["evaluations"], snaps[-1] if snaps else None,
                          out / "descriptor_distributions.png"):
        made.append("descriptor_distributions.png")

    replay_result = None
    if args.compare_fixed:
        r = args.compare_fixed
        replay_result = replay(run["evaluations"], run["config"], [[r[0], r[1]], [r[2], r[3]]])
        if replay_result:
            plot_replay(replay_result, out / "fixed_vs_sliding.png")
            made.append("fixed_vs_sliding.png")

    write_summary(run, made, replay_result, out / "summary.md")
    print((out / "summary.md").read_text(encoding="utf-8"))
    print(f"Figures written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
