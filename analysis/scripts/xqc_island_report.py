"""Per-gene and per-generation report for one LLM-GE XQC island run, plus a figure.

Everything comes from files the run itself saved (nothing from logs or memory):
  <run>/global_data/global_gen_<N>.pkl      GLOBAL_DATA (genes touched in gen N),
                                            GLOBAL_DATA_ANCESTRY (parents, operator)
  <run>/island_*/checkpoint_gen_<N>.pkl     MAP-Elites archive after gen N
  sota/XQC/llmge/models/network_<id>.py     the gene the LLM wrote
  sota/XQC/llmge/results/<id>_results.csv   score written by train_xqc_gene.py
  sota/XQC/llmge/stats/<id>_stats.json      per-seed returns, config, error message

Generation of a gene: 0 if its ancestry is just CREATED (initial population; these
are taken from the ancestry log), otherwise the first global_gen_<N>.pkl that lists it. Status: "ok" (trained),
"invalid" (rejected by validation or crashed; -999999 sentinel), "no_result"
(no results file: the eval job hit its wall time). Fitness is always read from
the results CSV, because GLOBAL_DATA_HIST can hold stale -inf entries.

Run from the repo root with the main uv env (unpickling needs deap):
    uv run python analysis/scripts/xqc_island_report.py xqc_island_run2 \
        --out analysis/results/xqc_island_run2 --seed-mean 6970
Writes genes.csv, generations.csv, fitness_by_generation.png into --out and copies
every gene file, results CSV and stats JSON of the run into --out/data/.

Multi-objective runs (LLMGE_ARCHIVE=mome: each archive cell holds a Pareto front)
are detected from the checkpoint. genes.csv then also has speed (mean_return_forward),
effort (-mean_return_ctrl), foot_contact_0/1 and the gene's foot-contact cell;
generations.csv has the archive metrics, all with the fixed reference point
REFERENCE = (speed 0, effort 600) so they compare across runs:
  archive_cells / coverage_pct   filled cells of the MAP_BINS x MAP_BINS grid
  archive_genes                  genes on all cell fronts
  moqd_score                     sum over cells of the hypervolume of the cell's front
  global_hypervolume             hypervolume of the Pareto front of every archive gene
  max_speed, mean_speed          over the genes trained so far (status ok)
Extra figures: pareto_front.png (speed vs effort of every trained gene, the
non-dominated ones joined as a staircase) and foot_contact_map.png (best score in
each foot-contact cell of the final archive).
analysis/results is gitignored (**/results/), so commit a report with git add -f.
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import pickle
import shutil

import numpy as np

LLMGE = "sota/XQC/llmge"
FAILED = -999999.0
REFERENCE = (0.0, 600.0)   # (speed, effort); constants_Mujoco.MOME_REFERENCE_POINT = (0, -600)
MAP_BINS = 32              # constants_Mujoco.py, LLMGE_ARCHIVE=mome


def is_mome(archive):
    """MOME checkpoints map each cell to a list of individuals (its front)."""
    return any(len(v) and isinstance(v[0], list) for v in archive.values())


def members(archive):
    return [ind for v in archive.values() for ind in v] if is_mome(archive) else list(archive.values())


def hypervolume(points):
    """Area of (speed, effort) points' dominated region: more speed and less effort
    are better, bounded by REFERENCE."""
    pts = sorted({(s, e) for s, e in points if s > REFERENCE[0] and e < REFERENCE[1]}, key=lambda p: -p[0])
    hv, low = 0.0, REFERENCE[1]
    for s, e in pts:
        if e < low:
            hv += (s - REFERENCE[0]) * (low - e)
            low = e
    return hv


def pareto(points):
    """Non-dominated (speed, effort) points, sorted by effort."""
    return sorted(p for p in set(points)
                  if not any(q[0] >= p[0] and q[1] <= p[1] and q != p for q in points))


def cell_of(fc0, fc1):
    return tuple(min(int(min(max(x, 0.0), 1.0) * MAP_BINS), MAP_BINS - 1) for x in (fc0, fc1))


def load_cfg(path):
    spec = importlib.util.spec_from_file_location("gene", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {**mod.get_architecture_kwargs(), **mod.get_xqc_kwargs()}


def read_result(gene):
    path = f"{LLMGE}/results/{gene}_results.csv"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        row = list(csv.DictReader(f))[-1]
    return {k: float(v) for k, v in row.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory, e.g. xqc_island_run2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed-mean", type=float, help="seed config's mean on the same seeds/budget (reference line)")
    args = ap.parse_args()

    gens = sorted(int(p.rsplit("_", 1)[1][:-4]) for p in glob.glob(f"{args.run}/global_data/global_gen_*.pkl"))
    first_seen, ancestry = {}, {}
    for g in gens:
        with open(f"{args.run}/global_data/global_gen_{g}.pkl", "rb") as f:
            gd = pickle.load(f)
        for k in gd["GLOBAL_DATA"]:
            first_seen.setdefault(k, g)
        ancestry.update(gd["GLOBAL_DATA_ANCESTRY"])

    seed_cfg = load_cfg(f"{LLMGE}/network.py")
    os.makedirs(f"{args.out}/data/models", exist_ok=True)
    os.makedirs(f"{args.out}/data/results", exist_ok=True)
    os.makedirs(f"{args.out}/data/stats", exist_ok=True)

    # The initial population is only guaranteed to be in the ancestry log.
    genes = set(first_seen) | {k for k, a in ancestry.items() if a["MUTATE_TYPE"] == ["CREATED"] and k != "network"}
    rows = []
    for gene in genes:
        anc = ancestry.get(gene, {"GENES": [gene], "MUTATE_TYPE": ["CREATED"]})
        gen = 0 if anc["MUTATE_TYPE"] == ["CREATED"] else first_seen[gene]
        model = f"{LLMGE}/models/network_{gene}.py"
        text = open(model).read() if os.path.exists(model) else ""
        origin = "seed population" if gen == 0 else ("crossover" if "GeneCrossed" in text else "mutation")
        parent = anc["GENES"][-2] if len(anc["GENES"]) > 1 else ""
        res = read_result(gene)
        stats_path = f"{LLMGE}/stats/{gene}_stats.json"
        stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
        if res is None:
            status = "no_result"
        elif res["mean_reward"] <= FAILED:
            status = "invalid"
        else:
            status = "ok"
        diff = {}
        if status == "ok":
            cfg = stats.get("config") or load_cfg(model)
            diff = {k: v for k, v in cfg.items() if seed_cfg.get(k) != v}
        mo = {}
        if res is not None and "foot_contact_0" in res:
            ok = status == "ok"
            mo = {"speed": res["mean_return_forward"] if ok else "",
                  "effort": -res["mean_return_ctrl"] if ok else "",
                  "foot_contact_0": res["foot_contact_0"] if ok else "",
                  "foot_contact_1": res["foot_contact_1"] if ok else "",
                  "cell": " ".join(map(str, cell_of(res["foot_contact_0"], res["foot_contact_1"]))) if ok else ""}
        rows.append({
            "gene_id": gene, "generation": gen, "origin": origin, "operator": anc["MUTATE_TYPE"][-1],
            "parent": parent, "status": status, "error": stats.get("error", ""),
            "mean_reward": res["mean_reward"] if status == "ok" else "",
            "std_reward": res["std_reward"] if status == "ok" else "",
            "per_seed_return": json.dumps([round(x, 1) for x in stats.get("per_seed_return", [])]),
            "mean_distance": res["mean_distance"] if status == "ok" else "",
            "mean_control_cost": res["mean_control_cost"] if status == "ok" else "",
            "actor_params": int(res["param_count"]) if status == "ok" else "",
            "train_minutes": round(res["train_time"] / 60, 1) if res else "",
            "changes_vs_seed": json.dumps(diff),
            **mo,
        })
        for src, dst in [(model, "models"), (f"{LLMGE}/results/{gene}_results.csv", "results"), (stats_path, "stats")]:
            if os.path.exists(src):
                shutil.copy2(src, f"{args.out}/data/{dst}/")
    rows.sort(key=lambda r: (r["generation"], r["status"] != "ok", -(r["mean_reward"] or 0), r["gene_id"]))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"{args.out}/genes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    score = {r["gene_id"]: r["mean_reward"] for r in rows if r["status"] == "ok"}
    objs = {r["gene_id"]: (r["speed"], r["effort"]) for r in rows if r["status"] == "ok" and r.get("speed") != ""
            and "speed" in r}
    gen_rows, best, archive = [], -np.inf, {}
    for g in [0] + gens:
        rs = [r for r in rows if r["generation"] == g]
        ok = [r["mean_reward"] for r in rs if r["status"] == "ok"]
        best = max([best] + ok)
        cells = qd = ""
        mo = {}
        ckpt = glob.glob(f"{args.run}/island_*/checkpoint_gen_{g}.pkl")
        if ckpt:
            with open(ckpt[0], "rb") as f:
                archive = pickle.load(f)["MAP_ELITES_ARCHIVE"]
            cells = len(archive)
            qd = round(sum(score.get(ind[0], 0.0) for ind in members(archive)), 1)
            if is_mome(archive):
                done = [objs[r["gene_id"]] for r in rows if r["generation"] <= g and r["gene_id"] in objs]
                mo = {"coverage_pct": round(100 * cells / MAP_BINS ** 2, 2),
                      "archive_genes": len(members(archive)),
                      "moqd_score": round(sum(hypervolume([objs[i[0]] for i in v if i[0] in objs])
                                              for v in archive.values()), 1),
                      "global_hypervolume": round(hypervolume([objs[i[0]] for i in members(archive)
                                                               if i[0] in objs]), 1),
                      "max_speed": round(max(s for s, _ in done), 1) if done else "",
                      "mean_speed": round(float(np.mean([s for s, _ in done])), 1) if done else ""}
        gen_rows.append({
            "generation": g, "evaluated": len(rs), "ok": len(ok),
            "invalid": sum(r["status"] == "invalid" for r in rs),
            "no_result": sum(r["status"] == "no_result" for r in rs),
            "crossover_invalid": sum(r["status"] == "invalid" and r["origin"] == "crossover" for r in rs),
            "crossover_total": sum(r["origin"] == "crossover" for r in rs),
            "best_this_gen": round(max(ok), 1) if ok else "", "best_so_far": round(best, 1),
            "median_ok": round(float(np.median(ok)), 1) if ok else "",
            "archive_cells": cells, "archive_qd_score": qd,
            **mo,
        })
    with open(f"{args.out}/generations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in gen_rows for k in r)), restval="")
        w.writeheader()
        w.writerows(gen_rows)
    plot(rows, gen_rows, args, f"{args.out}/fitness_by_generation.png")
    if objs:
        plot_pareto(rows, f"{args.out}/pareto_front.png")
        if archive and is_mome(archive):
            plot_cells(archive, score, f"{args.out}/foot_contact_map.png")
    for r in gen_rows:
        print(r)


def plot(rows, gen_rows, args, path):
    """One panel: every trained gene as a light dot, the best gene of each generation
    as a dark dot, and a line through the best score found so far at each generation.
    The seed config on the same seeds is the dashed line. Counts of invalid and
    timed-out genes are in generations.csv rather than on the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dot, ink, muted, grid, bg = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    gens = [r["generation"] for r in gen_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(bg)
    rng = np.random.default_rng(0)
    for g in gens:
        ys = [r["mean_reward"] for r in rows if r["generation"] == g and r["status"] == "ok"]
        ax.scatter(g + rng.uniform(-0.15, 0.15, len(ys)), ys, s=34, color=dot, alpha=0.45,
                   edgecolor=bg, linewidth=1, zorder=3, label="one gene" if g == gens[0] else None)
    best = [r["best_so_far"] for r in gen_rows]
    ax.plot(gens, best, color=ink, lw=2, marker="o", ms=8, markeredgecolor=bg, markeredgewidth=2, zorder=4,
            label="best gene found so far")
    for g, b, prev in zip(gens, best, [None] + best[:-1]):
        if b != prev:
            ax.text(g, b + 160, f"{b:,.0f}", ha="center", va="bottom", fontsize=9, color=ink)
    if args.seed_mean:
        ax.axhline(args.seed_mean, color=muted, lw=1.5, ls="--", zorder=2, label=f"original settings ({args.seed_mean:,.0f})")
    ax.set_xlim(gens[0] - 0.5, gens[-1] + 0.5)
    ax.set_xticks(gens)
    ax.set_xlabel("Generation", color=ink)
    ax.set_ylabel("Score (average of 3 seeds)", color=ink)
    ax.set_title("Score of every gene the XQC island trained", loc="left", fontsize=12, color=ink)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, fontsize=9, frameon=False)
    ax.set_facecolor(bg)
    ax.grid(axis="y", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(path, dpi=160)

def _style(ax, muted, grid, bg):
    ax.set_facecolor(bg)
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted)


def plot_pareto(rows, path):
    """Speed vs effort of every trained gene; the Pareto front (no other gene is both
    faster and cheaper) is drawn as a staircase: each step is the best speed you can
    get for at most that much effort."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dot, ink, muted, grid, bg = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    pts = [(r["speed"], r["effort"]) for r in rows if r["status"] == "ok" and r.get("speed", "") != ""]
    front = pareto(pts)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(bg)
    ax.scatter([e for _, e in pts], [s for s, _ in pts], s=30, color=dot, alpha=0.4, edgecolor=bg,
               linewidth=1, zorder=3, label="one gene")
    xs, ys = [], []
    for s, e in front:   # sorted by effort; speed rises along the front
        xs += [e, e]
        ys += [ys[-1] if ys else 0.0, s]
    xs.append(max(e for _, e in pts))
    ys.append(ys[-1])
    ax.plot(xs[1:], ys[1:], color=ink, lw=2, zorder=4, drawstyle="default", label="Pareto front")
    ax.scatter([e for _, e in front], [s for s, _ in front], s=60, color=ink, edgecolor=bg, linewidth=2, zorder=5)
    ax.set_xlabel("Effort (sum of 0.1 x squared motor commands per episode, lower is better)", color=ink)
    ax.set_ylabel("Speed (sum of forward reward per episode)", color=ink)
    ax.set_title("Speed vs effort of every trained gene", loc="left", fontsize=12, color=ink)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2, fontsize=9, frameon=False)
    _style(ax, muted, grid, bg)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def plot_cells(archive, score, path):
    """Best score in each filled foot-contact cell of the final archive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, bg = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    grid_vals = np.full((MAP_BINS, MAP_BINS), np.nan)
    for cell, front in archive.items():
        vals = [score[i[0]] for i in front if i[0] in score]
        if vals:
            grid_vals[cell[1], cell[0]] = max(vals)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor(bg)
    im = ax.imshow(grid_vals, origin="lower", extent=(0, 1, 0, 1), cmap="Blues", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("Best score in cell", color=ink)
    ax.set_xlabel("Back foot on the ground (share of steps)", color=ink)
    ax.set_ylabel("Front foot on the ground (share of steps)", color=ink)
    ax.set_title(f"Filled cells: {len(archive)} of {MAP_BINS * MAP_BINS}", loc="left", fontsize=12, color=ink)
    _style(ax, muted, grid, bg)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()
