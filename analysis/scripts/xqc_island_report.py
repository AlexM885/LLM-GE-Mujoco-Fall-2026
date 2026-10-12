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
        })
        for src, dst in [(model, "models"), (f"{LLMGE}/results/{gene}_results.csv", "results"), (stats_path, "stats")]:
            if os.path.exists(src):
                shutil.copy2(src, f"{args.out}/data/{dst}/")
    rows.sort(key=lambda r: (r["generation"], r["status"] != "ok", -(r["mean_reward"] or 0), r["gene_id"]))
    with open(f"{args.out}/genes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    score = {r["gene_id"]: r["mean_reward"] for r in rows if r["status"] == "ok"}
    gen_rows, best = [], -np.inf
    for g in [0] + gens:
        rs = [r for r in rows if r["generation"] == g]
        ok = [r["mean_reward"] for r in rs if r["status"] == "ok"]
        best = max([best] + ok)
        cells = qd = ""
        ckpt = glob.glob(f"{args.run}/island_*/checkpoint_gen_{g}.pkl")
        if ckpt:
            with open(ckpt[0], "rb") as f:
                archive = pickle.load(f)["MAP_ELITES_ARCHIVE"]
            cells = len(archive)
            qd = round(sum(score.get(ind[0], 0.0) for ind in archive.values()), 1)
        gen_rows.append({
            "generation": g, "evaluated": len(rs), "ok": len(ok),
            "invalid": sum(r["status"] == "invalid" for r in rs),
            "no_result": sum(r["status"] == "no_result" for r in rs),
            "crossover_invalid": sum(r["status"] == "invalid" and r["origin"] == "crossover" for r in rs),
            "crossover_total": sum(r["origin"] == "crossover" for r in rs),
            "best_this_gen": round(max(ok), 1) if ok else "", "best_so_far": round(best, 1),
            "median_ok": round(float(np.median(ok)), 1) if ok else "",
            "archive_cells": cells, "archive_qd_score": qd,
        })
    with open(f"{args.out}/generations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gen_rows[0]))
        w.writeheader()
        w.writerows(gen_rows)
    plot(rows, gen_rows, args, f"{args.out}/fitness_by_generation.png")
    for r in gen_rows:
        print(r)


def plot(rows, gen_rows, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok_c, bad_c, none_c, ink, muted, grid = "#2a78d6", "#eb6834", "#8d8b85", "#0b0b0b", "#52514e", "#e4e3df"
    gens = [r["generation"] for r in gen_rows]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    fig.patch.set_facecolor("#fcfcfb")
    rng = np.random.default_rng(0)
    for g in gens:
        ys = [r["mean_reward"] for r in rows if r["generation"] == g and r["status"] == "ok"]
        ax.scatter(g + rng.uniform(-0.18, 0.18, len(ys)), ys, s=36, color=ok_c, alpha=0.75,
                   edgecolor="#fcfcfb", linewidth=1, zorder=3)
    ax.step(gens, [r["best_so_far"] for r in gen_rows], where="post", color=ink, lw=2, zorder=4)
    top = max((r for r in rows if r["status"] == "ok"), key=lambda r: r["mean_reward"])
    ax.annotate(f"best so far: {top['mean_reward']:,.0f}\n(gen {top['generation']}, {top['gene_id'][:8]}…)",
                (top["generation"], top["mean_reward"]), xytext=(-12, 0), textcoords="offset points",
                ha="right", va="center", fontsize=9, color=ink)
    if args.seed_mean:
        ax.axhline(args.seed_mean, color=muted, lw=1.5, ls="--", zorder=2)
        ax.text(gens[-1] + 0.45, args.seed_mean, f"seed config,\nsame seeds: {args.seed_mean:,.0f}",
                va="center", ha="right", fontsize=8.5, color=muted,
                bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1))
    ax.set_ylabel("Return (mean of 3 seeds, 100k steps)", color=ink)
    ax.set_title("Every evaluated gene, XQC island on HalfCheetah-v4", loc="left", fontsize=12, color=ink)
    ok_n = np.array([r["ok"] for r in gen_rows]); bad_n = np.array([r["invalid"] for r in gen_rows])
    none_n = np.array([r["no_result"] for r in gen_rows])
    bx.bar(gens, ok_n, 0.6, color=ok_c, label="trained", edgecolor="#fcfcfb", linewidth=2)
    bx.bar(gens, bad_n, 0.6, bottom=ok_n, color=bad_c, label="invalid gene", edgecolor="#fcfcfb", linewidth=2)
    bx.bar(gens, none_n, 0.6, bottom=ok_n + bad_n, color=none_c, label="timed out (1.5 h)",
           edgecolor="#fcfcfb", linewidth=2)
    bx.set_ylabel("Genes", color=ink)
    bx.set_xlabel("Generation", color=ink)
    bx.set_xticks(gens)
    bx.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3, bbox_to_anchor=(0, 1.32))
    for a in (ax, bx):
        a.set_facecolor("#fcfcfb")
        a.grid(axis="y", color=grid, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(muted)
        a.tick_params(colors=muted)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(path, dpi=160)


if __name__ == "__main__":
    main()
