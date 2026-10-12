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
3. Search efficiency: best-so-far against the number of genes evaluated, and its
   average over an equal evaluation budget (area under that curve).
4. Gene diversity: distinct network architectures and distinct full settings
   among each arm's valid genes (read with Shazeb's gene validator).
5. Post-hoc checks, when posthoc_submit.py jobs have finished: each arm's
   selected genes retrained with new seeds -> re-checked top-5 reward,
   improvement over the seed gene, best reward under changed conditions
   (robustness of the archive), and spread of gait features MESB never optimised.
6. Writes compare.csv, compare.md (tables + verdict) and compare.png.

Verdict (primary metrics, chosen before the results; QD and coverage are
reported but not used, since MESB shapes the grid they are measured on):
MESB helps if its best-under-changed-conditions score is higher AND its
re-checked top-5 reward is not lower than the fixed arm's by more than 2%.

    uv run python scripts/mesb/matched/compare_arms.py --pair xqc_matched_p1
"""

import argparse
import csv
import importlib.util
import itertools
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


def add_evaluation_counts(table):
    total = 0
    for row in table:
        total += row["evaluated"]
        row["evaluations_so_far"] = total


def mean_best_over_budget(table, budget):
    """Average best-so-far over the first ``budget`` evaluations (step function)."""
    area, used = 0.0, 0
    for row in table:
        take = min(row["evaluated"], budget - used)
        if take <= 0 or row["best_so_far"] is None:
            used += max(take, 0)
            continue
        area += take * row["best_so_far"]
        used += take
    return round(area / used, 1) if used else None


ARCH_KEYS = ("hidden_dims_critic", "hidden_dims_actor", "use_batch_norm", "use_weight_norm",
             "use_layer_norm", "pre_activation_bn", "normalize_last_layer", "skip_connections")


def gene_diversity(rows, report_dir):
    """Distinct architectures / full settings among valid genes (missing keys = XQC default)."""
    spec = importlib.util.spec_from_file_location(
        "xqc_gene_validator", REPO_ROOT / "sota" / "XQC" / "train_xqc_gene.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    archs, configs, read = set(), set(), 0
    for r in rows:
        if not r["ok"]:
            continue
        for path in (Path(report_dir) / "data" / "models" / f"network_{r['gene_id']}.py",
                     REPO_ROOT / "sota" / "XQC" / "llmge" / "models" / f"network_{r['gene_id']}.py"):
            if path.exists():
                try:
                    _, cfg = validator.gene_overrides(str(path))
                except Exception:  # noqa: BLE001
                    break
                norm = {k: json.dumps(v) for k, v in cfg.items()}
                archs.add(tuple(norm.get(k, "default") for k in ARCH_KEYS))
                configs.add(tuple(sorted(norm.items())))
                read += 1
                break
    return {"valid_genes_read": read, "distinct_architectures": len(archs),
            "distinct_settings": len(configs)}


def posthoc_summary(pair_dir):
    """Metrics from posthoc_submit.py results, or None if not available yet."""
    ph = pair_dir / "posthoc"
    sel_path = ph / "selection.json"
    if not sel_path.exists():
        return None
    sel = json.loads(sel_path.read_text())
    selection, k_top = sel["genes"], sel.get("k_top", 5)
    results = {}
    for group, genes in selection.items():
        for g in genes:
            f = ph / group / f"{g['gene_id']}.json"
            if f.exists():
                d = json.loads(f.read_text())
                if d.get("status") == "success":
                    results[(group, g["gene_id"])] = dict(d, role=g["role"], run_reward=g.get("run_reward"))
    seed = results.get(("seed", "seed"))
    feats = ("back_foot_contact", "front_foot_contact", "torso_height_mean", "torso_angle_std")
    pooled = [r["gait"]["mean"] for (grp, _), r in results.items() if grp in ARMS and r.get("gait")]
    stats = {k: (st.mean(p[k] for p in pooled), (st.pstdev([p[k] for p in pooled]) or 1.0))
             for k in feats} if pooled else {}
    out = {"seed_recheck": round(seed["mean_reward"], 1) if seed else None, "k_top": k_top, "arms": {}}
    for arm in ARMS:
        mine = [r for (grp, _), r in results.items() if grp == arm]
        expected = len(selection.get(arm, []))
        if not mine:
            out["arms"][arm] = {"checked": 0, "expected": expected}
            continue
        tops = [r["mean_reward"] for r in mine if r["role"] == "top"]
        best = max(r["mean_reward"] for r in mine)
        conds = sorted({c for r in mine for c in r.get("perturbations", {})})
        best_cond = {c: max(r["perturbations"][c]["mean_return"] for r in mine if c in r["perturbations"])
                     for c in conds}
        vecs = [[(r["gait"]["mean"][k] - stats[k][0]) / stats[k][1] for k in feats]
                for r in mine if r.get("gait")]
        pairs = list(itertools.combinations(vecs, 2))
        a = {"checked": len(mine), "expected": expected,
             "recheck_top5_mean": round(st.mean(tops), 1) if tops else None,
             "recheck_best": round(best, 1),
             "run_reward_top5_mean": round(st.mean(r["run_reward"] for r in mine if r["role"] == "top"), 1)
             if tops else None,
             "best_under_changes": {c: round(v, 1) for c, v in best_cond.items()},
             "robustness_score": round(st.mean(best_cond.values()), 1) if best_cond else None,
             "robustness_retained": (round(st.mean(best_cond.values()) / best, 3)
                                     if best_cond and best > 0 else None),
             "gait_spread": round(st.mean(math.dist(x, y) for x, y in pairs), 3) if pairs else None}
        if seed:
            sv = seed["mean_reward"]
            a["improvement_over_seed_best"] = round((best - sv) / abs(sv), 3) if sv else None
            a["improvement_over_seed_top5"] = round((a["recheck_top5_mean"] - sv) / abs(sv), 3) \
                if sv and a["recheck_top5_mean"] is not None else None
        out["arms"][arm] = a
    return out


def verdict(post):
    """Primary rule; None when the post-hoc checks are not complete."""
    if not post or any(post["arms"].get(a, {}).get("checked", 0) == 0 for a in ARMS):
        return None
    f, m = post["arms"]["fixed"], post["arms"]["mesb"]
    if f.get("robustness_score") is None or m.get("robustness_score") is None:
        return None
    robust = m["robustness_score"] > f["robustness_score"]
    ft, mt = f["recheck_top5_mean"] or 0.0, m["recheck_top5_mean"] or 0.0
    not_worse = mt >= ft - 0.02 * abs(ft)
    return robust, not_worse


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
    for arm in ARMS:
        add_evaluation_counts(table[arm])
    budget = min(table[arm][-1]["evaluations_so_far"] for arm in ARMS)
    efficiency = {arm: mean_best_over_budget(table[arm], budget) for arm in ARMS}
    diversity = {arm: gene_diversity(genes[arm], out / arm) for arm in ARMS}
    post = posthoc_summary(pair_dir)

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
    lines = [f"# Matched pair {args.pair}: fixed grid vs MESB", ""]
    lines += ["## Primary metrics (post-hoc checks)", ""]
    if post and all(post["arms"].get(a, {}).get("checked") for a in ARMS):
        fa, ma = post["arms"]["fixed"], post["arms"]["mesb"]
        lines += [f"Genes checked: fixed {fa['checked']}/{fa['expected']}, "
                  f"MESB {ma['checked']}/{ma['expected']}; "
                  f"seed gene re-checked: {post['seed_recheck']}", "",
                  "| metric | fixed | MESB |", "|---|---|---|"]
        k = post["k_top"]
        for key, label in (("recheck_top5_mean", f"re-checked top-{k} reward (new seeds)"),
                           ("run_reward_top5_mean", f"same {k} genes, reward during the run"),
                           ("recheck_best", "re-checked best reward"),
                           ("improvement_over_seed_top5", f"top-{k} vs seed gene (fraction)"),
                           ("improvement_over_seed_best", "best vs seed gene (fraction)"),
                           ("robustness_score", "best reward under changed conditions (mean)"),
                           ("robustness_retained", "share of best reward kept under changes"),
                           ("gait_spread", "gait spread (mean pairwise distance)")):
            lines.append(f"| {label} | {fa.get(key)} | {ma.get(key)} |")
        for cond in sorted(set(fa.get("best_under_changes", {})) | set(ma.get("best_under_changes", {}))):
            lines.append(f"| best under {cond} | {fa['best_under_changes'].get(cond)} | "
                         f"{ma['best_under_changes'].get(cond)} |")
    else:
        lines += ["Post-hoc checks not available yet: run scripts/mesb/matched/posthoc_submit.py "
                  f"--pair {args.pair}, wait for its jobs, then rerun this script."]
    lines += ["", "## Search and diversity during the run", "",
              "| metric | fixed | MESB |", "|---|---|---|",
              f"| mean best-so-far over the first {budget} evaluations | {efficiency['fixed']} | "
              f"{efficiency['mesb']} |"]
    for key in ("best_so_far", "top10_mean_so_far"):
        lines.append(f"| {key} (final) | {last['fixed'][key]} | {last['mesb'][key]} |")
    for key in ("distinct_architectures", "distinct_settings", "valid_genes_read"):
        lines.append(f"| {key} | {diversity['fixed'][key]} | {diversity['mesb'][key]} |")
    totals = {arm: {k: sum(r[k] for r in table[arm]) for k in ("evaluated", "ok", "invalid", "no_result")}
              for arm in ARMS}
    for k in ("evaluated", "ok", "invalid", "no_result"):
        lines.append(f"| genes {k} (all generations) | {totals['fixed'][k]} | {totals['mesb'][k]} |")
    lines += ["", "## Secondary (archive-based; MESB shapes the grid, so not used for the verdict)", "",
              "| metric (final, cumulative) | fixed | MESB |", "|---|---|---|"]
    for key in ("shared_sliding_cells", "shared_sliding_qd", "shared_fixed_cells", "shared_fixed_qd"):
        lines.append(f"| {key} | {last['fixed'][key]} | {last['mesb'][key]} |")
    v = verdict(post)
    lines += ["", "## Verdict", "",
              "Rule chosen before the results: MESB helps if its best reward under changed conditions "
              "is higher AND its re-checked top-5 reward is not lower than the fixed arm's by more "
              "than 2%. One pair is evidence, not proof.", ""]
    if v is None:
        lines.append("- pending: post-hoc checks incomplete")
    else:
        robust, not_worse = v
        lines += [f"- best under changed conditions: {'MESB higher' if robust else 'MESB not higher'}",
                  "- re-checked top-5 reward: "
                  + ("MESB not worse" if not_worse else "MESB worse by more than 2%"),
                  "- verdict for this pair: " + ("MESB helps" if (robust and not_worse) else "not shown")]
    (out / "posthoc_summary.json").write_text(json.dumps(
        {"posthoc": post, "efficiency_budget": budget, "efficiency": efficiency,
         "diversity": diversity}, indent=2))
    (out / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
        for arm, style in (("fixed", "o-"), ("mesb", "s-")):
            g = [r["generation"] for r in table[arm]]
            axes[0].plot([r["evaluations_so_far"] for r in table[arm]],
                         [r["best_so_far"] for r in table[arm]], style, label=arm)
            axes[1].plot(g, [r["top10_mean_so_far"] for r in table[arm]], style, label=arm)
            axes[2].plot(g, [r["shared_sliding_cells"] for r in table[arm]], style, label=arm)
            axes[3].plot(g, [r["shared_sliding_qd"] for r in table[arm]], style, label=arm)
        for ax, title, xl in zip(axes, ("best reward vs genes evaluated", "top-10 mean reward",
                                        "cells on shared grid (secondary)",
                                        "QD on shared grid (secondary)"),
                                 ("genes evaluated", "generation", "generation", "generation")):
            ax.set_title(title)
            ax.set_xlabel(xl)
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
