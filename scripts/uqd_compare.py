"""Compare held-out final evaluations across runs and configurations.

    python scripts/uqd_compare.py mujoco_rl_output/uqd/*_seed* [--stage final_eval|aria/final_eval] \\
        [--report best_only] [--out-prefix mujoco_rl_output/uqd/comparison]

For every run directory with <run>/<stage>/final_eval.json this collects the
corrected-archive metrics and the measured search cost, groups runs by
preset, and writes <prefix>.csv (one row per run), <prefix>.md and
<prefix>.json (mean, SD and count per preset).

Reproducibility-Scores (UQD paper, Sec. VII-C2): each corrected cell's
descriptor variance (trace) or fitness variance is divided by the largest
value observed in that cell across *all runs passed to this script*, and the
score is the sum over the archive of 1 - normalised variance. The value
therefore depends on which runs are compared together; compare scores only
within one invocation.

Only descriptive statistics are reported. Any significance test (the papers
use Wilcoxon rank-sum with Bonferroni/Holm correction across all pairwise
comparisons) must be run on the per-run rows and corrected for the number of
comparisons made.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.uqd.budget import CostLedger  # noqa: E402

METRICS = ("coverage", "filled_cells", "qd_score_raw", "qd_score_normalized", "qd_score_loss",
           "max_fitness", "mean_fitness", "p_score", "mean_p_in_cell", "search_cell_retention",
           "correct_cell_fraction", "fitness_sd_median", "fitness_sem_mean", "ndv_mean",
           "heldout_candidate_failures", "missing_heldout_results")


def load_runs(run_dirs, stage, report):
    runs = []
    for run_dir in run_dirs:
        path = os.path.join(run_dir, stage, "final_eval.json")
        if not os.path.exists(path):
            print(f"skip {run_dir}: no {stage}/final_eval.json", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rep = data["metrics"]["reports"].get(report)
        if rep is None:
            print(f"skip {run_dir}: no '{report}' report", file=sys.stderr)
            continue
        # Search cost always comes from the run itself; for a post-hoc stage
        # (aria/final_eval) the stage's own cost is reported next to it.
        ledger_path = os.path.join(run_dir, "cost_ledger.json")
        search_cost = data.get("search_cost") or {}
        if stage != "final_eval" and os.path.exists(ledger_path):
            with open(ledger_path, encoding="utf-8") as f:
                search_cost = CostLedger.from_list(json.load(f)).summary()
        runs.append({"run_dir": run_dir, "data": data, "report": rep, "search_cost": search_cost,
                     "stage_cost": data.get("search_cost") if stage != "final_eval" else None})
    return runs


def reproducibility_scores(runs):
    max_dvar, max_fvar = {}, {}
    for run in runs:
        for cell, e in run["report"]["corrected_archive"].items():
            if e.get("descriptor_var") is not None:
                max_dvar[cell] = max(max_dvar.get(cell, 0.0), float(sum(e["descriptor_var"])))
            if e.get("fitness_sd") is not None:
                max_fvar[cell] = max(max_fvar.get(cell, 0.0), float(e["fitness_sd"]) ** 2)
    for run in runs:
        d_score = f_score = 0.0
        for cell, e in run["report"]["corrected_archive"].items():
            if e.get("descriptor_var") is not None:
                m = max_dvar[cell]
                d_score += 1.0 - (sum(e["descriptor_var"]) / m if m > 0 else 0.0)
            if e.get("fitness_sd") is not None:
                m = max_fvar[cell]
                f_score += 1.0 - ((e["fitness_sd"] ** 2) / m if m > 0 else 0.0)
        run["reproducibility_score"] = d_score
        run["fitness_reproducibility_score"] = f_score


def row_for(run):
    data = run["data"]
    cost = run["search_cost"]
    stage_cost = run["stage_cost"] or {}
    row = {
        "preset": data.get("preset"), "category": data.get("category"), "run_seed": data.get("run_seed"),
        "stage_labels": json.dumps(data.get("labels", {})), "run_dir": run["run_dir"],
        "heldout_episodes": data.get("heldout_seed_count"),
        "search_env_steps_measured": cost.get("search_env_steps_measured"),
        "search_entries_with_unknown_env_steps": sum(
            v.get("entries_with_unknown_env_steps", 0) for v in cost.values() if isinstance(v, dict)),
        "search_wall_seconds_measured": sum(
            v.get("wall_seconds_measured", 0) for v in cost.values() if isinstance(v, dict)),
        "aria_env_steps_measured": (stage_cost.get("aria") or {}).get("env_steps_measured"),
        "final_eval_env_steps_measured": ((data.get("final_eval_cost") or {}).get("final_eval") or {})
        .get("env_steps_measured"),
        "reproducibility_score": run["reproducibility_score"],
        "fitness_reproducibility_score": run["fitness_reproducibility_score"],
    }
    for m in METRICS:
        row[m] = run["report"].get(m)
    return row


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["preset"], []).append(r)
    numeric = [k for k in rows[0] if k not in ("preset", "category", "run_seed", "stage_labels", "run_dir")]
    out = {}
    for preset, rs in sorted(groups.items()):
        out[preset] = {"runs": len(rs), "category": rs[0]["category"]}
        for k in numeric:
            vals = [r[k] for r in rs if isinstance(r[k], (int, float)) and r[k] is not None]
            out[preset][k] = {"mean": float(np.mean(vals)) if vals else None,
                              "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                              "n": len(vals)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--stage", default="final_eval")
    parser.add_argument("--report", default="best_only", choices=("best_only", "in_cell_selector"))
    parser.add_argument("--out-prefix", default="uqd_comparison")
    args = parser.parse_args()
    runs = load_runs(args.run_dirs, args.stage, args.report)
    if not runs:
        raise SystemExit("no runs with final evaluations found")
    reproducibility_scores(runs)
    rows = [row_for(r) for r in runs]
    with open(args.out_prefix + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarise(rows)
    with open(args.out_prefix + ".json", "w", encoding="utf-8") as f:
        json.dump({"stage": args.stage, "report": args.report, "summary": summary}, f, indent=1)
    cols = ("coverage", "qd_score_normalized", "qd_score_loss", "mean_p_in_cell", "correct_cell_fraction",
            "reproducibility_score", "search_env_steps_measured")
    lines = [f"Stage `{args.stage}`, report `{args.report}`; mean (SD) over runs.", "",
             "| preset | category | runs | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 3)]
    for preset, s in summary.items():
        cells = []
        for c in cols:
            v = s[c]
            cells.append("n/a" if v["mean"] is None else
                         f"{v['mean']:.4g}" + (f" ({v['sd']:.2g})" if v["sd"] is not None else ""))
        lines.append(f"| {preset} | {s['category']} | {s['runs']} | " + " | ".join(cells) + " |")
    with open(args.out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
