import argparse
import json
import pathlib
import statistics


KNOWN_ARMS = {
    "off": {
        "best_new": 1428.3395743144476,
        "rewards": [1428.3395743144476, 507.98, 494.70, 472.96, 426.31, 393.19, 392.47, 315.66],
        "note": "prior no-feedback control",
    },
    "telemetry": {
        "best_new": 1144.1934867355794,
        "rewards": [1144.1934867355794, 495.94, 433.46, 420.55, 416.27, 402.90, 351.92, 243.08],
        "note": "prior telemetry control",
    },
    "visual_bookend32": {
        "best_new": 571.0675629015499,
        "rewards": [571.0675629015499, 455.87, 443.51, 425.75, 402.84, 388.92, 387.27, 387.27],
        "note": "prior descriptive Gemma",
    },
    "visual_actionable32": {
        "best_new": 658.202908258407,
        "rewards": [658.202908258407, 528.48, 449.97, 430.92, 424.85, 423.52, 419.70, 367.58],
        "note": "prior actionable Gemma",
    },
    "visual_temporal64": {
        "best_new": 460.1720891731058,
        "rewards": [460.1720891731058, 451.76489071439346, 435.2116944622813, 429.67886133588837,
                    392.8516440910794, 370.7924255812184, 330.0363002826429, 307.97012899407986],
        "note": "prior 64-frame uniform Gemma",
    },
}


def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def reward_from_stats(path):
    data = load_json(path) or {}
    value = data.get("mean_reward", data.get("fitness", data.get("reward")))
    return None if value is None else float(value)


def summarize_arm(path):
    baseline = load_json(path / "baseline_snapshot.json")
    if not baseline:
        return None
    baseline_ids = set(baseline.get("genes", []))
    stats_dir = path / "sota" / "MujocoRL" / "stats"
    rows = []
    for stats in stats_dir.glob("*_stats.json"):
        gene = stats.name[:-11]
        if gene in baseline_ids:
            continue
        reward = reward_from_stats(stats)
        if reward is None:
            continue
        rows.append({"gene": gene, "reward": reward})
    rows.sort(key=lambda row: row["reward"], reverse=True)

    jobs = load_json(path / "arm_jobs.json") or {}
    selected = []
    selected_path = path / "selected_parent_genes.csv"
    if selected_path.exists():
        selected = [x for x in selected_path.read_text().strip().split(",") if x]
    cache_dir = path / "sota" / "MujocoRL" / "behavior_cache"
    cache_count = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0
    return {
        "arm": path.name,
        "path": str(path),
        "baseline_best": baseline.get("best"),
        "baseline_n_stats": baseline.get("n_stats"),
        "n_new": len(rows),
        "best_new": rows[0]["reward"] if rows else None,
        "mean_new": statistics.mean([r["reward"] for r in rows]) if rows else None,
        "median_new": statistics.median([r["reward"] for r in rows]) if rows else None,
        "ranked_new": rows,
        "selected_parent_count": len(selected),
        "feedback_cache_count": cache_count,
        "jobs": jobs,
    }


def enrich_known():
    out = []
    for arm, data in KNOWN_ARMS.items():
        rewards = data["rewards"]
        out.append({
            "arm": arm,
            "path": None,
            "baseline_best": [2441.591774805161, "xXxnzEF5yh9ynYbalpLGoXp2x9a"],
            "baseline_n_stats": 144,
            "n_new": len(rewards),
            "best_new": max(rewards),
            "mean_new": statistics.mean(rewards),
            "median_new": statistics.median(rewards),
            "ranked_new": [{"gene": None, "reward": r} for r in sorted(rewards, reverse=True)],
            "selected_parent_count": None,
            "feedback_cache_count": None,
            "note": data["note"],
        })
    return out


def write_markdown(summaries, output):
    lines = [
        "# MuJoCo VLM Iteration Report",
        "",
        "| Arm | New evals | Best new reward | Mean new reward | Cache | Parents | Note |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(summaries, key=lambda x: (x["best_new"] is not None, x["best_new"] or -1), reverse=True):
        best = "" if row["best_new"] is None else f"{row['best_new']:.2f}"
        mean = "" if row["mean_new"] is None else f"{row['mean_new']:.2f}"
        cache = "" if row["feedback_cache_count"] is None else str(row["feedback_cache_count"])
        parents = "" if row["selected_parent_count"] is None else str(row["selected_parent_count"])
        lines.append(
            f"| {row['arm']} | {row['n_new']} | {best} | {mean} | {cache} | {parents} | {row.get('note', '')} |"
        )
    lines += [
        "",
        "Top new individuals by arm:",
        "",
    ]
    for row in summaries:
        lines.append(f"## {row['arm']}")
        if not row["ranked_new"]:
            lines.append("No completed new stats yet.")
            lines.append("")
            continue
        for item in row["ranked_new"][:8]:
            gene = item["gene"] or ""
            lines.append(f"- {item['reward']:.2f} {gene}")
        lines.append("")
    output.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--include-known", action="store_true")
    parser.add_argument("--json-out", type=pathlib.Path, default=None)
    parser.add_argument("--md-out", type=pathlib.Path, default=None)
    args = parser.parse_args()

    summaries = enrich_known() if args.include_known else []
    for child in sorted(args.root.iterdir() if args.root.exists() else []):
        if child.is_dir():
            summary = summarize_arm(child)
            if summary:
                summaries.append(summary)

    payload = {"root": str(args.root), "arms": summaries}
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.md_out:
        write_markdown(summaries, args.md_out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
