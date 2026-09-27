import argparse
import json
import os


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def short(text, limit=160):
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", default="sota/MujocoRL/behavior_captures")
    parser.add_argument("--cache-dir", default="sota/MujocoRL/behavior_cache")
    args = parser.parse_args()

    print("VLM feedback summary")

    captures = []
    if os.path.isdir(args.capture_root):
        for gene in sorted(os.listdir(args.capture_root)):
            manifest = load_json(os.path.join(args.capture_root, gene, "manifest.json"))
            telemetry = load_json(os.path.join(args.capture_root, gene, "telemetry.json"))
            if manifest:
                captures.append((gene, manifest, telemetry or {}))

    print(f"\nCaptured parents: {len(captures)}")
    for gene, manifest, telemetry in captures:
        episodes = telemetry.get("episodes", [])
        returns = [ep.get("return") for ep in episodes if ep.get("return") is not None]
        mean_return = sum(returns) / len(returns) if returns else None
        print(
            f"  {gene}: frames={manifest.get('num_frames')} "
            f"episodes={manifest.get('episodes')} "
            f"capture_sec={manifest.get('capture_seconds')} "
            f"mean_return={mean_return:.2f}" if mean_return is not None else
            f"  {gene}: frames={manifest.get('num_frames')} "
            f"episodes={manifest.get('episodes')} "
            f"capture_sec={manifest.get('capture_seconds')}"
        )

    records = []
    if os.path.isdir(args.cache_dir):
        for name in sorted(os.listdir(args.cache_dir)):
            if not name.endswith(".json"):
                continue
            record = load_json(os.path.join(args.cache_dir, name))
            if record:
                records.append(record)

    print(f"\nCached VLM observations: {len(records)}")
    for record in records:
        stats = record.get("stats", {})
        print(
            f"  {record.get('parent_gene')}: mode={record.get('mode')} "
            f"backend={stats.get('backend')} frames={stats.get('frame_count')} "
            f"observe_sec={stats.get('observe_seconds')} "
            f"uncertainty={record.get('uncertainty')}"
        )
        print(f"    posture: {short(record.get('visible_posture'))}")
        print(f"    motion: {short(record.get('motion_pattern'))}")
        print(f"    instability: {short(record.get('instability'))}")


if __name__ == "__main__":
    main()
