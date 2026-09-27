from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


GENE_RE = re.compile(r"xXx[A-Za-z0-9]+")
JOB_ID_RE = re.compile(r"Job ID:\s+(\d+)")
JOB_NAME_RE = re.compile(r"Job name:\s+(.+)")
GEN_RE = re.compile(r"STARTING GENERATION:\s*(\d+)")
ARCHIVE_RE = re.compile(
    r"MAP-Elites:\s*(?P<added>\d+) added,\s*(?P<replaced>\d+) replaced,\s*"
    r"(?P<filled>\d+)/(?P<total>\d+) cells filled \((?P<coverage>[-+0-9.]+)% coverage\)"
)
SUBMITTED_RE = re.compile(r"Submitted batch job\s+(\d+)")
PROMPT_BLOCK_RE = re.compile(
    r"\*+\s*PROMPT TO LLM\s*\*+\n(?P<body>.*?)(?=\n\*+\s*TEXT FROM LLM\s*\*+|\Z)",
    re.DOTALL,
)
LLM_TEXT_BLOCK_RE = re.compile(
    r"\*+\s*TEXT FROM LLM\s*\*+\n(?P<body>.*?)(?=\n\*+\s*CODE FROM LLM\s*\*+|\Z)",
    re.DOTALL,
)


@dataclass
class JobLog:
    path: Path
    relpath: str
    category: str
    job_id: str
    job_name: str
    status: str
    mtime: float
    size: int
    genes: list[str]
    submitted_jobs: list[str]
    latest_generation: int | None
    archive_line: str
    error_hint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-dir", default="pace_vlm_results")
    parser.add_argument("--refresh", type=int, default=30)
    parser.add_argument("--tail-lines", type=int, default=180)
    args, _ = parser.parse_known_args()
    return args


ARGS = parse_args()

st.set_page_config(page_title="LLM-GE Live Monitor", layout="wide")


def js_autorefresh(seconds: int) -> None:
    if seconds <= 0:
        return
    components.html(
        f"""
        <script>
        const delay = {seconds * 1000};
        setTimeout(() => window.parent.location.reload(), delay);
        </script>
        """,
        height=0,
    )


def read_text(path: Path, limit_bytes: int | None = None) -> str:
    try:
        if limit_bytes is None or path.stat().st_size <= limit_bytes:
            return path.read_text(errors="ignore")
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - limit_bytes))
            return handle.read().decode(errors="ignore")
    except OSError:
        return ""


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return {}


def tail(text: str, lines: int) -> str:
    return "\n".join(text.splitlines()[-lines:])


def fmt_age(ts: float) -> str:
    if not ts:
        return "unknown"
    delta = max(0, datetime.now().timestamp() - ts)
    if delta < 90:
        return f"{delta:.0f}s ago"
    if delta < 3600:
        return f"{delta / 60:.1f}m ago"
    if delta < 86400:
        return f"{delta / 3600:.1f}h ago"
    return f"{delta / 86400:.1f}d ago"


def classify_log(text: str) -> tuple[str, str]:
    lowered = text.lower()
    if "traceback" in lowered or "importerror" in lowered or "error:" in lowered:
        hint = ""
        for line in reversed(text.splitlines()):
            if any(token in line.lower() for token in ["traceback", "error", "importerror", "exception"]):
                hint = line.strip()
                break
        return "failed", hint
    if "cancelled" in lowered or "oom" in lowered or "time limit" in lowered:
        return "failed", "cancelled/OOM/time limit"
    if "begin slurm epilog" in lowered or "job done" in lowered or "evalutated all genes" in lowered:
        return "completed", ""
    if text.strip():
        return "running/unknown", ""
    return "empty", ""


def discover_run_dirs(start: Path) -> list[Path]:
    candidates = []
    for path in sorted(start.glob("*")):
        if not path.is_dir():
            continue
        markers = ["run_job_outputs", "stats", "mujoco_rl_output", "behavior_cache", "behavior_captures"]
        if any((path / marker).exists() for marker in markers):
            candidates.append(path)
    return candidates


@st.cache_data(ttl=5, show_spinner=False)
def load_jobs(run_dir_str: str) -> list[JobLog]:
    run_dir = Path(run_dir_str)
    logs = sorted((run_dir / "run_job_outputs").glob("**/*.out"))
    jobs: list[JobLog] = []
    for path in logs:
        text = read_text(path, limit_bytes=900_000)
        status, error_hint = classify_log(text)
        job_id = JOB_ID_RE.search(text)
        job_name = JOB_NAME_RE.search(text)
        gens = [int(match.group(1)) for match in GEN_RE.finditer(text)]
        archive_matches = list(ARCHIVE_RE.finditer(text))
        archive_line = archive_matches[-1].group(0) if archive_matches else ""
        submitted = SUBMITTED_RE.findall(text)
        genes = sorted(set(GENE_RE.findall(text)))
        try:
            stat = path.stat()
        except OSError:
            continue
        jobs.append(
            JobLog(
                path=path,
                relpath=str(path.relative_to(run_dir)),
                category=path.parent.name,
                job_id=job_id.group(1) if job_id else path.stem,
                job_name=job_name.group(1).strip() if job_name else "",
                status=status,
                mtime=stat.st_mtime,
                size=stat.st_size,
                genes=genes,
                submitted_jobs=submitted,
                latest_generation=max(gens) if gens else None,
                archive_line=archive_line,
                error_hint=error_hint,
            )
        )
    return sorted(jobs, key=lambda job: job.mtime, reverse=True)


@st.cache_data(ttl=5, show_spinner=False)
def load_stats(run_dir_str: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    run_dir = Path(run_dir_str)
    stat_roots = [run_dir / "stats", run_dir / "sota" / "MujocoRL" / "stats"]
    stat_files = []
    for root in stat_roots:
        stat_files.extend(sorted(root.glob("*_stats.json")))
    for path in stat_files:
        data = read_json(path)
        if not data:
            continue
        rows.append(
            {
                "gene_id": data.get("gene_id", path.stem.replace("_stats", "")),
                "reward": data.get("mean_reward"),
                "std": data.get("std_reward"),
                "distance": data.get("mean_distance"),
                "foot_0": data.get("foot_contact_0"),
                "foot_1": data.get("foot_contact_1"),
                "timesteps": data.get("timesteps"),
                "train_sec": data.get("train_time_sec"),
                "score_band": data.get("score_band"),
                "path": str(path),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("reward", ascending=False, na_position="last")


@st.cache_data(ttl=5, show_spinner=False)
def load_vlm(run_dir_str: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    run_dir = Path(run_dir_str)
    cache_roots = [run_dir / "behavior_cache", run_dir / "sota" / "MujocoRL" / "behavior_cache"]
    cache_files = []
    for root in cache_roots:
        cache_files.extend(sorted(root.glob("*.json")))
    for path in cache_files:
        data = read_json(path)
        stats = data.get("stats") or {}
        rows.append(
            {
                "parent_gene": data.get("parent_gene"),
                "determinable": data.get("determinable"),
                "uncertainty": data.get("uncertainty"),
                "motion": data.get("motion_pattern"),
                "posture": data.get("visible_posture"),
                "instability": data.get("instability"),
                "backend": stats.get("backend"),
                "observe_sec": stats.get("observe_seconds"),
                "frames": data.get("frames"),
                "model": data.get("observer_model_id"),
                "path": str(path),
            }
        )
    return pd.DataFrame(rows)


def telemetry_brief(telemetry: dict[str, Any]) -> str:
    lines = []
    for episode in telemetry.get("episodes", []):
        index = episode.get("index", "?")
        lines.append(
            "  episode {index}: seed={seed} return={ret:.2f} steps={steps} "
            "terminated={terminated}".format(
                index=index,
                seed=episode.get("seed", "?"),
                ret=float(episode.get("return") or 0.0),
                steps=episode.get("steps", "?"),
                terminated=episode.get("terminated", True),
            )
        )
        lines.append(
            "    distance={distance:.2f} mean_torso_z={mean_z:.3f} "
            "final_torso_z={final_z:.3f} final_pitch={pitch:.3f}".format(
                distance=float(episode.get("distance") or 0.0),
                mean_z=float(episode.get("mean_torso_z") or 0.0),
                final_z=float(episode.get("final_torso_z") or 0.0),
                pitch=float(episode.get("final_pitch") or 0.0),
            )
        )
        lines.append(
            "    foot_contact=({foot0:.3f}, {foot1:.3f}) "
            "mean_ctrl_cost={ctrl:.4f} termination_step={term}".format(
                foot0=float(episode.get("foot_contact_0") or 0.0),
                foot1=float(episode.get("foot_contact_1") or 0.0),
                ctrl=float(episode.get("mean_control_cost") or 0.0),
                term=episode.get("termination_step", episode.get("steps", "?")),
            )
        )
    return "\n".join(lines) if lines else "  (no telemetry available)"


def vlm_input_text(manifest: dict[str, Any], telemetry: dict[str, Any]) -> str:
    frame_entries = manifest.get("frame_index") or []
    frame_lines = [
        f"  frame {entry.get('index')}: {Path(entry.get('file', '')).name} "
        f"(episode={entry.get('episode')}, step={entry.get('step')})"
        for entry in frame_entries
    ]
    return (
        "System prompt:\n"
        "You are a careful locomotion observer. You describe only what the supplied "
        "evidence supports and say when a behaviour cannot be determined. You never "
        "diagnose hyperparameters or prescribe fixes.\n\n"
        "User prompt template with concrete evidence:\n"
        "You are inspecting one trained parent policy from a quality-diversity search over\n"
        "Walker2d-v5 policy source code. Your job is to describe the measured behaviour\n"
        "of this single trained realization, so a separate code-editing model can choose\n"
        "one bounded change.\n\n"
        "Evidence rules: use only the numerical telemetry and the ordered frames. "
        "Frame index 0 is earliest. Return exactly one JSON object.\n\n"
        "MODE: visual\n"
        "FRAMES:\n"
        f"{chr(10).join(frame_lines) if frame_lines else '  (no frames supplied)'}\n\n"
        "NUMERICAL TELEMETRY:\n"
        f"{telemetry_brief(telemetry)}\n"
    )


def find_log_snippets(jobs: list[JobLog], needle: str, context: int = 2) -> list[dict[str, str]]:
    snippets = []
    for job in jobs:
        if not needle or needle not in " ".join(job.genes) and needle not in job.relpath:
            text = read_text(job.path, limit_bytes=1_500_000)
        else:
            text = read_text(job.path, limit_bytes=1_500_000)
        if needle not in text:
            continue
        lines = text.splitlines()
        hits = [i for i, line in enumerate(lines) if needle in line]
        for hit in hits[:4]:
            start = max(0, hit - context)
            end = min(len(lines), hit + context + 1)
            snippets.append(
                {
                    "file": job.relpath,
                    "job_id": job.job_id,
                    "category": job.category,
                    "snippet": "\n".join(lines[start:end]),
                }
            )
    return snippets


def child_genes_for_parent(jobs: list[JobLog], parent_gene: str) -> list[str]:
    children: list[str] = []
    pattern = re.compile(
        rf"Mutating:\s*{re.escape(parent_gene)}\s+and Replacing with:\s*(xXx[A-Za-z0-9]+)"
    )
    for job in jobs:
        if job.category != "islands":
            continue
        text = read_text(job.path, limit_bytes=2_000_000)
        children.extend(pattern.findall(text))
    return sorted(set(children))


@st.cache_data(ttl=5, show_spinner=False)
def load_generated_files(run_dir_str: str) -> pd.DataFrame:
    root = Path(run_dir_str) / "mujoco_rl_output"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/*")):
        if not path.is_file():
            continue
        suffix = path.suffix
        if suffix not in {".txt", ".sh", ".py"} and not path.name.endswith("_feedback.txt"):
            continue
        gene_match = GENE_RE.search(path.name)
        kind = "other"
        if path.name.endswith("_feedback.txt"):
            kind = "vlm feedback"
        elif path.name.endswith("_model.txt"):
            kind = "llm output"
        elif path.name.endswith("_model.sh"):
            kind = "model script"
        elif path.suffix == ".sh":
            kind = "submitted script"
        rows.append(
            {
                "gene": gene_match.group(0) if gene_match else "",
                "kind": kind,
                "file": path.name,
                "path": str(path),
                "modified": path.stat().st_mtime,
            }
        )
    return pd.DataFrame(rows).sort_values("modified", ascending=False) if rows else pd.DataFrame()


def artifact_dir(run_dir: Path, name: str) -> Path:
    direct = run_dir / name
    if direct.exists():
        return direct
    mujoco = run_dir / "sota" / "MujocoRL" / name
    if mujoco.exists():
        return mujoco
    return direct


def run_squeue(job_ids: list[str]) -> pd.DataFrame:
    ids = [job_id for job_id in sorted(set(job_ids)) if job_id.isdigit()]
    if not ids:
        return pd.DataFrame()
    cmd = [
        "squeue",
        "-j",
        ",".join(ids),
        "-h",
        "-o",
        "%i|%T|%M|%L|%R|%j",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return pd.DataFrame()
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("|", 5)
        if len(parts) == 6:
            rows.append(
                {
                    "job_id": parts[0],
                    "state": parts[1],
                    "elapsed": parts[2],
                    "remaining": parts[3],
                    "node/reason": parts[4],
                    "name": parts[5],
                }
            )
    return pd.DataFrame(rows)


def latest_island_text(jobs: list[JobLog]) -> str:
    island_jobs = [job for job in jobs if job.category == "islands"]
    if not island_jobs:
        return ""
    return read_text(island_jobs[0].path, limit_bytes=1_500_000)


def metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label, "n/a" if value is None or value == "" else value, help=help_text)


def status_badge(status: str) -> str:
    if status == "completed":
        return "complete"
    if status == "failed":
        return "failed"
    if status == "running/unknown":
        return "active?"
    return status


workspace = Path.cwd()
run_candidates = discover_run_dirs(workspace)
default_run = workspace / ARGS.run_dir
if default_run.exists() and default_run not in run_candidates:
    run_candidates.insert(0, default_run)

st.title("LLM-GE Live Run Monitor")
st.caption("Read-only local dashboard for Slurm logs, LLM prompts, VLM feedback, stats, and rollout captures.")

with st.sidebar:
    st.header("Run")
    labels = [str(path) for path in run_candidates] or [str(default_run)]
    chosen = st.selectbox("Run directory", labels, index=0)
    custom = st.text_input("Custom run directory", value=chosen)
    run_dir = Path(custom).expanduser().resolve()
    refresh = st.number_input("Refresh every seconds", min_value=0, max_value=3600, value=ARGS.refresh, step=5)
    tail_lines = st.number_input("Log tail lines", min_value=20, max_value=2000, value=ARGS.tail_lines, step=20)
    st.divider()
    st.code(
        "rsync -avz nsani3@login-phoenix.pace.gatech.edu:/storage/ice1/5/8/nsani3/llmge-vlm-run-20260926/{run_job_outputs,behavior_cache,behavior_captures,stats,results,mujoco_rl_output,mujoco_islands_run} ./pace_vlm_results/",
        language="bash",
    )
    st.caption("Run that rsync in another terminal during a live PACE run; this app will notice new files on refresh.")

if refresh:
    js_autorefresh(int(refresh))

if not run_dir.exists():
    st.error(f"Run directory does not exist: {run_dir}")
    st.stop()

jobs = load_jobs(str(run_dir))
stats_df = load_stats(str(run_dir))
vlm_df = load_vlm(str(run_dir))
generated_df = load_generated_files(str(run_dir))
island_text = latest_island_text(jobs)

submitted_ids: list[str] = []
for job in jobs:
    submitted_ids.extend(job.submitted_jobs)
    if job.job_id.isdigit():
        submitted_ids.append(job.job_id)

latest_generation = None
if island_text:
    gens = [int(match.group(1)) for match in GEN_RE.finditer(island_text)]
    latest_generation = max(gens) if gens else None

archive_matches = list(ARCHIVE_RE.finditer(island_text))
latest_archive = archive_matches[-1].groupdict() if archive_matches else {}

top_reward = None
top_gene = None
if not stats_df.empty and "reward" in stats_df:
    finite = stats_df[pd.to_numeric(stats_df["reward"], errors="coerce").apply(lambda x: pd.notna(x) and math.isfinite(x))]
    if not finite.empty:
        top = finite.iloc[0]
        top_reward = f"{top['reward']:.2f}"
        top_gene = top["gene_id"]

top_cols = st.columns(6)
with top_cols[0]:
    metric_card("Generation", latest_generation)
with top_cols[1]:
    filled = latest_archive.get("filled")
    total = latest_archive.get("total")
    metric_card("Archive Cells", f"{filled}/{total}" if filled and total else None)
with top_cols[2]:
    metric_card("Coverage", f"{latest_archive.get('coverage')}%" if latest_archive else None)
with top_cols[3]:
    metric_card("Best Reward", top_reward, top_gene)
with top_cols[4]:
    metric_card("Logs", len(jobs))
with top_cols[5]:
    metric_card("VLM Cached", len(vlm_df))

tabs = st.tabs(["Overview", "Jobs & Logs", "Stats", "VLM Inspector", "Prompts & Files", "Captures", "File Viewer"])

with tabs[0]:
    st.subheader("Run Health")
    status_counts = pd.Series([job.status for job in jobs]).value_counts().rename_axis("status").reset_index(name="count")
    if not status_counts.empty:
        st.dataframe(status_counts, use_container_width=True, hide_index=True)
    failed = [job for job in jobs if job.status == "failed"]
    if failed:
        st.warning(f"{len(failed)} log(s) look failed.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "status": status_badge(job.status),
                        "category": job.category,
                        "job_id": job.job_id,
                        "file": job.relpath,
                        "hint": job.error_hint,
                        "modified": fmt_age(job.mtime),
                    }
                    for job in failed
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    if latest_archive:
        st.subheader("Latest Archive Update")
        st.write(
            f"{latest_archive['added']} added, {latest_archive['replaced']} replaced, "
            f"{latest_archive['filled']}/{latest_archive['total']} cells filled "
            f"({latest_archive['coverage']}% coverage)."
        )
    if not stats_df.empty:
        st.subheader("Reward Leaderboard")
        st.dataframe(stats_df.head(10), use_container_width=True, hide_index=True)
    if island_text:
        with st.expander("Latest island/controller log tail", expanded=False):
            st.code(tail(island_text, int(tail_lines)), language="text")

with tabs[1]:
    st.subheader("Slurm Logs")
    live_query = st.checkbox("Query squeue for discovered job IDs", value=False)
    if live_query:
        queue_df = run_squeue(submitted_ids)
        if queue_df.empty:
            st.info("No matching active jobs returned by squeue, or squeue is unavailable locally.")
        else:
            st.dataframe(queue_df, use_container_width=True, hide_index=True)

    job_rows = [
        {
            "status": status_badge(job.status),
            "category": job.category,
            "job_id": job.job_id,
            "job_name": job.job_name,
            "generation": job.latest_generation,
            "genes_seen": len(job.genes),
            "submitted": ", ".join(job.submitted_jobs[-6:]),
            "modified": fmt_age(job.mtime),
            "file": job.relpath,
            "hint": job.error_hint,
        }
        for job in jobs
    ]
    st.dataframe(pd.DataFrame(job_rows), use_container_width=True, hide_index=True)

    if jobs:
        selected_log = st.selectbox("Open log", [job.relpath for job in jobs])
        selected_job = next(job for job in jobs if job.relpath == selected_log)
        text = read_text(selected_job.path, limit_bytes=2_500_000)
        st.download_button("Download log", text, file_name=selected_job.path.name)
        st.code(tail(text, int(tail_lines)), language="text")

        prompt_match = PROMPT_BLOCK_RE.search(text)
        llm_match = LLM_TEXT_BLOCK_RE.search(text)
        if prompt_match or llm_match:
            st.subheader("LLM Blocks From This Log")
            if prompt_match:
                with st.expander("Prompt sent to LLM", expanded=True):
                    st.code(prompt_match.group("body").strip(), language="text")
            if llm_match:
                with st.expander("Text returned by LLM", expanded=True):
                    st.code(llm_match.group("body").strip(), language="text")

with tabs[2]:
    st.subheader("Evaluation Stats")
    if stats_df.empty:
        st.info("No stats JSON files found.")
    else:
        search = st.text_input("Filter gene", key="stats_filter")
        view = stats_df
        if search:
            view = view[view["gene_id"].str.contains(search, case=False, na=False)]
        st.dataframe(view, use_container_width=True, hide_index=True)
        if not view.empty:
            chart_df = view[["gene_id", "reward"]].dropna().head(20).set_index("gene_id")
            st.bar_chart(chart_df)
            selected_stat = st.selectbox("Open stat JSON", view["path"].tolist())
            st.code(json.dumps(read_json(Path(selected_stat)), indent=2), language="json")

with tabs[3]:
    st.subheader("VLM Inspector")
    if vlm_df.empty:
        st.info("No VLM cache JSON files found.")
    else:
        st.dataframe(vlm_df, use_container_width=True, hide_index=True)
        labels = [
            f"{row.parent_gene} | {row.uncertainty} | {Path(row.path).name[:12]}"
            for row in vlm_df.itertuples()
        ]
        selected_label = st.selectbox("Select observed parent", labels)
        selected_index = labels.index(selected_label)
        selected_vlm = Path(vlm_df.iloc[selected_index]["path"])
        data = read_json(selected_vlm)
        parent_gene = data.get("parent_gene", "")
        cap_dir = artifact_dir(run_dir, "behavior_captures") / parent_gene
        manifest = read_json(cap_dir / "manifest.json")
        telemetry = read_json(cap_dir / "telemetry.json")
        stats = data.get("stats") or {}
        raw = (stats.get("usage") or {}).get("raw_output", "")

        meta_cols = st.columns(5)
        with meta_cols[0]:
            metric_card("Parent", parent_gene)
        with meta_cols[1]:
            metric_card("Observer", data.get("observer_model_id"))
        with meta_cols[2]:
            metric_card("Backend", stats.get("backend"))
        with meta_cols[3]:
            metric_card("Observe Sec", stats.get("observe_seconds"))
        with meta_cols[4]:
            metric_card("Frames", data.get("frames"))

        st.markdown("**Provenance**")
        provenance = {
            "local_run_dir": str(run_dir),
            "cache_file": str(selected_vlm),
            "capture_dir": str(cap_dir),
            "remote_model_checkpoint": manifest.get("model"),
            "remote_genome_file": manifest.get("genome"),
            "captured_at": manifest.get("created_at"),
            "observed_at_unix": stats.get("observed_at"),
            "mode": data.get("mode"),
            "prompt_revision": data.get("prompt_revision"),
            "schema_version": data.get("schema_version"),
            "checkpoint_sha256": data.get("checkpoint_sha256"),
            "genome_sha256": data.get("genome_sha256"),
            "generation_context": "Observed after checkpoint_gen_1 and used as parent feedback during generation 2 mutation"
            if parent_gene != "seed" else "Seed policy observation, not an evolved generation parent",
        }
        st.code(json.dumps(provenance, indent=2), language="json")

        io_tabs = st.tabs(["Video & Frames", "Text Input", "Raw VLM Output", "Mutation Feedback", "Log Evidence", "Full JSON"])
        with io_tabs[0]:
            st.markdown("**Human preview video**")
            video_path = cap_dir / "rollout.mp4"
            if video_path.exists():
                st.video(str(video_path))
            else:
                st.info("No rollout.mp4 found for this capture.")
            frame_entries = manifest.get("frame_index") or []
            frame_paths = [cap_dir / entry.get("file", "") for entry in frame_entries]
            existing_frames = [path for path in frame_paths if path.exists()]
            if existing_frames:
                captions = [
                    f"frame {entry.get('index')} | ep {entry.get('episode')} | step {entry.get('step')}"
                    for entry, path in zip(frame_entries, frame_paths)
                    if path.exists()
                ]
                st.markdown("**Actual image inputs sent to the VLM**")
                st.image([str(path) for path in existing_frames], caption=captions, width=150)
            st.markdown("**Capture manifest**")
            st.code(json.dumps(manifest, indent=2), language="json")

        with io_tabs[1]:
            st.markdown("**Exact kind of text evidence passed alongside the images**")
            st.code(vlm_input_text(manifest, telemetry), language="text")
            with st.expander("Full telemetry JSON"):
                st.code(json.dumps(telemetry, indent=2), language="json")

        with io_tabs[2]:
            st.markdown("**Raw model output before parsing**")
            st.code(raw, language="json")
            parsed_fields = {
                "determinable": data.get("determinable"),
                "visible_posture": data.get("visible_posture"),
                "motion_pattern": data.get("motion_pattern"),
                "instability": data.get("instability"),
                "supporting_frames": data.get("supporting_frames"),
                "uncertainty": data.get("uncertainty"),
                "notes": data.get("notes"),
            }
            st.markdown("**Parsed observer fields**")
            st.code(json.dumps(parsed_fields, indent=2), language="json")

        with io_tabs[3]:
            st.markdown("**Final feedback text injected into the code-mutation prompt**")
            st.code(data.get("feedback_text", ""), language="text")
            children = child_genes_for_parent(jobs, parent_gene)
            if children:
                st.markdown("**Generated child files from this parent**")
                rows = []
                for child in children:
                    for path in sorted((run_dir / "mujoco_rl_output").glob(f"**/{child}*")):
                        rows.append({"child_gene": child, "file": str(path.relative_to(run_dir))})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with io_tabs[4]:
            snippets = find_log_snippets(jobs, parent_gene)
            if not snippets:
                st.info("No log snippets found for this parent gene.")
            for item in snippets:
                with st.expander(f"{item['category']} job {item['job_id']} | {item['file']}", expanded=False):
                    st.code(item["snippet"], language="text")

        with io_tabs[5]:
            st.markdown("**Full cache JSON**")
            st.code(json.dumps(data, indent=2), language="json")

with tabs[4]:
    st.subheader("Generated Prompts, Scripts, and VLM Feedback")
    if generated_df.empty:
        st.info("No generated LLM files found.")
    else:
        kind_filter = st.multiselect("Kind", sorted(generated_df["kind"].unique()), default=sorted(generated_df["kind"].unique()))
        file_view = generated_df[generated_df["kind"].isin(kind_filter)]
        st.dataframe(
            file_view.assign(modified=file_view["modified"].apply(fmt_age)),
            use_container_width=True,
            hide_index=True,
        )
        selected_file = st.selectbox("Open generated file", file_view["path"].tolist())
        path = Path(selected_file)
        text = read_text(path, limit_bytes=2_500_000)
        st.download_button("Download file", text, file_name=path.name)
        lang = "bash" if path.suffix == ".sh" else "python" if path.suffix == ".py" else "text"
        st.code(text, language=lang)

with tabs[5]:
    st.subheader("Rollout Captures")
    capture_root = artifact_dir(run_dir, "behavior_captures")
    captures = sorted([path for path in capture_root.glob("*") if path.is_dir()])
    if not captures:
        st.info("No behavior captures found.")
    else:
        selected_capture = st.selectbox("Capture", [path.name for path in captures])
        cap_dir = capture_root / selected_capture
        manifest = read_json(cap_dir / "manifest.json")
        telemetry = read_json(cap_dir / "telemetry.json")
        cols = st.columns([1, 1])
        with cols[0]:
            st.markdown("**Manifest**")
            st.code(json.dumps(manifest, indent=2), language="json")
        with cols[1]:
            st.markdown("**Telemetry**")
            st.code(json.dumps(telemetry, indent=2), language="json")
        video_path = cap_dir / "rollout.mp4"
        if video_path.exists():
            st.video(str(video_path))
        frames = sorted(cap_dir.glob("*.png")) + sorted(cap_dir.glob("*.jpg"))
        if frames:
            st.image([str(path) for path in frames[:16]], caption=[path.name for path in frames[:16]], width=160)

with tabs[6]:
    st.subheader("Run File Viewer")
    all_files = sorted([path for path in run_dir.glob("**/*") if path.is_file()])
    if not all_files:
        st.info("No files found.")
    else:
        rels = [str(path.relative_to(run_dir)) for path in all_files]
        query = st.text_input("Filter files", key="file_filter")
        options = [rel for rel in rels if query.lower() in rel.lower()] if query else rels
        selected = st.selectbox("File", options)
        path = run_dir / selected
        st.write(f"`{path}`")
        if path.suffix.lower() in {".mp4", ".mov", ".webm"}:
            st.video(str(path))
        elif path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            st.image(str(path), use_container_width=True)
        else:
            text = read_text(path, limit_bytes=2_500_000)
            st.download_button("Download", text, file_name=path.name)
            language = "json" if path.suffix == ".json" else "bash" if path.suffix == ".sh" else "text"
            st.code(text, language=language)
