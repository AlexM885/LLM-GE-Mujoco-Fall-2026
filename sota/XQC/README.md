# XQC baseline (resumable, PACE-ICE)

XQC (Palenicek et al., ICLR 2026, [paper](https://arxiv.org/abs/2509.25174)) vendored from
[danielpalenicek/xqc](https://github.com/danielpalenicek/xqc) at commit `9a6832b` (MIT, see
`LICENSE`, upstream readme in `README.upstream.md`). The upstream training script is kept
unchanged as `train_parallel.py`. `train_xqc.py` runs the same loop with checkpoint/resume,
Slurm signal handling and a fixed per-run output format.

Changes vs upstream:
- `train_xqc.py`, `xqc/checkpoint.py`: resume, checkpoints, logging (the learner itself is unchanged)
- the training action space is seeded (upstream warm-up actions were unseeded, so fixed-seed runs were not reproducible)
- `requirements.txt`: MuJoCo-only subset of upstream pins; `requirements.lock.txt` is the resolved PACE env

## Configuration

Everything is a Hydra config value, set in `conf/config.yaml`, an experiment file in
`conf/experiment/`, or on the command line. No source edits per experiment.

| Key | Default | Meaning |
|---|---|---|
| `env` | `h1-walk-v0` | env name, e.g. `env=HalfCheetah-v4` (select with `env=`, not `env.name=`) |
| `seed` / `num_seeds` | 0 / 10 | first seed / number of seeds run in parallel (vmap) |
| `max_steps` | 1,000,000 | total env steps |
| `eval_interval` / `eval_episodes` | 50,000 / 10 | evaluation schedule |
| `checkpoint_interval` | 50,000 | env steps between checkpoints (plus one on every stop signal) |
| `keep_checkpoints` | 2 | complete checkpoints kept |
| `save_replay_buffer` | true | include the replay buffer (needed for a faithful resume) |
| `run_dir` | `./runs/<env>/seed<seed>` | all outputs for the run |
| `agent.*` | `conf/agent/xqc.yaml` | learner hyperparameters, e.g. `agent.critic_lr=1e-4` |
| `batch_size`, `updates_per_step`, `replay_buffer_size`, `start_training` | 256, 2, 1M, 5000 | training loop |

Experiments: `+experiment=halfcheetah_v4_baseline` (1M steps, 1 seed, eval every 25k),
`+experiment=smoke` (a few thousand steps).

## Outputs (`run_dir`)

| File | Content |
|---|---|
| `eval.csv` | `env, seed, env_step, return, wall_time_s, slurm_job_id`, one row per seed per eval |
| `train_metrics.jsonl` | learner metrics every `log_interval` steps |
| `summary.json` | last and best mean return, steps, wall time, `finished` |
| `config.json` | fully resolved config |
| `checkpoints/step_*/`, `checkpoints/latest` | agent, optimizer, RNG, replay buffer, env state |
| `DONE` | written when `max_steps` is reached |
| `run.args`, `chain/`, `slurm/` | Hydra overrides, job chain bookkeeping and Slurm logs (PACE only) |

## Run on PACE-ICE

Everything lives in scratch under `XQC_BASE` (default `~/scratch/xqc-baseline`, resolved to
`/storage/ice1/...`): the repo clone, venv, caches, tmp and runs. `pace/env.sh` redirects
pip/uv/XDG/wandb/matplotlib caches and TMPDIR there.

```bash
# one-time setup, on a login node
B=$(readlink -f ~/scratch)/xqc-baseline; mkdir -p $B && cd $B
git clone --branch shazeb/xqc-baseline --single-branch https://github.com/AlexM885/LLM-GE-Mujoco-Fall-2026.git repo
cd repo && bash sota/XQC/pace/install_env.sh
sbatch -p ice-gpu --gres=gpu:1 sota/XQC/pace/check_env.sbatch   # compute-node check -> $B/checks/

# launch: RUN_NAME, chain length, then Hydra overrides
bash sota/XQC/pace/launch_chain.sh halfcheetah_v4_s0 -n 2 -p ice-gpu -- +experiment=halfcheetah_v4_baseline

# resume or extend: same RUN_NAME, no overrides (reuses run.args, continues from the latest checkpoint)
bash sota/XQC/pace/launch_chain.sh halfcheetah_v4_s0 -n 1 -p ice-gpu

# monitor / stop
squeue -u $USER
cat $B/runs/halfcheetah_v4_s0/summary.json
scancel $(cat $B/runs/halfcheetah_v4_s0/chain/jobs.txt)
```

How the chain works:
- Each job is at most 8 h (the launcher refuses more). Slurm sends `USR1` 300 s before the
  limit (`--signal-lead`); the job script forwards it and Python checkpoints and exits 0.
  `max_runtime_s` is a fallback that stops 120 s before the limit.
- Jobs are linked with `afterany`, so a crash or node failure does not strand the rest of
  the chain; the next job resumes from the latest checkpoint. `afterok` would leave every
  later job pending forever after one non-zero exit.
- A job that finds or writes `DONE` cancels the chain's remaining pending jobs.
- Two consecutive jobs that exit without advancing the checkpoint cancel the rest of the
  chain (no crash loops). Details are in `run_dir/chain/history.log`.
- A run never silently restarts from scratch: if checkpoint dirs exist but none is
  complete, or the config differs from the checkpoint's (other than logging/eval keys),
  training refuses to start.

## Run locally

```bash
python train_xqc.py +experiment=smoke run_dir=runs/smoke     # Ctrl-C-safe: kill -USR1 <pid> checkpoints and exits
python train_xqc.py +experiment=smoke run_dir=runs/smoke     # same command resumes
python tools/compare_runs.py runs/A runs/B                   # check a resumed run against an uninterrupted one
```
