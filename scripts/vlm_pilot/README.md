# VLM aux-reward pilot runbook

End-to-end procedure for comparing the VLM-authored auxiliary training
reward (Direction A) against the no-feedback me-base control, with
paired-seed statistics (Direction C). Scoped so the first PACE run
answers a single yes/no: does the VLM arm beat the control with
`P(VLM > control) > 0.85` and `|Cohen's d| > 0.8` across 5 paired
seeds? If yes, scale to the full run. If no, iterate on the aux-reward
prompt before spending more compute.

## Scripts in this folder

```
00_capture_parents.sh      capture rollouts from a seed gene (needed by the real pilot)
01_emit_aux_reward.sh      run Gemma 3 27B once to emit the aux_reward cache
02_train_arm.sbatch        sbatch array that trains 1 gene × N seeds in one arm
03_collect_and_report.sh   gather per-seed stats and run the paired-stats report
smoke_pilot.sh             local smoke run (no PACE, no GPU, mock VLM)
```

All real-pilot scripts expect `$VLM_PILOT_ROOT` to point at a shared
scratch directory (e.g. `/storage/ice1/5/8/nsani3/vlm-pilot-<date>`).
Everything one experiment produces lives under that root.

## Smoke pilot (local, no GPU)

For validating the plumbing end-to-end before claiming any PACE time.
Takes <30 s.

```
scripts/vlm_pilot/smoke_pilot.sh
```

What it does:
1. Writes a hand-crafted sandbox-valid aux_reward cache JSON to a tmp dir.
2. Confirms `train_rl_vlm._load_aux_reward_cache` consumes it.
3. Writes synthetic per-seed stats for 5 seeds × 2 arms.
4. Runs the paired-seed report and prints it.

If this fails, do not spend GPU time. The schema contract is broken.

## Real pilot on PACE

Shape: **one gene** (the baseline seed network), **5 training seeds per
arm**, **2 arms** (VLM-aux and no-feedback control), **1 generation**,
`250 000` training timesteps per child.

### Step 0: SSH in and set the root

```
ssh nsani3@login-ice.pace.gatech.edu
cd $HOME/path/to/this/repo
export VLM_PILOT_ROOT=/storage/ice1/5/8/nsani3/vlm-pilot-$(date +%Y%m%d-%H%M)
mkdir -p "$VLM_PILOT_ROOT"
```

### Step 1: capture parent rollouts for the VLM to look at

```
bash scripts/vlm_pilot/00_capture_parents.sh
```

This trains the baseline seed gene once (~500k steps), captures 2
rollouts of 1000 steps with video + telemetry, and leaves them under
`$VLM_PILOT_ROOT/captures/network/`.

### Step 2: run the VLM aux_reward emitter

```
bash scripts/vlm_pilot/01_emit_aux_reward.sh
```

Loads Gemma 3 27B on one GPU, sees the captured rollout, writes
`$VLM_PILOT_ROOT/aux_reward_cache.json`. Record validation state
should be `"ok"` or `"repaired"`; if it's `"fallback"`, inspect
`aux_reward_cache.json`'s `fallback_reason` and iterate on
`templates/AuxReward/emit.txt`.

### Step 3: run both arms with paired seeds

The two arms are launched as **sbatch arrays over seeds 0..4**. They can
run in parallel if the cluster has capacity.

```
# VLM-aux arm (reads $VLM_PILOT_ROOT/aux_reward_cache.json)
ARM=vlm sbatch --array=0-4 scripts/vlm_pilot/02_train_arm.sbatch

# No-feedback control arm (cache env var is unset, same binary)
ARM=control sbatch --array=0-4 scripts/vlm_pilot/02_train_arm.sbatch
```

Each array task:
- sets `SEED=$SLURM_ARRAY_TASK_ID`
- for `ARM=vlm`, exports `LLMGE_AUX_REWARD_CACHE`
- for `ARM=control`, unsets it, so the exact same binary runs with no
  shaping
- trains the baseline seed gene for 250k steps with that seed
- writes `$VLM_PILOT_ROOT/<arm>/seed_<n>.json` with
  `{seed, best_reward, mean_reward, foot_contact_0, foot_contact_1}`

### Step 4: paired-seed report

```
bash scripts/vlm_pilot/03_collect_and_report.sh
```

Writes `$VLM_PILOT_ROOT/report.md` with the per-seed table, rliable IQM,
probability-of-improvement, paired t, and Cohen's d.

## Decision rule

| condition                              | action                                                    |
|----------------------------------------|-----------------------------------------------------------|
| `P(VLM > control) >= 0.85` and `d > 0.8` | Scale to full run (Task #8)                               |
| `0.60 <= P < 0.85`                     | Iterate on the emit template, re-run pilot               |
| `P < 0.60`                             | Reconsider the arm (likely issue in λ, anneal, or prompt) |

## Expected pilot cost

- 1 × capture job: 1 × A100/L40, ~45 min
- 1 × VLM emit job: 1 × A100/L40, ~5 min
- 10 × TD3 training jobs (5 seeds × 2 arms): each ~45 min on A100 at 250k
  steps (longer on L40). Parallelizable, so wall-clock ≈ 45 min if the
  cluster has slots.

Target: 15 GPU-hours for the full pilot.
