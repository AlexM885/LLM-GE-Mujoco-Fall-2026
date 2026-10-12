# MuJoCo RL (TD3 / Walker2d-v5)

This directory trains and evaluates the LLM-evolved policy genes (`models/network_<gene_id>.py`) against `Walker2d-v5` using Stable-Baselines3's TD3. `train_rl.py` is the training entrypoint; `eval.py` defines the environment/fitness spec and runs evaluation episodes.

## Seeding

Both scripts take a seed argument for reproducible runs:

```bash
# Train + evaluate a gene with a fixed seed
python train_rl.py -network models.network_<gene_id> -timesteps 500000 -seed 0

# Re-evaluate a saved model standalone, with a fixed seed
python eval.py --model path/to/model.zip --seed 0
```

- `-seed` (`train_rl.py`) and `--seed` (`eval.py`) both default to `0`.
- In training, the seed pins Python/NumPy/PyTorch RNG state (via SB3's `set_random_seed`), the environment's first reset, the action space, and TD3's own internals.
- In evaluation, each episode is reset with `seed + episode_index`, so episodes differ from each other but repeat exactly across runs.
- Output files are namespaced by seed — `trained_models/<gene_id>_seed<seed>.zip`, `results/<gene_id>_seed<seed>_results.csv`, `stats/<gene_id>_seed<seed>_stats.json` — so re-running the same gene under different seeds doesn't overwrite earlier results.
- The GA driver passes a seed through automatically via `MUJOCO_EVAL_SEED` in `src/cfg/constants_Mujoco.py` (env-var overridable, default `0`), threaded into `EVAL_RUNLINE`.

**Exact reproducibility requires CPU.** GPU runs (CUDA/cuDNN) are not guaranteed to match bit-for-bit even with the same seed; this was verified on CPU-only PACE ICE compute nodes across 5 seeds (2–3 repetitions each), with every same-seed run matching exactly on all recorded metrics and every seed producing a distinct result.
