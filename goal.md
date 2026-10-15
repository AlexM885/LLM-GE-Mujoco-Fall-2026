# VLM aux-reward on HalfCheetah XQC — tracker

**Branch:** `neel/vlm-on-halfcheetah-xqc` (off `shazeb/halfcheetah-xqc`)
**Parent run infrastructure:** Shazeb's multi-objective XQC island on
HalfCheetah-v4 (`train_gene_xqc.py`, `mesb_archive`, foot-contact MO
MAP-Elites).
**Related branch:** `neel/vlm-on-me-base` — same VLM infrastructure
targeting Walker2d-v5 on Alex's ME-Base; iter-5 showed +390 mean / +431
IQM / −27% std over control with the Gemma 3 12B IT bonus-form prompt
at N=10, 500k steps.

## Research question (HalfCheetah variant)

Does the same VLM-authored auxiliary reward pipeline that worked on
Walker2d-v5 generalise to **HalfCheetah-v4 under the XQC multi-objective
MAP-Elites** island? The hypothesis is that per-generation Gemma 3 12B
IT shaping accelerates exploration on HalfCheetah too — specifically,
that the bonus-form prompt + optional PBRS wrap avoids the stand-still
and survival traps we characterized on Walker2d.

Success criteria (pilot, N=10 paired seeds, 500k steps per child):
- `vlm > control` mean with Cohen's d (paired) ≥ 0.3, and
- IQM gain ≥ +300, and
- P(VLM > control) ≥ 0.70

If the pilot passes, scale to the XQC island full chain (multi-gen,
foot-contact MOQD) and compare final MOQD score / hypervolume / Pareto
staircase against the matched baseline.

## Ported from neel/vlm-on-me-base (what works)

All VLM infrastructure code is identical except the emit prompt.
Specifically the following are ported verbatim (119 unit tests passing):

- `AuxRewardWrapper` with centering, STRIDE anneal, **PBRS mode** (Ng
  1999: `F = γΦ(s') − Φ(s)` leaves optimal policy invariant; eliminates
  stand-still and survival traps by construction).
- `aux_reward_sandbox` AST whitelist loader.
- `vlm_aux_reward.py` emitter against Gemma 3 12B IT with:
  - v2-bonus prompt (steers toward bonus-form shaping)
  - optional Eureka-style reflection round (reads previous cache +
    per-seed outcome, revises)
  - --capture-root override so captures outside the default dir work.
- `aux_reward_observer.sh` PACE sbatch (uses the writable
  `/storage/ice1/5/8/nsani3/hf_cache/` cache with the Gemma 3 12B IT
  weights already downloaded for the Walker2d run).
- Analysis: numpy-only `vlm_aux_reward_report.py` with IQM + bootstrap
  CI + probability-of-improvement (replaces broken rliable→arch
  dependency).

## HalfCheetah-specific changes

- `train_rl_vlm.py`: `DEFAULT_ENV_ID = "HalfCheetah-v4"` (was Walker2d-v5).
- `templates/AuxReward/emit.txt`: rewritten for HalfCheetah-v4
  observation layout (17-d but different indices: `obs[8]` is forward
  velocity, `obs[2:5]`/`obs[5:8]` are back/front leg angles, `obs[11:17]`
  are leg angular velocities). Few-shot examples adapted:
  * back-vs-front leg symmetry bonus
  * body pitch stability bonus
  * stride-rhythm bonus (back/front leg alternation)
  * joint-velocity smoothness bonus
- `scripts/vlm_pilot/02_train_arm.sbatch`: `PILOT_ENV` env var now
  parametrises the Gymnasium env id (default HalfCheetah-v4).

## Status log

| timestamp (UTC) | phase | notes |
|---|---|---|
| 2026-10-15T07:30 | branch created | Off `origin/shazeb/halfcheetah-xqc` (sha 764e63dbe). VLM infra ported from `neel/vlm-on-me-base` c-committed sha a8d2af848. 119 unit tests passing locally. emit.txt rewritten for HalfCheetah obs layout. train_rl_vlm.py default env flipped to HalfCheetah-v4. Pilot sbatch now takes `PILOT_ENV` so the same scripts can target either env. HF cache (Gemma 3 12B IT, 35 GB) already populated on PACE scratch from Walker2d run — reused as-is. |
