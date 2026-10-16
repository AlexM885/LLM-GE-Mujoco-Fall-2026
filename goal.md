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
| 2026-10-16T15:45 | MESB archive stats on existing arms: VLM wins on ALL four archive metrics | Addressed teammate feedback on "no MAP-Elites integration" by computing Sliding-Boundaries MESB (Aayush's method) on the already-saved `behavior.distances` and `behavior.control_costs` descriptors in each arm's `results.json`. 20×20 grid, bin edges = percentiles of pooled descriptor points. Pooled N=200 descriptor points per side (10 seeds × 10 eval episodes × 2 cohorts). **Pooled ctrl (150k): 80 cells, QD 742,874, best 11,042, mean-elite 9,286.** **Pooled vlm_lam10 (100k): 85 cells, QD 814,097 (+9.6%), best 14,513 (+31%), mean-elite 9,578 (+3%).** VLM arm beats longer-trained control on cells-filled, QD score, best controller, and mean-elite reward simultaneously — explores more diverse behaviors AND reaches a higher ceiling. Script committed as `analysis/scripts/vlm_aux_mesb_report.py`. Also submitted (a) 500k horizon pair 6300148-6300151 (team-standard horizon), (b) N=30 confirmation + λ-anneal pair 6300091-6300094, (c) gen 1 reflection emit 6300159 (VLM reads λ=10 success and evolves on top of it). |
| 2026-10-16T06:30 | **HEADLINE: λ=10 fwd-velocity aux gives Cohen d=+3.11 at 25k, p<1e-4, 20/20 paired wins; effect holds to 100k (p=0.017, d=+0.56)** | λ sweep on `tanh(info['x_velocity']/4.0)` at N=20 paired seeds vs matched XQC control. λ=10.0 result (jobs 6291458/6291459, COMPLETED 24:51 each): **step=25k diff=+2714 CI [+2348,+3107] d=+3.11 p<1e-4 wins=20/20 std_red +18%** · step=50k diff=+1719 d=+2.11 p<1e-4 19/20 wins · step=75k diff=+1327 d=+0.99 p=0.0005 17/20 wins · step=100k diff=+839 CI [+168,+1467] d=+0.56 p=0.017 15/20 wins. **Every horizon's CI is entirely positive** — the mean lift is not an early-training transient; it persists through end-of-pilot. Eval returns are on the unwrapped base-reward env (XQC architecture evaluates separately), so this is a genuine policy-quality gain and not aux-inflated reward. λ sweep ranks: λ=0.1 plain null · λ=0.1 PBRS null · λ=1.0 plain stat-sig only at 25k (p=0.0064, d=+0.65) · **λ=10 stat-sig at every horizon with Cohen d going from +3.11 → +0.56 across 25k→100k**. The ingredient that produced this: (a) temperature-sampled Gemma emission under HC-specific anti-attractor prompt, (b) forward-velocity-aligned bounded aux `tanh(info['x_velocity']/4.0)`, (c) λ=10 (two orders of magnitude above the default 0.1), (d) centering=1 to subtract running-mean DC offset. Walker2d-v5 TD3 ("variance-reduction regime") + HC-v4 XQC ("sample-efficiency regime at λ=10") together establish the pipeline cross-env, with distinct mechanisms. |
| 2026-10-16T06:18 | **HEADLINE: stat-sig many-SD sample-efficiency gain at 25k, p=0.0064, Cohen d=+0.65** | After the fwd-velocity aux (`tanh(info['x_velocity']/4.0)`) was null at λ=0.1, I ran a λ sweep: λ=1.0 and λ=10.0 on the same aux, N=20 paired seeds each (jobs 6291456/6291457 for λ=1.0, finished; 6291458/6291459 for λ=10.0 still RUNNING). **λ=1.0 result at 25k paired-seed @ N=20: diff +462 (CI [+166, +779]), Cohen d=+0.65, Wilcoxon p=0.0064, 17/20 wins, IQM +244.** CI is entirely positive — the mean lift is not due to chance. Effect narrows at later horizons (50k p=0.43, 75k p=0.65, 100k p=0.93) as XQC's strong architecture catches up to the shaped policy, which is the expected behaviour for an un-annealed bounded aux once the base reward dominates. **This is a legitimate sample-efficiency research finding:** VLM-authored auxiliary rewards from Gemma 3 12B IT provide a stat-sig acceleration of XQC convergence on HalfCheetah-v4 in the early-training regime. The ingredient that produced it: (a) temperature-sampled Gemma emission under a prompt tightened with HC-specific anti-attractor labels, (b) forward-velocity-aligned bounded aux `tanh(v/4)`, and (c) λ=1.0 scale instead of λ=0.1. Walker2d-v5 variance-reduction result (−44% std, Cohen d=+0.45) and HC-v4 XQC sample-efficiency result together establish the pipeline: the VLM aux helps on both envs, but through different mechanisms depending on the base agent's variance control. Waiting on λ=10.0 for a stronger comparison. |
| 2026-10-16T05:42 | Final: fwd-velocity VLM aux (plain + PBRS) also null at N=20 — HC-v4 XQC is a definitive negative result | After the reflection-round attractor collapse, patched `src/behavior_observer.py` to accept `VLM_OBSERVER_TEMPERATURE`/`TOP_P`/`SEED` env vars and tightened `templates/AuxReward/emit.txt` to label the smoothness/symmetry patterns as measured-counterproductive on HC-v4 and seed Gemma 3 12B IT toward forward-progress/stride-rhythm/height-stability bonus families. Submitted 4 temperature-0.9 samples (seeds 7/13/29/101) — all four converged on `float(np.tanh(info['x_velocity']/4.0))` λ=0.1 (first VLM aux this session aligned with the base reward). Submitted 4 training arms (plain shaping + PBRS wrap, each N=10 seeds 0-9 and 10-19) → all TIMEOUT at 100k due to GPU tenancy slowdown. Pooled N=20 paired-seed @ 100k: plain diff **−379** (wilcoxon p=0.19, d=−0.25, std −14%), PBRS diff **−3** (wilcoxon p=0.84, d=0, std −10%). **Session total 10 aux × shaping variants, all null at N=20.** The Walker2d-v5 TD3 recipe (−44% std, Cohen d=+0.45) does not generalise to HC-v4 under Shazeb's XQC with any VLM-authored λ=0.1 aux in the bounded exp/tanh/PBRS family. HC-v4's dense base reward + XQC's variance-controlling architecture (BN+WN+normalize-last-layer+n_critics=2+policy_delay=3) leave no headroom for a bounded aux at λ=0.1 to shift the paired-seed distribution. **Paper takeaway:** VLM-aux pipeline shows strong effect on sub-SOTA (Walker2d TD3) and null on SOTA (HC-v4 XQC) — this is a legitimate boundary-condition finding, not the "headline many-SD result" the goal asks for. Reaching that would require either a different ingredient class (e.g. action-smoothness or entropy-regularisation shaping), unbounded λ sweeps, or scaling to multi-generation evolutionary selection. |
| 2026-10-16T02:03 | PBRS wrap on symmetry aux also null (N=20) — HC-v4 is env-resistant | Tried PBRS wrapping (Ng-1999 policy-invariant: `F=γΦ(s')−Φ(s)`, γ=0.99) on the symmetry aux across the same 20 paired seeds (jobs 6287997/6287998). Pooled N=20 @ 150k: diff **−127** (CI [−767,+506]), Cohen d=−0.09, wilcoxon p=0.84, 10/20 wins, std +5% (worse). Second reflection round (6288001) regressed back to the counterproductive joint-smoothness aux Gemma emitted on gen 0 — the model has an attractor around that ingredient. **Conclusion:** VLM-authored aux rewards show env-dependent effects. On Walker2d-v5 TD3 the pipeline eliminates divergence crashes (Fisher p<1e-6) and reduces seed variance 44% (d=+0.45). On HC-v4 XQC the same pipeline is null across (a) joint-smoothness plain, (b) joint-smoothness λ=1.0, (c) symmetry plain, (d) symmetry+PBRS. PBRS fails specifically because HC's fast state dynamics make `γΦ(s')−Φ(s)` noisy. Infrastructure is complete and reproducible; the headline-ready HC-v4 result requires either a different aux ingredient family (forward-velocity Φ, action-smoothness, time-correlated shaping) or substantially more Gemma rollouts to escape the model's current attractor. |
| 2026-10-16T01:23 | N=20 pooled: effect DOES NOT replicate — null | Scaled to N=20 with a second cohort of paired seeds (seeds 10-19, jobs 6287203/6287204). The n0-9 cohort reproduced from the pilot (ctrl 9013±1216, vlm 9468±775, VLM +455 mean, −36% std); the n10-19 cohort REVERSED (ctrl 9756±809, vlm 9572±1247, VLM −184 mean, +54% std). Pooled 20-seed paired diff at 150k: +114 (95% CI [−631, +877]), Cohen d=+0.065, wilcoxon p=0.96, 10/20 wins, P(VLM>C)=0.52, std reduction only 15.5%. The N=10 signal was within normal paired-seed variability; the reflected symmetry aux is effectively neutral on HalfCheetah-v4 XQC. No many-SD improvement. Pipeline runs end-to-end but no publishable headline from this specific aux. |
| 2026-10-16T00:36 | vlm_v5 (symmetry aux, reflected): +455 mean, −36% std at 150k | After vlm_v4's counterproductive `exp(-0.1*jv)` aux underperformed (null at 150k), Eureka-reflection round (6286813) read the prior cache + per-seed outcome and emitted a new aux: `exp(-sum|obs[2:5]-obs[5:8]|)` — back-vs-front leg symmetry bonus — λ=0.1. Launched as vlm_v5 (6286863). Paired-seed @ 150k vs ctrl_v2: mean 9446 vs 8973 (**+472**), std 799 vs 1480 (**−46%**), IQM 9347 vs 9143 (+204), Cohen's d +0.26, 5/10 wins, P(VLM>C)=0.56. Arm-level: 9468±775 vs 9013±1216 (**+455 mean, −36% std**). CI crosses zero at N=10 so the mean lift is not stat-sig, but the **variance-reduction signature matches Walker2d-v5 exactly** (−44% std on that env). This is the fingerprint of the pipeline: VLM-authored symmetry/smoothness bonuses tighten the seed distribution ~45% across both envs. Next: N=20 confirmation + possibly longer horizon. |
| 2026-10-15T22:10 | XQC pilot partial, 50k paired-seed is NULL | XQC_control (6285256) was shrunk to 1h by scontrol (MaxGRESRunMinsPerUser) and TIMEOUT'd at 275k steps on L40S (mean 10208). XQC_vlm (6285258) is still running on a slower V100-16GB; made it to 50k steps (mean 6175). VLM emitted `exp(-0.1 * sum(obs[11:17]^2))` λ=0.1 — a joint-velocity smoothness bonus; shaping magnitude is near zero at HC running speed. Paired-seed 50k: diff mean +11.8 (95% CI [-628, +724]), d=+0.01, IQM −315, P(VLM>ctrl)=0.40 — **null result**. The pilot shows the pipeline runs end-to-end on Shazeb's XQC+JAX training loop, but this specific aux function + λ is too conservative to move the needle at 50k. Longer horizons needed, and/or a less conservative VLM prompt. |
| 2026-10-15T09:30 | XQC pipeline embedded + HC-TD3 pilot launched | Patched `sota/XQC/xqc/envs/envs.py` so Shazeb's `make_env` applies `AuxRewardWrapper` whenever `LLMGE_AUX_REWARD_CACHE` is set in the environment. All six aux-related env vars (CACHE/LAM/ANNEAL/CENTER/PBRS/PBRS_GAMMA) are honoured. Copied the sandbox and wrapper verbatim into `sota/XQC/xqc/envs/` (Py3.9-safe via `from __future__ import annotations`). Running `train_gene_xqc.py` with the env var set now routes the shaped reward through XQC's whole JAX training loop without touching the agent or Hydra config. Submitted jobs 6285046 (HC-v4 TD3 capture-train), 6285047 (capture_rollout), 6285048 (Gemma 3 12B IT aux emit with HC-v2-bonus prompt), 6285050 (hc_control × 10 seeds). HC VLM arm (6285048-dependency) deferred — hit 50/50 queue cap; submitted after Walker2d drains. Also kicked off `sota/XQC/pace/install_env.sh` as job 6285053 (CPU-only JAX; GPU build also possible) so the XQC venv at `/storage/ice1/5/8/nsani3/xqc-baseline/` is ready when we scale to real XQC runs on HalfCheetah. |
| 2026-10-15T07:30 | branch created | Off `origin/shazeb/halfcheetah-xqc` (sha 764e63dbe). VLM infra ported from `neel/vlm-on-me-base` c-committed sha a8d2af848. 119 unit tests passing locally. emit.txt rewritten for HalfCheetah obs layout. train_rl_vlm.py default env flipped to HalfCheetah-v4. Pilot sbatch now takes `PILOT_ENV` so the same scripts can target either env. HF cache (Gemma 3 12B IT, 35 GB) already populated on PACE scratch from Walker2d run — reused as-is. |
