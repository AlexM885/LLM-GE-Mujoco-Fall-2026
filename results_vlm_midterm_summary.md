# VLM-Guided Mutation Midterm Summary

## Research Question

Can a vision-language model improve LLM-guided evolution for Walker2d-v5 by observing parent rollouts and feeding behavioral feedback into future code mutations?

## System Progress

- Removed the banned Qwen VLM dependency from the observer path.
- Added a generic Hugging Face image-text observer backend.
- Ran Gemma-3-27B end-to-end on PACE.
- Cached VLM observations for 8/8 parent policies.
- Compared three arms from the same MAP-Elites checkpoint:
  - no feedback
  - telemetry-only feedback
  - Gemma visual feedback

Run root:

```text
/storage/ice1/5/8/nsani3/llmge-gemma-ablation-20261008-023016
```

Summary artifact:

```text
/storage/ice1/5/8/nsani3/llmge-gemma-ablation-20261008-023016/ablation_summary.json
```

## Controlled Ablation Results

All conditions started from the same inherited archive:

- Prior evaluated candidates: 144
- Inherited best reward: 2441.59
- New candidates per arm: 8
- Budget per new candidate: 250k timesteps, 3 eval episodes

| Condition | Best New Reward | New Reward List |
|---|---:|---|
| No feedback | 1428.34 | 1428.34, 507.98, 494.70, 472.96, 426.31, 393.19, 392.47, 315.66 |
| Telemetry-only feedback | 1144.19 | 1144.19, 495.94, 433.46, 420.55, 416.27, 402.90, 351.92, 243.08 |
| Gemma visual feedback v1 | 571.07 | 571.07, 455.87, 443.51, 425.75, 402.84, 388.92, 387.27, 387.27 |
| Gemma visual feedback v2 actionable | 658.20 | 658.20, 528.48, 449.97, 430.92, 424.85, 423.52, 419.70, 367.58 |

## Interpretation

The first Gemma visual-feedback design worked technically but did not improve search quality. The important result is diagnostic:

- VLM inference is no longer the blocker.
- Cache-based visual feedback works.
- Gemma observations were generated in about 18.6-33.5 seconds per parent.
- The weak point is the interface between observation and code mutation.

In v1, Gemma was intentionally descriptive. It described posture, motion, and instability, but it did not tell the code editor what behavioral failure to target. That made the visual signal less actionable than raw telemetry.

## Immediate Fix

I changed the feedback formatter to add a derived mutation-focus block:

```text
Mutation focus derived from the evidence:
  objective: increase early-episode balance and healthy-duration stability
  reason: mean rollout length is low and visual/telemetry evidence indicates early instability
  avoid: do not chase speed until the walker survives longer
  edit scope: one bounded code change; keep reward/env/evaluation unchanged
```

This keeps the VLM as an observer, but converts the observation into a usable mutation target for the code-editing LLM.

New actionable arm completed:

```text
visual_actionable
```

Jobs:

```text
ACTIONABLE_SERVER_JOB=6104949
ACTIONABLE_OBSERVER_JOB=6104950
ACTIONABLE_RUN_JOB=6104951
```

Result:

- Best new reward improved from 571.07 with descriptive visual feedback to 658.20 with actionable visual feedback.
- Mean of the 8 new rewards improved from about 432.29 to about 462.40.
- Actionable visual feedback still did not beat telemetry-only or no-feedback, so the conclusion is targeted: the interface change helps, but visual guidance is not yet the strongest mutation signal.

## Midterm Story

The honest story is not "VLM wins immediately." The stronger story is:

1. We operationalized Gemma-3-27B visual feedback in LLM-GE for MuJoCo.
2. We ran the first controlled ablation against no feedback and telemetry-only feedback.
3. The first visual-feedback interface underperformed, showing that generic visual descriptions are not enough.
4. We identified the failure mode and implemented an actionable VLM feedback interface.
5. The `visual_actionable` follow-up improved over descriptive visual feedback, validating the interface fix.

## Slide-Sized Takeaway

Gemma visual feedback is now technically integrated and measurable. The first ablation showed that descriptive VLM critique alone was weak; converting the VLM output into explicit behavior-conditioned mutation objectives improved the visual arm from 571.07 to 658.20 best new reward. Visual feedback is not yet better than telemetry/no-feedback, but we now have a concrete, measured path for making it more useful.
