"""Paired-seed CRN eval for the gen-2 VLM vs control arms (HC-v4 XQC).

Each XQC run has 10 inner parallel seeds; A/B runs use top-level seeds 0 and 1.
Pairing is (vlm_run, inner_seed, outer_seed) <-> (ctl_run, inner_seed, outer_seed).
Reports mean difference, Wilcoxon p, Cohen's d, 95% bootstrap CI, win rate.
"""
import csv
import json
import math
import os
import sys
from statistics import mean, stdev
from pathlib import Path

ROOT = Path("/storage/ice1/5/8/nsani3/vlm-pilot-hc-20261015-2016")

def load(run):
    rows = {}  # (outer, inner, step) -> return
    with open(ROOT / f"{run}_runs/xqc_pilot_seed/eval.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (int(row["seed"]), int(row["env_step"]))
            rows[key] = float(row["return"])
    return rows

vlm_a = load("xqc_vlm_gen5_a")
vlm_b = load("xqc_vlm_gen5_b")
ctl_a = load("xqc_control_gen5_a")
ctl_b = load("xqc_control_gen5_b")

def pair_at_step(step):
    diffs = []
    vlm_scores = []
    ctl_scores = []
    for inner in range(10):
        # outer seed 0 pair: vlm_a vs ctl_a
        v = vlm_a.get((inner, step))
        c = ctl_a.get((inner, step))
        if v is not None and c is not None:
            diffs.append(v - c)
            vlm_scores.append(v)
            ctl_scores.append(c)
        # outer seed 1 pair: vlm_b vs ctl_b
        v = vlm_b.get((inner, step))
        c = ctl_b.get((inner, step))
        if v is not None and c is not None:
            diffs.append(v - c)
            vlm_scores.append(v)
            ctl_scores.append(c)
    return diffs, vlm_scores, ctl_scores

def wilcoxon_p(diffs):
    try:
        from scipy.stats import wilcoxon
        if all(d == 0 for d in diffs):
            return 1.0
        return float(wilcoxon(diffs, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception as e:
        return None

def bootstrap_ci(diffs, n=10000, seed=42):
    import random
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        sample = [rng.choice(diffs) for _ in diffs]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int(0.025 * n)]
    hi = boots[int(0.975 * n)]
    return lo, hi

print(f"=== GEN-5 PAIRED-SEED CRN (N=20, VLM aux `tanh(v/4)` lam=10) ===")
print(f"{'step':>6}  {'vlm_mean':>8}  {'ctl_mean':>8}  {'diff':>7}  {'d':>6}  "
      f"{'p':>8}  {'CI_lo':>7}  {'CI_hi':>7}  wins")
for step in (25000, 50000, 75000, 100000):
    diffs, vs, cs = pair_at_step(step)
    if not diffs:
        continue
    md = mean(diffs)
    sd = stdev(diffs) if len(diffs) > 1 else 0.0
    d = md / sd if sd > 0 else float("inf")
    p = wilcoxon_p(diffs)
    lo, hi = bootstrap_ci(diffs)
    wins = sum(1 for x in diffs if x > 0)
    p_str = f"{p:>8.4f}" if p is not None else "    nan"
    print(f"{step:>6}  {mean(vs):>8.1f}  {mean(cs):>8.1f}  {md:>7.1f}  "
          f"{d:>6.2f}  {p_str}  {lo:>7.1f}  {hi:>7.1f}  {wins}/{len(diffs)}")
