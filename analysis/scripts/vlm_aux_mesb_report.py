"""MESB archive stats from existing XQC results.json descriptors.

Each arm's results.json has behavior.distances and behavior.control_costs
lists of length num_seeds*num_eval_episodes (100). We bin those into a
20x20 Sliding-Boundaries archive (Aayush's method) with cell fitness =
the max corresponding base reward in that cell, then report coverage,
QD score, cells filled, and best-per-cell — the exact metrics Alex and
Shazeb report.
"""
import json
import numpy as np
import sys
from pathlib import Path

root = Path("/storage/ice1/5/8/nsani3/vlm-pilot-hc-20261015-2016")

ARMS = {
    "control_v2  (seeds 0-9, 150k)":    "xqc_control_v2",
    "control_n20 (seeds 10-19, 150k)":  "xqc_control_n20",
    "vlm_lam10_a (seeds 0-9, 100k)":    "xqc_vlm_fwdvel_lam10_a",
    "vlm_lam10_b (seeds 10-19, 100k)":  "xqc_vlm_fwdvel_lam10_b",
}

def load_arm(arm):
    j = json.load(open(root/arm/"results.json"))
    beh = j["behavior"]
    rewards = j["rewards"]  # per-episode base rewards
    # 100 (distance, control_cost, reward) triples
    d = np.array(beh["distances"])
    c = np.array(beh["control_costs"])
    r = np.array(rewards)
    assert len(d) == len(c) == len(r), f"{arm}: lengths {len(d)}/{len(c)}/{len(r)}"
    return d, c, r

# Collect every point from every arm, then set MESB boundaries from pooled quantiles.
all_pts = []
for arm in ARMS.values():
    d, c, r = load_arm(arm)
    all_pts.append((d, c, r))

d_pool = np.concatenate([p[0] for p in all_pts])
c_pool = np.concatenate([p[1] for p in all_pts])

# Sliding-Boundaries: 20x20 grid, bin edges at percentiles of pooled descriptors.
NBINS = 20
d_edges = np.quantile(d_pool, np.linspace(0, 1, NBINS+1))
c_edges = np.quantile(c_pool, np.linspace(0, 1, NBINS+1))
# nudge last edge up to include it
d_edges[-1] += 1e-9; c_edges[-1] += 1e-9

def archive(arm_pts):
    d, c, r = arm_pts
    cells = {}
    for di, ci, ri in zip(d, c, r):
        bi = int(np.searchsorted(d_edges, di, side='right') - 1)
        bj = int(np.searchsorted(c_edges, ci, side='right') - 1)
        bi = max(0, min(NBINS-1, bi))
        bj = max(0, min(NBINS-1, bj))
        k = (bi, bj)
        if k not in cells or cells[k] < ri:
            cells[k] = ri
    occupied = len(cells)
    coverage = occupied / (NBINS*NBINS)
    qd_score = float(sum(cells.values()))
    best = float(max(cells.values()))
    mean_elite = float(np.mean(list(cells.values())))
    return occupied, coverage, qd_score, best, mean_elite

print(f"MESB (20x20 Sliding Boundaries, pooled percentile edges), N=100 pts/arm:")
print(f"  d_range  = [{d_pool.min():.1f}, {d_pool.max():.1f}]")
print(f"  c_range  = [{c_pool.min():.1f}, {c_pool.max():.1f}]")
print()
print(f"{'arm':40} occ   cov%     QD-score    best   mean-elite")
print(f"{'-'*40} ----  ------   ---------   ------   --------")
for name, arm in ARMS.items():
    occ, cov, qd, best, me = archive(load_arm(arm))
    print(f"{name:40} {occ:4d}  {cov*100:5.1f}%  {qd:9,.0f}   {best:6,.0f}   {me:7,.0f}")

# Pool VLM arms vs pool control arms
def pool(arms):
    pts = [load_arm(a) for a in arms]
    return (np.concatenate([p[0] for p in pts]),
            np.concatenate([p[1] for p in pts]),
            np.concatenate([p[2] for p in pts]))

ctrl_pts = pool(["xqc_control_v2", "xqc_control_n20"])
vlm_pts = pool(["xqc_vlm_fwdvel_lam10_a", "xqc_vlm_fwdvel_lam10_b"])
print()
print(f"POOLED (N=200 pts each)")
print(f"{'ctrl pooled':40}  ", archive(ctrl_pts))
print(f"{'vlm_lam10 pooled':40}  ", archive(vlm_pts))
