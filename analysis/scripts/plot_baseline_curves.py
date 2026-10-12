"""Learning curves of the three LLM-GE trainers' seed configs on HalfCheetah-v4
(figure emade.wiki/files/hc_ppo_td3_xqc_sw.png, notebook Oct 5 2026).

Inputs (copied unchanged into analysis/results/hc_baseline_curves/; byte-identical to PACE):
  ppo_s{0,1,2}.csv  <- $XQC_RUNS/baseline_ppo_halfcheetah_v4/seed{0,1,2}/eval.csv
  td3_s{0,1,2}.csv  <- $XQC_RUNS/baseline_td3_halfcheetah_v4/seed{0,1,2}/eval.csv
                       (sota/MujocoRL/baseline_curve.py via pace/launch_baseline.sh, commit 25dc2ee)
  xqc_10seeds.csv   <- $XQC_RUNS/halfcheetah_v4_10seeds/eval.csv (sota/XQC, +experiment=halfcheetah_v4_10seeds)
Each row is one evaluation: mean return of 10 deterministic episodes at env_step,
every 25k steps. The figure shows mean ± 1 std across seeds at each step.

    python analysis/scripts/plot_baseline_curves.py analysis/results/hc_baseline_curves out.png
"""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d, out = sys.argv[1], sys.argv[2]
frames = [pd.read_csv(f"{d}/{a}_s{s}.csv").assign(algo=a.upper()) for a in ("ppo","td3") for s in range(3)]
x = pd.read_csv(f"{d}/xqc_10seeds.csv").assign(algo="XQC")
df = pd.concat(frames + [x[["seed","env_step","return","wall_time_s","algo"]]])
df = df[(df.env_step % 25000 == 0) & (df.env_step > 0) & (df.env_step <= 1_000_000)]
g = df.groupby(["algo","env_step"])["return"].agg(["mean","std","count"]).reset_index()
colors = {"XQC": "#2a78d6", "TD3": "#eb6834", "PPO": "#1baf7a"}
n = {"XQC": 10, "TD3": 3, "PPO": 3}
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
for a in ("XQC","TD3","PPO"):
    s = g[g.algo == a]
    ax.fill_between(s.env_step/1e3, s["mean"]-s["std"], s["mean"]+s["std"], color=colors[a], alpha=0.15, lw=0)
    ax.plot(s.env_step/1e3, s["mean"], color=colors[a], lw=2, label=f"{a} ({n[a]} seeds)")
    last = s.iloc[-1]
    ax.annotate(f"{a}  {last['mean']:,.0f}", (last.env_step/1e3, last["mean"]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color="#333333")
ax.axvline(100, color="#888888", ls=":", lw=1)
ax.text(108, 3200, "LLM-GE budget\n(100k × 3 seeds)", fontsize=8, color="#555555", va="top")
ax.set_xlabel("Environment steps (thousands)"); ax.set_ylabel("Evaluation return (10 episodes)")
ax.set_title("HalfCheetah-v4: seed configs of the three LLM-GE trainers (mean ± 1 std)", fontsize=10, loc="left")
ax.set_xlim(0, 1130); ax.grid(axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.legend(loc="lower right", bbox_to_anchor=(0.97, 0.18), frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(out)
w = df.groupby("algo")["wall_time_s"].max()
for st in (100_000, 250_000, 500_000, 1_000_000):
    print(st, {a: f"{r['mean']:.0f} ± {r['std']:.0f}" for a, r in g[g.env_step == st].set_index("algo").iterrows()})
print("max wall time (s):", w.to_dict())
