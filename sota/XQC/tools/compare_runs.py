"""Compare two XQC run dirs: final checkpoint (agent, buffer, RNG) and eval.csv.

Used to check that an interrupted-and-resumed run matches an uninterrupted
one. Bit-exact equality is expected on CPU. On GPU, XLA reductions are not
guaranteed deterministic, so small float differences there are normal.

  python tools/compare_runs.py RUN_A RUN_B
"""

import csv
import os
import pickle
import sys

import numpy as np


def latest(run_dir):
    root = os.path.join(run_dir, "checkpoints")
    return os.path.join(root, open(os.path.join(root, "latest")).read().strip())


def max_abs_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.dtype.kind in "fc":
        return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    return 0.0 if np.array_equal(a, b) else float("inf")


def main(a_dir, b_dir):
    ca, cb = latest(a_dir), latest(b_dir)
    print(f"A: {ca}\nB: {cb}")
    ok = True

    la = pickle.load(open(os.path.join(ca, "agent.pkl"), "rb"))
    lb = pickle.load(open(os.path.join(cb, "agent.pkl"), "rb"))
    diffs = [max_abs_diff(x, y) for x, y in zip(la, lb)]
    n_exact = sum(d == 0.0 for d in diffs)
    print(f"agent arrays: {len(la)} vs {len(lb)}, bit-exact {n_exact}/{len(diffs)}, max abs diff {max(diffs):.3e}")
    ok &= len(la) == len(lb) and n_exact == len(diffs)

    with np.load(os.path.join(ca, "buffer.npz")) as ba, np.load(os.path.join(cb, "buffer.npz")) as bb:
        for k in ba.files:
            d = max_abs_diff(ba[k], bb[k])
            print(f"buffer.{k}: max abs diff {d:.3e}")
            ok &= d == 0.0

    sa = pickle.load(open(os.path.join(ca, "state.pkl"), "rb"))
    sb = pickle.load(open(os.path.join(cb, "state.pkl"), "rb"))
    same_np = all(max_abs_diff(x, y) == 0.0 for x, y in zip(sa["numpy_random"], sb["numpy_random"]))
    print(f"env_step {sa['env_step']} vs {sb['env_step']}, loop {sa['loop']['i']} vs {sb['loop']['i']}, "
          f"numpy RNG equal {same_np}, observations diff {max_abs_diff(sa['observations'], sb['observations']):.3e}")
    ok &= sa["env_step"] == sb["env_step"] and same_np

    ea = list(csv.DictReader(open(os.path.join(a_dir, "eval.csv"))))
    eb = list(csv.DictReader(open(os.path.join(b_dir, "eval.csv"))))
    print(f"{'env_step':>9} {'seed':>4} {'return A':>12} {'return B':>12}")
    for ra, rb in zip(ea, eb):
        print(f"{ra['env_step']:>9} {ra['seed']:>4} {float(ra['return']):12.3f} {float(rb['return']):12.3f}")
    same_eval = [(r["env_step"], r["seed"], r["return"]) for r in ea] == [(r["env_step"], r["seed"], r["return"]) for r in eb]
    print(f"eval rows {len(ea)} vs {len(eb)}, identical: {same_eval}")
    ok &= same_eval

    print("RESULT:", "IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
