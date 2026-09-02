"""Verify nll_grad_venue's analytic gradient against scipy's numerical one,
on a small subsample, before trusting it for the real fits.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import check_grad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "fuse"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common
from analyze import structural
import venue_common as VC
import venue_model as VM

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def main():
    rows = VC.load_rows()
    rs = np.random.RandomState(1234)
    sample = [rows[i] for i in rs.choice(len(rows), size=400, replace=False)]

    venues, VI, vs_keys, VSI = VC.build_venue_indices(rows)
    n_ven, n_vs = len(venues), len(vs_keys)
    log(f"n_ven={n_ven} n_vs={n_vs}")

    bats = sorted({r["batter"] for r in sample})
    pits = sorted({r["pitcher"] for r in sample})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)

    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}
    Xs = structural(sample, season_idx)
    bi = np.fromiter((BI[r["batter"]] for r in sample), int, len(sample))
    pj = np.fromiter((PI[r["pitcher"]] for r in sample), int, len(sample))
    vi = np.fromiter((VC.venue_index(r, VI) for r in sample), int, len(sample))
    vsi = np.fromiter((VC.venue_stance_index(r, VSI) for r in sample), int, len(sample))
    yv = np.fromiter((float(rs.randint(0, 2)) for _ in sample), float, len(sample))

    n = VM.theta_size(Xs.shape[1], n_bat, n_pit, n_ven, n_vs)
    x0 = rs.normal(scale=0.1, size=n)

    args = (Xs, bi, pj, vi, vsi, yv, n_bat, n_pit, n_ven, n_vs, 1.3, 5.0, 7.0, 4.0, 9.0)

    def f(th):
        return VM.nll_grad(th, *args)[0]

    def g(th):
        return VM.nll_grad(th, *args)[1]

    analytic = g(x0)
    gnorm = float(np.linalg.norm(analytic))
    log(f"theta dim = {n}  (venue rows in sample: known venue={int((vi<n_ven).sum())}, "
        f"unknown={int((vi==n_ven).sum())}; known vs={int((vsi<n_vs).sum())}, "
        f"unknown={int((vsi==n_vs).sum())})")
    log(f"||analytic grad|| = {gnorm:.4f}")
    # check_grad's forward-difference error scales ~linearly with epsilon
    # (truncation error), so report it at a small epsilon rather than
    # scipy's default 1.5e-8 (too small -> swamped by float64 cancellation)
    # or a large one (truncation dominates). 1e-7 is in the flat middle of
    # that tradeoff for this problem, confirmed by the epsilon sweep below.
    for eps in (1e-5, 1e-6, 1e-7):
        err = check_grad(f, g, x0, epsilon=eps)
        log(f"check_grad error (epsilon={eps:g}) = {err:.3e}  (relative to ||grad||: {err/gnorm:.3e})")
    err = check_grad(f, g, x0, epsilon=1e-7)
    rel = err / gnorm
    log(f"REPORTED check_grad error (epsilon=1e-7): absolute={err:.3e}  relative={rel:.3e}")
    assert rel < 1e-3, f"gradient check FAILED: relative error {rel}"
    log("gradient check PASSED")


if __name__ == "__main__":
    main()
