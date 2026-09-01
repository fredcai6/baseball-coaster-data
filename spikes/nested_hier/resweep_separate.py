"""Supplementary: the initial separate_only reference grid (5..1000) was
monotonically WORSE as lambda_gate increased and never beat null even at its
weakest tested point (lambda_gate=5 -> val 4.02691, worse than null 4.01172).
That means the true optimum sits below 5 -- consistent with per-gate player
latents being fit off much smaller, noisier per-player-per-gate subsamples
than the flat/shared models, so a lambda calibrated for the flat GLLVM's
full-100k-row sum-NLL scale is already too weak *relative* to a single
gate's smaller row count share, and yet still needs to go lower, not higher,
to find where deltas alone (no shared component) can add value at all.

This script re-runs ONLY the separate_only CV with a wider, lower grid, on
the exact same inner train/val split and d used by fit.py's main run, then
patches result.json's separate_only block and limit_check accordingly. It
does not touch the test set or the hier model's own numbers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import fit  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def main():
    result = json.loads((HERE / "result.json").read_text())
    d_star = result["d"]
    lam_shared_star = result["lambda_shared"]

    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]

    bat_ids = sorted(set(r["batter"] for r in tr))
    pit_ids = sorted(set(r["pitcher"] for r in tr))
    bat_idx = {b: i for i, b in enumerate(bat_ids)}
    pit_idx = {q: i for i, q in enumerate(pit_ids)}
    n_bat, n_pit = len(bat_ids), len(pit_ids)
    seasons = sorted(set(r["season"] for r in rows))
    season_idx = {s: i for i, s in enumerate(seasons)}

    D_tr = fit.build_design(tr, bat_idx, pit_idx, season_idx, n_bat, n_pit)
    D_te = fit.build_design(te, bat_idx, pit_idx, season_idx, n_bat, n_pit)
    gs_tr_full = fit.gate_slices(D_tr)
    gs_te_full = fit.gate_slices(D_te)

    inner_train_g, inner_val_g = fit.make_inner_split(tr)
    itr = [r for r in tr if r["game_id"] in inner_train_g]
    ival = [r for r in tr if r["game_id"] in inner_val_g]
    ibat_ids = sorted(set(r["batter"] for r in itr))
    ipit_ids = sorted(set(r["pitcher"] for r in itr))
    ibat_idx = {b: i for i, b in enumerate(ibat_ids)}
    ipit_idx = {q: i for i, q in enumerate(ipit_ids)}
    in_bat, in_pit = len(ibat_ids), len(ipit_ids)
    D_itr = fit.build_design(itr, ibat_idx, ipit_idx, season_idx, in_bat, in_pit)
    D_ival = fit.build_design(ival, ibat_idx, ipit_idx, season_idx, in_bat, in_pit)
    gs_itr = fit.gate_slices(D_itr)
    gs_ival = fit.gate_slices(D_ival)

    struct0 = {g: fit.fit_struct_only_one_gate(gs_itr[g], fit.N_BRANCH[g]) for g in fit.GATE_ORDER}
    alpha0_i = {g: struct0[g][0] for g in fit.GATE_ORDER}
    beta0_i = {g: struct0[g][1] for g in fit.GATE_ORDER}

    wide_grid = [0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 80.0, 300.0, 1000.0]
    scores = []
    for lam in wide_grid:
        t0 = time.time()
        f = fit.alt_fit("separate_only", gs_itr, d_star, None, lam, in_bat, in_pit,
                         alpha0_i, beta0_i, fit.CV_ROUNDS, fit.CV_BLOCK_MAXITER, seed=0)
        vd = fit.joint_deviance_from_fit(f, gs_ival, D_ival["y"], in_bat, in_pit, d_star)
        scores.append({"lambda_gate": lam, "val_deviance": vd, "fit_sec": time.time() - t0})
        log(f"separate_only lambda_gate={lam:g} val_dev={vd:.5f} ({time.time()-t0:.1f}s)")

    best = min(scores, key=lambda r: r["val_deviance"])
    lam_sep_star = best["lambda_gate"]
    log(f"NEW selected separate_only lambda*={lam_sep_star:g} val_dev={best['val_deviance']:.5f}")

    # final refit on full train at the new best lambda, evaluate on test once
    alpha0_f = {}
    beta0_f = {}
    for g in fit.GATE_ORDER:
        a_g, b_g = fit.fit_struct_only_one_gate(gs_tr_full[g], fit.N_BRANCH[g], maxiter=300)
        alpha0_f[g] = a_g
        beta0_f[g] = b_g

    f_sep = fit.alt_fit("separate_only", gs_tr_full, d_star, None, lam_sep_star, n_bat, n_pit,
                         alpha0_f, beta0_f, fit.FINAL_ROUNDS, fit.FINAL_BLOCK_MAXITER, seed=4000)
    separate_only_test_dev = fit.joint_deviance_from_fit(f_sep, gs_te_full, D_te["y"], n_bat, n_pit, d_star)
    log(f"NEW separate_only (Variant A stand-in) test deviance = {separate_only_test_dev:.5f}")

    result["sibling_reference"]["separate_only"] = {
        "description": "Variant A stand-in: L_shared frozen at 0, independent per-gate latents. "
                        "NOTE: superseded the original coarse grid (5..1000), which never beat null "
                        "even at its weakest point -- see resweep_separate.py docstring.",
        "lambda_gate": lam_sep_star,
        "val_deviance": best["val_deviance"],
        "test_deviance": separate_only_test_dev,
        "lambda_grid": scores,
        "original_coarse_grid": result["sibling_reference"]["separate_only"]["lambda_grid"],
    }
    lo_end = result["lambda_gate_curve"][0]
    result["limit_check"]["separate_only_val_deviance"] = best["val_deviance"]
    result["limit_check"]["lo_end_minus_separate_only"] = lo_end["val_deviance"] - best["val_deviance"]

    (HERE / "result.json").write_text(json.dumps(result, indent=2, default=float))
    log("patched result.json")


if __name__ == "__main__":
    main()
