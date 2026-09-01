"""Out-of-fold offsets: the honest version of every interaction measurement.

The problem this fixes
----------------------
nested_sep's latents were fitted on ALL training games, so every experiment
that evaluated on training rows -- the -0.01410 boosting gap (boost/fit.py:141
splits sorted(train_g)), final.py's Phase 1 and Phase 2, temper.py,
recal_gap.py -- scored a model on data it had already seen. frozen_check.py
showed what that cost: the offset is 3.84862 in-sample against 3.94846 on the
frozen test, and a recalibration fitted in-sample HURTS the frozen test by
+0.03195. The "over-shrunk, amplify it" diagnosis was fitted to that gap.

The fix
-------
K-fold cross-fitting over the training GAMES. For each fold, refit all five
gates on the other four folds and predict the held-out one. Every training row
then carries an offset from a model that never saw it, so calibration and
interaction can be measured without the contamination.

Hyperparameters (d*, lambda*) are reused from the full fit rather than
re-selected per fold. That is a mild optimism -- the grid saw all of train --
but it is second-order next to refitting the latents, and it keeps each fold
to one alt_fit per gate instead of a full CV sweep.

Sanity gate: OOF deviance must land near the frozen-test 3.94846, NOT near the
in-sample 3.84862. If it comes out low, the cross-fit leaked and nothing below
it is worth reading.
"""
import sys, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "nested_sep"))
sys.path.insert(0, str(HERE.parent / "boost"))
sys.path.insert(0, str(HERE.parent / "fuse"))
import common, gates
import fit as NS
import analyze as FZ
import fit as BF   # noqa -- resolved below

# `fit` collides between nested_sep and boost; import each explicitly.
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


NS = _load("ns_fit", HERE.parent / "nested_sep" / "fit.py")
BF = _load("boost_fit", HERE.parent / "boost" / "fit.py")
CTRL = _load("boost_control", HERE.parent / "boost" / "control.py")

N_FOLDS = 5
FOLD_SEED = 31337
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def dev(logits, y):
    return -2.0 * np.mean(logits[np.arange(len(y)), y] - logsumexp(logits, axis=1))


def fit_fold(fit_rows, y_fit, d_per_gate, lam_per_gate, seasons):
    """Fit all five gates on `fit_rows`; return an npz-shaped dict."""
    a = gates.assign(y_fit)
    season_idx = {s: i for i, s in enumerate(seasons)}
    z = {}
    for gate in gates.GATE_ORDER:
        K = gates.N_BRANCH[gate]
        idx, br = a[gate]
        grows = [fit_rows[i] for i in idx]
        bat_ids, pit_ids, bidx, pidx = NS.build_ids(grows)
        D = NS.build_design(grows, bidx, pidx, season_idx, len(bat_ids), len(pit_ids))
        D["y"] = br
        alpha0, beta0 = NS.fit_struct_only(D, K, maxiter=400)
        d = d_per_gate[gate]
        lam = lam_per_gate[gate]
        alpha, beta, Lbat, Fbat, Lpit, Fpit, _ = NS.alt_fit(
            D, K, d, d, lam, lam, seed=1000, n_bat=len(bat_ids), n_pit=len(pit_ids),
            alpha0=alpha0, beta0=beta0, rounds=NS.ALT_ROUNDS_FINAL,
            block_maxiter=NS.BLOCK_MAXITER_FINAL)
        z[f"{gate}_Lbat"], z[f"{gate}_Fbat"] = Lbat, Fbat
        z[f"{gate}_Lpit"], z[f"{gate}_Fpit"] = Lpit, Fpit
        z[f"{gate}_bat_ids"] = np.array(bat_ids)
        z[f"{gate}_pit_ids"] = np.array(pit_ids)
        z[f"{gate}_alpha"], z[f"{gate}_beta"] = alpha, beta
    return z


def main():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    y = np.array([r["y"] for r in tr])
    yte = np.array([r["y"] for r in te])
    seasons = sorted({r["season"] for r in rows})
    season_idx = {s: i for i, s in enumerate(seasons)}
    log(f"train={len(tr)} test={len(te)}")

    res = json.loads((HERE.parent / "nested_sep" / "result.json").read_text())
    d_per_gate = {k: int(v) for k, v in res["d_per_gate"].items()}
    lam_per_gate = {k: float(v) for k, v in res["lambda_per_gate"].items()}
    log(f"reusing d*={d_per_gate} lambda*={lam_per_gate}")

    g = sorted(train_g)
    rs = np.random.RandomState(FOLD_SEED)
    rs.shuffle(g)
    folds = [set(g[i::N_FOLDS]) for i in range(N_FOLDS)]

    oof = np.zeros((len(tr), len(common.CATEGORIES)))
    for k, fold in enumerate(folds):
        mask = np.array([r["game_id"] in fold for r in tr])
        fit_rows = [r for r, m in zip(tr, mask) if not m]
        pred_rows = [r for r, m in zip(tr, mask) if m]
        z = fit_fold(fit_rows, y[~mask], d_per_gate, lam_per_gate, seasons)
        P = FZ.nested_category_probs(pred_rows, z, False, season_idx)
        oof[mask] = P
        d_fold = dev(np.log(np.maximum(P, 1e-300)), y[mask])
        log(f"fold {k}: fit={len(fit_rows)} pred={len(pred_rows)}  "
            f"out-of-fold deviance = {d_fold:.5f}")

    off_oof = np.log(np.maximum(oof, 1e-300))
    d_oof = dev(off_oof, y)

    zfull = np.load(HERE.parent / "nested_sep" / "latent.npz", allow_pickle=True)
    off_in = np.log(np.maximum(FZ.nested_category_probs(tr, zfull, False, season_idx), 1e-300))
    off_te = np.log(np.maximum(FZ.nested_category_probs(te, zfull, False, season_idx), 1e-300))
    d_in, d_te = dev(off_in, y), dev(off_te, yte)

    log("")
    log(f"SANITY  in-sample (full-train fit)  = {d_in:.5f}")
    log(f"SANITY  out-of-fold                 = {d_oof:.5f}")
    log(f"SANITY  frozen test                 = {d_te:.5f}")
    ok = abs(d_oof - d_te) < abs(d_oof - d_in)
    log(f"  -> OOF sits {'NEAR TEST (clean)' if ok else 'NEAR IN-SAMPLE (LEAK)'}")
    if not ok:
        log("ABORT: cross-fit leaked; nothing below is trustworthy")
        return

    # ---- calibration learned on honest predictions, applied to the real model
    k_ = oof.shape[1]

    def f(p):
        return dev(off_oof * p[:k_] + p[k_:], y)

    r = minimize(f, np.concatenate([np.ones(k_), np.zeros(k_)]), method="L-BFGS-B")
    a, b = r.x[:k_], r.x[k_:]
    cal_te = dev(off_te * a + b, yte)
    log("")
    log("=== calibration fitted OUT-OF-FOLD, scored on the frozen test ===")
    log(f"  raw frozen test          = {d_te:.5f}")
    log(f"  + OOF-fitted affine      = {cal_te:.5f}  ({cal_te - d_te:+.5f})")
    log(f"  (in-sample-fitted affine was +0.03195 -- frozen_check.py)")
    log("  per-category scales (a>1 amplify, a<1 shrink):")
    for i, c in enumerate(common.CATEGORIES):
        log(f"    {c:6} a={a[i]:+.4f}  b={b[i]:+.4f}")

    # ---- the interaction gap, above an honest baseline
    z5 = np.load(HERE.parent / "gllvm" / "latent.npz", allow_pickle=True)
    axes = {}
    for side in ("bat", "pit"):
        B = z5[f"L{side}"] @ z5[f"F{side}"].T
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        axes[side] = (U * S)[:, :5]
    bat_ids, pit_ids = list(z5["bat_ids"]), list(z5["pit_ids"])
    _, Ab, Ap, _, _ = BF.additive_offset_and_coords(
        tr, z5, {x: i for i, x in enumerate(bat_ids)},
        {x: i for i, x in enumerate(pit_ids)}, season_idx,
        len(bat_ids), len(pit_ids), axes)
    X = np.hstack([Ab, Ap])

    gg = sorted(train_g)
    rs2 = np.random.RandomState(11)
    rs2.shuffle(gg)
    itr = set(gg[: int(len(gg) * 0.8)])
    mg = np.array([r["game_id"] in itr for r in tr])

    out = {}
    for tag, O in (("in-sample offset", off_in), ("out-of-fold offset", off_oof)):
        base = BF.deviance(O[~mg], y[~mg])
        add = CTRL.boost_additive(Ab[mg], Ap[mg], O[mg], y[mg],
                                  Ab[~mg], Ap[~mg], O[~mg], y[~mg])
        joint, rnd = BF.boost(X[mg], O[mg], y[mg], X[~mg], O[~mg], y[~mg])
        log("")
        log(f"=== interaction gap, {tag} ===")
        log(f"  base={base:.5f}  additive={add:.5f}  joint={joint:.5f}")
        log(f"  GAP = {joint - add:+.5f}")
        out[tag] = dict(base=float(base), additive=float(add),
                        joint=float(joint), gap=float(joint - add))

    out["sanity"] = dict(in_sample=float(d_in), oof=float(d_oof), frozen_test=float(d_te))
    out["calibration"] = dict(raw_test=float(d_te), cal_test=float(cal_te),
                              delta=float(cal_te - d_te), a=a.tolist(), b=b.tolist())
    (HERE / "crossfit_result.json").write_text(json.dumps(out, indent=1) + "\n")
    log("\nwrote crossfit_result.json")


if __name__ == "__main__":
    main()
