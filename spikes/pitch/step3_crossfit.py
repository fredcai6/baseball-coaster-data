"""Task 3 + Task 4: 5-fold cross-fit of the step1a binarised-tree model, then
the falsification test of whether per-node x per-SIDE regularisation already
absorbed the miscalibration that an older, worse-regularised model
(nested_sep, one lambda per gate shared across batters/pitchers) needed an
out-of-fold affine recalibration to fix.

CROSS-FIT OPTIMISM (same choice as spikes/pitch/crossfit.py made for a
different model): lam_bat, lam_pit, psi are reused PER NODE from
step1_result.json rather than re-selected per fold. The grid search that
picked them already saw all of train, so this is a mild optimism. It is
second-order here because there is no latent structure in step1a to refit --
each fold only has to re-solve the per-player effects and structural
coefficients, not rerun a hyperparameter sweep.

Tree walk (the part that's easy to get wrong): each of the 9 NODES in step1.py
is a binary split with a `reach` set (which of the 10 categories enter this
node) and a `pos` set (which of those get the node's fitted probability `p`;
the rest of `reach` get `1-p`). A category's total probability is the PRODUCT,
over every node whose `reach` contains it, of p-or-(1-p) at that node. There is
no early exit: you evaluate every node and multiply in whichever factor
applies (or skip the node entirely if the category isn't in `reach`). Because
the 9 nodes form a tree, each category is in `reach` for exactly the nodes on
its root-to-leaf path, so this always produces a proper distribution --
verified below by checking the 10 category probabilities sum to 1 per row.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # step1.py
sys.path.insert(0, str(HERE.parent))             # common.py
sys.path.insert(0, str(HERE.parent / "fuse"))    # analyze.structural

import common          # noqa: E402
import step1 as S1     # noqa: E402
from analyze import structural  # noqa: E402

N_FOLDS = 5
FOLD_SEED = 20260830  # recorded; arbitrary choice, matches today's date
T0 = time.time()
CATS = common.CATEGORIES
K = len(CATS)


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def dev(logP, y):
    """Multinomial deviance = 2 x mean NLL, given LOG probabilities."""
    return -2.0 * np.mean(logP[np.arange(len(y)), y])


def build_index(rows):
    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    return ({b: i for i, b in enumerate(bats)},
             {p: i for i, p in enumerate(pits)}, len(bats), len(pits))


def pack_fit(rows_subset, reach, pos, BI, PI, season_idx):
    """Same subsetting as step1.py's pack(): a node is trained only on rows
    whose TRUE outcome falls in its reach set."""
    sub = [r for r in rows_subset if r["y"] in reach]
    Xs = structural(sub, season_idx)
    bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
    pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
    yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
    return Xs, bi, pj, yv


def fit_all_nodes(fit_rows, node_hp, BI, PI, season_idx):
    """Fit all nine nodes on `fit_rows`, reusing hyperparameters from
    step1_result.json. Returns {node_name: theta}."""
    n_bat, n_pit = len(BI), len(PI)
    thetas = {}
    for name, reach, pos in S1.NODES:
        Xs, bi, pj, yv = pack_fit(fit_rows, reach, pos, BI, PI, season_idx)
        hp = node_hp[name]
        thetas[name] = S1.fit(Xs, bi, pj, yv, n_bat, n_pit,
                              hp["psi"], hp["lam_bat"], hp["lam_pit"])
    return thetas


def category_probs(rows, thetas, node_hp, BI, PI, season_idx):
    """Walk the tree for EVERY row (regardless of that row's true outcome) to
    produce a full 10-category distribution. No early exit -- every node that
    has a category in `reach` contributes a factor for that category."""
    n_bat, n_pit = len(BI), len(PI)
    Xs = structural(rows, season_idx)
    ps = Xs.shape[1]
    bi = np.fromiter((BI[r["batter"]] for r in rows), int, len(rows))
    pj = np.fromiter((PI[r["pitcher"]] for r in rows), int, len(rows))
    P = np.ones((len(rows), K))
    for name, reach, pos in S1.NODES:
        th = thetas[name]
        hp = node_hp[name]
        eta = (th[0] + Xs @ th[1:1 + ps] + th[1 + ps:1 + ps + n_bat][bi]
               + th[1 + ps + n_bat:][pj])
        p, omp, _, _ = S1.ao_prob(eta, hp["psi"])
        for ci in range(K):
            if ci in pos:
                P[:, ci] *= p
            elif ci in reach:
                P[:, ci] *= omp
    return P


def main():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    y_tr = np.array([r["y"] for r in tr])
    y_te = np.array([r["y"] for r in te])
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}
    BI, PI, n_bat, n_pit = build_index(rows)
    log(f"train={len(tr)} test={len(te)} batters={n_bat} pitchers={n_pit}")

    res = json.loads((HERE / "step1_result.json").read_text())
    node_hp = {d["node"]: d for d in res["nodes"]}
    log(f"reusing per-node hyperparameters from step1_result.json: "
        f"{ {k: (v['lam_bat'], v['lam_pit'], v['psi']) for k, v in node_hp.items()} }")

    # ---- full-train (step1a) fit -- the model that scores 3.94729 -----------
    log("fitting full-train model on all TRAIN rows ...")
    thetas_full = fit_all_nodes(tr, node_hp, BI, PI, season_idx)
    log("full-train fit done")

    P_in = category_probs(tr, thetas_full, node_hp, BI, PI, season_idx)
    P_te = category_probs(te, thetas_full, node_hp, BI, PI, season_idx)

    row_sums_in = P_in.sum(axis=1)
    row_sums_te = P_te.sum(axis=1)
    assert np.allclose(row_sums_in, 1.0, atol=1e-10), \
        f"train row sums off: max err {np.abs(row_sums_in - 1).max():.3e}"
    assert np.allclose(row_sums_te, 1.0, atol=1e-10), \
        f"test row sums off: max err {np.abs(row_sums_te - 1).max():.3e}"
    log("sanity: all category probabilities sum to 1 (train + test) to 1e-10")

    logP_in = np.log(np.maximum(P_in, 1e-300))
    logP_te = np.log(np.maximum(P_te, 1e-300))
    d_in = dev(logP_in, y_tr)
    d_te = dev(logP_te, y_te)
    log(f"in-sample deviance (full-train fit on TRAIN) = {d_in:.5f}")
    log(f"frozen-test deviance (full-train fit)         = {d_te:.5f}  (reference 3.94729)")

    # ---- 5-fold cross-fit over TRAINING games --------------------------------
    g = sorted(train_g)
    rs = np.random.RandomState(FOLD_SEED)
    rs.shuffle(g)
    folds = [set(g[i::N_FOLDS]) for i in range(N_FOLDS)]
    log(f"fold seed={FOLD_SEED}  fold sizes={[len(f) for f in folds]} games")

    oof = np.zeros((len(tr), K))
    fold_devs = []
    for k, fold in enumerate(folds):
        mask = np.array([r["game_id"] in fold for r in tr])
        fit_rows = [r for r, m in zip(tr, mask) if not m]
        pred_rows = [r for r, m in zip(tr, mask) if m]
        thetas_k = fit_all_nodes(fit_rows, node_hp, BI, PI, season_idx)
        Pk = category_probs(pred_rows, thetas_k, node_hp, BI, PI, season_idx)
        assert np.allclose(Pk.sum(axis=1), 1.0, atol=1e-10)
        oof[mask] = Pk
        d_fold = dev(np.log(np.maximum(Pk, 1e-300)), y_tr[mask])
        fold_devs.append(float(d_fold))
        log(f"fold {k}: fit={len(fit_rows)} pred={len(pred_rows)}  "
            f"out-of-fold deviance = {d_fold:.5f}")

    logP_oof = np.log(np.maximum(oof, 1e-300))
    d_oof = dev(logP_oof, y_tr)

    log("")
    log(f"SANITY  in-sample (full-train fit)  = {d_in:.5f}")
    log(f"SANITY  out-of-fold                 = {d_oof:.5f}")
    log(f"SANITY  frozen test                 = {d_te:.5f}")
    ok = abs(d_oof - d_te) < abs(d_oof - d_in)
    log(f"  -> OOF sits {'NEAR TEST (clean)' if ok else 'NEAR IN-SAMPLE (LEAK)'}")

    np.savez(HERE / "step3_oof.npz", oof=oof, y=y_tr,
             game_ids=np.array([r["game_id"] for r in tr]),
             categories=np.array(CATS))
    log("wrote step3_oof.npz")

    if not ok:
        log("ABORT: cross-fit leaked; task 4 not run, nothing below would be trustworthy")
        out = dict(fold_seed=FOLD_SEED, n_folds=N_FOLDS,
                   note=("lam_bat/lam_pit/psi reused per node from "
                         "step1_result.json, not reselected per fold (mild optimism)"),
                   sanity=dict(in_sample=float(d_in), oof=float(d_oof),
                              frozen_test=float(d_te), passed=False),
                   fold_deviances=fold_devs)
        (HERE / "step3_result.json").write_text(json.dumps(out, indent=1) + "\n")
        log("wrote step3_result.json (sanity-gate FAILED)")
        return

    # ---- Task 4: 20-param per-category affine recalibration on OOF log-probs
    def f(params):
        a = params[:K]
        b = params[K:]
        z = logP_oof * a + b
        z = z - logsumexp(z, axis=1, keepdims=True)
        return -2.0 * np.mean(z[np.arange(len(y_tr)), y_tr])

    r = minimize(f, np.concatenate([np.ones(K), np.zeros(K)]), method="L-BFGS-B",
                 options={"maxiter": 2000})
    a_hat, b_hat = r.x[:K], r.x[K:]
    log(f"calibration fit converged={r.success}  OOF deviance after cal={r.fun:.5f}")

    z_te = logP_te * a_hat + b_hat
    z_te = z_te - logsumexp(z_te, axis=1, keepdims=True)
    d_te_cal = -2.0 * np.mean(z_te[np.arange(len(y_te)), y_te])

    ref_a = {"K": 0.92, "BB": 0.86, "HBP": 1.12, "F": 0.99, "G": 0.93,
             "1B": 0.78, "2B": 0.60, "3B": 0.65, "HR": 1.16, "OTHER": 0.89}

    log("")
    log("=== calibration fitted OUT-OF-FOLD, scored on the frozen test ===")
    log(f"  raw frozen test          = {d_te:.5f}")
    log(f"  + OOF-fitted affine      = {d_te_cal:.5f}  ({d_te_cal - d_te:+.5f})")
    log("  (nested_sep, one lambda per gate shared bat/pit, gained -0.00243 from this)")
    log("  per-category scales (a>1 amplify, a<1 shrink):")
    for i, c in enumerate(CATS):
        log(f"    {c:6} a={a_hat[i]:+.4f}  b={b_hat[i]:+.4f}   "
            f"(nested_sep OOF reference a={ref_a[c]:.2f})")

    out = dict(
        fold_seed=FOLD_SEED, n_folds=N_FOLDS,
        note=("lam_bat/lam_pit/psi reused per node from step1_result.json, "
              "not reselected per fold (mild optimism)"),
        sanity=dict(in_sample=float(d_in), oof=float(d_oof),
                   frozen_test=float(d_te), passed=True),
        fold_deviances=fold_devs,
        calibration=dict(raw_test=float(d_te), cal_test=float(d_te_cal),
                         delta=float(d_te_cal - d_te),
                         categories=CATS,
                         a=a_hat.tolist(), b=b_hat.tolist(),
                         nested_sep_reference_a=[ref_a[c] for c in CATS]),
    )
    (HERE / "step3_result.json").write_text(json.dumps(out, indent=1) + "\n")
    log("wrote step3_result.json")


if __name__ == "__main__":
    main()
