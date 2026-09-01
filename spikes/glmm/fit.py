"""SPIKE 1: ridge-penalized (additive, "GLMM-flavored") multinomial logistic
regression for plate-appearance outcomes.

Linear predictor (Powers/Hastie/Tibshirani 2018, Sec 5.1, minus the stadium
term which the harness's row schema does not carry):

    eta_ik = alpha_k + beta_{batter_i,k} + gamma_{pitcher_i,k}
             + zeta_k * home_i + theta_k * opposite_hand_i + season_k

An L2 (ridge) penalty on the batter and pitcher coefficient blocks is exactly
a Gaussian random-effect prior on those blocks -- the penalized-likelihood
form of a mixed model with a batter random intercept-by-category and a
pitcher random intercept-by-category, crossed. Everything else (intercept,
season, home, handedness) is an unpenalized fixed effect.

Implementation choice: hand-rolled L-BFGS-B (scipy.optimize.minimize,
analytic gradient) rather than sklearn.linear_model.LogisticRegression.
sklearn's multinomial solver applies a single scalar C to every non-intercept
coefficient; there is no way to tell it "penalize the batter/pitcher blocks
but leave season/home/handedness alone" short of rescaling columns (an
approximation the task description explicitly flags as awkward). The model
here also uses a reference-category parameterization (K-1 free coefficient
columns, with the most frequent category's logit pinned to 0) instead of the
free (over-parameterized, softmax-invariant) K-column form, which keeps the
Hessian non-singular along the unpenalized directions and lets L-BFGS-B
converge cleanly without needing to penalize the intercept at all.

Run: ./.venv/bin/python spikes/glmm/fit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

OUT = HERE
CATEGORIES = common.CATEGORIES
K = len(CATEGORIES)
REF_CAT = "F"  # most frequent category in training data (24259 PA) -- fixes
               # the softmax shift-invariance without penalizing intercepts.
REF_IDX = CATEGORIES.index(REF_CAT)
NONREF = [i for i in range(K) if i != REF_IDX]
KM1 = K - 1
RNG_SEED = 20260830


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def tto_bucket(t):
    return t if t <= 3 else 4  # buckets: 1, 2, 3, "4+" (4 and 5 pooled -- n=1498 combined)


def opp_hand_flags(r):
    """(opposite_hand, unknown_handedness). Switch hitters always count as
    opposite-hand per spec. Missing either side of the matchup -> a distinct
    'unknown' indicator level (opposite_hand held at its reference value of 0)
    rather than dropping the row or imputing a guessed hand."""
    bats, throws = r["bats"], r["throws"]
    if bats is None or throws is None:
        return 0.0, 1.0
    if bats == "S":
        return 1.0, 0.0
    return (1.0 if bats != throws else 0.0), 0.0


def build_vocab(rows, key):
    ids = sorted({r[key] for r in rows})
    return {v: i for i, v in enumerate(ids)}


class Design:
    """Sparse design matrix builder with named, sliceable column blocks so
    the ridge penalty can be applied to exactly the batter/pitcher blocks."""

    def __init__(self, seasons, tto_ref=1, batter_vocab=None, pitcher_vocab=None,
                 include_home=True, include_hand=True, include_tto=True,
                 include_batter=True, include_pitcher=True):
        self.seasons = seasons  # sorted list; seasons[0] is the reference level
        self.tto_ref = tto_ref
        self.batter_vocab = batter_vocab or {}
        self.pitcher_vocab = pitcher_vocab or {}
        self.include_home = include_home
        self.include_hand = include_hand
        self.include_tto = include_tto
        self.include_batter = include_batter
        self.include_pitcher = include_pitcher

        blocks = ["intercept", "season"]
        if include_home:
            blocks.append("home")
        if include_hand:
            blocks += ["opp_hand", "unknown_hand"]
        if include_tto:
            blocks.append("tto")
        if include_batter:
            blocks.append("batter")
        if include_pitcher:
            blocks.append("pitcher")
        self.block_order = blocks

    def transform(self, rows):
        n = len(rows)
        parts = []
        slices = {}
        col = 0

        def add(mat, name):
            nonlocal col
            mat = sparse.csr_matrix(mat)
            parts.append(mat)
            slices[name] = slice(col, col + mat.shape[1])
            col += mat.shape[1]

        add(np.ones((n, 1)), "intercept")

        season_idx = {s: i for i, s in enumerate(self.seasons)}
        s_rows, s_cols, s_data = [], [], []
        for i, r in enumerate(rows):
            j = season_idx.get(r["season"])
            if j is not None and j > 0:  # drop reference season (first level)
                s_rows.append(i); s_cols.append(j - 1); s_data.append(1.0)
        add(sparse.coo_matrix((s_data, (s_rows, s_cols)),
                               shape=(n, max(0, len(self.seasons) - 1))), "season")

        if self.include_home:
            home = np.array([[1.0 if r["batting_is_home"] else 0.0] for r in rows])
            add(home, "home")

        if self.include_hand:
            opp = np.zeros((n, 1)); unk = np.zeros((n, 1))
            for i, r in enumerate(rows):
                o, u = opp_hand_flags(r)
                opp[i, 0] = o; unk[i, 0] = u
            add(opp, "opp_hand")
            add(unk, "unknown_hand")

        if self.include_tto:
            buckets = sorted({tto_bucket(r["tto"]) for r in rows} | {self.tto_ref})
            b_idx = {b: i for i, b in enumerate(buckets)}
            t_rows, t_cols, t_data = [], [], []
            for i, r in enumerate(rows):
                j = b_idx[tto_bucket(r["tto"])]
                if j > 0:
                    t_rows.append(i); t_cols.append(j - 1); t_data.append(1.0)
            add(sparse.coo_matrix((t_data, (t_rows, t_cols)),
                                   shape=(n, max(0, len(buckets) - 1))), "tto")

        if self.include_batter:
            r_idx, c_idx = [], []
            for i, r in enumerate(rows):
                j = self.batter_vocab.get(r["batter"])
                if j is not None:
                    r_idx.append(i); c_idx.append(j)
            add(sparse.coo_matrix((np.ones(len(r_idx)), (r_idx, c_idx)),
                                   shape=(n, len(self.batter_vocab))), "batter")

        if self.include_pitcher:
            r_idx, c_idx = [], []
            for i, r in enumerate(rows):
                j = self.pitcher_vocab.get(r["pitcher"])
                if j is not None:
                    r_idx.append(i); c_idx.append(j)
            add(sparse.coo_matrix((np.ones(len(r_idx)), (r_idx, c_idx)),
                                   shape=(n, len(self.pitcher_vocab))), "pitcher")

        X = sparse.hstack(parts, format="csr")
        return X, slices


# --------------------------------------------------------------------------
# Penalized multinomial fit (reference-category parameterization)
# --------------------------------------------------------------------------

def fit_model(X, y, penalty_slices, maxiter=400, tol=1e-8):
    """penalty_slices: dict name -> (slice, lambda). Categories not in
    `penalty_slices` get lambda=0 (unpenalized)."""
    n, p = X.shape
    y = np.asarray(y)
    Xd = X  # sparse csr

    def fun(flat_B):
        B = flat_B.reshape(p, KM1)
        lin = Xd.dot(B)  # n x KM1 dense
        full = np.zeros((n, K))
        full[:, NONREF] = lin
        m = full.max(axis=1, keepdims=True)
        ex = np.exp(full - m)
        Z = ex.sum(axis=1, keepdims=True)
        logZ = m[:, 0] + np.log(Z[:, 0])
        full_y = full[np.arange(n), y]
        nll = np.sum(logZ - full_y)

        P = ex / Z  # n x K, softmax probabilities
        Y = np.zeros((n, K))
        Y[np.arange(n), y] = 1.0
        R = (P - Y)[:, NONREF]  # n x KM1 residual, only nonref columns matter
        grad = Xd.T.dot(R)  # p x KM1

        pen = 0.0
        for name, (sl, lam) in penalty_slices.items():
            if lam:
                block = B[sl, :]
                pen += 0.5 * lam * np.sum(block ** 2)
                grad[sl, :] += lam * block

        return nll + pen, grad.ravel()

    x0 = np.zeros(p * KM1)
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                    options={"maxiter": maxiter, "ftol": tol, "gtol": 1e-6})
    B = res.x.reshape(p, KM1)
    return B, res


def predict_proba(X, B):
    n = X.shape[0]
    lin = X.dot(B)
    full = np.zeros((n, K))
    full[:, NONREF] = lin
    m = full.max(axis=1, keepdims=True)
    ex = np.exp(full - m)
    P = ex / ex.sum(axis=1, keepdims=True)
    return P


def penalty_slices_for(slices, lam_batter, lam_pitcher):
    out = {}
    if "batter" in slices:
        out["batter"] = (slices["batter"], lam_batter)
    if "pitcher" in slices:
        out["pitcher"] = (slices["pitcher"], lam_pitcher)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    t_start = time.time()
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    ys_te = np.array([r["y"] for r in te])
    ys_tr = np.array([r["y"] for r in tr])

    seasons = sorted({r["season"] for r in tr})
    batter_vocab = build_vocab(tr, "batter")
    pitcher_vocab = build_vocab(tr, "pitcher")
    print(f"train PA {len(tr)}  test PA {len(te)}  batters {len(batter_vocab)}  "
          f"pitchers {len(pitcher_vocab)}  seasons {seasons}")

    results = []
    null_p = common.null_model(tr)
    results.append(common.report("null", te, [null_p] * len(te)))

    # ---- lambda selection: 5-fold CV by game, on the FULL model, single
    # shared lambda for both player blocks. Reused for all ablation-ladder
    # steps below rather than re-tuned per step (budget: one grid search).
    print("\n--- cross-validating ridge strength (grouped by game) ---")
    lam_grid = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    train_games_sorted = sorted(train_g)
    rng = np.random.RandomState(RNG_SEED)
    perm = rng.permutation(len(train_games_sorted))
    n_folds = 5
    fold_of_game = {}
    for rank, gidx in enumerate(perm):
        fold_of_game[train_games_sorted[gidx]] = rank % n_folds

    cv_scores = {lam: [] for lam in lam_grid}
    t_cv = time.time()
    for f in range(n_folds):
        fold_train = [r for r in tr if fold_of_game[r["game_id"]] != f]
        fold_val = [r for r in tr if fold_of_game[r["game_id"]] == f]
        bv = build_vocab(fold_train, "batter")
        pv = build_vocab(fold_train, "pitcher")
        seas = sorted({r["season"] for r in fold_train})
        design = Design(seas, batter_vocab=bv, pitcher_vocab=pv)
        Xf, slices = design.transform(fold_train)
        Xv, _ = design.transform(fold_val)
        yf = np.array([r["y"] for r in fold_train])
        yv = np.array([r["y"] for r in fold_val])
        for lam in lam_grid:
            pen = penalty_slices_for(slices, lam, lam)
            B, res = fit_model(Xf, yf, pen, maxiter=200)
            probs = predict_proba(Xv, B)
            dev = common.deviance(probs, yv)
            cv_scores[lam].append(dev)
        print(f"  fold {f+1}/{n_folds} done ({time.time()-t_cv:.0f}s elapsed)")

    cv_mean = {lam: float(np.mean(v)) for lam, v in cv_scores.items()}
    best_lambda = min(cv_mean, key=cv_mean.get)
    print("CV mean deviance by lambda:", {k: round(v, 5) for k, v in cv_mean.items()})
    print(f"chosen lambda = {best_lambda}  (CV time {time.time()-t_cv:.0f}s)")

    # ---- ablation ladder, all on the full training set, evaluated on the
    # frozen test split. Player blocks use `best_lambda` throughout.
    print("\n--- ablation ladder ---")

    def fit_and_eval(name, include_batter, include_pitcher, include_home,
                      include_hand, include_tto):
        design = Design(seasons, batter_vocab=batter_vocab, pitcher_vocab=pitcher_vocab,
                         include_home=include_home, include_hand=include_hand,
                         include_tto=include_tto, include_batter=include_batter,
                         include_pitcher=include_pitcher)
        Xtr, slices = design.transform(tr)
        Xte, _ = design.transform(te)
        pen = penalty_slices_for(slices, best_lambda, best_lambda)
        t0 = time.time()
        B, res = fit_model(Xtr, ys_tr, pen)
        dt = time.time() - t0
        probs = predict_proba(Xte, B)
        out = common.report(name, te, probs, extra={
            "runtime_sec": round(dt, 1), "n_params": int(B.size),
            "converged": bool(res.success),
        })
        return out, B, slices, design

    ladder = []
    ladder.append(fit_and_eval("intercept+season", False, False, False, False, False))
    ladder.append(fit_and_eval("+batter", True, False, False, False, False))
    ladder.append(fit_and_eval("+pitcher", False, True, False, False, False))
    both_out = fit_and_eval("+both(batter+pitcher)", True, True, False, False, False)
    ladder.append(both_out)
    hand_out = fit_and_eval("+home+handedness", True, True, True, True, False)
    ladder.append(hand_out)
    full_out = fit_and_eval("+TTO (full model)", True, True, True, True, True)
    ladder.append(full_out)

    results.extend(r[0] for r in ladder)

    full_result, full_B, full_slices, full_design = full_out

    # ---- sanity check: fitted batter/pitcher K-effects vs observed K rate
    def observed_rate(rows_, key, cat_idx):
        from collections import defaultdict
        n_by = defaultdict(int); k_by = defaultdict(int)
        for r in rows_:
            n_by[r[key]] += 1
            if r["y"] == cat_idx:
                k_by[r[key]] += 1
        return {k: k_by[k] / n_by[k] for k in n_by if n_by[k] >= 20}

    k_idx = CATEGORIES.index("K")
    nonref_pos = {cat: j for j, cat in enumerate(NONREF)}
    k_col = nonref_pos[k_idx]  # position of K's coefficient column among the KM1 free columns

    batter_slice = full_slices["batter"]
    pitcher_slice = full_slices["pitcher"]
    batter_ids = sorted(batter_vocab, key=batter_vocab.get)
    pitcher_ids = sorted(pitcher_vocab, key=pitcher_vocab.get)

    batter_K_eff = full_B[batter_slice, k_col]
    pitcher_K_eff = full_B[pitcher_slice, k_col]

    obs_batter_k = observed_rate(tr, "batter", k_idx)
    obs_pitcher_k = observed_rate(tr, "pitcher", k_idx)

    bx, by = [], []
    for i, bid in enumerate(batter_ids):
        if bid in obs_batter_k:
            bx.append(batter_K_eff[i]); by.append(obs_batter_k[bid])
    px, py = [], []
    for i, pid in enumerate(pitcher_ids):
        if pid in obs_pitcher_k:
            px.append(pitcher_K_eff[i]); py.append(obs_pitcher_k[pid])

    corr_batter_k = float(np.corrcoef(bx, by)[0, 1]) if len(bx) > 2 else None
    corr_pitcher_k = float(np.corrcoef(px, py)[0, 1]) if len(px) > 2 else None
    print(f"\nsanity check: corr(fitted batter K coef, observed K% | n>=20) = {corr_batter_k:.3f}"
          f"  (n={len(bx)})")
    print(f"sanity check: corr(fitted pitcher K coef, observed K% | n>=20) = {corr_pitcher_k:.3f}"
          f"  (n={len(px)})")

    total_time = time.time() - t_start

    # ---- deliverables ----
    result = {
        "model": "ridge-penalized multinomial logistic (additive GLMM baseline)",
        "test_pa": len(te),
        "deviance": full_result["deviance"],
        "null_deviance": results[0]["deviance"],
        "lambda": best_lambda,
        "lambda_selection": "5-fold GroupKFold-by-game CV on the full model, "
                             f"grid={lam_grid}, mean deviance={cv_mean}",
        "runtime_sec": round(total_time, 1),
        "n_params": int(full_B.size),
        "reference_category": REF_CAT,
        "ablation_ladder": results,
        "sanity_check": {
            "corr_batter_K_coef_vs_observed_Kpct": corr_batter_k,
            "corr_pitcher_K_coef_vs_observed_Kpct": corr_pitcher_k,
            "n_batters_used": len(bx),
            "n_pitchers_used": len(px),
        },
        "n_batters_train": len(batter_vocab),
        "n_pitchers_train": len(pitcher_vocab),
        "n_batters_test_unseen_in_train": len({r["batter"] for r in te} - set(batter_vocab)),
        "n_pitchers_test_unseen_in_train": len({r["pitcher"] for r in te} - set(pitcher_vocab)),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2))

    np.savez(OUT / "effects.npz",
             batter_coef=full_B[batter_slice, :],   # (n_batters, 9) -- nonref categories
             pitcher_coef=full_B[pitcher_slice, :],  # (n_pitchers, 9)
             nonref_categories=np.array(NONREF),
             all_categories=np.array(CATEGORIES, dtype=object),
             reference_category=np.array(REF_CAT, dtype=object))
    (OUT / "ids.json").write_text(json.dumps({
        "batter_ids": batter_ids, "pitcher_ids": pitcher_ids,
        "nonref_category_order": [CATEGORIES[i] for i in NONREF],
        "reference_category": REF_CAT,
        "note": "effects.npz rows are in this batter_ids/pitcher_ids order; "
                "columns are in nonref_category_order (the reference category's "
                "coefficient is fixed at 0 and not stored).",
    }, indent=2))

    Xte_full, _ = full_design.transform(te)
    full_probs = predict_proba(Xte_full, full_B)
    np.savez(OUT / "residuals.npz",
             probs=full_probs, y=ys_te,
             game_id=np.array([r["game_id"] for r in te], dtype=object),
             batter=np.array([r["batter"] for r in te], dtype=object),
             pitcher=np.array([r["pitcher"] for r in te], dtype=object),
             categories=np.array(CATEGORIES, dtype=object))

    print(f"\nTotal runtime: {total_time:.0f}s")
    print("Wrote result.json, effects.npz, ids.json, residuals.npz to", OUT)


if __name__ == "__main__":
    main()
