"""SPIKE 3/3: GLLVM-style hybrid for plate-appearance outcomes.

Model
-----
For batters and (separately) pitchers, the player effect on the 10-category
log-odds is factorized explicitly instead of left as a free matrix:

    B_bat = L_bat @ F_bat.T      L_bat: (n_batters x d)   F_bat: (10 x d)
    B_pit = L_pit @ F_pit.T      L_pit: (n_pitchers x d)  F_pit: (10 x d)

Full linear predictor for PA i, category k:

    eta[i,k] = alpha[k]
             + Xs[i,:] @ beta[:,k]                 # home / platoon / season, all explicit
             + L_bat[batter_i,:] @ F_bat[k,:]
             + L_pit[pitcher_i,:] @ F_pit[k,:]

probs[i,:] = softmax(eta[i,:]).

This is the GLLVM synthesis of the two sibling spikes: an explicit Gaussian
L2 penalty on L (the random-effect / player-shrinkage part, borrowed from
the ridge/GLMM spike) is combined with a fixed, small rank d shared across
ALL players (the low-rank / borrow-across-categories part, borrowed from
NPMR). Unlike NPMR's nuclear-norm relaxation (convex), fixing d and directly
parametrizing L and F is NON-CONVEX in the product L @ F.T -- see "Fitting"
below for how we deal with that, and result.json / README.md for how bad it
turns out to be in practice.

Fitting
-------
We fit by ALTERNATING block minimization rather than one joint L-BFGS-B call
over all parameters:

  1. Fix F (both bat and pit) and the structural terms (alpha, beta); solve
     for L. Given F fixed, eta is LINEAR in L, so this is an ordinary
     (ridge-penalized) multinomial logistic regression -- convex, and
     scipy's L-BFGS-B converges properly.
  2. Fix L; solve for F, alpha, beta jointly. Given L fixed, eta is linear
     in (F, alpha, beta) too -- also convex.
  3. Alternate until the penalized loss stops improving or a round budget
     is hit.

We tried the "one big L-BFGS-B over everything" approach first (as the spike
brief allows) and it FAILED silently: starting L and F from small random
values, the joint optimizer converges in ~60-90 iterations to L ≈ F ≈ 0 --
a real saddle point of the bilinear objective, not a numerical accident. The
gradient at (L, F) near zero is itself near zero (it is a product), so
L-BFGS-B's gradient-norm stopping rule is satisfied almost immediately,
before either factor has moved anywhere. Alternating avoids this: each
block subproblem is solved holding a NON-zero other block fixed, so the
first block update (L given random, non-collapsed F) sees a real gradient
and moves to a real optimum. We verified independently (a saturated,
non-low-rank per-player ridge multinomial model, no F factorization at all)
that batter identity alone carries real signal -- Dirichlet/ridge-shrunk
empirical batter rates beat the null by ~0.03 deviance -- so the zero
collapse in the joint approach was an optimization failure, not an absence
of signal. This is the single biggest practical finding of this spike: see
README.md.

Because the alternating fits are still non-convex in the OUTER loop (the
order and randomness of initialization can matter), we run multiple random
restarts and report the SPREAD of test deviance across them -- the
"is the non-convexity a practical problem" question the brief asks for.

d and penalty selection
------------------------
d is swept over {1,2,3,4,5}. For each d we grid-search the penalty
strength lambda (tied: the same lambda multiplies ||L||^2 and ||F||^2 for
both batters and pitchers -- this is the classical variational form of
nuclear-norm regularization, Srebro/Rennie/Jaakkola 2005: minimizing
lambda*(||L||^2+||F||^2) subject to L@F.T = M is equivalent, at the optimum,
to lambda * ||M||_* (nuclear norm) -- which is exactly NPMR's penalty. Tying
the two Frobenius penalties together is what connects this spike to NPMR's
column-pooling story while ALSO shrinkage-penalizing L on its own, which is
the GLMM piece NPMR lacks.

Selection is by a SINGLE inner train/validation split BY GAME within the
training games only (80/20, independent random split, never touching the
frozen test set). We use a single split rather than k-fold CV to fit the
~20-30 minute compute budget -- see README.md for the explicit tradeoff.

Compute budget notes (read before assuming a number is exact)
---------------------------------------------------------------
- Inner-CV grid: d in {1..5} x lambda in {8, 20, 50} = 15 fits, 1 restart
  each, capped alternation rounds (ALT_ROUNDS_CV) and capped L-BFGS-B
  iterations per block (BLOCK_MAXITER_CV). This is a coarse grid, not an
  exhaustive one -- explicitly capped to fit the time budget.
- Final refit at the selected (d*, lambda*): N_RESTARTS_FINAL random
  restarts on the FULL training set with a larger per-block iteration cap,
  to (a) get the best available fit and (b) measure init-sensitivity.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "/home/tommy/projects/baseball-coaster-data/spikes")
import common  # noqa: E402

OUTDIR = "/home/tommy/projects/baseball-coaster-data/spikes/gllvm"
K = len(common.CATEGORIES)

# ---------------------------------------------------------------- budget --
D_GRID = [1, 2, 3, 4, 5]
LAMBDA_GRID = [8.0, 20.0, 50.0]
ALT_ROUNDS_CV = 5
BLOCK_MAXITER_CV = 60
ALT_ROUNDS_FINAL = 10
BLOCK_MAXITER_FINAL = 150
N_RESTARTS_FINAL = 5
LAMBDA_STRUCT = 1e-3  # tiny ridge on alpha/beta, only for softmax identifiability
INNER_VAL_FRAC = 0.2
INNER_SEED = 13


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


T0 = time.time()


# --------------------------------------------------------- data plumbing --
def hand_state(bats, throws):
    """same / opposite / unknown, explicit third level instead of dropping rows.

    Switch hitters (bats == 'S') always take the platoon advantage, so they
    are coded as facing an 'opposite' pitcher regardless of throws.
    """
    if bats is None or throws is None:
        return "unknown"
    if bats == "S":
        return "opposite"
    return "opposite" if bats != throws else "same"


def build_design(rows, bat_idx, pit_idx, season_idx, n_bat, n_pit):
    n = len(rows)
    n_seasons = len(season_idx)
    bi = np.array([bat_idx.get(r["batter"], n_bat) for r in rows], dtype=np.int64)
    pi = np.array([pit_idx.get(r["pitcher"], n_pit) for r in rows], dtype=np.int64)
    p = 3 + (n_seasons - 1)  # home, opposite-hand, unknown-hand, season dummies
    Xs = np.zeros((n, p), dtype=np.float64)
    Xs[:, 0] = [1.0 if r["batting_is_home"] else 0.0 for r in rows]
    hs = [hand_state(r["bats"], r["throws"]) for r in rows]
    Xs[:, 1] = [1.0 if h == "opposite" else 0.0 for h in hs]
    Xs[:, 2] = [1.0 if h == "unknown" else 0.0 for h in hs]
    for i, r in enumerate(rows):
        si = season_idx[r["season"]]
        if si > 0:
            Xs[i, 3 + si - 1] = 1.0
    y = np.array([r["y"] for r in rows], dtype=np.int64)
    return dict(n=n, bi=bi, pi=pi, Xs=Xs, y=y, p=p)


# --------------------------------------------------------------- model ----
def forward(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, d):
    Lbat_full = np.vstack([Lbat, np.zeros((1, d))])
    Lpit_full = np.vstack([Lpit, np.zeros((1, d))])
    Lb_sel = Lbat_full[D["bi"]]
    Lp_sel = Lpit_full[D["pi"]]
    eta = alpha[None, :] + D["Xs"] @ beta + Lb_sel @ Fbat.T + Lp_sel @ Fpit.T
    return eta, Lb_sel, Lp_sel


def softmax_nll_grad(eta, y):
    """SUM (not mean) NLL and its gradient wrt eta.

    Using the sum rather than the per-row mean matters a lot here: player
    parameters only receive gradient from THEIR rows (tens to hundreds),
    while a mean-NLL divides every gradient by the FULL dataset size
    (~100k). That makes the raw signal reaching a player's L-row roughly
    (rows_for_player / total_rows) ~ 1e-3, so any all-parameter-shared
    lambda that is not itself ~1e-3 crushes L to zero regardless of true
    signal strength. We hit exactly this bug during development (see the
    module docstring) before switching to sum-NLL, where lambda has a
    sane O(1)-O(100) scale independent of dataset size.
    """
    eta = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(eta)
    Z = ex.sum(axis=1, keepdims=True)
    probs = ex / Z
    n = eta.shape[0]
    nll = -np.sum(np.log(np.maximum(probs[np.arange(n), y], 1e-300)))
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y] = 1.0
    dEta = probs - onehot
    return nll, dEta, probs


def fit_struct_only(D, alpha0=None, beta0=None, maxiter=300):
    n, p = D["n"], D["p"]
    if alpha0 is None:
        alpha0 = np.zeros(K)
    if beta0 is None:
        beta0 = np.zeros((p, K))

    def og(theta):
        alpha = theta[:K]
        beta = theta[K:].reshape(p, K)
        eta = alpha[None, :] + D["Xs"] @ beta
        nll, dEta, _ = softmax_nll_grad(eta, D["y"])
        loss = nll + LAMBDA_STRUCT * (np.sum(alpha**2) + np.sum(beta**2))
        d_alpha = dEta.sum(axis=0) + 2 * LAMBDA_STRUCT * alpha
        d_beta = D["Xs"].T @ dEta + 2 * LAMBDA_STRUCT * beta
        return loss, np.concatenate([d_alpha.ravel(), d_beta.ravel()])

    theta0 = np.concatenate([alpha0.ravel(), beta0.ravel()])
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return res.x[:K], res.x[K:].reshape(p, K)


def solve_L_given_F(D, alpha, beta, Fbat, Fpit, d, lam, L0bat, L0pit, n_bat, n_pit, maxiter):
    def og(theta):
        Lbat = theta[: n_bat * d].reshape(n_bat, d)
        Lpit = theta[n_bat * d :].reshape(n_pit, d)
        eta, Lb_sel, Lp_sel = forward(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, d)
        nll, dEta, _ = softmax_nll_grad(eta, D["y"])
        loss = nll + lam * (np.sum(Lbat**2) + np.sum(Lpit**2))
        d_Lbat_full = np.zeros((n_bat + 1, d))
        np.add.at(d_Lbat_full, D["bi"], dEta @ Fbat)
        d_Lpit_full = np.zeros((n_pit + 1, d))
        np.add.at(d_Lpit_full, D["pi"], dEta @ Fpit)
        d_Lbat = d_Lbat_full[:n_bat] + 2 * lam * Lbat
        d_Lpit = d_Lpit_full[:n_pit] + 2 * lam * Lpit
        return loss, np.concatenate([d_Lbat.ravel(), d_Lpit.ravel()])

    theta0 = np.concatenate([L0bat.ravel(), L0pit.ravel()])
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return res.x[: n_bat * d].reshape(n_bat, d), res.x[n_bat * d :].reshape(n_pit, d), res.fun


def solve_F_given_L(D, Lbat, Lpit, d, lam, alpha0, beta0, F0bat, F0pit, maxiter):
    p = D["p"]

    def og(theta):
        i = 0
        alpha = theta[i : i + K]
        i += K
        beta = theta[i : i + p * K].reshape(p, K)
        i += p * K
        Fbat = theta[i : i + K * d].reshape(K, d)
        i += K * d
        Fpit = theta[i:].reshape(K, d)
        eta, Lb_sel, Lp_sel = forward(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, d)
        nll, dEta, _ = softmax_nll_grad(eta, D["y"])
        loss = nll + lam * (np.sum(Fbat**2) + np.sum(Fpit**2)) + LAMBDA_STRUCT * (np.sum(alpha**2) + np.sum(beta**2))
        d_alpha = dEta.sum(axis=0) + 2 * LAMBDA_STRUCT * alpha
        d_beta = D["Xs"].T @ dEta + 2 * LAMBDA_STRUCT * beta
        d_Fbat = dEta.T @ Lb_sel + 2 * lam * Fbat
        d_Fpit = dEta.T @ Lp_sel + 2 * lam * Fpit
        return loss, np.concatenate([d_alpha.ravel(), d_beta.ravel(), d_Fbat.ravel(), d_Fpit.ravel()])

    theta0 = np.concatenate([alpha0.ravel(), beta0.ravel(), F0bat.ravel(), F0pit.ravel()])
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    i = 0
    alpha = res.x[i : i + K]
    i += K
    beta = res.x[i : i + p * K].reshape(p, K)
    i += p * K
    Fbat = res.x[i : i + K * d].reshape(K, d)
    i += K * d
    Fpit = res.x[i:].reshape(K, d)
    return alpha, beta, Fbat, Fpit, res.fun


def predict(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, d):
    eta, _, _ = forward(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, d)
    eta = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(eta)
    return ex / ex.sum(axis=1, keepdims=True)


def alt_fit(D, d, lam, seed, n_bat, n_pit, alpha0, beta0, rounds, block_maxiter, f_scale=1.0):
    rng = np.random.default_rng(seed)
    Fbat = rng.normal(scale=f_scale, size=(K, d))
    Fpit = rng.normal(scale=f_scale, size=(K, d))
    Lbat = np.zeros((n_bat, d))
    Lpit = np.zeros((n_pit, d))
    alpha, beta = alpha0.copy(), beta0.copy()
    prev_loss = None
    for r in range(rounds):
        Lbat, Lpit, lossL = solve_L_given_F(D, alpha, beta, Fbat, Fpit, d, lam, Lbat, Lpit, n_bat, n_pit, block_maxiter)
        alpha, beta, Fbat, Fpit, lossF = solve_F_given_L(D, Lbat, Lpit, d, lam, alpha, beta, Fbat, Fpit, block_maxiter)
        if prev_loss is not None and abs(prev_loss - lossF) < 1e-6 * max(1.0, abs(prev_loss)):
            prev_loss = lossF
            break
        prev_loss = lossF
    return alpha, beta, Lbat, Fbat, Lpit, Fpit, prev_loss


# ------------------------------------------------------------- pipeline --
def make_inner_split(train_rows, seed=INNER_SEED, val_frac=INNER_VAL_FRAC):
    """Split TRAIN games (never test) into inner-train / inner-val, by game,
    stratified by season -- mirrors common.get_split's method but is local
    and transient (not written anywhere, not shared across spikes)."""
    import random

    by_season = {}
    for r in train_rows:
        by_season.setdefault(r["season"], set()).add(r["game_id"])
    rng = random.Random(seed)
    itr, ival = set(), set()
    for season, games in sorted(by_season.items()):
        g = sorted(games)
        rng.shuffle(g)
        cut = int(len(g) * (1 - val_frac))
        itr.update(g[:cut])
        ival.update(g[cut:])
    return itr, ival


def main():
    log("loading data")
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    log(f"train={len(tr)} test={len(te)}")

    p_null = common.null_model(tr)
    null_dev = common.deviance([p_null] * len(te), [r["y"] for r in te])
    log(f"null deviance on test = {null_dev:.5f}")

    # id maps built from TRAIN only; unseen players at eval time fall back to
    # an implicit all-zero latent row (population-average effect) -- the
    # graceful "replacement level" NPMR had to construct by hand.
    bat_ids = sorted(set(r["batter"] for r in tr))
    pit_ids = sorted(set(r["pitcher"] for r in tr))
    bat_idx = {b: i for i, b in enumerate(bat_ids)}
    pit_idx = {q: i for i, q in enumerate(pit_ids)}
    n_bat, n_pit = len(bat_ids), len(pit_ids)
    seasons = sorted(set(r["season"] for r in rows))
    season_idx = {s: i for i, s in enumerate(seasons)}
    log(f"n_bat(train)={n_bat} n_pit(train)={n_pit} seasons={seasons}")

    D_tr = build_design(tr, bat_idx, pit_idx, season_idx, n_bat, n_pit)
    D_te = build_design(te, bat_idx, pit_idx, season_idx, n_bat, n_pit)

    # -------------------------------------------------- inner CV: d, lambda
    inner_train_g, inner_val_g = make_inner_split(tr)
    itr = [r for r in tr if r["game_id"] in inner_train_g]
    ival = [r for r in tr if r["game_id"] in inner_val_g]
    log(f"inner split: itrain={len(itr)} ival={len(ival)}")

    ibat_ids = sorted(set(r["batter"] for r in itr))
    ipit_ids = sorted(set(r["pitcher"] for r in itr))
    ibat_idx = {b: i for i, b in enumerate(ibat_ids)}
    ipit_idx = {q: i for i, q in enumerate(ipit_ids)}
    in_bat, in_pit = len(ibat_ids), len(ipit_ids)

    D_itr = build_design(itr, ibat_idx, ipit_idx, season_idx, in_bat, in_pit)
    D_ival = build_design(ival, ibat_idx, ipit_idx, season_idx, in_bat, in_pit)

    ialpha0, ibeta0 = fit_struct_only(D_itr)
    struct_only_val_dev = common.deviance(
        list(predict(D_ival, ialpha0, ibeta0, np.zeros((in_bat, 1)), np.zeros((K, 1)), np.zeros((in_pit, 1)), np.zeros((K, 1)), 1)),
        D_ival["y"].tolist(),
    )
    log(f"[inner CV] structural-only (no latent) val deviance = {struct_only_val_dev:.5f}")

    cv_results = []
    for d in D_GRID:
        for lam in LAMBDA_GRID:
            t0 = time.time()
            alpha, beta, Lbat, Fbat, Lpit, Fpit, floss = alt_fit(
                D_itr, d, lam, seed=0, n_bat=in_bat, n_pit=in_pit,
                alpha0=ialpha0, beta0=ibeta0, rounds=ALT_ROUNDS_CV, block_maxiter=BLOCK_MAXITER_CV,
            )
            val_probs = predict(D_ival, alpha, beta, Lbat, Fbat, Lpit, Fpit, d)
            val_dev = common.deviance(list(val_probs), D_ival["y"].tolist())
            dt = time.time() - t0
            cv_results.append({"d": d, "lambda": lam, "val_deviance": val_dev, "fit_sec": dt})
            log(f"[inner CV] d={d} lambda={lam:g} val_dev={val_dev:.5f} ({dt:.1f}s)")

    best = min(cv_results, key=lambda r: r["val_deviance"])
    d_star, lam_star = best["d"], best["lambda"]
    log(f"selected d*={d_star} lambda*={lam_star:g} (val_dev={best['val_deviance']:.5f})")

    # ------------------------------------------------ final fit, restarts
    alpha0, beta0 = fit_struct_only(D_tr, maxiter=400)
    struct_only_test_dev = common.deviance(
        list(predict(D_te, alpha0, beta0, np.zeros((n_bat, 1)), np.zeros((K, 1)), np.zeros((n_pit, 1)), np.zeros((K, 1)), 1)),
        D_te["y"].tolist(),
    )
    log(f"structural-only (home/hand/season, no player latent) test deviance = {struct_only_test_dev:.5f}")

    restarts = []
    for seed in range(N_RESTARTS_FINAL):
        t0 = time.time()
        alpha, beta, Lbat, Fbat, Lpit, Fpit, floss = alt_fit(
            D_tr, d_star, lam_star, seed=1000 + seed, n_bat=n_bat, n_pit=n_pit,
            alpha0=alpha0, beta0=beta0, rounds=ALT_ROUNDS_FINAL, block_maxiter=BLOCK_MAXITER_FINAL,
        )
        probs_tr = predict(D_tr, alpha, beta, Lbat, Fbat, Lpit, Fpit, d_star)
        probs_te = predict(D_te, alpha, beta, Lbat, Fbat, Lpit, Fpit, d_star)
        train_dev = common.deviance(list(probs_tr), D_tr["y"].tolist())
        test_dev = common.deviance(list(probs_te), D_te["y"].tolist())
        dt = time.time() - t0
        restarts.append(
            dict(seed=seed, train_dev=train_dev, test_dev=test_dev, train_penalized_loss=floss,
                 fit_sec=dt, alpha=alpha, beta=beta, Lbat=Lbat, Fbat=Fbat, Lpit=Lpit, Fpit=Fpit,
                 probs_te=probs_te)
        )
        log(f"[final restart {seed}] train_dev={train_dev:.5f} test_dev={test_dev:.5f} penalized_train_loss={floss:.2f} ({dt:.1f}s)")

    # canonical fit = lowest TRAIN penalized loss (never selects on test)
    canonical = min(restarts, key=lambda r: r["train_penalized_loss"])
    test_devs = [r["test_dev"] for r in restarts]
    spread = max(test_devs) - min(test_devs)
    log(f"restart test-deviance spread: min={min(test_devs):.5f} max={max(test_devs):.5f} spread={spread:.5f}")
    log(f"canonical restart (best train loss) = seed {canonical['seed']}, test_dev={canonical['test_dev']:.5f}")

    runtime_sec = time.time() - T0

    result = {
        "model": "gllvm_hybrid",
        "test_pa": len(te),
        "deviance": canonical["test_dev"],
        "null_deviance": null_dev,
        "d": d_star,
        "penalty": {"lambda_L": lam_star, "lambda_F": lam_star, "lambda_struct": LAMBDA_STRUCT, "tied": True},
        "runtime_sec": runtime_sec,
        "n_restarts": N_RESTARTS_FINAL,
        "restart_deviance_spread": spread,
        "restart_test_deviances": test_devs,
        "structural_only_test_deviance": struct_only_test_dev,
        "cv_grid": cv_results,
        "canonical_restart_seed": canonical["seed"],
        "alternation": "block coordinate descent (L given F, then F+alpha+beta given L); "
                        "joint one-shot L-BFGS-B over all params was tried first and collapsed "
                        "L,F to ~0 (see fit.py module docstring)",
        "inner_cv": {"method": "single 80/20 game-level split within train", "seed": INNER_SEED},
    }

    with open(f"{OUTDIR}/result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    log("wrote result.json")

    np.savez(
        f"{OUTDIR}/latent.npz",
        Lbat=canonical["Lbat"], Fbat=canonical["Fbat"],
        Lpit=canonical["Lpit"], Fpit=canonical["Fpit"],
        alpha=canonical["alpha"], beta=canonical["beta"],
        bat_ids=np.array(bat_ids), pit_ids=np.array(pit_ids),
        categories=np.array(common.CATEGORIES),
    )
    log("wrote latent.npz")

    np.savez(
        f"{OUTDIR}/residuals.npz",
        probs=canonical["probs_te"], y=D_te["y"],
        game_id=np.array([r["game_id"] for r in te]),
        batter=np.array([r["batter"] for r in te]),
        pitcher=np.array([r["pitcher"] for r in te]),
    )
    log("wrote residuals.npz")

    return result, canonical, restarts, bat_ids, pit_ids, tr, te, bat_idx, pit_idx


if __name__ == "__main__":
    main()
