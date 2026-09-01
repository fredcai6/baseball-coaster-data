"""SPIKE VARIANT A: nested-GLLVM with a SEPARATE latent space per gate.

The gate tree (frozen in spikes/gates.py) turns one 10-category PA outcome
into a sequence of conditional choices:

    root(TTO vs CONTACT) -> tto(K/BB/HBP) | contact(OUT/HIT/OTHER)
                                              -> out(F/G) | hit(1B/2B/3B/HR)

The continuation-ratio likelihood factorizes EXACTLY across gates (see
gates.saturated_check, diff ~1e-16), so joint deviance is directly
comparable to the flat spikes:

    null 4.01172   ridge/GLMM 3.95550   NPMR 3.95424   GLLVM(flat) 3.95563

This variant gives EACH gate its own independent player latent spaces:

    B_bat^(g) = L_bat^(g) @ F_bat^(g).T     L_bat^(g): (n_bat_g x d_g)
    B_pit^(g) = L_pit^(g) @ F_pit^(g).T     F_bat^(g): (K_g x d_g)

Nothing is shared across gates -- not d, not lambda, not the latent
coordinates. This is the most flexible (most parameter-hungry) of the three
sequential-GLLVM variants; the question it answers is whether that
flexibility earns back its statistical cost versus the flat GLLVM and versus
sharing structure across gates (a sibling variant's job).

Because nothing is shared, the five gates' likelihoods do not couple at all
under this variant -- each is an independent small multinomial logistic GLLVM
problem, fit with the SAME block-coordinate-descent recipe as the flat GLLVM
spike (spikes/gllvm/fit.py), which documented that a naive joint L-BFGS over
L and F together collapses both to ~0 (a real saddle point of the bilinear
objective, not a numerical accident) and needs alternating block descent
instead. We do the same here, per gate.

We exploit the decoupling for parallelism: the five gates are fit in
separate worker PROCESSES (ProcessPoolExecutor), not just separate function
calls, since gate CV grids are the most expensive step and this repo has 24
cores to spend on 5 independent problems.

Per-gate CV grid (d, lambda) is deliberately small -- see BUDGET NOTES at the
bottom of this docstring and in README.md -- because with 5 independent
gates to fit, an exhaustive per-gate sweep the size of the flat spike's
would blow the ~25 minute compute budget for this spike by 5x.

Rank ceiling per gate: a K_g-branch softmax's per-player contribution is a
linear functional of a d-dim latent row; for K_g=2 (root, out) any d >= 1 is
equivalent up to reparameterization (both L and F are free, so d=1 already
spans every achievable per-player scalar effect). We therefore only grid
d=1 for the two binary gates and grid d up to K_g-1 for the multi-branch
ones, instead of wasting compute on provably-redundant larger d.

DIPS check (free validation, see README.md): for each gate we ALSO fit
batter-latent-only (d_pit=0) and pitcher-latent-only (d_bat=0) models at the
gate's selected (d*, lambda*) and report the deviance ladder
structural-only -> +batter -> +pitcher -> +both. McCracken's DIPS (1999)
predicts pitchers dominate at the TTO gate and barely matter at OUT/HIT.
d_bat=0 or d_pit=0 works for free in this parametrization: an (n, 0) latent
times a (K, 0) loading is an (n, K) zero matrix, so no special-casing is
needed in forward()/solve_*() -- confirmed empirically before relying on it.
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SPIKES_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, SPIKES_DIR)
import common  # noqa: E402
import gates  # noqa: E402

OUTDIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------- budget --
# See BUDGET NOTES at the end of this file / README.md for the explicit
# time-vs-thoroughness tradeoff. All capped deliberately to fit ~25 min
# total wall clock across 5 PARALLEL gate processes.
LAMBDA_GRID = [8.0, 15.0, 25.0, 40.0, 60.0, 90.0]
D_GRID_BY_GATE = {
    "root": [1],       # K=2: d=1 already spans every achievable effect
    "tto": [1, 2],     # K=3
    "contact": [1, 2],  # K=3
    "out": [1],        # K=2
    "hit": [1, 2, 3],  # K=4
}
ALT_ROUNDS_CV = 6
BLOCK_MAXITER_CV = 70
ALT_ROUNDS_FINAL = 12
BLOCK_MAXITER_FINAL = 180
N_RESTARTS_FINAL = 7
LAMBDA_STRUCT = 1e-3
INNER_VAL_FRAC = 0.2
INNER_SEED = 13
FIELDS = ["batter", "pitcher", "bats", "throws", "batting_is_home", "season", "game_id"]


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


T0 = time.time()


# --------------------------------------------------------- data plumbing --
def hand_state(bats, throws):
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
    p = 3 + (n_seasons - 1)
    Xs = np.zeros((n, p), dtype=np.float64)
    Xs[:, 0] = [1.0 if r["batting_is_home"] else 0.0 for r in rows]
    hs = [hand_state(r["bats"], r["throws"]) for r in rows]
    Xs[:, 1] = [1.0 if h == "opposite" else 0.0 for h in hs]
    Xs[:, 2] = [1.0 if h == "unknown" else 0.0 for h in hs]
    for i, r in enumerate(rows):
        si = season_idx[r["season"]]
        if si > 0:
            Xs[i, 3 + si - 1] = 1.0
    return dict(n=n, bi=bi, pi=pi, Xs=Xs, p=p)


# --------------------------------------------------------------- model ----
def forward(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, n_bat, n_pit):
    d_bat, d_pit = Lbat.shape[1], Lpit.shape[1]
    Lbat_full = np.vstack([Lbat, np.zeros((1, d_bat))])
    Lpit_full = np.vstack([Lpit, np.zeros((1, d_pit))])
    Lb_sel = Lbat_full[D["bi"]]
    Lp_sel = Lpit_full[D["pi"]]
    eta = alpha[None, :] + D["Xs"] @ beta + Lb_sel @ Fbat.T + Lp_sel @ Fpit.T
    return eta, Lb_sel, Lp_sel


def softmax_nll_grad(eta, y):
    """Sum (not mean) NLL, see flat gllvm/fit.py docstring for why: mean-NLL
    dilutes the gradient reaching a player's latent row by ~1/n_rows, which
    makes any dataset-size-independent lambda crush L to zero."""
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


def fit_struct_only(D, K, alpha0=None, beta0=None, maxiter=300):
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


def solve_L_given_F(D, K, alpha, beta, Fbat, Fpit, lam_bat, lam_pit, L0bat, L0pit,
                     n_bat, n_pit, maxiter):
    d_bat, d_pit = Fbat.shape[1], Fpit.shape[1]

    def og(theta):
        Lbat = theta[: n_bat * d_bat].reshape(n_bat, d_bat)
        Lpit = theta[n_bat * d_bat:].reshape(n_pit, d_pit)
        eta, Lb_sel, Lp_sel = forward(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, n_bat, n_pit)
        nll, dEta, _ = softmax_nll_grad(eta, D["y"])
        loss = nll + lam_bat * np.sum(Lbat**2) + lam_pit * np.sum(Lpit**2)
        d_Lbat_full = np.zeros((n_bat + 1, d_bat))
        np.add.at(d_Lbat_full, D["bi"], dEta @ Fbat)
        d_Lpit_full = np.zeros((n_pit + 1, d_pit))
        np.add.at(d_Lpit_full, D["pi"], dEta @ Fpit)
        d_Lbat = d_Lbat_full[:n_bat] + 2 * lam_bat * Lbat
        d_Lpit = d_Lpit_full[:n_pit] + 2 * lam_pit * Lpit
        return loss, np.concatenate([d_Lbat.ravel(), d_Lpit.ravel()])

    theta0 = np.concatenate([L0bat.ravel(), L0pit.ravel()])
    if theta0.size == 0:
        return L0bat, L0pit, og(theta0)[0]
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return (res.x[: n_bat * d_bat].reshape(n_bat, d_bat),
            res.x[n_bat * d_bat:].reshape(n_pit, d_pit), res.fun)


def solve_F_given_L(D, K, Lbat, Lpit, lam_bat, lam_pit, alpha0, beta0, F0bat, F0pit, maxiter):
    p = D["p"]
    d_bat, d_pit = Lbat.shape[1], Lpit.shape[1]
    n_bat, n_pit = Lbat.shape[0], Lpit.shape[0]

    def og(theta):
        i = 0
        alpha = theta[i:i + K]
        i += K
        beta = theta[i:i + p * K].reshape(p, K)
        i += p * K
        Fbat = theta[i:i + K * d_bat].reshape(K, d_bat)
        i += K * d_bat
        Fpit = theta[i:].reshape(K, d_pit)
        eta, Lb_sel, Lp_sel = forward(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, n_bat, n_pit)
        nll, dEta, _ = softmax_nll_grad(eta, D["y"])
        loss = (nll + lam_bat * np.sum(Fbat**2) + lam_pit * np.sum(Fpit**2)
                + LAMBDA_STRUCT * (np.sum(alpha**2) + np.sum(beta**2)))
        d_alpha = dEta.sum(axis=0) + 2 * LAMBDA_STRUCT * alpha
        d_beta = D["Xs"].T @ dEta + 2 * LAMBDA_STRUCT * beta
        d_Fbat = dEta.T @ Lb_sel + 2 * lam_bat * Fbat
        d_Fpit = dEta.T @ Lp_sel + 2 * lam_pit * Fpit
        return loss, np.concatenate([d_alpha.ravel(), d_beta.ravel(), d_Fbat.ravel(), d_Fpit.ravel()])

    theta0 = np.concatenate([alpha0.ravel(), beta0.ravel(), F0bat.ravel(), F0pit.ravel()])
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    i = 0
    alpha = res.x[i:i + K]
    i += K
    beta = res.x[i:i + p * K].reshape(p, K)
    i += p * K
    Fbat = res.x[i:i + K * d_bat].reshape(K, d_bat)
    i += K * d_bat
    Fpit = res.x[i:].reshape(K, d_pit)
    return alpha, beta, Fbat, Fpit, res.fun


def predict(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, n_bat, n_pit):
    eta, _, _ = forward(D, alpha, beta, Lbat, Fbat, Lpit, Fpit, n_bat, n_pit)
    eta = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(eta)
    return ex / ex.sum(axis=1, keepdims=True)


def alt_fit(D, K, d_bat, d_pit, lam_bat, lam_pit, seed, n_bat, n_pit, alpha0, beta0,
            rounds, block_maxiter, f_scale=1.0):
    rng = np.random.default_rng(seed)
    Fbat = rng.normal(scale=f_scale, size=(K, d_bat))
    Fpit = rng.normal(scale=f_scale, size=(K, d_pit))
    Lbat = np.zeros((n_bat, d_bat))
    Lpit = np.zeros((n_pit, d_pit))
    alpha, beta = alpha0.copy(), beta0.copy()
    prev_loss = None
    for r in range(rounds):
        Lbat, Lpit, lossL = solve_L_given_F(D, K, alpha, beta, Fbat, Fpit, lam_bat, lam_pit,
                                             Lbat, Lpit, n_bat, n_pit, block_maxiter)
        alpha, beta, Fbat, Fpit, lossF = solve_F_given_L(D, K, Lbat, Lpit, lam_bat, lam_pit,
                                                          alpha, beta, Fbat, Fpit, block_maxiter)
        if prev_loss is not None and abs(prev_loss - lossF) < 1e-6 * max(1.0, abs(prev_loss)):
            prev_loss = lossF
            break
        prev_loss = lossF
    return alpha, beta, Lbat, Fbat, Lpit, Fpit, prev_loss


def dev(probs, y):
    eps = 1e-12
    n = len(y)
    return -2.0 * np.mean(np.log(np.maximum(probs[np.arange(n), y], eps)))


# ------------------------------------------------------------- per gate --
def build_ids(rows):
    bat_ids = sorted(set(r["batter"] for r in rows))
    pit_ids = sorted(set(r["pitcher"] for r in rows))
    return bat_ids, pit_ids, {b: i for i, b in enumerate(bat_ids)}, {q: i for i, q in enumerate(pit_ids)}


def fit_gate(args):
    """Runs entirely inside a worker process for one gate. Self-contained:
    only common/gates/numpy/scipy needed, imported at module load in the
    worker (this module is re-imported by each process)."""
    (gate, K, branch_names, tr_rows, y_tr, te_rows, y_te,
     inner_train_games, inner_val_games, seasons, D_GRID) = args
    t_gate0 = time.time()

    def glog(msg):
        print(f"[gate={gate} {time.time()-t_gate0:6.1f}s] {msg}", flush=True)

    season_idx = {s: i for i, s in enumerate(seasons)}

    bat_ids, pit_ids, bat_idx, pit_idx = build_ids(tr_rows)
    n_bat, n_pit = len(bat_ids), len(pit_ids)
    D_tr = build_design(tr_rows, bat_idx, pit_idx, season_idx, n_bat, n_pit)
    D_tr["y"] = y_tr
    D_te = build_design(te_rows, bat_idx, pit_idx, season_idx, n_bat, n_pit)
    D_te["y"] = y_te
    glog(f"n_bat={n_bat} n_pit={n_pit} tr_n={D_tr['n']} te_n={D_te['n']}")

    itr_mask = np.array([r["game_id"] in inner_train_games for r in tr_rows])
    ival_mask = ~itr_mask
    itr_rows = [r for r, m in zip(tr_rows, itr_mask) if m]
    ival_rows = [r for r, m in zip(tr_rows, itr_mask) if not m]
    y_itr = y_tr[itr_mask]
    y_ival = y_tr[ival_mask]

    ibat_ids, ipit_ids, ibat_idx, ipit_idx = build_ids(itr_rows)
    in_bat, in_pit = len(ibat_ids), len(ipit_ids)
    D_itr = build_design(itr_rows, ibat_idx, ipit_idx, season_idx, in_bat, in_pit)
    D_itr["y"] = y_itr
    D_ival = build_design(ival_rows, ibat_idx, ipit_idx, season_idx, in_bat, in_pit)
    D_ival["y"] = y_ival
    glog(f"inner split itr={D_itr['n']} ival={D_ival['n']}")

    ialpha0, ibeta0 = fit_struct_only(D_itr, K)
    struct_val = dev(predict(D_ival, ialpha0, ibeta0, np.zeros((in_bat, 0)), np.zeros((K, 0)),
                              np.zeros((in_pit, 0)), np.zeros((K, 0)), in_bat, in_pit), D_ival["y"])
    glog(f"structural-only val_dev={struct_val:.5f}")

    cv_results = []
    for d in D_GRID:
        for lam in LAMBDA_GRID:
            t0 = time.time()
            alpha, beta, Lbat, Fbat, Lpit, Fpit, floss = alt_fit(
                D_itr, K, d, d, lam, lam, seed=0, n_bat=in_bat, n_pit=in_pit,
                alpha0=ialpha0, beta0=ibeta0, rounds=ALT_ROUNDS_CV, block_maxiter=BLOCK_MAXITER_CV)
            val_probs = predict(D_ival, alpha, beta, Lbat, Fbat, Lpit, Fpit, in_bat, in_pit)
            val_dev = dev(val_probs, D_ival["y"])
            cv_results.append({"d": d, "lambda": lam, "val_deviance": val_dev, "fit_sec": time.time() - t0})
            glog(f"CV d={d} lambda={lam:g} val_dev={val_dev:.5f} ({time.time()-t0:.1f}s)")

    best = min(cv_results, key=lambda r: r["val_deviance"])
    d_star, lam_star = best["d"], best["lambda"]
    glog(f"selected d*={d_star} lambda*={lam_star:g} (val_dev={best['val_deviance']:.5f})")

    alpha0, beta0 = fit_struct_only(D_tr, K, maxiter=400)
    struct_test = dev(predict(D_te, alpha0, beta0, np.zeros((n_bat, 0)), np.zeros((K, 0)),
                               np.zeros((n_pit, 0)), np.zeros((K, 0)), n_bat, n_pit), D_te["y"])
    glog(f"structural-only test_dev={struct_test:.5f}")

    restarts = []
    for seed in range(N_RESTARTS_FINAL):
        t0 = time.time()
        alpha, beta, Lbat, Fbat, Lpit, Fpit, floss = alt_fit(
            D_tr, K, d_star, d_star, lam_star, lam_star, seed=1000 + seed, n_bat=n_bat, n_pit=n_pit,
            alpha0=alpha0, beta0=beta0, rounds=ALT_ROUNDS_FINAL, block_maxiter=BLOCK_MAXITER_FINAL)
        probs_te = predict(D_te, alpha, beta, Lbat, Fbat, Lpit, Fpit, n_bat, n_pit)
        test_dev = dev(probs_te, D_te["y"])
        restarts.append(dict(seed=seed, test_dev=test_dev, train_penalized_loss=floss,
                              fit_sec=time.time() - t0, alpha=alpha, beta=beta,
                              Lbat=Lbat, Fbat=Fbat, Lpit=Lpit, Fpit=Fpit, probs_te=probs_te))
        glog(f"final restart {seed} test_dev={test_dev:.5f} penloss={floss:.2f} ({time.time()-t0:.1f}s)")

    canonical = min(restarts, key=lambda r: r["train_penalized_loss"])
    test_devs = [r["test_dev"] for r in restarts]
    spread = max(test_devs) - min(test_devs)
    glog(f"restart spread={spread:.6f} canonical seed={canonical['seed']} test_dev={canonical['test_dev']:.5f}")

    # ---- DIPS ladder: batter-only (d_pit=0) and pitcher-only (d_bat=0),
    # reusing the selected (d*, lambda*) rather than re-running CV for each
    # ablation -- an approximation (the single-sided optimum lambda could
    # differ slightly) explicitly noted in README.md / result.json.
    a1, b1, Lbat_bo, Fbat_bo, Lpit_bo, Fpit_bo, _ = alt_fit(
        D_tr, K, d_star, 0, lam_star, lam_star, seed=2000, n_bat=n_bat, n_pit=n_pit,
        alpha0=alpha0, beta0=beta0, rounds=ALT_ROUNDS_FINAL, block_maxiter=BLOCK_MAXITER_FINAL)
    bat_only_test = dev(predict(D_te, a1, b1, Lbat_bo, Fbat_bo, Lpit_bo, Fpit_bo, n_bat, n_pit), D_te["y"])
    glog(f"DIPS batter-only test_dev={bat_only_test:.5f}")

    a2, b2, Lbat_po, Fbat_po, Lpit_po, Fpit_po, _ = alt_fit(
        D_tr, K, 0, d_star, lam_star, lam_star, seed=3000, n_bat=n_bat, n_pit=n_pit,
        alpha0=alpha0, beta0=beta0, rounds=ALT_ROUNDS_FINAL, block_maxiter=BLOCK_MAXITER_FINAL)
    pit_only_test = dev(predict(D_te, a2, b2, Lbat_po, Fbat_po, Lpit_po, Fpit_po, n_bat, n_pit), D_te["y"])
    glog(f"DIPS pitcher-only test_dev={pit_only_test:.5f}")

    n_params = (K + D_tr["p"] * K  # alpha, beta
                + n_bat * d_star + K * d_star  # bat latent + loadings
                + n_pit * d_star + K * d_star)  # pit latent + loadings

    return dict(
        gate=gate, K=K, branch_names=branch_names,
        d_star=d_star, lambda_star=lam_star, cv_grid=cv_results,
        struct_val=struct_val, struct_test=struct_test,
        bat_only_test=bat_only_test, pit_only_test=pit_only_test,
        both_test=canonical["test_dev"], n_params=n_params,
        n_bat=n_bat, n_pit=n_pit, tr_n=D_tr["n"], te_n=D_te["n"],
        restart_test_devs=test_devs, restart_spread=spread,
        canonical_seed=canonical["seed"],
        alpha=canonical["alpha"], beta=canonical["beta"],
        Lbat=canonical["Lbat"], Fbat=canonical["Fbat"],
        Lpit=canonical["Lpit"], Fpit=canonical["Fpit"],
        bat_ids=bat_ids, pit_ids=pit_ids,
        probs_te=canonical["probs_te"], y_te=D_te["y"],
        te_idx_order=None,  # filled in by caller from gates.assign ordering
        gate_sec=time.time() - t_gate0,
    )


# ------------------------------------------------------------- pipeline --
def make_inner_split(train_rows, seed=INNER_SEED, val_frac=INNER_VAL_FRAC):
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


def trim(r):
    return {k: r[k] for k in FIELDS}


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

    y_tr = np.array([r["y"] for r in tr])
    y_te = np.array([r["y"] for r in te])
    a_tr = gates.assign(y_tr)
    a_te = gates.assign(y_te)

    inner_train_games, inner_val_games = make_inner_split(tr)
    seasons = sorted(set(r["season"] for r in rows))

    jobs = []
    for gate in gates.GATE_ORDER:
        K = gates.N_BRANCH[gate]
        branch_names = [b for b, _ in gates.GATES[gate]]
        tr_idx, tr_br = a_tr[gate]
        te_idx, te_br = a_te[gate]
        gate_tr_rows = [trim(tr[i]) for i in tr_idx]
        gate_te_rows = [trim(te[i]) for i in te_idx]
        jobs.append((gate, K, branch_names, gate_tr_rows, tr_br, gate_te_rows, te_br,
                     inner_train_games, inner_val_games, seasons, D_GRID_BY_GATE[gate]))
        log(f"gate={gate} K={K} tr_reach={len(tr_idx)} te_reach={len(te_idx)}")

    log(f"dispatching {len(jobs)} gates to worker processes")
    results = {}
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        futs = {ex.submit(fit_gate, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            gate = futs[fut]
            res = fut.result()
            results[gate] = res
            log(f"gate={gate} DONE in {res['gate_sec']:.1f}s "
                f"d*={res['d_star']} lambda*={res['lambda_star']:g} "
                f"both_test={res['both_test']:.5f}")

    # ---------------------------------------------------- joint deviance --
    gate_logp_te = {}
    for gate in gates.GATE_ORDER:
        res = results[gate]
        gate_logp_te[gate] = np.log(np.maximum(res["probs_te"], 1e-300))
    joint_test_dev = gates.joint_deviance(gate_logp_te, y_te)
    log(f"JOINT test deviance = {joint_test_dev:.6f}  (null={null_dev:.5f})")

    total_n_params = sum(r["n_params"] for r in results.values())
    total_runtime = time.time() - T0

    per_gate_deviance = {g: results[g]["both_test"] for g in gates.GATE_ORDER}
    d_per_gate = {g: results[g]["d_star"] for g in gates.GATE_ORDER}
    lambda_per_gate = {g: results[g]["lambda_star"] for g in gates.GATE_ORDER}
    restart_spread = {g: results[g]["restart_spread"] for g in gates.GATE_ORDER}

    dips_ladder = {
        g: {
            "branches": results[g]["branch_names"],
            "structural_only": results[g]["struct_test"],
            "batter_only": results[g]["bat_only_test"],
            "pitcher_only": results[g]["pit_only_test"],
            "both": results[g]["both_test"],
            "batter_gain": results[g]["struct_test"] - results[g]["bat_only_test"],
            "pitcher_gain": results[g]["struct_test"] - results[g]["pit_only_test"],
        }
        for g in gates.GATE_ORDER
    }

    result = {
        "model": "nested_gllvm_separate_latents",
        "joint_test_deviance": joint_test_dev,
        "null_deviance": null_dev,
        "reference": {"ridge_glmm": 3.95550, "npmr": 3.95424, "gllvm_flat": 3.95563},
        "per_gate_deviance": per_gate_deviance,
        "d_per_gate": d_per_gate,
        "lambda_per_gate": lambda_per_gate,
        "n_params": total_n_params,
        "runtime_sec": total_runtime,
        "restart_spread": restart_spread,
        "dips_ladder": dips_ladder,
        "cv_grid_per_gate": {g: results[g]["cv_grid"] for g in gates.GATE_ORDER},
        "n_bat_per_gate": {g: results[g]["n_bat"] for g in gates.GATE_ORDER},
        "n_pit_per_gate": {g: results[g]["n_pit"] for g in gates.GATE_ORDER},
        "budget_notes": {
            "lambda_grid": LAMBDA_GRID,
            "d_grid_by_gate": D_GRID_BY_GATE,
            "alt_rounds_cv": ALT_ROUNDS_CV, "block_maxiter_cv": BLOCK_MAXITER_CV,
            "alt_rounds_final": ALT_ROUNDS_FINAL, "block_maxiter_final": BLOCK_MAXITER_FINAL,
            "n_restarts_final": N_RESTARTS_FINAL,
            "dips_ablations_reuse_selected_lambda": True,
            "inner_cv": "single 80/20 game-level split within train, shared across gates",
        },
    }

    with open(OUTDIR / "result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    log("wrote result.json")

    npz_kwargs = {}
    for g in gates.GATE_ORDER:
        r = results[g]
        npz_kwargs[f"{g}_Lbat"] = r["Lbat"]
        npz_kwargs[f"{g}_Fbat"] = r["Fbat"]
        npz_kwargs[f"{g}_Lpit"] = r["Lpit"]
        npz_kwargs[f"{g}_Fpit"] = r["Fpit"]
        npz_kwargs[f"{g}_bat_ids"] = np.array(r["bat_ids"])
        npz_kwargs[f"{g}_pit_ids"] = np.array(r["pit_ids"])
        npz_kwargs[f"{g}_branch_names"] = np.array(r["branch_names"])
        npz_kwargs[f"{g}_alpha"] = r["alpha"]
        npz_kwargs[f"{g}_beta"] = r["beta"]
    np.savez(OUTDIR / "latent.npz", **npz_kwargs)
    log("wrote latent.npz")

    log(f"TOTAL runtime = {total_runtime:.1f}s")
    return result


if __name__ == "__main__":
    main()
