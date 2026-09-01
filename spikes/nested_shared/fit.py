"""VARIANT B of the sequential-GLLVM spikes: ONE shared player latent,
gate-specific loadings.

Model
-----
Each batter has a single latent vector L_bat[i] in R^d, used at EVERY gate.
Each pitcher has a single latent vector L_pit[j] in R^d, used at EVERY gate.
Each gate g has its own loading matrices F_bat^(g), F_pit^(g) (n_branch_g x d)
and its own structural intercept/coefficients alpha^(g), beta^(g):

    eta^(g)[i,:] = alpha^(g) + Xs[i,:] @ beta^(g)
                 + L_bat[batter_i,:] @ F_bat^(g).T
                 + L_pit[pitcher_i,:] @ F_pit^(g).T

    P(branch | reached gate g) = softmax(eta^(g))

Total log-likelihood factorizes exactly across gates (gates.joint_deviance),
so this is directly comparable to the flat models (null 4.01172, ridge/GLMM
3.95550, NPMR 3.95424, flat GLLVM 3.95563) and to sibling nested spikes.

A player has ONE position in style space; each gate is a different lens on
it (a different F). This costs barely more than the flat GLLVM: L stays
(n_bat + n_pit) x d, and F totals 14 branch-rows across all 5 gates
(2+3+3+2+4) vs 10 categories flat.

Algorithm -- the crux
----------------------
The likelihood factorizes across gates, but the shared L couples them:

  - solve_F_given_L splits into 5 INDEPENDENT per-gate solves (smaller than
    the flat 10-category problem, and embarrassingly parallel -- we just
    loop, it's cheap enough not to bother with a process pool).
  - solve_L_given_F is where the coupling lives: a player's row in L
    accumulates gradient from EVERY gate rows for that player reach. That
    sum-over-gates is the whole point of sharing -- it is a single L-BFGS-B
    call over (n_bat+n_pit)*d parameters, with the objective summed across
    all 5 gates' softmax NLLs.
  - Convexity: convex in L given F (linear-in-L eta, softmax NLL is convex),
    convex in F (all of F^(g), alpha^(g), beta^(g)) given L, non-convex
    jointly. Same story as the flat GLLVM in spikes/gllvm/fit.py, which
    documents a joint L-BFGS-B collapsing L,F to ~0 -- we use the same
    alternating block-coordinate fix here without re-deriving it.
  - The ragged part: each gate has a different row subset (a batter/pitcher
    appears in "root" always, but only in "hit" if their PA reached the
    HIT branch of "contact") and a different branch count (2 to 4). All the
    plumbing below (`build_gate_data`, the missing-player "+1 zero row"
    trick borrowed from spikes/gllvm/fit.py) exists to keep that bookkeeping
    correct; see the shapes asserted in `sanity_check_shapes`.

DIPS free-validation, loadings-agreement, and restart-sensitivity are
computed after the main fit; see README.md and loadings.md for the writeup.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "/home/tommy/projects/baseball-coaster-data/spikes")
import common  # noqa: E402
import gates  # noqa: E402

OUTDIR = "/home/tommy/projects/baseball-coaster-data/spikes/nested_shared"
GATE_ORDER = gates.GATE_ORDER
LAMBDA_STRUCT = 1e-3

# ---------------------------------------------------------------- budget --
# ~25 minute compute budget. A first pass with a coarse grid (d in {1,2,3},
# 2 lambdas, short rounds) finished the ENTIRE pipeline -- CV + 5 restarts +
# full DIPS ladder -- in ~121s, far under budget (each per-gate softmax is
# over 2-4 branches, not 10, so despite 5 gates the per-round FLOP count is
# actually LOWER than the flat GLLVM's; see fit.py module docstring). That
# headroom is spent here: d matches the flat GLLVM's full 1..5 sweep, lambda
# gets a finer grid, and rounds/iterations are raised so the reported numbers
# are converged fits, not budget-truncated ones. N_RESTARTS_FINAL is kept at
# 5 to stay apples-to-apples with the flat GLLVM's restart-spread number.
D_GRID = [1, 2, 3, 4, 5]
LAMBDA_GRID = [10.0, 20.0, 40.0, 60.0]
ALT_ROUNDS_CV = 5
BLOCK_MAXITER_CV = 50
ALT_ROUNDS_FINAL = 10
BLOCK_MAXITER_FINAL = 120
N_RESTARTS_FINAL = 5
INNER_VAL_FRAC = 0.2
INNER_SEED = 13
DIPS_ROUNDS = 5
DIPS_MAXITER = 50

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


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
    y = np.array([r["y"] for r in rows], dtype=np.int64)
    return dict(n=n, bi=bi, pi=pi, Xs=Xs, y=y, p=p)


def build_gate_data(D):
    """Per gate: row subset (idx into D's rows), branch labels, and the
    sliced batter/pitcher-id / structural-design rows for that subset."""
    a = gates.assign(D["y"])
    gd = {}
    for g in GATE_ORDER:
        idx, br = a[g]
        gd[g] = dict(
            idx=idx, br=np.asarray(br),
            bi=D["bi"][idx], pi=D["pi"][idx], Xs=D["Xs"][idx],
            n_branch=gates.N_BRANCH[g],
        )
    return gd


def sanity_check_shapes(gd):
    for g in GATE_ORDER:
        gdg = gd[g]
        n = len(gdg["idx"])
        assert gdg["br"].shape == (n,)
        assert gdg["bi"].shape == (n,)
        assert gdg["pi"].shape == (n,)
        assert gdg["Xs"].shape[0] == n
        assert gdg["br"].max() < gdg["n_branch"]


# --------------------------------------------------------------- model ----
def softmax_nll_grad(eta, y):
    """SUM (not mean) NLL and its gradient wrt eta -- see spikes/gllvm/fit.py
    module docstring for why sum-NLL is what keeps a single global lambda
    sane across parameter blocks of very different row counts (a gate with
    26k rows vs one with 100k rows)."""
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


def forward_gate(gdg, alpha_g, beta_g, Lbat, Fbat_g, Lpit, Fpit_g, d):
    n_bat, n_pit = Lbat.shape[0], Lpit.shape[0]
    Lbat_full = np.vstack([Lbat, np.zeros((1, d))])
    Lpit_full = np.vstack([Lpit, np.zeros((1, d))])
    Lb_sel = Lbat_full[gdg["bi"]]
    Lp_sel = Lpit_full[gdg["pi"]]
    eta = alpha_g[None, :] + gdg["Xs"] @ beta_g + Lb_sel @ Fbat_g.T + Lp_sel @ Fpit_g.T
    return eta, Lb_sel, Lp_sel


def predict_gate(gdg, alpha_g, beta_g, Lbat, Fbat_g, Lpit, Fpit_g, d):
    eta, _, _ = forward_gate(gdg, alpha_g, beta_g, Lbat, Fbat_g, Lpit, Fpit_g, d)
    eta = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(eta)
    return ex / ex.sum(axis=1, keepdims=True)


def fit_struct_only_gate(gdg, maxiter=200):
    nb, p = gdg["n_branch"], gdg["Xs"].shape[1]

    def og(theta):
        alpha = theta[:nb]
        beta = theta[nb:].reshape(p, nb)
        eta = alpha[None, :] + gdg["Xs"] @ beta
        nll, dEta, _ = softmax_nll_grad(eta, gdg["br"])
        loss = nll + LAMBDA_STRUCT * (np.sum(alpha**2) + np.sum(beta**2))
        d_alpha = dEta.sum(axis=0) + 2 * LAMBDA_STRUCT * alpha
        d_beta = gdg["Xs"].T @ dEta + 2 * LAMBDA_STRUCT * beta
        return loss, np.concatenate([d_alpha.ravel(), d_beta.ravel()])

    theta0 = np.zeros(nb + p * nb)
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return res.x[:nb], res.x[nb:].reshape(p, nb)


# ---------------------------------------- shared-L alternating block fit --
def solve_F_given_L_all_gates(gate_data, Lbat, Lpit, d, lam, alpha0, beta0, F0bat, F0pit, maxiter):
    """INDEPENDENT per-gate solve: fixing L, each gate's (alpha,beta,Fbat,
    Fpit) only appears in that gate's own NLL term, so this splits cleanly
    into 5 small convex problems. No coupling here -- that's next."""
    alpha, beta, Fbat, Fpit = {}, {}, {}, {}
    total_loss = 0.0
    for g in GATE_ORDER:
        gdg = gate_data[g]
        nb, p = gdg["n_branch"], gdg["Xs"].shape[1]

        def og(theta, gdg=gdg, nb=nb, p=p):
            i = 0
            al = theta[i:i + nb]; i += nb
            be = theta[i:i + p * nb].reshape(p, nb); i += p * nb
            Fb = theta[i:i + nb * d].reshape(nb, d); i += nb * d
            Fp = theta[i:].reshape(nb, d)
            eta, Lb_sel, Lp_sel = forward_gate(gdg, al, be, Lbat, Fb, Lpit, Fp, d)
            nll, dEta, _ = softmax_nll_grad(eta, gdg["br"])
            loss = nll + lam * (np.sum(Fb**2) + np.sum(Fp**2)) + LAMBDA_STRUCT * (np.sum(al**2) + np.sum(be**2))
            d_al = dEta.sum(axis=0) + 2 * LAMBDA_STRUCT * al
            d_be = gdg["Xs"].T @ dEta + 2 * LAMBDA_STRUCT * be
            d_Fb = dEta.T @ Lb_sel + 2 * lam * Fb
            d_Fp = dEta.T @ Lp_sel + 2 * lam * Fp
            return loss, np.concatenate([d_al.ravel(), d_be.ravel(), d_Fb.ravel(), d_Fp.ravel()])

        theta0 = np.concatenate([alpha0[g].ravel(), beta0[g].ravel(), F0bat[g].ravel(), F0pit[g].ravel()])
        res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
        i = 0
        al = res.x[i:i + nb]; i += nb
        be = res.x[i:i + p * nb].reshape(p, nb); i += p * nb
        Fb = res.x[i:i + nb * d].reshape(nb, d); i += nb * d
        Fp = res.x[i:].reshape(nb, d)
        alpha[g], beta[g], Fbat[g], Fpit[g] = al, be, Fb, Fp
        total_loss += res.fun
    return alpha, beta, Fbat, Fpit, total_loss


def solve_L_given_F_shared(gate_data, alpha, beta, Fbat, Fpit, d, lam, L0bat, L0pit, n_bat, n_pit, maxiter):
    """THE coupling step: one L-BFGS-B call over (n_bat+n_pit)*d parameters,
    with the objective and gradient SUMMED across all 5 gates. A player's
    L-row gets gradient contributions from every gate a PA of theirs
    reached -- that sum is what makes sharing do any work at all."""
    def og(theta):
        Lbat = theta[:n_bat * d].reshape(n_bat, d)
        Lpit = theta[n_bat * d:].reshape(n_pit, d)
        total_loss = lam * (np.sum(Lbat**2) + np.sum(Lpit**2))
        d_Lbat_full = np.zeros((n_bat + 1, d))
        d_Lpit_full = np.zeros((n_pit + 1, d))
        for g in GATE_ORDER:
            gdg = gate_data[g]
            eta, Lb_sel, Lp_sel = forward_gate(gdg, alpha[g], beta[g], Lbat, Fbat[g], Lpit, Fpit[g], d)
            nll, dEta, _ = softmax_nll_grad(eta, gdg["br"])
            total_loss += nll
            np.add.at(d_Lbat_full, gdg["bi"], dEta @ Fbat[g])
            np.add.at(d_Lpit_full, gdg["pi"], dEta @ Fpit[g])
        d_Lbat = d_Lbat_full[:n_bat] + 2 * lam * Lbat
        d_Lpit = d_Lpit_full[:n_pit] + 2 * lam * Lpit
        return total_loss, np.concatenate([d_Lbat.ravel(), d_Lpit.ravel()])

    theta0 = np.concatenate([L0bat.ravel(), L0pit.ravel()])
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return res.x[:n_bat * d].reshape(n_bat, d), res.x[n_bat * d:].reshape(n_pit, d), res.fun


def alt_fit_shared(gate_data, d, lam, seed, n_bat, n_pit, alpha0, beta0, rounds, block_maxiter, f_scale=0.3):
    rng = np.random.default_rng(seed)
    Fbat = {g: rng.normal(scale=f_scale, size=(gate_data[g]["n_branch"], d)) for g in GATE_ORDER}
    Fpit = {g: rng.normal(scale=f_scale, size=(gate_data[g]["n_branch"], d)) for g in GATE_ORDER}
    Lbat = np.zeros((n_bat, d))
    Lpit = np.zeros((n_pit, d))
    alpha = {g: alpha0[g].copy() for g in GATE_ORDER}
    beta = {g: beta0[g].copy() for g in GATE_ORDER}
    prev_loss = None
    for r in range(rounds):
        Lbat, Lpit, lossL = solve_L_given_F_shared(gate_data, alpha, beta, Fbat, Fpit, d, lam, Lbat, Lpit, n_bat, n_pit, block_maxiter)
        alpha, beta, Fbat, Fpit, lossF = solve_F_given_L_all_gates(gate_data, Lbat, Lpit, d, lam, alpha, beta, Fbat, Fpit, block_maxiter)
        if prev_loss is not None and abs(prev_loss - lossF) < 1e-6 * max(1.0, abs(prev_loss)):
            prev_loss = lossF
            break
        prev_loss = lossF
    return alpha, beta, Lbat, Fbat, Lpit, Fpit, prev_loss


def joint_logp(gate_data, alpha, beta, Lbat, Fbat, Lpit, Fpit, d):
    logp = {}
    for g in GATE_ORDER:
        probs = predict_gate(gate_data[g], alpha[g], beta[g], Lbat, Fbat[g], Lpit, Fpit[g], d)
        logp[g] = np.log(np.maximum(probs, 1e-300))
    return logp


def per_gate_deviance(gate_data, alpha, beta, Lbat, Fbat, Lpit, Fpit, d):
    out = {}
    for g in GATE_ORDER:
        probs = predict_gate(gate_data[g], alpha[g], beta[g], Lbat, Fbat[g], Lpit, Fpit[g], d)
        br = gate_data[g]["br"]
        nll = -np.log(np.maximum(probs[np.arange(len(br)), br], 1e-300))
        out[g] = float(2.0 * nll.mean())
    return out


def count_params(n_bat, n_pit, d, gate_data):
    n = n_bat * d + n_pit * d  # shared L
    for g in GATE_ORDER:
        nb, p = gate_data[g]["n_branch"], gate_data[g]["Xs"].shape[1]
        n += nb + p * nb + 2 * nb * d  # alpha, beta, Fbat, Fpit per gate
    return int(n)


# --------------------------------------------------- DIPS ladder fitting --
def fit_gate_variant(gdg_tr, gdg_te, n_bat, n_pit, d, lam, use_bat, use_pit,
                      rounds=DIPS_ROUNDS, maxiter=DIPS_MAXITER, seed=0):
    """Standalone (NOT shared-L) low-rank fit for a single gate, used only
    for the DIPS deviance ladder. use_bat/use_pit toggle which side's
    latent is active; 'neither' reduces to the structural-only baseline."""
    nb = gdg_tr["n_branch"]
    p = gdg_tr["Xs"].shape[1]
    alpha, beta = fit_struct_only_gate(gdg_tr, maxiter=200)
    rng = np.random.default_rng(seed)
    Fbat = rng.normal(scale=0.3, size=(nb, d)) if use_bat else np.zeros((nb, d))
    Fpit = rng.normal(scale=0.3, size=(nb, d)) if use_pit else np.zeros((nb, d))
    Lbat = np.zeros((n_bat, d))
    Lpit = np.zeros((n_pit, d))

    def fwd(gdg, al, be, Lb, Fb, Lp, Fp):
        Lb_full = np.vstack([Lb, np.zeros((1, d))])
        Lp_full = np.vstack([Lp, np.zeros((1, d))])
        Lb_sel = Lb_full[gdg["bi"]]
        Lp_sel = Lp_full[gdg["pi"]]
        eta = al[None, :] + gdg["Xs"] @ be
        if use_bat:
            eta = eta + Lb_sel @ Fb.T
        if use_pit:
            eta = eta + Lp_sel @ Fp.T
        return eta, Lb_sel, Lp_sel

    if use_bat or use_pit:
        for _ in range(rounds):
            # --- L given F ---
            def og_L(theta):
                i = 0
                if use_bat:
                    Lb = theta[i:i + n_bat * d].reshape(n_bat, d); i += n_bat * d
                else:
                    Lb = Lbat
                Lp = theta[i:].reshape(n_pit, d) if use_pit else Lpit
                eta, Lb_sel, Lp_sel = fwd(gdg_tr, alpha, beta, Lb, Fbat, Lp, Fpit)
                nll, dEta, _ = softmax_nll_grad(eta, gdg_tr["br"])
                loss = nll
                grads = []
                if use_bat:
                    loss += lam * np.sum(Lb**2)
                    g_full = np.zeros((n_bat + 1, d))
                    np.add.at(g_full, gdg_tr["bi"], dEta @ Fbat)
                    grads.append((g_full[:n_bat] + 2 * lam * Lb).ravel())
                if use_pit:
                    loss += lam * np.sum(Lp**2)
                    g_full = np.zeros((n_pit + 1, d))
                    np.add.at(g_full, gdg_tr["pi"], dEta @ Fpit)
                    grads.append((g_full[:n_pit] + 2 * lam * Lp).ravel())
                return loss, np.concatenate(grads)

            theta0 = np.concatenate(
                ([Lbat.ravel()] if use_bat else []) + ([Lpit.ravel()] if use_pit else []))
            res = minimize(og_L, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
            i = 0
            if use_bat:
                Lbat = res.x[i:i + n_bat * d].reshape(n_bat, d); i += n_bat * d
            if use_pit:
                Lpit = res.x[i:].reshape(n_pit, d)

            # --- F, alpha, beta given L ---
            def og_F(theta):
                i = 0
                al = theta[i:i + nb]; i += nb
                be = theta[i:i + p * nb].reshape(p, nb); i += p * nb
                Fb = theta[i:i + nb * d].reshape(nb, d) if use_bat else Fbat
                if use_bat:
                    i += nb * d
                Fp = theta[i:].reshape(nb, d) if use_pit else Fpit
                eta, Lb_sel, Lp_sel = fwd(gdg_tr, al, be, Lbat, Fb, Lpit, Fp)
                nll, dEta, _ = softmax_nll_grad(eta, gdg_tr["br"])
                loss = nll + LAMBDA_STRUCT * (np.sum(al**2) + np.sum(be**2))
                d_al = dEta.sum(axis=0) + 2 * LAMBDA_STRUCT * al
                d_be = gdg_tr["Xs"].T @ dEta + 2 * LAMBDA_STRUCT * be
                grads = [d_al.ravel(), d_be.ravel()]
                if use_bat:
                    loss += lam * np.sum(Fb**2)
                    grads.append((dEta.T @ Lb_sel + 2 * lam * Fb).ravel())
                if use_pit:
                    loss += lam * np.sum(Fp**2)
                    grads.append((dEta.T @ Lp_sel + 2 * lam * Fp).ravel())
                return loss, np.concatenate(grads)

            theta0 = np.concatenate(
                [alpha.ravel(), beta.ravel()]
                + ([Fbat.ravel()] if use_bat else [])
                + ([Fpit.ravel()] if use_pit else []))
            res = minimize(og_F, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
            i = 0
            alpha = res.x[i:i + nb]; i += nb
            beta = res.x[i:i + p * nb].reshape(p, nb); i += p * nb
            if use_bat:
                Fbat = res.x[i:i + nb * d].reshape(nb, d); i += nb * d
            if use_pit:
                Fpit = res.x[i:].reshape(nb, d)

    eta_te, _, _ = fwd(gdg_te, alpha, beta, Lbat, Fbat, Lpit, Fpit)
    eta_te = eta_te - eta_te.max(axis=1, keepdims=True)
    ex = np.exp(eta_te)
    probs_te = ex / ex.sum(axis=1, keepdims=True)
    br_te = gdg_te["br"]
    dev_te = float(2.0 * np.mean(-np.log(np.maximum(probs_te[np.arange(len(br_te)), br_te], 1e-300))))
    return dev_te


def dips_ladder(gate_data_tr, gate_data_te, n_bat, n_pit, d, lam):
    ladder = {}
    for g in GATE_ORDER:
        gdg_tr, gdg_te = gate_data_tr[g], gate_data_te[g]
        none_dev = fit_gate_variant(gdg_tr, gdg_te, n_bat, n_pit, d, lam, False, False)
        bat_dev = fit_gate_variant(gdg_tr, gdg_te, n_bat, n_pit, d, lam, True, False)
        pit_dev = fit_gate_variant(gdg_tr, gdg_te, n_bat, n_pit, d, lam, False, True)
        both_dev = fit_gate_variant(gdg_tr, gdg_te, n_bat, n_pit, d, lam, True, True)
        ladder[g] = dict(none=none_dev, batter_only=bat_dev, pitcher_only=pit_dev, both=both_dev)
        log(f"[DIPS] gate={g:8s} none={none_dev:.5f} bat={bat_dev:.5f} pit={pit_dev:.5f} both={both_dev:.5f}")
    return ladder


# ------------------------------------------------------------- pipeline --
def make_inner_split(train_rows, seed=INNER_SEED, val_frac=INNER_VAL_FRAC):
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
    gate_data_tr = build_gate_data(D_tr)
    gate_data_te = build_gate_data(D_te)
    sanity_check_shapes(gate_data_tr)
    sanity_check_shapes(gate_data_te)
    for g in GATE_ORDER:
        log(f"  gate {g:8s} n_branch={gate_data_tr[g]['n_branch']} "
            f"train_rows={len(gate_data_tr[g]['idx'])} test_rows={len(gate_data_te[g]['idx'])}")

    nested_dev, flat_dev, diff = gates.saturated_check(D_tr["y"])
    log(f"gates.saturated_check on TRAIN y: nested={nested_dev:.6f} flat={flat_dev:.6f} diff={diff:.2e}")
    assert diff < 1e-6, "gate factorization broken -- stop"

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
    gd_itr = build_gate_data(D_itr)
    gd_ival = build_gate_data(D_ival)

    ialpha0, ibeta0 = {}, {}
    for g in GATE_ORDER:
        a0, b0 = fit_struct_only_gate(gd_itr[g])
        ialpha0[g], ibeta0[g] = a0, b0
    struct_logp = {g: np.log(np.maximum(
        predict_gate(gd_ival[g], ialpha0[g], ibeta0[g], np.zeros((in_bat, 1)), np.zeros((gd_ival[g]["n_branch"], 1)),
                     np.zeros((in_pit, 1)), np.zeros((gd_ival[g]["n_branch"], 1)), 1), 1e-300)) for g in GATE_ORDER}
    struct_only_val_dev = gates.joint_deviance(struct_logp, D_ival["y"])
    log(f"[inner CV] structural-only (no latent) val deviance = {struct_only_val_dev:.5f}")

    cv_results = []
    for d in D_GRID:
        for lam in LAMBDA_GRID:
            t0 = time.time()
            alpha, beta, Lbat, Fbat, Lpit, Fpit, floss = alt_fit_shared(
                gd_itr, d, lam, seed=0, n_bat=in_bat, n_pit=in_pit,
                alpha0=ialpha0, beta0=ibeta0, rounds=ALT_ROUNDS_CV, block_maxiter=BLOCK_MAXITER_CV,
            )
            logp = joint_logp(gd_ival, alpha, beta, Lbat, Fbat, Lpit, Fpit, d)
            val_dev = gates.joint_deviance(logp, D_ival["y"])
            dt = time.time() - t0
            cv_results.append({"d": d, "lambda": lam, "val_deviance": float(val_dev), "fit_sec": dt})
            log(f"[inner CV] d={d} lambda={lam:g} val_dev={val_dev:.5f} ({dt:.1f}s)")

    best = min(cv_results, key=lambda r: r["val_deviance"])
    d_star, lam_star = best["d"], best["lambda"]
    log(f"selected d*={d_star} lambda*={lam_star:g} (val_dev={best['val_deviance']:.5f})")

    # ------------------------------------------------ final fit, restarts
    alpha0, beta0 = {}, {}
    for g in GATE_ORDER:
        a0, b0 = fit_struct_only_gate(gate_data_tr[g], maxiter=300)
        alpha0[g], beta0[g] = a0, b0
    struct_logp_te = {g: np.log(np.maximum(
        predict_gate(gate_data_te[g], alpha0[g], beta0[g], np.zeros((n_bat, 1)), np.zeros((gate_data_te[g]["n_branch"], 1)),
                     np.zeros((n_pit, 1)), np.zeros((gate_data_te[g]["n_branch"], 1)), 1), 1e-300)) for g in GATE_ORDER}
    struct_only_test_dev = gates.joint_deviance(struct_logp_te, D_te["y"])
    log(f"structural-only (home/hand/season, no player latent) test deviance = {struct_only_test_dev:.5f}")

    restarts = []
    for seed in range(N_RESTARTS_FINAL):
        t0 = time.time()
        alpha, beta, Lbat, Fbat, Lpit, Fpit, floss = alt_fit_shared(
            gate_data_tr, d_star, lam_star, seed=1000 + seed, n_bat=n_bat, n_pit=n_pit,
            alpha0=alpha0, beta0=beta0, rounds=ALT_ROUNDS_FINAL, block_maxiter=BLOCK_MAXITER_FINAL,
        )
        logp_tr = joint_logp(gate_data_tr, alpha, beta, Lbat, Fbat, Lpit, Fpit, d_star)
        logp_te = joint_logp(gate_data_te, alpha, beta, Lbat, Fbat, Lpit, Fpit, d_star)
        train_dev = gates.joint_deviance(logp_tr, D_tr["y"])
        test_dev = gates.joint_deviance(logp_te, D_te["y"])
        dt = time.time() - t0
        restarts.append(dict(seed=seed, train_dev=float(train_dev), test_dev=float(test_dev),
                              train_penalized_loss=floss, fit_sec=dt,
                              alpha=alpha, beta=beta, Lbat=Lbat, Fbat=Fbat, Lpit=Lpit, Fpit=Fpit))
        log(f"[final restart {seed}] train_dev={train_dev:.5f} test_dev={test_dev:.5f} "
            f"penalized_train_loss={floss:.2f} ({dt:.1f}s)")

    canonical = min(restarts, key=lambda r: r["train_penalized_loss"])
    test_devs = [r["test_dev"] for r in restarts]
    spread = max(test_devs) - min(test_devs)
    log(f"restart test-deviance spread: min={min(test_devs):.5f} max={max(test_devs):.5f} spread={spread:.6f}")
    log(f"canonical restart (best train loss) = seed {canonical['seed']}, test_dev={canonical['test_dev']:.5f}")

    per_gate_dev = per_gate_deviance(gate_data_te, canonical["alpha"], canonical["beta"],
                                      canonical["Lbat"], canonical["Fbat"], canonical["Lpit"], canonical["Fpit"], d_star)
    log(f"per-gate test deviance: {per_gate_dev}")

    # ------------------------------------------------------------ DIPS ---
    log("running DIPS ladder (per-gate bat-only / pit-only / both)")
    ladder = dips_ladder(gate_data_tr, gate_data_te, n_bat, n_pit, d_star, lam_star)

    n_params = count_params(n_bat, n_pit, d_star, gate_data_tr)
    runtime_sec = time.time() - T0

    result = {
        "model": "nested_shared_gllvm_variant_b",
        "joint_test_deviance": canonical["test_dev"],
        "null_deviance": float(null_dev),
        "per_gate_deviance": per_gate_dev,
        "d": d_star,
        "lambda": lam_star,
        "n_params": n_params,
        "runtime_sec": runtime_sec,
        "restart_spread": spread,
        # extra detail kept for the writeup, not required by the spec but cheap to keep
        "test_pa": len(te),
        "structural_only_test_deviance": float(struct_only_test_dev),
        "cv_grid": cv_results,
        "restart_test_deviances": test_devs,
        "canonical_restart_seed": canonical["seed"],
        "dips_ladder": ladder,
        "inner_cv": {"method": "single 80/20 game-level split within train", "seed": INNER_SEED},
        "budget_note": f"D_GRID={D_GRID} LAMBDA_GRID={LAMBDA_GRID} capped for ~25min budget; "
                        "see README.md",
    }

    with open(f"{OUTDIR}/result.json", "w") as fh:
        json.dump(result, fh, indent=2)
    log("wrote result.json")

    np.savez(
        f"{OUTDIR}/latent.npz",
        Lbat=canonical["Lbat"], Lpit=canonical["Lpit"],
        bat_ids=np.array(bat_ids), pit_ids=np.array(pit_ids),
        **{f"Fbat_{g}": canonical["Fbat"][g] for g in GATE_ORDER},
        **{f"Fpit_{g}": canonical["Fpit"][g] for g in GATE_ORDER},
        **{f"alpha_{g}": canonical["alpha"][g] for g in GATE_ORDER},
        **{f"branches_{g}": np.array([b for b, _ in gates.GATES[g]]) for g in GATE_ORDER},
    )
    log("wrote latent.npz")

    return result, canonical, restarts, ladder, bat_ids, pit_ids, gate_data_tr, gate_data_te, d_star, lam_star


if __name__ == "__main__":
    main()
