"""SPIKE: hierarchical latents on the sequential-GLLVM gate tree.

    VARIANT C of three sequential-GLLVM spikes.

Two siblings fit the extremes of a single family:

    A) every gate gets its OWN independent latent space for a player
       (max flexibility, max parameters)
    B) every player gets ONE latent, reused at every gate, only the
       per-gate loadings (F) differ (max parsimony)

This spike fits the model that CONTAINS both as limiting cases:

    L^(g)[i] = L_shared[i] + delta^(g)[i]

    penalty = lambda_shared * (||L_shared_bat||^2 + ||L_shared_pit||^2)
            + lambda_gate   * sum_g (||delta_bat^(g)||^2 + ||delta_pit^(g)||^2)

As lambda_gate -> infinity, delta^(g) -> 0 and this collapses to Variant B
(shared). As lambda_gate -> 0, delta^(g) is unpenalized and can absorb
anything, so L_shared becomes irrelevant and each gate decouples -- Variant A
(separate). The CV-selected lambda_gate is therefore a MEASUREMENT of which
sibling's assumption the data actually supports, not an argument for either.

Gate tree (frozen, spikes/gates.py):

    root -> TTO (tto: K/BB/HBP) | CONTACT
    contact -> OUT (out: F/G) | HIT (hit: 1B/2B/3B/HR) | OTHER

Linear predictor at gate g, branch k, row i:

    eta_g[i,k] = alpha_g[k] + Xs[i,:] @ beta_g[:,k]
               + L_bat^(g)[batter_i,:]  @ Fbat_g[k,:]
               + L_pit^(g)[pitcher_i,:] @ Fpit_g[k,:]

Fitting: block coordinate descent, cribbed from spikes/gllvm/fit.py's
documented lesson. Naive one-shot L-BFGS-B over L_shared, all deltas, and all
F jointly collapses everything to ~0 (same bilinear saddle GLLVM hit, and
WORSE here: L_shared and delta^(g) are additively confounded -- any (L_shared,
delta) and (L_shared + c, delta - c) give the identical L^(g) -- so without
the two penalties pulling in different directions the split isn't even
identified, let alone found by a gradient method). We alternate:

  Block 1 (given F, alpha, beta fixed): solve for L_shared and all deltas
  JOINTLY across all 5 gates in one convex problem (eta is linear in these
  given F fixed; sum-NLL + quadratic penalties is convex).

  Block 2 (given L fixed, i.e. given every L^(g)): solve F_g, alpha_g,
  beta_g PER GATE, independently (gates don't interact once L is fixed).
  Also convex.

A tiny, fixed (never swept) ridge on F (LAMBDA_F_STRUCT) is required for
scale identifiability: with L2 only on L, the bilinear rescaling
(L*c, F/c) sends the penalty on L to 0 as c->0 while leaving the fit
unchanged, so F is otherwise free to diverge. A nonzero, even tiny, penalty
on F makes the AM-GM argument bite and pins down a finite scale. This mirrors
GLLVM's LAMBDA_STRUCT anchor on alpha/beta, just applied to F instead.

Three fitting "modes" share this machinery (see `solve_L_given_F`):
  - "hier":      both L_shared and all deltas are free params (the spike).
  - "shared_only":  deltas frozen at 0 -- this IS Variant B done honestly,
                     with its own CV'd lambda_shared, used as a reference
                     point to check the lambda_gate->infinity limit.
  - "separate_only": L_shared frozen at 0 -- this IS Variant A done
                     honestly, with its own CV'd lambda_gate, used as a
                     reference point to check the lambda_gate->0 limit.

DIPS ladder: per gate, three more small fits (batter-latent-only,
pitcher-latent-only, both) on that gate's rows alone, at the SAME (d,
lambda_shared) used elsewhere, to see whether pitcher identity explains the
TTO gate much more than the OUT/HIT gates -- McCracken's DIPS claim, ported
onto the recast gate tree.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import gates  # noqa: E402

K_ALL = len(common.CATEGORIES)
GATE_ORDER = gates.GATE_ORDER          # ["root", "tto", "contact", "out", "hit"]
N_BRANCH = gates.N_BRANCH

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------- budget -----
# ~25 minute wall-clock budget for the whole script. Everything below is
# capped hard so a smoke test and the real run both finish in time; see
# README.md "Compute budget" for the actual wall-clock breakdown observed.
D_GRID = [1, 2, 3, 4]
LAMBDA_SHARED_GRID = [10.0, 30.0, 80.0]
LAMBDA_GATE_GRID = [1e-3, 1e-2, 1e-1, 1.0, 5.0, 20.0, 80.0, 300.0,
                     1500.0, 8000.0, 5e4, 1e6]
LAMBDA_SEPARATE_GRID = [5.0, 20.0, 80.0, 300.0, 1000.0]
LAMBDA_F_STRUCT = 1e-3
LAMBDA_ALPHABETA = 1e-3

CV_ROUNDS = 4
CV_BLOCK_MAXITER = 35
FINAL_ROUNDS = 8
FINAL_BLOCK_MAXITER = 100
N_RESTARTS_HIER = 3
N_RESTARTS_REF = 2
DIPS_ROUNDS = 5
DIPS_BLOCK_MAXITER = 50

INNER_VAL_FRAC = 0.2
INNER_SEED = 13


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


def gate_slices(D):
    """Per-gate (idx, branch, Xs, bi, pi) built once from a full design D."""
    a = gates.assign(D["y"])
    out = {}
    for g in GATE_ORDER:
        idx, br = a[g]
        out[g] = dict(idx=idx, br=br, Xs=D["Xs"][idx], bi=D["bi"][idx], pi=D["pi"][idx])
    return out


# --------------------------------------------------------------- model ----
def softmax_nll_grad(eta, y):
    """SUM (not mean) NLL and its gradient wrt eta -- see gllvm/fit.py for
    why sum, not mean, matters here (player rows are a tiny fraction of the
    dataset, so mean-NLL crushes their gradient relative to any O(1) lambda)."""
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


def gate_eta(gslice, alpha_g, beta_g, Lbat_full_g, Fbat_g, Lpit_full_g, Fpit_g):
    Lb_sel = Lbat_full_g[gslice["bi"]]
    Lp_sel = Lpit_full_g[gslice["pi"]]
    eta = alpha_g[None, :] + gslice["Xs"] @ beta_g + Lb_sel @ Fbat_g.T + Lp_sel @ Fpit_g.T
    return eta, Lb_sel, Lp_sel


def pad(L, n):
    """Append the implicit zero row used for unseen batters/pitchers."""
    return np.vstack([L, np.zeros((1, L.shape[1]))])


# ---------------------------------------------- block 1: solve L given F --
def solve_L_given_F(mode, gslices, alpha, beta, Fbat, Fpit, d, lam_shared, lam_gate,
                     Lshared_bat0, Lshared_pit0, delta_bat0, delta_pit0,
                     n_bat, n_pit, maxiter):
    """One convex solve for L_shared and/or per-gate deltas, jointly across
    all gates (mode='hier'), or for just one side (mode='shared_only' freezes
    delta==0; mode='separate_only' freezes L_shared==0)."""
    use_shared = mode in ("hier", "shared_only")
    use_delta = mode in ("hier", "separate_only")
    gates_here = list(gslices.keys())

    slices_spec = []
    off = 0
    if use_shared:
        slices_spec.append(("Lshared_bat", off, off + n_bat * d)); off += n_bat * d
        slices_spec.append(("Lshared_pit", off, off + n_pit * d)); off += n_pit * d
    if use_delta:
        for g in gates_here:
            slices_spec.append((f"delta_bat_{g}", off, off + n_bat * d)); off += n_bat * d
            slices_spec.append((f"delta_pit_{g}", off, off + n_pit * d)); off += n_pit * d
    total = off

    def unpack(theta):
        d_ = {}
        for name, a, b in slices_spec:
            d_[name] = theta[a:b].reshape(-1, d)
        Lshared_bat = d_.get("Lshared_bat", np.zeros((n_bat, d)))
        Lshared_pit = d_.get("Lshared_pit", np.zeros((n_pit, d)))
        delta_bat = {g: d_.get(f"delta_bat_{g}", np.zeros((n_bat, d))) for g in gates_here}
        delta_pit = {g: d_.get(f"delta_pit_{g}", np.zeros((n_pit, d))) for g in gates_here}
        return Lshared_bat, Lshared_pit, delta_bat, delta_pit

    def og(theta):
        Lshared_bat, Lshared_pit, delta_bat, delta_pit = unpack(theta)
        loss = 0.0
        grad = {name: np.zeros((b - a,)) for name, a, b in slices_spec}
        d_Lshared_bat_acc = np.zeros((n_bat, d))
        d_Lshared_pit_acc = np.zeros((n_pit, d))
        for g in gates_here:
            Lbat_g = Lshared_bat + delta_bat[g]
            Lpit_g = Lshared_pit + delta_pit[g]
            Lbat_full = pad(Lbat_g, n_bat)
            Lpit_full = pad(Lpit_g, n_pit)
            eta, Lb_sel, Lp_sel = gate_eta(gslices[g], alpha[g], beta[g], Lbat_full, Fbat[g], Lpit_full, Fpit[g])
            nll, dEta, _ = softmax_nll_grad(eta, gslices[g]["br"])
            loss += nll
            d_Lbat_full = np.zeros((n_bat + 1, d))
            np.add.at(d_Lbat_full, gslices[g]["bi"], dEta @ Fbat[g])
            d_Lpit_full = np.zeros((n_pit + 1, d))
            np.add.at(d_Lpit_full, gslices[g]["pi"], dEta @ Fpit[g])
            g_bat = d_Lbat_full[:n_bat]
            g_pit = d_Lpit_full[:n_pit]
            if use_shared:
                d_Lshared_bat_acc += g_bat
                d_Lshared_pit_acc += g_pit
            if use_delta:
                grad[f"delta_bat_{g}"] += (g_bat + 2 * lam_gate * delta_bat[g]).ravel()
                grad[f"delta_pit_{g}"] += (g_pit + 2 * lam_gate * delta_pit[g]).ravel()
                loss += lam_gate * (np.sum(delta_bat[g] ** 2) + np.sum(delta_pit[g] ** 2))
        if use_shared:
            grad["Lshared_bat"] = (d_Lshared_bat_acc + 2 * lam_shared * Lshared_bat).ravel()
            grad["Lshared_pit"] = (d_Lshared_pit_acc + 2 * lam_shared * Lshared_pit).ravel()
            loss += lam_shared * (np.sum(Lshared_bat ** 2) + np.sum(Lshared_pit ** 2))
        gvec = np.concatenate([grad[name] for name, _, _ in slices_spec]) if slices_spec else np.zeros(0)
        return loss, gvec

    theta0_parts = []
    for name, a, b in slices_spec:
        if name == "Lshared_bat":
            theta0_parts.append(Lshared_bat0.ravel())
        elif name == "Lshared_pit":
            theta0_parts.append(Lshared_pit0.ravel())
        elif name.startswith("delta_bat_"):
            theta0_parts.append(delta_bat0[name[len("delta_bat_"):]].ravel())
        else:
            theta0_parts.append(delta_pit0[name[len("delta_pit_"):]].ravel())
    theta0 = np.concatenate(theta0_parts) if theta0_parts else np.zeros(0)

    if total == 0:
        return Lshared_bat0, Lshared_pit0, delta_bat0, delta_pit0, 0.0

    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    Lshared_bat, Lshared_pit, delta_bat, delta_pit = unpack(res.x)
    if not use_shared:
        Lshared_bat, Lshared_pit = np.zeros((n_bat, d)), np.zeros((n_pit, d))
    if not use_delta:
        delta_bat = {g: np.zeros((n_bat, d)) for g in gates_here}
        delta_pit = {g: np.zeros((n_pit, d)) for g in gates_here}
    return Lshared_bat, Lshared_pit, delta_bat, delta_pit, res.fun


# ------------------------------------------ block 2: solve F, alpha, beta -
def solve_F_given_L_one_gate(gslice, Lbat_full_g, Lpit_full_g, d, alpha0_g, beta0_g,
                              Fbat0_g, Fpit0_g, k_g, maxiter):
    p = gslice["Xs"].shape[1]

    def og(theta):
        i = 0
        alpha = theta[i:i + k_g]; i += k_g
        beta = theta[i:i + p * k_g].reshape(p, k_g); i += p * k_g
        Fbat = theta[i:i + k_g * d].reshape(k_g, d); i += k_g * d
        Fpit = theta[i:].reshape(k_g, d)
        eta, Lb_sel, Lp_sel = gate_eta(gslice, alpha, beta, Lbat_full_g, Fbat, Lpit_full_g, Fpit)
        nll, dEta, _ = softmax_nll_grad(eta, gslice["br"])
        loss = (nll + LAMBDA_F_STRUCT * (np.sum(Fbat ** 2) + np.sum(Fpit ** 2))
                + LAMBDA_ALPHABETA * (np.sum(alpha ** 2) + np.sum(beta ** 2)))
        d_alpha = dEta.sum(axis=0) + 2 * LAMBDA_ALPHABETA * alpha
        d_beta = gslice["Xs"].T @ dEta + 2 * LAMBDA_ALPHABETA * beta
        d_Fbat = dEta.T @ Lb_sel + 2 * LAMBDA_F_STRUCT * Fbat
        d_Fpit = dEta.T @ Lp_sel + 2 * LAMBDA_F_STRUCT * Fpit
        return loss, np.concatenate([d_alpha.ravel(), d_beta.ravel(), d_Fbat.ravel(), d_Fpit.ravel()])

    theta0 = np.concatenate([alpha0_g.ravel(), beta0_g.ravel(), Fbat0_g.ravel(), Fpit0_g.ravel()])
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    i = 0
    alpha = res.x[i:i + k_g]; i += k_g
    beta = res.x[i:i + p * k_g].reshape(p, k_g); i += p * k_g
    Fbat = res.x[i:i + k_g * d].reshape(k_g, d); i += k_g * d
    Fpit = res.x[i:].reshape(k_g, d)
    return alpha, beta, Fbat, Fpit, res.fun


def fit_struct_only_one_gate(gslice, k_g, maxiter=200):
    p = gslice["Xs"].shape[1]

    def og(theta):
        alpha = theta[:k_g]
        beta = theta[k_g:].reshape(p, k_g)
        eta = alpha[None, :] + gslice["Xs"] @ beta
        nll, dEta, _ = softmax_nll_grad(eta, gslice["br"])
        loss = nll + LAMBDA_ALPHABETA * (np.sum(alpha ** 2) + np.sum(beta ** 2))
        d_alpha = dEta.sum(axis=0) + 2 * LAMBDA_ALPHABETA * alpha
        d_beta = gslice["Xs"].T @ dEta + 2 * LAMBDA_ALPHABETA * beta
        return loss, np.concatenate([d_alpha.ravel(), d_beta.ravel()])

    theta0 = np.zeros(k_g + p * k_g)
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return res.x[:k_g], res.x[k_g:].reshape(p, k_g)


# ------------------------------------------------------------ alt fit -----
def alt_fit(mode, gslices, d, lam_shared, lam_gate, n_bat, n_pit, alpha0, beta0,
            rounds, block_maxiter, seed):
    rng = np.random.default_rng(seed)
    gates_here = list(gslices.keys())
    Fbat = {g: rng.normal(scale=0.3, size=(N_BRANCH[g], d)) for g in gates_here}
    Fpit = {g: rng.normal(scale=0.3, size=(N_BRANCH[g], d)) for g in gates_here}
    Lshared_bat = np.zeros((n_bat, d))
    Lshared_pit = np.zeros((n_pit, d))
    delta_bat = {g: np.zeros((n_bat, d)) for g in gates_here}
    delta_pit = {g: np.zeros((n_pit, d)) for g in gates_here}
    alpha = {g: alpha0[g].copy() for g in gates_here}
    beta = {g: beta0[g].copy() for g in gates_here}
    prev_loss = None
    for r in range(rounds):
        Lshared_bat, Lshared_pit, delta_bat, delta_pit, lossL = solve_L_given_F(
            mode, gslices, alpha, beta, Fbat, Fpit, d, lam_shared, lam_gate,
            Lshared_bat, Lshared_pit, delta_bat, delta_pit, n_bat, n_pit, block_maxiter)
        total_lossF = 0.0
        for g in gates_here:
            Lbat_g = Lshared_bat + delta_bat[g]
            Lpit_g = Lshared_pit + delta_pit[g]
            Lbat_full = pad(Lbat_g, n_bat)
            Lpit_full = pad(Lpit_g, n_pit)
            a_g, b_g, Fb_g, Fp_g, lossF = solve_F_given_L_one_gate(
                gslices[g], Lbat_full, Lpit_full, d, alpha[g], beta[g], Fbat[g], Fpit[g],
                N_BRANCH[g], block_maxiter)
            alpha[g], beta[g], Fbat[g], Fpit[g] = a_g, b_g, Fb_g, Fp_g
            total_lossF += lossF
        if prev_loss is not None and abs(prev_loss - total_lossF) < 1e-6 * max(1.0, abs(prev_loss)):
            prev_loss = total_lossF
            break
        prev_loss = total_lossF
    return dict(alpha=alpha, beta=beta, Lshared_bat=Lshared_bat, Lshared_pit=Lshared_pit,
                delta_bat=delta_bat, delta_pit=delta_pit, Fbat=Fbat, Fpit=Fpit,
                penalized_loss=prev_loss)


def predict_gate(g, gslice, fit, n_bat, n_pit, d):
    Lbat_g = fit["Lshared_bat"] + fit["delta_bat"].get(g, np.zeros((n_bat, d)))
    Lpit_g = fit["Lshared_pit"] + fit["delta_pit"].get(g, np.zeros((n_pit, d)))
    Lbat_full = pad(Lbat_g, n_bat)
    Lpit_full = pad(Lpit_g, n_pit)
    eta, _, _ = gate_eta(gslice, fit["alpha"][g], fit["beta"][g], Lbat_full, fit["Fbat"][g],
                          Lpit_full, fit["Fpit"][g])
    eta = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(eta)
    return ex / ex.sum(axis=1, keepdims=True)


def joint_deviance_from_fit(fit, gslices, y_full, n_bat, n_pit, d):
    gate_logp = {}
    for g in gslices:
        probs = predict_gate(g, gslices[g], fit, n_bat, n_pit, d)
        gate_logp[g] = np.log(np.maximum(probs, 1e-300))
    return gates.joint_deviance(gate_logp, y_full)


def per_gate_deviance_from_fit(fit, gslices, n_bat, n_pit, d):
    out = {}
    for g in gslices:
        probs = predict_gate(g, gslices[g], fit, n_bat, n_pit, d)
        br = gslices[g]["br"]
        nll = -np.log(np.maximum(probs[np.arange(len(br)), br], 1e-300))
        out[g] = float(2.0 * nll.mean())
    return out


def n_params_for(mode, d, n_bat, n_pit, p, gates_here=GATE_ORDER):
    use_shared = mode in ("hier", "shared_only")
    use_delta = mode in ("hier", "separate_only")
    total = 0
    if use_shared:
        total += (n_bat + n_pit) * d
    if use_delta:
        total += len(gates_here) * (n_bat + n_pit) * d
    for g in gates_here:
        k = N_BRANCH[g]
        total += k              # alpha
        total += p * k          # beta
        total += k * 2 * d      # Fbat + Fpit
    return total


# ----------------------------------------------------- DIPS ladder fits ---
def solve_L_single_gate(gslice, alpha, beta, Fbat, Fpit, d, lam, use_bat, use_pit,
                         Lbat0, Lpit0, n_bat, n_pit, maxiter):
    spec = []
    off = 0
    if use_bat:
        spec.append(("bat", off, off + n_bat * d)); off += n_bat * d
    if use_pit:
        spec.append(("pit", off, off + n_pit * d)); off += n_pit * d
    total = off
    if total == 0:
        return Lbat0, Lpit0, 0.0

    def unpack(theta):
        Lbat = theta[spec[0][1]:spec[0][2]].reshape(n_bat, d) if use_bat else np.zeros((n_bat, d))
        if use_pit:
            i = 1 if use_bat else 0
            Lpit = theta[spec[i][1]:spec[i][2]].reshape(n_pit, d)
        else:
            Lpit = np.zeros((n_pit, d))
        return Lbat, Lpit

    def og(theta):
        Lbat, Lpit = unpack(theta)
        Lbat_full, Lpit_full = pad(Lbat, n_bat), pad(Lpit, n_pit)
        eta, Lb_sel, Lp_sel = gate_eta(gslice, alpha, beta, Lbat_full, Fbat, Lpit_full, Fpit)
        nll, dEta, _ = softmax_nll_grad(eta, gslice["br"])
        loss = nll
        grads = []
        if use_bat:
            d_Lbat_full = np.zeros((n_bat + 1, d))
            np.add.at(d_Lbat_full, gslice["bi"], dEta @ Fbat)
            g_bat = d_Lbat_full[:n_bat] + 2 * lam * Lbat
            loss += lam * np.sum(Lbat ** 2)
            grads.append(g_bat.ravel())
        if use_pit:
            d_Lpit_full = np.zeros((n_pit + 1, d))
            np.add.at(d_Lpit_full, gslice["pi"], dEta @ Fpit)
            g_pit = d_Lpit_full[:n_pit] + 2 * lam * Lpit
            loss += lam * np.sum(Lpit ** 2)
            grads.append(g_pit.ravel())
        return loss, np.concatenate(grads)

    theta0_parts = []
    if use_bat:
        theta0_parts.append(Lbat0.ravel())
    if use_pit:
        theta0_parts.append(Lpit0.ravel())
    theta0 = np.concatenate(theta0_parts)
    res = minimize(og, theta0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    Lbat, Lpit = unpack(res.x)
    return Lbat, Lpit, res.fun


def alt_fit_single_gate(gslice, k, d, lam, use_bat, use_pit, n_bat, n_pit, rounds, block_maxiter, seed):
    rng = np.random.default_rng(seed)
    Fbat = rng.normal(scale=0.3, size=(k, d)) if use_bat else np.zeros((k, d))
    Fpit = rng.normal(scale=0.3, size=(k, d)) if use_pit else np.zeros((k, d))
    Lbat = np.zeros((n_bat, d))
    Lpit = np.zeros((n_pit, d))
    alpha, beta = fit_struct_only_one_gate(gslice, k)
    prev_loss = None
    for r in range(rounds):
        Lbat, Lpit, lossL = solve_L_single_gate(gslice, alpha, beta, Fbat, Fpit, d, lam,
                                                 use_bat, use_pit, Lbat, Lpit, n_bat, n_pit, block_maxiter)
        Lbat_full, Lpit_full = pad(Lbat, n_bat), pad(Lpit, n_pit)
        alpha, beta, Fbat, Fpit, lossF = solve_F_given_L_one_gate(
            gslice, Lbat_full, Lpit_full, d, alpha, beta, Fbat, Fpit, k, block_maxiter)
        if prev_loss is not None and abs(prev_loss - lossF) < 1e-6 * max(1.0, abs(prev_loss)):
            prev_loss = lossF
            break
        prev_loss = lossF
    return dict(alpha=alpha, beta=beta, Lbat=Lbat, Lpit=Lpit, Fbat=Fbat, Fpit=Fpit, penalized_loss=prev_loss)


def predict_single_gate(gslice, fit, n_bat, n_pit):
    Lbat_full, Lpit_full = pad(fit["Lbat"], n_bat), pad(fit["Lpit"], n_pit)
    eta, _, _ = gate_eta(gslice, fit["alpha"], fit["beta"], Lbat_full, fit["Fbat"], Lpit_full, fit["Fpit"])
    eta = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(eta)
    return ex / ex.sum(axis=1, keepdims=True)


def single_gate_deviance(gslice, probs):
    br = gslice["br"]
    nll = -np.log(np.maximum(probs[np.arange(len(br)), br], 1e-300))
    return float(2.0 * nll.mean())


# ------------------------------------------------------------- pipeline --
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
    p_struct = D_tr["p"]

    gs_tr_full = gate_slices(D_tr)
    gs_te_full = gate_slices(D_te)
    for g in GATE_ORDER:
        log(f"gate {g:8s} train_rows={len(gs_tr_full[g]['idx']):7d} test_rows={len(gs_te_full[g]['idx']):6d} branches={N_BRANCH[g]}")

    # -------------------------------------------------- inner split (CV) --
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
    gs_itr = gate_slices(D_itr)
    gs_ival = gate_slices(D_ival)

    struct0 = {g: fit_struct_only_one_gate(gs_itr[g], N_BRANCH[g]) for g in GATE_ORDER}
    alpha0_i = {g: struct0[g][0] for g in GATE_ORDER}
    beta0_i = {g: struct0[g][1] for g in GATE_ORDER}

    def fit_mode_inner(mode, d, lam_shared, lam_gate, seed=0, rounds=CV_ROUNDS, maxiter=CV_BLOCK_MAXITER):
        return alt_fit(mode, gs_itr, d, lam_shared, lam_gate, in_bat, in_pit, alpha0_i, beta0_i,
                        rounds, maxiter, seed)

    def val_joint_dev(fit, d):
        return joint_deviance_from_fit(fit, gs_ival, D_ival["y"], in_bat, in_pit, d)

    # ------------------------------------------------------- pick d ------
    # Cheap proxy: sweep d using the shared_only mode (single L, per-gate F)
    # at a fixed moderate lambda_shared. This is the same d used everywhere
    # below (hier, shared_only, separate_only, DIPS) so the lambda_gate
    # curve compares apples to apples.
    log("selecting d via shared_only proxy sweep")
    d_scores = []
    for d in D_GRID:
        t0 = time.time()
        f = fit_mode_inner("shared_only", d, lam_shared=30.0, lam_gate=None)
        vd = val_joint_dev(f, d)
        d_scores.append({"d": d, "val_deviance": vd, "fit_sec": time.time() - t0})
        log(f"  d={d} val_dev={vd:.5f} ({time.time()-t0:.1f}s)")
    d_star = min(d_scores, key=lambda r: r["val_deviance"])["d"]
    log(f"selected d*={d_star}")

    # ----------------------------------------------- pick lambda_shared --
    log("selecting lambda_shared via shared_only")
    lam_shared_scores = []
    for lam in LAMBDA_SHARED_GRID:
        t0 = time.time()
        f = fit_mode_inner("shared_only", d_star, lam_shared=lam, lam_gate=None)
        vd = val_joint_dev(f, d_star)
        lam_shared_scores.append({"lambda_shared": lam, "val_deviance": vd, "fit_sec": time.time() - t0})
        log(f"  lambda_shared={lam:g} val_dev={vd:.5f} ({time.time()-t0:.1f}s)")
    lam_shared_star = min(lam_shared_scores, key=lambda r: r["val_deviance"])["lambda_shared"]
    shared_only_val_best = min(r["val_deviance"] for r in lam_shared_scores)
    log(f"selected lambda_shared*={lam_shared_star:g} (val_dev={shared_only_val_best:.5f})")

    # ------------------------------------------- separate_only reference -
    log("fitting separate_only reference (own lambda_gate CV)")
    lam_sep_scores = []
    for lam in LAMBDA_SEPARATE_GRID:
        t0 = time.time()
        f = fit_mode_inner("separate_only", d_star, lam_shared=None, lam_gate=lam)
        vd = val_joint_dev(f, d_star)
        lam_sep_scores.append({"lambda_gate": lam, "val_deviance": vd, "fit_sec": time.time() - t0})
        log(f"  separate_only lambda_gate={lam:g} val_dev={vd:.5f} ({time.time()-t0:.1f}s)")
    lam_sep_star = min(lam_sep_scores, key=lambda r: r["val_deviance"])["lambda_gate"]
    separate_only_val_best = min(r["val_deviance"] for r in lam_sep_scores)
    log(f"selected separate_only lambda*={lam_sep_star:g} (val_dev={separate_only_val_best:.5f})")

    # --------------------------------------- MAIN EVENT: lambda_gate curve
    log("sweeping lambda_gate for the hierarchical (hier) model -- THE headline result")
    lambda_gate_curve = []
    for lam in LAMBDA_GATE_GRID:
        t0 = time.time()
        f = fit_mode_inner("hier", d_star, lam_shared=lam_shared_star, lam_gate=lam)
        vd = val_joint_dev(f, d_star)
        lambda_gate_curve.append({"lambda_gate": lam, "val_deviance": vd, "fit_sec": time.time() - t0})
        log(f"  lambda_gate={lam:g} val_dev={vd:.5f} ({time.time()-t0:.1f}s)")
    best_curve_pt = min(lambda_gate_curve, key=lambda r: r["val_deviance"])
    lam_gate_star = best_curve_pt["lambda_gate"]
    log(f"selected lambda_gate*={lam_gate_star:g} (val_dev={best_curve_pt['val_deviance']:.5f})")

    lo_end = lambda_gate_curve[0]
    hi_end = lambda_gate_curve[-1]
    log(f"limit check: hier@lambda_gate={lo_end['lambda_gate']:g} val_dev={lo_end['val_deviance']:.5f} "
        f"vs separate_only val_dev={separate_only_val_best:.5f} "
        f"(diff={lo_end['val_deviance']-separate_only_val_best:+.5f})")
    log(f"limit check: hier@lambda_gate={hi_end['lambda_gate']:g} val_dev={hi_end['val_deviance']:.5f} "
        f"vs shared_only val_dev={shared_only_val_best:.5f} "
        f"(diff={hi_end['val_deviance']-shared_only_val_best:+.5f})")

    # ------------------------------------------------ final fits on FULL TRAIN
    alpha0_f = {}
    beta0_f = {}
    for g in GATE_ORDER:
        a_g, b_g = fit_struct_only_one_gate(gs_tr_full[g], N_BRANCH[g], maxiter=300)
        alpha0_f[g] = a_g
        beta0_f[g] = b_g
    struct_only_test_dev = joint_deviance_from_fit(
        dict(alpha=alpha0_f, beta=beta0_f,
             Lshared_bat=np.zeros((n_bat, 1)), Lshared_pit=np.zeros((n_pit, 1)),
             delta_bat={g: np.zeros((n_bat, 1)) for g in GATE_ORDER},
             delta_pit={g: np.zeros((n_pit, 1)) for g in GATE_ORDER},
             Fbat={g: np.zeros((N_BRANCH[g], 1)) for g in GATE_ORDER},
             Fpit={g: np.zeros((N_BRANCH[g], 1)) for g in GATE_ORDER}),
        gs_te_full, D_te["y"], n_bat, n_pit, 1)
    log(f"structural-only (no player latent) test deviance = {struct_only_test_dev:.5f}")

    log(f"final HIER fit: d={d_star} lambda_shared={lam_shared_star:g} lambda_gate={lam_gate_star:g}, "
        f"{N_RESTARTS_HIER} restarts")
    restarts = []
    for seed in range(N_RESTARTS_HIER):
        t0 = time.time()
        f = alt_fit("hier", gs_tr_full, d_star, lam_shared_star, lam_gate_star, n_bat, n_pit,
                    alpha0_f, beta0_f, FINAL_ROUNDS, FINAL_BLOCK_MAXITER, seed=2000 + seed)
        test_dev = joint_deviance_from_fit(f, gs_te_full, D_te["y"], n_bat, n_pit, d_star)
        train_dev = joint_deviance_from_fit(f, gs_tr_full, D_tr["y"], n_bat, n_pit, d_star)
        dt = time.time() - t0
        restarts.append(dict(seed=seed, fit=f, test_dev=test_dev, train_dev=train_dev,
                              penalized_loss=f["penalized_loss"], fit_sec=dt))
        log(f"  [restart {seed}] train_dev={train_dev:.5f} test_dev={test_dev:.5f} "
            f"penalized_loss={f['penalized_loss']:.2f} ({dt:.1f}s)")

    canonical = min(restarts, key=lambda r: r["penalized_loss"])
    test_devs = [r["test_dev"] for r in restarts]
    restart_spread = max(test_devs) - min(test_devs)
    log(f"restart spread: min={min(test_devs):.5f} max={max(test_devs):.5f} spread={restart_spread:.5f}")
    log(f"canonical = seed {canonical['seed']} test_dev={canonical['test_dev']:.5f}")

    joint_test_dev = canonical["test_dev"]
    per_gate_dev = per_gate_deviance_from_fit(canonical["fit"], gs_te_full, n_bat, n_pit, d_star)
    log(f"per-gate test deviance: {per_gate_dev}")

    # ---------------------------- final shared_only / separate_only refits
    log("final refit: shared_only reference on full train")
    f_shared = alt_fit("shared_only", gs_tr_full, d_star, lam_shared_star, None, n_bat, n_pit,
                        alpha0_f, beta0_f, FINAL_ROUNDS, FINAL_BLOCK_MAXITER, seed=3000)
    shared_only_test_dev = joint_deviance_from_fit(f_shared, gs_te_full, D_te["y"], n_bat, n_pit, d_star)
    log(f"shared_only (Variant B stand-in) test deviance = {shared_only_test_dev:.5f}")

    log("final refit: separate_only reference on full train")
    f_sep = alt_fit("separate_only", gs_tr_full, d_star, None, lam_sep_star, n_bat, n_pit,
                     alpha0_f, beta0_f, FINAL_ROUNDS, FINAL_BLOCK_MAXITER, seed=4000)
    separate_only_test_dev = joint_deviance_from_fit(f_sep, gs_te_full, D_te["y"], n_bat, n_pit, d_star)
    log(f"separate_only (Variant A stand-in) test deviance = {separate_only_test_dev:.5f}")

    # ------------------------------------------------------ DIPS ladder --
    log("DIPS ladder: per-gate batter-only / pitcher-only / both")
    dips = {}
    for g in GATE_ORDER:
        k = N_BRANCH[g]
        row = {}
        for label, use_bat, use_pit in [("batter_only", True, False),
                                         ("pitcher_only", False, True),
                                         ("both", True, True)]:
            f = alt_fit_single_gate(gs_tr_full[g], k, d_star, lam_shared_star, use_bat, use_pit,
                                     n_bat, n_pit, DIPS_ROUNDS, DIPS_BLOCK_MAXITER, seed=5000)
            probs = predict_single_gate(gs_te_full[g], f, n_bat, n_pit)
            dev = single_gate_deviance(gs_te_full[g], probs)
            row[label] = dev
        dips[g] = row
        log(f"  gate={g:8s} batter_only={row['batter_only']:.5f} pitcher_only={row['pitcher_only']:.5f} both={row['both']:.5f}")
    # struct-only-per-gate reference (no latent at all) for the ladder's floor
    dips_struct = {}
    for g in GATE_ORDER:
        a_g, b_g = alpha0_f[g], beta0_f[g]
        k = N_BRANCH[g]
        eta = a_g[None, :] + gs_te_full[g]["Xs"] @ b_g
        eta = eta - eta.max(axis=1, keepdims=True)
        ex = np.exp(eta)
        probs = ex / ex.sum(axis=1, keepdims=True)
        dips_struct[g] = single_gate_deviance(gs_te_full[g], probs)
    log(f"DIPS ladder struct-only floor: {dips_struct}")

    runtime_sec = time.time() - T0

    n_params = n_params_for("hier", d_star, n_bat, n_pit, p_struct)

    result = {
        "model": "nested_hier (Variant C: shared + per-gate-shrunk-deviation latents)",
        "test_pa": len(te),
        "joint_test_deviance": joint_test_dev,
        "null_deviance": null_dev,
        "structural_only_test_deviance": struct_only_test_dev,
        "per_gate_deviance": per_gate_dev,
        "d": d_star,
        "lambda_shared": lam_shared_star,
        "lambda_gate": lam_gate_star,
        "lambda_gate_curve": lambda_gate_curve,
        "n_params": n_params,
        "runtime_sec": runtime_sec,
        "restart_spread": restart_spread,
        "restart_test_deviances": test_devs,
        "canonical_restart_seed": canonical["seed"],
        "d_selection": d_scores,
        "lambda_shared_selection": lam_shared_scores,
        "sibling_reference": {
            "shared_only": {
                "description": "Variant B stand-in: delta frozen at 0, one L per player reused at every gate",
                "lambda_shared": lam_shared_star,
                "val_deviance": shared_only_val_best,
                "test_deviance": shared_only_test_dev,
            },
            "separate_only": {
                "description": "Variant A stand-in: L_shared frozen at 0, independent per-gate latents",
                "lambda_gate": lam_sep_star,
                "val_deviance": separate_only_val_best,
                "test_deviance": separate_only_test_dev,
                "lambda_grid": lam_sep_scores,
            },
        },
        "limit_check": {
            "lo_end_lambda_gate": lo_end["lambda_gate"],
            "lo_end_val_deviance": lo_end["val_deviance"],
            "separate_only_val_deviance": separate_only_val_best,
            "lo_end_minus_separate_only": lo_end["val_deviance"] - separate_only_val_best,
            "hi_end_lambda_gate": hi_end["lambda_gate"],
            "hi_end_val_deviance": hi_end["val_deviance"],
            "shared_only_val_deviance": shared_only_val_best,
            "hi_end_minus_shared_only": hi_end["val_deviance"] - shared_only_val_best,
        },
        "dips_ladder": dips,
        "dips_ladder_struct_only_floor": dips_struct,
        "alternation": "block coordinate descent: block 1 solves L_shared + all per-gate deltas "
                        "jointly across gates (convex given F); block 2 solves F/alpha/beta per gate "
                        "independently (convex given L). One-shot joint L-BFGS-B was not attempted here "
                        "-- gllvm/fit.py already demonstrated it collapses L,F to 0, and this "
                        "parameterization is MORE prone to it (L_shared, delta additively confounded).",
        "inner_cv": {"method": "single 80/20 game-level split within train", "seed": INNER_SEED},
    }

    with open(HERE / "result.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    log("wrote result.json")

    np.savez(
        HERE / "latent.npz",
        Lshared_bat=canonical["fit"]["Lshared_bat"], Lshared_pit=canonical["fit"]["Lshared_pit"],
        **{f"delta_bat_{g}": canonical["fit"]["delta_bat"][g] for g in GATE_ORDER},
        **{f"delta_pit_{g}": canonical["fit"]["delta_pit"][g] for g in GATE_ORDER},
        **{f"Fbat_{g}": canonical["fit"]["Fbat"][g] for g in GATE_ORDER},
        **{f"Fpit_{g}": canonical["fit"]["Fpit"][g] for g in GATE_ORDER},
        **{f"alpha_{g}": canonical["fit"]["alpha"][g] for g in GATE_ORDER},
        **{f"beta_{g}": canonical["fit"]["beta"][g] for g in GATE_ORDER},
        bat_ids=np.array(bat_ids), pit_ids=np.array(pit_ids),
        gate_order=np.array(GATE_ORDER), d=d_star,
    )
    log("wrote latent.npz")

    log(f"TOTAL RUNTIME {runtime_sec:.1f}s")
    return result


if __name__ == "__main__":
    main()
