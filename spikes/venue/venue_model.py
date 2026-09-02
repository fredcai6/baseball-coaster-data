"""Venue main-effect + venue x stance deviation, layered on step1's per-node
binary-tree fit. Does NOT modify spikes/pitch/step1.py -- reuses its
`ao_prob` (the Aranda-Ordaz link) and `LAM_STRUCT` unmodified, and re-derives
`nll_grad` / `fit` / `node_dev` with two extra additive terms, following the
exact same np.bincount pattern step1.py uses for the batter/pitcher terms.

Parameter layout (theta):
    [alpha, beta(p), b(n_bat), q(n_pit), v(n_ven), w(n_vs)]

v is one coefficient per park (n_ven = 16, ridge lam_ven). w is one
coefficient per (park, stance) pair (n_vs = 32, ridge lam_vs), nested on top
of the venue main effect AND the existing hand_opposite platoon column in
structural() -- w captures venue-specific L/R asymmetry, not the general
platoon effect, which stays in beta.

Rows whose venue (or stance) is unknown get a SENTINEL index one past the
end of v (or w). Implemented via the append-a-zero-row trick already used in
spikes/fuse/analyze.py's nested_category_probs for rostered-elsewhere
players: eta indexes into a padded array [v, 0.0], so an unknown-venue row
contributes exactly 0 to that term, and because the sentinel is never part
of the optimized vector, its "coefficient" cannot drift away from zero and
the gradient never flows to it (np.bincount's minlength+1 bin is computed
and then discarded).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pitch"))
import step1  # noqa: E402  (reuse ao_prob + LAM_STRUCT unmodified)

ao_prob = step1.ao_prob
LAM_STRUCT = step1.LAM_STRUCT
EPS = step1.EPS


def theta_size(ps, n_bat, n_pit, n_ven, n_vs):
    return 1 + ps + n_bat + n_pit + n_ven + n_vs


def unpack(th, ps, n_bat, n_pit, n_ven, n_vs):
    i = 0
    alpha = th[i]; i += 1
    beta = th[i:i + ps]; i += ps
    b = th[i:i + n_bat]; i += n_bat
    q = th[i:i + n_pit]; i += n_pit
    v = th[i:i + n_ven]; i += n_ven
    w = th[i:i + n_vs]; i += n_vs
    return alpha, beta, b, q, v, w


def eta_of(th, Xs, bi, pj, vi, vsi, n_bat, n_pit, n_ven, n_vs):
    ps = Xs.shape[1]
    alpha, beta, b, q, v, w = unpack(th, ps, n_bat, n_pit, n_ven, n_vs)
    v_pad = np.concatenate([v, [0.0]])
    w_pad = np.concatenate([w, [0.0]])
    return alpha + Xs @ beta + b[bi] + q[pj] + v_pad[vi] + w_pad[vsi]


def nll_grad(th, Xs, bi, pj, vi, vsi, yv, n_bat, n_pit, n_ven, n_vs, psi,
             lam_b, lam_p, lam_ven, lam_vs):
    ps = Xs.shape[1]
    alpha, beta, b, q, v, w = unpack(th, ps, n_bat, n_pit, n_ven, n_vs)
    v_pad = np.concatenate([v, [0.0]])
    w_pad = np.concatenate([w, [0.0]])
    eta = alpha + Xs @ beta + b[bi] + q[pj] + v_pad[vi] + w_pad[vsi]
    p, omp, e, u = ao_prob(eta, psi)
    nll = -np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
    nll += 0.5 * (lam_b * b @ b + lam_p * q @ q + LAM_STRUCT * beta @ beta
                  + lam_ven * v @ v + lam_vs * w @ w)
    g_eta = -(yv - p) * (e / u) / p
    grad_v = np.bincount(vi, weights=g_eta, minlength=n_ven + 1)[:n_ven] + lam_ven * v
    grad_w = np.bincount(vsi, weights=g_eta, minlength=n_vs + 1)[:n_vs] + lam_vs * w
    grad = np.concatenate((
        [g_eta.sum()],
        Xs.T @ g_eta + LAM_STRUCT * beta,
        np.bincount(bi, weights=g_eta, minlength=n_bat) + lam_b * b,
        np.bincount(pj, weights=g_eta, minlength=n_pit) + lam_p * q,
        grad_v,
        grad_w,
    ))
    return nll, grad


def fit(Xs, bi, pj, vi, vsi, yv, n_bat, n_pit, n_ven, n_vs, psi, lam_b, lam_p,
        lam_ven, lam_vs, x0=None):
    n = theta_size(Xs.shape[1], n_bat, n_pit, n_ven, n_vs)
    x0 = np.zeros(n) if x0 is None else x0
    r = minimize(nll_grad, x0,
                 args=(Xs, bi, pj, vi, vsi, yv, n_bat, n_pit, n_ven, n_vs,
                       psi, lam_b, lam_p, lam_ven, lam_vs),
                 jac=True, method="L-BFGS-B",
                 options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-8})
    return r.x


def node_dev(th, Xs, bi, pj, vi, vsi, yv, n_bat, n_pit, n_ven, n_vs, psi):
    eta = eta_of(th, Xs, bi, pj, vi, vsi, n_bat, n_pit, n_ven, n_vs)
    p, omp, _, _ = ao_prob(eta, psi)
    return -2.0 * np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
