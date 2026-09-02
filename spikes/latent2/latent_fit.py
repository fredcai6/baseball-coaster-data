"""LATENT2: re-test low-rank player parameterisation vs shape D's free
per-node player effects, under the CURRENT winning architecture (shape D,
step6_shapes.py) and with roughly EQUAL hyperparameter tuning freedom.

Why this file exists
---------------------
Every prior latent attempt in this repo lost to the free-effects arm, but
every one of them was handicapped relative to it:
  - step1b_fixed.py used ONE global lambda across L, M, f, g, all 9 nodes,
    both sides -- 1 tuned hyperparameter, against the free arm's 18
    (lam_bat x lam_pit, per node). It also inherited psi rather than
    re-selecting it, and ran on shape A, pre-covariate-fix, pre-plateau-
    tie-break.
  - flat GLLVM (spikes/gllvm/fit.py) ties lambda across bat/pit and sweeps
    it flat across all 10 categories at once, no per-node structure at all.

This spike fixes that by exploiting a structural fact: because L is SHARED
across all 9 nodes, the classic rescaling degeneracy (L -> L/c, f -> c*f)
is only ONE-DIMENSIONAL, not per-node. A per-node penalty on the loadings
f[n], g[n] is therefore identifiable, and the effective penalty on node n's
batter effect becomes sqrt(lam_L * lam_L*c_f[n]) -- i.e. a genuine per-node
shrinkage knob, just like the free arm has, that step1b_fixed never used.

Three arms, all on SHAPE D (step6_shapes.SHAPES["D"], the current winner,
3.94526 on the frozen test):

  Arm 1 (shared latent, per-side lambda):
    eta_n = alpha_n + Xs.beta_n + L[bat].f[n] + M[pit].g[n], L,M shared
    across nodes. Penalty lam_L on (L, all f), lam_M on (M, all g).
    Select (d, lam_L, lam_M).

  Arm 2 (arm 1 + per-node loading multipliers):
    f[n] penalised by lam_L * c_f[n], g[n] by lam_M * c_g[n]. c_f, c_g
    selected in a STAGED, greedy, single-pass per-node loop (NOT a full
    joint grid -- 9 nodes x 2 sides x 5 candidates would need a joint
    refit per candidate; a full joint grid over all 18 multipliers at once
    is computationally out of reach here, exactly as step1.py's own
    per-node lambda selection is staged rather than joint. This is
    reported, not hidden.)

  Arm 3 (hybrid: free effects + low-rank on top):
    eta_n = ... + L[bat].f[n] + b[bat,n] + M[pit].g[n] + q[pit,n], with b,
    q penalised by shape D's OWN selected lam_bat[n]/lam_pit[n] (pinned,
    not re-tuned), and the latent channel penalised by (lam_L, lam_M) as
    in arm 1. If no cross-node structure exists beyond what the free
    effects already carry, (L, f, M, g) should shrink to ~0 and this
    reduces to shape D exactly. Initialised so that at theta_0 the fit IS
    exactly shape D's own per-node free fit (SVD of the free-fit matrix
    gives L0, f0, M0, g0; b0 = B - L0 @ f0.T, q0 = Q - M0 @ g0.T), so any
    improvement from there is entirely the low-rank channel's doing.

Non-negotiable discipline (see task brief): select on inner validation only
(80/20 split BY GAME within train, same seed 90210 / same 80% cut convention
as step1.py and step6_shapes.py), compute the frozen test number ONCE per
arm at the end, plateau tie-break (LAM_TOL) + grid-edge detection on every
lambda selection, a null-beating sanity guard before reporting anything,
multiple random restarts on the final fit with reported spread, and a
scipy.optimize.check_grad gradient check before trusting any fit.

Does not modify step1.py, step6_shapes.py or common.py. Imports SHAPE D and
the plateau tie-break from step6_shapes, and ao_prob/fit/node_dev from
step1, unchanged.
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
SPIKES = os.path.dirname(HERE)
sys.path.insert(0, SPIKES)
sys.path.insert(0, os.path.join(SPIKES, "fuse"))
sys.path.insert(0, os.path.join(SPIKES, "pitch"))

import numpy as np
from scipy.optimize import minimize, check_grad
import common
from analyze import structural
import step1                    # ao_prob, fit (node), node_dev, EPS, LAM_STRUCT -- UNCHANGED
import step6_shapes as S6       # SHAPE D, select_plateau, LAM_TOL, PSI_GRID -- UNCHANGED

NODES = S6.SHAPES["D"]
NNODE = len(NODES)
NODE_NAMES = [nm for nm, _, _ in NODES]
EPS = step1.EPS
LAM_STRUCT = step1.LAM_STRUCT
ao_prob = step1.ao_prob
fit_node = step1.fit
node_dev = step1.node_dev
PSI_GRID = S6.PSI_GRID
select_plateau = S6.select_plateau
LAM_TOL = S6.LAM_TOL

NULL_DEV = 4.01172
SHAPE_D_TARGET = 3.94526   # frozen test, the number every arm must beat to "win"

D_GRID = [1, 2, 3, 4]
LAM_GRID = [15, 25, 40, 60, 90, 130, 190, 280]      # same grid as step1b_fixed
C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]                 # arm-2 per-node loading multipliers
N_RESTARTS = 5

RESULT_PATH = os.path.join(HERE, "result.json")
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------------ data --

def setup():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}

    g = sorted(train_g)
    rs = np.random.RandomState(90210)     # identical convention to step1/step6
    rs.shuffle(g)
    ifit = set(g[: int(0.8 * len(g))])

    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)

    def pack(rs_, reach, pos):
        sub = [r for r in rs_ if r["y"] in reach]
        Xs = structural(sub, season_idx)
        bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
        pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
        yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
        return Xs, bi, pj, yv

    itr_rows = [r for r in tr if r["game_id"] in ifit]
    ival_rows = [r for r in tr if r["game_id"] not in ifit]
    D_fit = [pack(itr_rows, rc, po) for _, rc, po in NODES]
    D_val = [pack(ival_rows, rc, po) for _, rc, po in NODES]
    D_tr = [pack(tr, rc, po) for _, rc, po in NODES]
    D_te = [pack(te, rc, po) for _, rc, po in NODES]
    ps = D_fit[0][0].shape[1]
    n_val, n_test = len(ival_rows), len(te)
    log(f"train {len(tr)} test {len(te)}  nodes {NNODE}  ps {ps}  "
        f"batters {n_bat} pitchers {n_pit}  n_val {n_val}")

    # shape D's OWN per-node lam_bat / lam_pit / psi, from step6_result.json --
    # the free arm's already-selected hyperparameters. Used (a) as the pinned
    # b/q penalty in arm 3, (b) as the inherited psi for all three arms, and
    # (c) to build the SVD warm-start / arm-3 residual init.
    s6 = json.load(open(os.path.join(SPIKES, "pitch", "step6_result.json")))
    s6_nodes = {n["node"]: n for n in s6["D"]["nodes"]}
    psi0 = np.array([s6_nodes[nm]["psi"] for nm in NODE_NAMES])
    lam_bat0 = np.array([s6_nodes[nm]["lam_bat"] for nm in NODE_NAMES])
    lam_pit0 = np.array([s6_nodes[nm]["lam_pit"] for nm in NODE_NAMES])
    log(f"shape D reference total (frozen test, from step6_result.json) = "
        f"{s6['D']['total_deviance']:.5f}")

    return dict(rows=rows, tr=tr, te=te, season_idx=season_idx, ifit=ifit,
                bats=bats, pits=pits, BI=BI, PI=PI, n_bat=n_bat, n_pit=n_pit,
                D_fit=D_fit, D_val=D_val, D_tr=D_tr, D_te=D_te, ps=ps,
                n_val=n_val, n_test=n_test,
                psi0=psi0, lam_bat0=lam_bat0, lam_pit0=lam_pit0)


# --------------------------------------------------------- node free fits --

def node_free_fits(data, psis, lam_bat, lam_pit, n_bat, n_pit, ps):
    """Refit shape D's OWN per-node free-effect model (step1.fit, unchanged)
    on `data` with the pinned (psis, lam_bat, lam_pit). Returns alpha(N),
    beta(N,ps), B(n_bat,N), Q(n_pit,N) -- the exact per-node free-effect
    matrices, used as the SVD warm-start source for all three arms and as
    the residual base for arm 3's init."""
    alpha = np.zeros(NNODE); beta = np.zeros((NNODE, ps))
    B = np.zeros((n_bat, NNODE)); Q = np.zeros((n_pit, NNODE))
    for n, (Xs, bi, pj, yv) in enumerate(data):
        th = fit_node(Xs, bi, pj, yv, n_bat, n_pit, psis[n], lam_bat[n], lam_pit[n])
        alpha[n] = th[0]; beta[n] = th[1:1 + ps]
        B[:, n] = th[1 + ps:1 + ps + n_bat]
        Q[:, n] = th[1 + ps + n_bat:]
    return alpha, beta, B, Q


def svd_init(B, Q, d):
    Ub, Sb, Vbt = np.linalg.svd(B, full_matrices=False)
    Uq, Sq, Vqt = np.linalg.svd(Q, full_matrices=False)
    L = Ub[:, :d] * np.sqrt(Sb[:d]); f = (Vbt[:d].T * np.sqrt(Sb[:d]))
    M = Uq[:, :d] * np.sqrt(Sq[:d]); g = (Vqt[:d].T * np.sqrt(Sq[:d]))
    return L, f, M, g


# ==================================================== ARM 1 / ARM 2 model --
# eta_n = alpha_n + Xs.beta_n + L[bi].f[n] + M[pj].g[n], L,M SHARED.
# Penalty: 0.5*lam_L*(||L||^2 + c_f[n]*||f[n]||^2 summed) + same for M/g/c_g.
# Arm 1 = arm 2 with c_f = c_g = 1 everywhere.

def sizes_shared(d, n_bat, n_pit, ps, nn=NNODE):
    return nn, nn * ps, nn * d, nn * d, n_bat * d, n_pit * d


def pack_shared(alpha, beta, f, g, L, M):
    return np.concatenate([alpha.ravel(), beta.ravel(), f.ravel(), g.ravel(),
                            L.ravel(), M.ravel()])


def unpack_shared(th, d, n_bat, n_pit, ps, nn=NNODE):
    o = 0
    alpha = th[o:o + nn]; o += nn
    beta = th[o:o + nn * ps].reshape(nn, ps); o += nn * ps
    f = th[o:o + nn * d].reshape(nn, d); o += nn * d
    g = th[o:o + nn * d].reshape(nn, d); o += nn * d
    L = th[o:o + n_bat * d].reshape(n_bat, d); o += n_bat * d
    M = th[o:o + n_pit * d].reshape(n_pit, d)
    return alpha, beta, f, g, L, M


def obj_shared(th, data, psis, d, n_bat, n_pit, lam_L, lam_M, c_f, c_g):
    ps = data[0][0].shape[1]
    nn = len(data)
    alpha, beta, f, g, L, M = unpack_shared(th, d, n_bat, n_pit, ps, nn)
    nll = 0.0
    gA = np.zeros(nn); gB = np.zeros((nn, ps))
    gF = np.zeros((nn, d)); gG = np.zeros((nn, d))
    gL = np.zeros((n_bat, d)); gM = np.zeros((n_pit, d))
    for n, (Xs, bi, pj, yv) in enumerate(data):
        Lb, Mp = L[bi], M[pj]
        eta = alpha[n] + Xs @ beta[n] + Lb @ f[n] + Mp @ g[n]
        p, omp, e, u = ao_prob(eta, psis[n])
        nll += -np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
        ge = -(yv - p) * (e / u) / p
        gA[n] = ge.sum()
        gB[n] = Xs.T @ ge + LAM_STRUCT * beta[n]
        gF[n] = Lb.T @ ge + lam_L * c_f[n] * f[n]
        gG[n] = Mp.T @ ge + lam_M * c_g[n] * g[n]
        np.add.at(gL, bi, ge[:, None] * f[n][None, :])
        np.add.at(gM, pj, ge[:, None] * g[n][None, :])
    nll += 0.5 * (lam_L * np.sum(L * L) + lam_M * np.sum(M * M)
                  + lam_L * np.sum(c_f[:, None] * f * f)
                  + lam_M * np.sum(c_g[:, None] * g * g)
                  + LAM_STRUCT * np.sum(beta * beta))
    gL += lam_L * L
    gM += lam_M * M
    return nll, pack_shared(gA, gB, gF, gG, gL, gM)


def score_shared(th, data, psis, d, n_bat, n_pit):
    ps = data[0][0].shape[1]
    alpha, beta, f, g, L, M = unpack_shared(th, d, n_bat, n_pit, ps, len(data))
    tot = 0.0
    for n, (Xs, bi, pj, yv) in enumerate(data):
        eta = alpha[n] + Xs @ beta[n] + L[bi] @ f[n] + M[pj] @ g[n]
        p, omp, _, _ = ao_prob(eta, psis[n])
        tot += -2.0 * np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
    return tot


def fit_shared(x0, data, psis, d, n_bat, n_pit, lam_L, lam_M, c_f, c_g, maxiter=600):
    r = minimize(obj_shared, x0, args=(data, psis, d, n_bat, n_pit, lam_L, lam_M, c_f, c_g),
                 jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-11})
    return r.x, r.fun


# ============================================================ ARM 3 model --
# eta_n = alpha_n + Xs.beta_n + L[bi].f[n] + b[n,bi] + M[pj].g[n] + q[n,pj]
# Latent channel: lam_L on (L, f), lam_M on (M, g), as in arm 1.
# Free channel:   b[n] penalised by lam_bat0[n] (PINNED), q[n] by lam_pit0[n].

def pack_arm3(alpha, beta, f, g, L, M, b, q):
    return np.concatenate([alpha.ravel(), beta.ravel(), f.ravel(), g.ravel(),
                            L.ravel(), M.ravel(), b.ravel(), q.ravel()])


def unpack_arm3(th, d, n_bat, n_pit, ps, nn=NNODE):
    o = 0
    alpha = th[o:o + nn]; o += nn
    beta = th[o:o + nn * ps].reshape(nn, ps); o += nn * ps
    f = th[o:o + nn * d].reshape(nn, d); o += nn * d
    g = th[o:o + nn * d].reshape(nn, d); o += nn * d
    L = th[o:o + n_bat * d].reshape(n_bat, d); o += n_bat * d
    M = th[o:o + n_pit * d].reshape(n_pit, d); o += n_pit * d
    b = th[o:o + nn * n_bat].reshape(nn, n_bat); o += nn * n_bat
    q = th[o:o + nn * n_pit].reshape(nn, n_pit)
    return alpha, beta, f, g, L, M, b, q


def obj_arm3(th, data, psis, d, n_bat, n_pit, lam_L, lam_M, lam_bat, lam_pit):
    ps = data[0][0].shape[1]
    nn = len(data)
    alpha, beta, f, g, L, M, b, q = unpack_arm3(th, d, n_bat, n_pit, ps, nn)
    nll = 0.0
    gA = np.zeros(nn); gB = np.zeros((nn, ps))
    gF = np.zeros((nn, d)); gG = np.zeros((nn, d))
    gL = np.zeros((n_bat, d)); gM = np.zeros((n_pit, d))
    gb = np.zeros((nn, n_bat)); gq = np.zeros((nn, n_pit))
    for n, (Xs, bi, pj, yv) in enumerate(data):
        Lb, Mp = L[bi], M[pj]
        eta = alpha[n] + Xs @ beta[n] + Lb @ f[n] + b[n][bi] + Mp @ g[n] + q[n][pj]
        p, omp, e, u = ao_prob(eta, psis[n])
        nll += -np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
        ge = -(yv - p) * (e / u) / p
        gA[n] = ge.sum()
        gB[n] = Xs.T @ ge + LAM_STRUCT * beta[n]
        gF[n] = Lb.T @ ge + lam_L * f[n]
        gG[n] = Mp.T @ ge + lam_M * g[n]
        np.add.at(gL, bi, ge[:, None] * f[n][None, :])
        np.add.at(gM, pj, ge[:, None] * g[n][None, :])
        gb[n] = np.bincount(bi, weights=ge, minlength=n_bat) + lam_bat[n] * b[n]
        gq[n] = np.bincount(pj, weights=ge, minlength=n_pit) + lam_pit[n] * q[n]
    nll += 0.5 * (lam_L * (np.sum(L * L) + np.sum(f * f))
                  + lam_M * (np.sum(M * M) + np.sum(g * g))
                  + np.sum(lam_bat[:, None] * b * b)
                  + np.sum(lam_pit[:, None] * q * q)
                  + LAM_STRUCT * np.sum(beta * beta))
    gL += lam_L * L
    gM += lam_M * M
    return nll, pack_arm3(gA, gB, gF, gG, gL, gM, gb, gq)


def score_arm3(th, data, psis, d, n_bat, n_pit):
    ps = data[0][0].shape[1]
    alpha, beta, f, g, L, M, b, q = unpack_arm3(th, d, n_bat, n_pit, ps, len(data))
    tot = 0.0
    for n, (Xs, bi, pj, yv) in enumerate(data):
        eta = alpha[n] + Xs @ beta[n] + L[bi] @ f[n] + b[n][bi] + M[pj] @ g[n] + q[n][pj]
        p, omp, _, _ = ao_prob(eta, psis[n])
        tot += -2.0 * np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
    return tot


def fit_arm3(x0, data, psis, d, n_bat, n_pit, lam_L, lam_M, lam_bat, lam_pit, maxiter=600):
    r = minimize(obj_arm3, x0, args=(data, psis, d, n_bat, n_pit, lam_L, lam_M, lam_bat, lam_pit),
                 jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-11})
    return r.x, r.fun


# --------------------------------------------- psi re-selection (arm 1) ---
# Cheap verification of "does re-selecting psi per node change anything",
# holding the SHARED L, M fixed at their jointly-fit values and re-optimising
# only that one node's own (alpha, beta, f, g) block -- a convex sub-problem
# for fixed psi, analogous to GLLVM's block-coordinate step. Sweeping the
# full joint model over psi per node (12 candidates x 9 nodes = 108 full
# joint refits) is not attempted -- this is the tractable stand-in the task
# brief allows ("at minimum verify... changes nothing").

def _node_block_obj(theta, Xs, Lb, Mp, yv, ps, d, psi, lam_L, lam_M, c_f_n, c_g_n):
    alpha = theta[0]; beta = theta[1:1 + ps]
    f = theta[1 + ps:1 + ps + d]; g = theta[1 + ps + d:1 + ps + 2 * d]
    eta = alpha + Xs @ beta + Lb @ f + Mp @ g
    p, omp, e, u = ao_prob(eta, psi)
    nll = -np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
    nll += 0.5 * (LAM_STRUCT * beta @ beta + lam_L * c_f_n * f @ f + lam_M * c_g_n * g @ g)
    ge = -(yv - p) * (e / u) / p
    grad = np.concatenate((
        [ge.sum()],
        Xs.T @ ge + LAM_STRUCT * beta,
        Lb.T @ ge + lam_L * c_f_n * f,
        Mp.T @ ge + lam_M * c_g_n * g,
    ))
    return nll, grad


def _node_block_dev(theta, Xs, Lb, Mp, yv, ps, d, psi):
    alpha = theta[0]; beta = theta[1:1 + ps]
    f = theta[1 + ps:1 + ps + d]; g = theta[1 + ps + d:1 + ps + 2 * d]
    eta = alpha + Xs @ beta + Lb @ f + Mp @ g
    p, omp, _, _ = ao_prob(eta, psi)
    return -2.0 * np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))


def resweep_node_psi(node_idx, F, V, L, M, d, ps, lam_L, lam_M, c_f_n, c_g_n,
                      inherited_psi, warm0=None):
    """Holding shared L, M fixed, re-fit node `node_idx`'s own (alpha, beta,
    f, g) at each psi in PSI_GRID (fit on F, score on V). Returns
    (best_psi, best_val_dev_raw, inherited_val_dev_raw) -- RAW (un-normalised
    by node row count) so the caller can sum across nodes and divide by the
    single shared n_val, matching the convention step1b_fixed's score()/n_val
    and step1's tot/n_test use (total deviance is summed raw log-lik across
    nodes, divided ONCE by the shared PA count, not per-node row counts)."""
    Xf, bif, pjf, yvf = F
    Xv, biv, pjv, yvv = V
    Lbf, Mpf = L[bif], M[pjf]
    Lbv, Mpv = L[biv], M[pjv]
    nv = max(1, len(yvv))  # only used to pick the argmin psi; scale-invariant there

    def val(psi, warm):
        x0 = np.zeros(1 + ps + 2 * d) if warm is None else warm
        r = minimize(_node_block_obj, x0,
                     args=(Xf, Lbf, Mpf, yvf, ps, d, psi, lam_L, lam_M, c_f_n, c_g_n),
                     jac=True, method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-11})
        raw = _node_block_dev(r.x, Xv, Lbv, Mpv, yvv, ps, d, psi)
        return raw / nv, raw, r.x

    inherited_norm, inherited_raw, warm = val(inherited_psi, warm0)
    best = (inherited_norm, inherited_psi, inherited_raw)
    for psi in PSI_GRID:
        dv, raw, warm = val(psi, warm)
        if dv < best[0]:
            best = (dv, psi, raw)
    return best[1], best[2], inherited_raw


# ------------------------------------------------------- structural-only --

def structural_only_dev(data, psis, n_bat, n_pit, ps, n_pa):
    """Sanity baseline: alpha + Xs.beta only, no player terms at all
    (b, q forced to ~0 via an astronomically large ridge). Total RAW
    deviance summed across the 9 nodes, divided ONCE by `n_pa` (the shared
    PA count) -- NOT by the sum of per-node subset sizes, which double-counts
    rows that reach multiple nodes and silently produces a much-too-small
    number. Same convention as score_shared/n_val, tot_ao/n_test elsewhere."""
    tot = 0.0
    for k, (Xs, bi, pj, yv) in enumerate(data):
        th = fit_node(Xs, bi, pj, yv, n_bat, n_pit, psis[k], 1e10, 1e10)
        tot += node_dev(th, Xs, bi, pj, yv, n_bat, n_pit, psis[k])
    return tot / n_pa


# ------------------------------------------------------------ grad checks --

def gradient_check_shared():
    rng = np.random.RandomState(42)
    N, d, ps, n_bat, n_pit, rows = 3, 2, 3, 6, 5, 25
    data = []
    for _ in range(N):
        Xs = rng.randn(rows, ps) * 0.3
        bi = rng.randint(0, n_bat, rows)
        pj = rng.randint(0, n_pit, rows)
        yv = rng.randint(0, 2, rows).astype(float)
        data.append((Xs, bi, pj, yv))
    psis = [0.7, 1.0, 1.4]
    lam_L, lam_M = 5.0, 8.0
    c_f = rng.uniform(0.3, 3.0, N)
    c_g = rng.uniform(0.3, 3.0, N)
    n_params = sum(sizes_shared(d, n_bat, n_pit, ps, N))
    th0 = rng.randn(n_params) * 0.3
    f_ = lambda x: obj_shared(x, data, psis, d, n_bat, n_pit, lam_L, lam_M, c_f, c_g)[0]
    g_ = lambda x: obj_shared(x, data, psis, d, n_bat, n_pit, lam_L, lam_M, c_f, c_g)[1]
    return check_grad(f_, g_, th0)


def gradient_check_arm3():
    rng = np.random.RandomState(43)
    N, d, ps, n_bat, n_pit, rows = 3, 2, 3, 6, 5, 25
    data = []
    for _ in range(N):
        Xs = rng.randn(rows, ps) * 0.3
        bi = rng.randint(0, n_bat, rows)
        pj = rng.randint(0, n_pit, rows)
        yv = rng.randint(0, 2, rows).astype(float)
        data.append((Xs, bi, pj, yv))
    psis = [0.7, 1.0, 1.4]
    lam_L, lam_M = 5.0, 8.0
    lam_bat = rng.uniform(1.0, 20.0, N)
    lam_pit = rng.uniform(1.0, 20.0, N)
    n_params = N + N * ps + 2 * N * d + n_bat * d + n_pit * d + N * n_bat + N * n_pit
    th0 = rng.randn(n_params) * 0.3
    f_ = lambda x: obj_arm3(x, data, psis, d, n_bat, n_pit, lam_L, lam_M, lam_bat, lam_pit)[0]
    g_ = lambda x: obj_arm3(x, data, psis, d, n_bat, n_pit, lam_L, lam_M, lam_bat, lam_pit)[1]
    return check_grad(f_, g_, th0)


if __name__ == "__main__":
    print("this module provides shared machinery; run arm1.py / arm2.py / arm3.py / run_all.py")
