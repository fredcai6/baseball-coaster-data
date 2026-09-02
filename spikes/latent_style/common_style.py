"""Shared machinery for the latent_style spike (split-half reliability +
external validity of latent2's arm-1 player latent).

Imports latent2/latent_fit.py and pitch/step1.py UNCHANGED (per the ground
rules: do not edit spikes/common.py, spikes/latent2/*, or spikes/pitch/*).
This file only adds new code in spikes/latent_style/.

Canonical fit discipline: arm1's own restart spread at (d=3, lam_L=lam_M=40)
was 8.35e-7 on the frozen test (spikes/latent2/result.json, "restart_spread"),
i.e. essentially zero -- multiple random restarts all land on the same
optimum. We therefore use N_RESTARTS_STYLE=3 (1 clean SVD-init + 2 jittered)
rather than repeating latent2's full 5, to keep each stage inside the 10
minute foreground budget; this is documented here rather than silently cut.
"""
import sys, os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKES = os.path.dirname(HERE)
for p in (os.path.join(SPIKES, "latent2"), SPIKES,
          os.path.join(SPIKES, "fuse"), os.path.join(SPIKES, "pitch")):
    if p not in sys.path:
        sys.path.insert(0, p)

import common                    # spikes/common.py -- DO NOT EDIT, only imported
from analyze import structural   # spikes/fuse/analyze.py
import latent_fit as LF          # spikes/latent2/latent_fit.py -- reuse by import

NODES = LF.NODES
NODE_NAMES = LF.NODE_NAMES
NNODE = LF.NNODE
D_STAR = 3
LAM_STAR = 40.0
N_RESTARTS_STYLE = 3

RESULT_PATH = os.path.join(HERE, "result.json")
RUNLOG_PATH = os.path.join(HERE, "run.log")
T0 = time.time()


def log(m):
    line = f"[{time.time()-T0:7.1f}s] {m}"
    print(line, flush=True)
    with open(RUNLOG_PATH, "a") as fh:
        fh.write(line + "\n")


def load_result():
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as fh:
            return json.load(fh)
    return {}


def save_result(out):
    with open(RESULT_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    log(f"wrote {RESULT_PATH}")


# --------------------------------------------------------------- universe --
# Player universe (bats/pits, BI/PI) and season_idx are fixed over ALL rows,
# same convention as latent_fit.setup() -- so a player missing from a half
# just gets an index that never appears in that half's data (shrinks to the
# prior/0), rather than changing the parameter vector's meaning.

def base_universe():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}
    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    s6 = json.load(open(os.path.join(SPIKES, "pitch", "step6_result.json")))
    s6_nodes = {n["node"]: n for n in s6["D"]["nodes"]}
    psi0 = np.array([s6_nodes[nm]["psi"] for nm in NODE_NAMES])
    lam_bat0 = np.array([s6_nodes[nm]["lam_bat"] for nm in NODE_NAMES])
    lam_pit0 = np.array([s6_nodes[nm]["lam_pit"] for nm in NODE_NAMES])
    return dict(rows=rows, train_g=train_g, test_g=test_g, season_idx=season_idx,
                bats=bats, pits=pits, BI=BI, PI=PI, n_bat=len(bats), n_pit=len(pits),
                psi0=psi0, lam_bat0=lam_bat0, lam_pit0=lam_pit0)


def pack(rows_sub, reach, pos, BI, PI, season_idx):
    sub = [r for r in rows_sub if r["y"] in reach]
    Xs = structural(sub, season_idx)
    bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
    pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
    yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
    return Xs, bi, pj, yv


def build_node_data(rows_sub, BI, PI, season_idx):
    return [pack(rows_sub, reach, pos, BI, PI, season_idx) for _, reach, pos in NODES]


def pa_counts(rows_sub, BI, PI):
    """Per-player training PA counts (every row touches root, so this is
    just a tally over the raw rows, not per-node)."""
    bc = np.zeros(len(BI)); pc = np.zeros(len(PI))
    for r in rows_sub:
        bc[BI[r["batter"]]] += 1
        pc[PI[r["pitcher"]]] += 1
    return bc, pc


# --------------------------------------------------- canonical arm1 fit ---

def fit_canonical_arm1(D_data, psi0, lam_bat0, lam_pit0, n_bat, n_pit, ps,
                        d=D_STAR, lam=LAM_STAR, n_restarts=N_RESTARTS_STYLE,
                        maxiter=900, tag="fit"):
    """Fit arm1 (shared latent, per-side lambda) at FIXED (d, lam_L=lam_M=lam)
    with inherited psi -- no hyperparameter search. Warm-starts from the SVD
    of shape D's own per-node free-effect fit (node_free_fits + svd_init,
    exactly latent2's convention), then N_RESTARTS_STYLE restarts of
    fit_shared, canonical = min penalized train loss (matches
    latent2/run_all.restart_final's rule).

    Returns dict with unpacked alpha,beta,f,g,L,M (canonical) AND the
    free-effect matrices B,Q,alpha_free,beta_free from node_free_fits --
    the same per-node free-effect refit Test 2(b) needs, computed once here
    and reused rather than refit twice.
    """
    c1 = np.ones(NNODE)
    t0 = time.time()
    alpha0, beta0, B, Q = LF.node_free_fits(D_data, psi0, lam_bat0, lam_pit0,
                                             n_bat, n_pit, ps)
    log(f"  [{tag}] node_free_fits done ({time.time()-t0:.1f}s)")
    Lm0, f0, M0, g0 = LF.svd_init(B, Q, d)
    x0_base = LF.pack_shared(alpha0, beta0, f0, g0, Lm0, M0)

    def jitter(x0, seed):
        if seed == 0:
            return x0.copy()
        rng = np.random.RandomState(9000 + seed)
        alpha, beta, f, g, Lm, M = LF.unpack_shared(x0, d, n_bat, n_pit, ps)
        f2 = f + rng.normal(scale=0.3 * (np.std(f) + 1e-6), size=f.shape)
        g2 = g + rng.normal(scale=0.3 * (np.std(g) + 1e-6), size=g.shape)
        L2 = Lm + rng.normal(scale=0.3 * (np.std(Lm) + 1e-6), size=Lm.shape)
        M2 = M + rng.normal(scale=0.3 * (np.std(M) + 1e-6), size=M.shape)
        return LF.pack_shared(alpha, beta, f2, g2, L2, M2)

    restarts = []
    for seed in range(n_restarts):
        t0 = time.time()
        x0 = jitter(x0_base, seed)
        th, train_loss = LF.fit_shared(x0, D_data, psi0, d, n_bat, n_pit,
                                        lam, lam, c1, c1, maxiter=maxiter)
        restarts.append(dict(seed=seed, theta=th, train_loss=float(train_loss)))
        log(f"  [{tag}] restart {seed}: train_loss={train_loss:.2f} ({time.time()-t0:.1f}s)")
    canonical = min(restarts, key=lambda r: r["train_loss"])
    spread = max(r["train_loss"] for r in restarts) - min(r["train_loss"] for r in restarts)
    log(f"  [{tag}] canonical seed={canonical['seed']}  train_loss spread={spread:.4f}")

    alpha, beta, f, g, Lm, M = LF.unpack_shared(canonical["theta"], d, n_bat, n_pit, ps)
    return dict(alpha=alpha, beta=beta, f=f, g=g, L=Lm, M=M,
                alpha_free=alpha0, beta_free=beta0, B_free=B, Q_free=Q,
                canonical_seed=int(canonical["seed"]), train_loss_spread=float(spread),
                d=d, lam=lam)


# ------------------------------------------------------------- Procrustes --

def orthogonal_procrustes_align(A, B):
    """Align B onto A by rotation/reflection only (no scaling -- see
    LATENT_STYLE.md: per-axis Pearson r is scale-invariant, so the optimal
    rotation from SVD(B^T A) is identical whether or not a scale factor is
    also fit; only the rotation resolves the axis-mixing ambiguity that
    matters for correlation). Both A, B pre-centred by the caller.
    Returns (R, B_aligned) with B_aligned = B @ R.
    """
    Mmat = B.T @ A
    U, S, Vt = np.linalg.svd(Mmat)
    R = U @ Vt
    return R, B @ R


def spearman_brown(r):
    """Full-sample reliability from a half-sample correlation."""
    return 2 * r / (1 + r)


# --------------------------------------------------------- misc utilities --

def pearson(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
