"""STEP 1b: latents shared ACROSS the nine binary nodes -- the real GLLVM here.

Step 1a gives every player nine independent node effects. That is a lot of
parameters for a 1.5k-game corpus, and it lets a player's strikeout effect and
his home-run effect be estimated with no help from each other.

This constrains them. Each batter gets ONE latent L_i in R^d used at every
node; each node n has its own loading f_n in R^d. The batter's nine-vector of
node effects is then L_i @ F.T -- forced into a d-dimensional subspace, so
nodes borrow strength from each other. Same for pitchers, independently.

    eta_n(i,j) = alpha_n + Xs beta_n + L_i . f_n + M_j . g_n

This is where a latent factorisation is NOT degenerate. Within a single binary
node it would be (a rank-d factorisation of a scalar is a scalar); across nine
nodes it is a genuine rank constraint on a 9-vector.

Initialised from step 1a's per-node ridge estimates via SVD, which is both a
good warm start and the natural rank-d projection of the unconstrained fit.
psi per node is inherited from step 1a rather than re-profiled -- noted as an
approximation, not a claim of joint optimality.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "fuse"))
import numpy as np
from scipy.optimize import minimize
import common
from analyze import structural
from step1 import NODES, ao_prob, fit as fit_node, EPS, LAM_STRUCT

D_GRID = [1, 2, 3]
LAM_LAT = [1.0, 3.0, 10.0, 30.0, 100.0]
LAM_LOAD = 1e-2
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def unpack(th, d, n_bat, n_pit, N, ps):
    o = 0
    alpha = th[o:o + N]; o += N
    beta = th[o:o + N * ps].reshape(N, ps); o += N * ps
    f = th[o:o + N * d].reshape(N, d); o += N * d
    g = th[o:o + N * d].reshape(N, d); o += N * d
    L = th[o:o + n_bat * d].reshape(n_bat, d); o += n_bat * d
    M = th[o:o + n_pit * d].reshape(n_pit, d)
    return alpha, beta, f, g, L, M


def obj(th, data, psis, d, n_bat, n_pit, lam_L, lam_M):
    N, ps = len(data), data[0][0].shape[1]
    alpha, beta, f, g, L, M = unpack(th, d, n_bat, n_pit, N, ps)
    nll = 0.0
    gA = np.zeros(N); gB = np.zeros((N, ps))
    gF = np.zeros((N, d)); gG = np.zeros((N, d))
    gL = np.zeros((n_bat, d)); gM = np.zeros((n_pit, d))
    for n, (Xs, bi, pj, yv) in enumerate(data):
        Lb, Mp = L[bi], M[pj]
        eta = alpha[n] + Xs @ beta[n] + Lb @ f[n] + Mp @ g[n]
        p, omp, e, u = ao_prob(eta, psis[n])
        nll += -np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
        ge = -(yv - p) * (e / u) / p
        gA[n] = ge.sum()
        gB[n] = Xs.T @ ge + LAM_STRUCT * beta[n]
        gF[n] = Lb.T @ ge + LAM_LOAD * f[n]
        gG[n] = Mp.T @ ge + LAM_LOAD * g[n]
        np.add.at(gL, bi, ge[:, None] * f[n][None, :])
        np.add.at(gM, pj, ge[:, None] * g[n][None, :])
    nll += 0.5 * (lam_L * np.sum(L * L) + lam_M * np.sum(M * M)
                  + LAM_LOAD * (np.sum(f * f) + np.sum(g * g))
                  + LAM_STRUCT * np.sum(beta * beta))
    gL += lam_L * L
    gM += lam_M * M
    return nll, np.concatenate([gA, gB.ravel(), gF.ravel(), gG.ravel(),
                                gL.ravel(), gM.ravel()])


def score(th, data, psis, d, n_bat, n_pit):
    N, ps = len(data), data[0][0].shape[1]
    alpha, beta, f, g, L, M = unpack(th, d, n_bat, n_pit, N, ps)
    tot = 0.0
    for n, (Xs, bi, pj, yv) in enumerate(data):
        eta = alpha[n] + Xs @ beta[n] + L[bi] @ f[n] + M[pj] @ g[n]
        p, omp, _, _ = ao_prob(eta, psis[n])
        tot += -2.0 * np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
    return tot


def main():
    prev = json.load(open(os.path.join(os.path.dirname(__file__), "step1_result.json")))
    psis = [n["psi"] for n in prev["nodes"]]
    lam_b = [n["lam_bat"] for n in prev["nodes"]]
    lam_p = [n["lam_pit"] for n in prev["nodes"]]
    log(f"inheriting psi={psis}")

    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}
    g_ = sorted(train_g)
    rs = np.random.RandomState(90210)
    rs.shuffle(g_)
    ifit = set(g_[: int(0.8 * len(g_))])

    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)
    N, ps = len(NODES), 5

    def pack(rs_, reach, pos):
        sub = [r for r in rs_ if r["y"] in reach]
        return (structural(sub, season_idx),
                np.fromiter((BI[r["batter"]] for r in sub), int, len(sub)),
                np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub)),
                np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub)))

    itr_rows = [r for r in tr if r["game_id"] in ifit]
    ival_rows = [r for r in tr if r["game_id"] not in ifit]
    D_fit = [pack(itr_rows, rc, po) for _, rc, po in NODES]
    D_val = [pack(ival_rows, rc, po) for _, rc, po in NODES]
    D_tr = [pack(tr, rc, po) for _, rc, po in NODES]
    D_te = [pack(te, rc, po) for _, rc, po in NODES]
    n_val, n_test = len(ival_rows), len(te)
    log(f"train {len(tr)} test {len(te)}  nodes {N}  batters {n_bat} pitchers {n_pit}")

    def init(data, d):
        """Rank-d SVD of the unconstrained per-node ridge estimates."""
        B = np.zeros((n_bat, N)); Q = np.zeros((n_pit, N))
        al = np.zeros(N); be = np.zeros((N, ps))
        for n, (Xs, bi, pj, yv) in enumerate(data):
            th = fit_node(Xs, bi, pj, yv, n_bat, n_pit, psis[n], lam_b[n], lam_p[n])
            al[n] = th[0]; be[n] = th[1:1 + ps]
            B[:, n] = th[1 + ps:1 + ps + n_bat]
            Q[:, n] = th[1 + ps + n_bat:]
        Ub, Sb, Vbt = np.linalg.svd(B, full_matrices=False)
        Uq, Sq, Vqt = np.linalg.svd(Q, full_matrices=False)
        L = Ub[:, :d] * np.sqrt(Sb[:d]); f = (Vbt[:d].T * np.sqrt(Sb[:d]))
        M = Uq[:, :d] * np.sqrt(Sq[:d]); g = (Vqt[:d].T * np.sqrt(Sq[:d]))
        return np.concatenate([al, be.ravel(), f.ravel(), g.ravel(),
                               L.ravel(), M.ravel()])

    best = None
    for d in D_GRID:
        x0 = init(D_fit, d)
        for lam in LAM_LAT:
            t0 = time.time()
            r = minimize(obj, x0, args=(D_fit, psis, d, n_bat, n_pit, lam, lam),
                         jac=True, method="L-BFGS-B",
                         options={"maxiter": 600, "ftol": 1e-11})
            v = score(r.x, D_val, psis, d, n_bat, n_pit) / n_val
            log(f"  d={d} lam={lam:<6g} val={v:.5f}  ({time.time()-t0:.1f}s)")
            if best is None or v < best[0]:
                best = (v, d, lam)
    _, d_star, lam_star = best
    log(f"selected d*={d_star} lambda*={lam_star:g}")

    x0 = init(D_tr, d_star)
    r = minimize(obj, x0, args=(D_tr, psis, d_star, n_bat, n_pit, lam_star, lam_star),
                 jac=True, method="L-BFGS-B", options={"maxiter": 900, "ftol": 1e-11})
    d_te = score(r.x, D_te, psis, d_star, n_bat, n_pit) / n_test
    log("")
    log(f"STEP 1b  shared-latent (d={d_star}) frozen test = {d_te:.5f}")
    log(f"  step 1a per-node x per-side + psi     = {prev['ao_side']:.5f}")
    log(f"  nested_sep                            = 3.94846")
    log(f"  nested_sep + OOF calibration          = 3.94603")

    np.savez(os.path.join(os.path.dirname(__file__), "step1b_latent.npz"),
             theta=r.x, d=d_star, lam=lam_star, psis=np.array(psis),
             bat_ids=np.array(bats), pit_ids=np.array(pits))
    json.dump({"d_star": d_star, "lam_star": lam_star, "test": d_te,
               "step1a": prev["ao_side"]},
              open(os.path.join(os.path.dirname(__file__), "step1b_result.json"), "w"),
              indent=1)
    log("wrote step1b_result.json")


if __name__ == "__main__":
    main()
