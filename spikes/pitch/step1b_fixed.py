"""STEP 1b FIXED: equal ridge penalty on (L, f) and (M, g) removes the scale
degeneracy that broke step1b.py.

Diagnosis (verified before writing this file): step1b.py's objective is
invariant under L -> L/c, f -> c*f (same for M, g), yet it penalised L and M
with a swept lambda while pinning the loading penalty LAM_LOAD = 1e-2. Its own
call site already used ONE lambda for both L and M --
    minimize(obj, x0, args=(..., lam, lam), ...)
-- so the only asymmetry was lambda-on-latents vs a fixed 1e-2 on loadings.
Minimising out the free rescaling shows the effective penalty on the batter
9-vector L_i @ F.T is sqrt(lam * 1e-2) * ||L|| * ||f||, so sweeping
lam in [1, 3, 10, 30, 100] only ever produced effective strengths in
[0.1, 1.0] -- a narrow, under-regularised band -- while leaving the optimiser
free to shrink L and inflate f (or vice versa) at ~zero cost, which is exactly
the ill-conditioning signature reported (d=2 fits converging FASTER than d=1,
13s vs 40s -- a stall, not efficiency). The result: inner-validation deviance
4.01-4.07, worse than the intercept-only null (4.01172), which a model with
free per-player terms cannot legitimately do.

Fix: penalise L, f, M, g all with the SAME lambda. Since
    min_{L,f: L f.T = B} ||L||^2 + ||f||^2  =  2 * nuclear_norm(B)
(the standard variational form of the nuclear norm), equal penalties turn this
into a well-posed low-rank regulariser on the 9-vector of node effects, with no
free rescaling direction left to exploit. Same argument for (M, g).

This file does not modify step1b.py or step1.py. It reuses NODES, ao_prob,
fit_node (for SVD warm-start), EPS, LAM_STRUCT from step1.py unchanged.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "fuse"))
import numpy as np
from scipy.optimize import minimize, check_grad
import common
from analyze import structural
from step1 import NODES, ao_prob, fit as fit_node, EPS, LAM_STRUCT

NULL_DEV = 4.01172
BEAT_TARGET = 3.94729  # step 1a: nine independent per-node effects

D_GRID = [1, 2, 3, 4]
# Wide enough that the pre-flight exploration (see step1b_fixed_result.json /
# task notes) found the minimum strictly inside [15, 280] for every d tested:
# d=1 min~lam60-100 (val 3.988), d=2 min~lam30-60 (3.973), d=3 min~lam30-60
# (3.965), d=4 min~lam60 (3.963). d=5 continued a marginal improvement
# (3.962) at ~5-10x the per-fit cost, so the grid stops at d=4.
LAM_GRID = [15, 25, 40, 60, 90, 130, 190, 280]
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


def obj(th, data, psis, d, n_bat, n_pit, lam):
    """Same as step1b.obj, EXCEPT f and g are penalised by `lam` too (was a
    pinned LAM_LOAD=1e-2, decoupled from the swept lambda on L/M -- the bug)."""
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
        gF[n] = Lb.T @ ge + lam * f[n]
        gG[n] = Mp.T @ ge + lam * g[n]
        np.add.at(gL, bi, ge[:, None] * f[n][None, :])
        np.add.at(gM, pj, ge[:, None] * g[n][None, :])
    nll += 0.5 * (lam * (np.sum(L * L) + np.sum(M * M)
                         + np.sum(f * f) + np.sum(g * g))
                  + LAM_STRUCT * np.sum(beta * beta))
    gL += lam * L
    gM += lam * M
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


def gradient_self_check():
    """Analytic vs numerical gradient of `obj` on a small synthetic problem,
    independent of the real data. Must be tiny before any real fit is trusted."""
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
    lam = 5.0
    n_params = N + N * ps + 2 * N * d + n_bat * d + n_pit * d
    th0 = rng.randn(n_params) * 0.3

    def f_(x):
        return obj(x, data, psis, d, n_bat, n_pit, lam)[0]

    def g_(x):
        return obj(x, data, psis, d, n_bat, n_pit, lam)[1]

    err = check_grad(f_, g_, th0)
    return err


def main():
    err = gradient_self_check()
    log(f"gradient check (analytic vs numerical, synthetic problem): error = {err:.3e}")
    if err > 1e-4:
        log("GRADIENT CHECK FAILED -- refusing to trust any fit from this objective.")
        json.dump({"status": "gradient_check_failed", "grad_error": float(err)},
                   open(os.path.join(os.path.dirname(__file__),
                                      "step1b_fixed_result.json"), "w"), indent=1)
        return

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
    sweep_log = []
    for d in D_GRID:
        x0 = init(D_fit, d)
        for lam in LAM_GRID:
            t0 = time.time()
            r = minimize(obj, x0, args=(D_fit, psis, d, n_bat, n_pit, lam),
                         jac=True, method="L-BFGS-B",
                         options={"maxiter": 600, "ftol": 1e-11})
            v = float(score(r.x, D_val, psis, d, n_bat, n_pit) / n_val)
            log(f"  d={d} lam={lam:<7g} val={v:.5f}  ({time.time()-t0:.1f}s)")
            sweep_log.append({"d": d, "lam": lam, "val": v})
            if best is None or v < best[0]:
                best = (v, d, lam)
    v_star, d_star, lam_star = best
    edge = bool(lam_star == LAM_GRID[0] or lam_star == LAM_GRID[-1])
    log(f"selected d*={d_star} lambda*={lam_star:g}  inner-val={v_star:.5f}  "
        f"grid_edge={edge}")

    if edge:
        log("LAMBDA SELECTED AT GRID EDGE -- widen LAM_GRID and re-run "
            "before trusting this result.")

    # MANDATORY GUARD: a model with free per-player terms cannot legitimately
    # lose to the intercept-only null. If it does, the fit is still broken.
    guard_pass = bool(v_star < NULL_DEV)
    log(f"NULL GUARD: inner-val {v_star:.5f} vs null {NULL_DEV}  "
        f"{'PASS' if guard_pass else 'FAIL'}")

    if not guard_pass:
        out = {"status": "guard_failed", "null": NULL_DEV,
               "inner_val_best": v_star, "d_star": int(d_star), "lam_star": float(lam_star),
               "grid_edge": edge, "grad_error": float(err), "sweep": sweep_log}
        json.dump(out, open(os.path.join(os.path.dirname(__file__),
                   "step1b_fixed_result.json"), "w"), indent=1)
        log("GUARD FAILED -- stopping. Not reporting a frozen-test number.")
        return

    x0 = init(D_tr, d_star)
    r = minimize(obj, x0, args=(D_tr, psis, d_star, n_bat, n_pit, lam_star),
                 jac=True, method="L-BFGS-B", options={"maxiter": 900, "ftol": 1e-11})
    d_te = float(score(r.x, D_te, psis, d_star, n_bat, n_pit) / n_test)
    log("")
    log(f"STEP 1b FIXED  shared-latent (d={d_star}, lam={lam_star:g}) "
        f"frozen test = {d_te:.5f}")
    log(f"  null                                   = {NULL_DEV:.5f}")
    log(f"  step 1a per-node x per-side + psi      = {BEAT_TARGET:.5f}")
    log(f"  nested_sep                             = 3.94846")
    log(f"  nested_sep + OOF calibration           = 3.94600")

    np.savez(os.path.join(os.path.dirname(__file__), "step1b_fixed_latent.npz"),
             theta=r.x, d=d_star, lam=lam_star, psis=np.array(psis),
             bat_ids=np.array(bats), pit_ids=np.array(pits))
    out = {"status": "ok", "grad_error": float(err), "null": NULL_DEV,
           "guard_pass": guard_pass, "d_star": int(d_star), "lam_star": float(lam_star),
           "grid_edge": edge, "inner_val": v_star, "test": d_te,
           "step1a": BEAT_TARGET, "beats_step1a": bool(d_te < BEAT_TARGET),
           "sweep": sweep_log}
    json.dump(out, open(os.path.join(os.path.dirname(__file__),
               "step1b_fixed_result.json"), "w"), indent=1)
    log("wrote step1b_fixed_result.json")


if __name__ == "__main__":
    main()
