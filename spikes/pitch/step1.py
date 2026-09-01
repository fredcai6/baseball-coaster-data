"""STEP 1a + 2: binarised tree, per-node x per-SIDE regularisation, fitted link.

Why per-side, and why per-node
------------------------------
The out-of-fold calibration in crossfit.py asked for wildly different
shrinkage by category -- doubles 0.60x, triples 0.65x, singles 0.78x, home
runs 1.16x. A single lambda per gate, shared between batters and pitchers,
cannot deliver that. Binarising makes the fix structural rather than post-hoc:
each node IS a category boundary, so a per-node lambda IS a per-category
lambda, and splitting it per side lets batters and pitchers shrink differently
(they have very different observation counts -- 772 batters vs 1220 pitchers).

NOTE ON LATENTS. There are deliberately none here. A binary node's player
contribution is a scalar, so a rank-d factorisation L_i . F collapses to one
number per player and buys nothing over a per-player effect. Latent structure
only pays ACROSS nodes, which is step1b.

Selection is STAGED, not joint: (lam_bat, lam_pit) at psi=1, then psi, then
(lam_bat, lam_pit) again at psi_hat. The full joint grid is 9 nodes x 5 x 5 x
12 and is not worth the compute. This is suboptimal and is reported as such.
Every link gets its OWN lambda -- sharing one across links is the bug that
manufactured a phantom -0.00847 in the first pass at this.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "fuse"))
import numpy as np
from scipy.optimize import minimize
import common
from analyze import structural

CI = common.CAT_INDEX
S = lambda *cs: frozenset(CI[c] for c in cs)
ALL = S(*common.CATEGORIES)

NODES = [
    ("root",    ALL,                                   S("K", "BB", "HBP")),
    ("tto_K",   S("K", "BB", "HBP"),                   S("K")),
    ("tto_BB",  S("BB", "HBP"),                        S("BB")),
    ("con_OTH", ALL - S("K", "BB", "HBP"),             S("OTHER")),
    ("con_OUT", ALL - S("K", "BB", "HBP", "OTHER"),    S("F", "G")),
    ("out_F",   S("F", "G"),                           S("F")),
    ("hit_1B",  S("1B", "2B", "3B", "HR"),             S("1B")),
    ("hit_2B",  S("2B", "3B", "HR"),                   S("2B")),
    ("hit_3B",  S("3B", "HR"),                         S("3B")),
]

LAM_GRID = [3.0, 10.0, 30.0, 100.0, 300.0]
PSI_GRID = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 10.0]
LAM_STRUCT = 1e-3
EPS = 1e-12
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def ao_prob(eta, psi):
    e = np.exp(np.clip(eta, -30, 30))
    u = 1.0 + psi * e
    log1mp = -np.log(u) / psi
    p = np.clip(-np.expm1(log1mp), EPS, 1 - EPS)
    return p, np.exp(log1mp), e, u


def nll_grad(th, Xs, bi, pj, yv, n_bat, n_pit, psi, lam_b, lam_p):
    ps = Xs.shape[1]
    alpha, beta = th[0], th[1:1 + ps]
    b = th[1 + ps:1 + ps + n_bat]
    q = th[1 + ps + n_bat:]
    eta = alpha + Xs @ beta + b[bi] + q[pj]
    p, omp, e, u = ao_prob(eta, psi)
    nll = -np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))
    nll += 0.5 * (lam_b * b @ b + lam_p * q @ q + LAM_STRUCT * beta @ beta)
    g_eta = -(yv - p) * (e / u) / p
    grad = np.concatenate((
        [g_eta.sum()],
        Xs.T @ g_eta + LAM_STRUCT * beta,
        np.bincount(bi, weights=g_eta, minlength=n_bat) + lam_b * b,
        np.bincount(pj, weights=g_eta, minlength=n_pit) + lam_p * q,
    ))
    return nll, grad


def fit(Xs, bi, pj, yv, n_bat, n_pit, psi, lam_b, lam_p, x0=None):
    n = 1 + Xs.shape[1] + n_bat + n_pit
    x0 = np.zeros(n) if x0 is None else x0
    r = minimize(nll_grad, x0, args=(Xs, bi, pj, yv, n_bat, n_pit, psi, lam_b, lam_p),
                 jac=True, method="L-BFGS-B",
                 options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-8})
    return r.x


def node_dev(th, Xs, bi, pj, yv, n_bat, n_pit, psi):
    ps = Xs.shape[1]
    eta = th[0] + Xs @ th[1:1 + ps] + th[1 + ps:1 + ps + n_bat][bi] \
        + th[1 + ps + n_bat:][pj]
    p, omp, _, _ = ao_prob(eta, psi)
    return -2.0 * np.sum(yv * np.log(p) + (1 - yv) * np.log(np.clip(omp, EPS, 1.0)))


def main():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}

    g = sorted(train_g)
    rs = np.random.RandomState(90210)
    rs.shuffle(g)
    ifit = set(g[: int(0.8 * len(g))])

    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)
    log(f"train {len(tr)} test {len(te)}  batters {n_bat} pitchers {n_pit}")

    def pack(rs_, reach, pos):
        sub = [r for r in rs_ if r["y"] in reach]
        Xs = structural(sub, season_idx)
        bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
        pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
        yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
        return Xs, bi, pj, yv

    n_test = len(te)
    tot = {"logit_shared": 0.0, "logit_side": 0.0, "ao_side": 0.0}
    results = []

    for name, reach, pos in NODES:
        F = pack([r for r in tr if r["game_id"] in ifit], reach, pos)
        V = pack([r for r in tr if r["game_id"] not in ifit], reach, pos)
        TR = pack(tr, reach, pos)
        TE = pack(te, reach, pos)
        nv = max(1, len(V[3]))

        def val(psi, lb, lp, warm=None):
            th = fit(*F, n_bat, n_pit, psi, lb, lp, warm)
            return node_dev(th, *V, n_bat, n_pit, psi) / nv, th

        # stage 0: shared lambda at logit (the link.py model, for attribution)
        best_sh = min(((val(1.0, l, l)[0], l) for l in LAM_GRID))
        # stage 1: per-side lambda at logit
        best_ls = None
        for lb in LAM_GRID:
            warm = None
            for lp in LAM_GRID:
                d, warm = val(1.0, lb, lp, warm)
                if best_ls is None or d < best_ls[0]:
                    best_ls = (d, lb, lp)
        # stage 2: psi at the stage-1 lambdas
        _, lb1, lp1 = best_ls
        best_psi = None
        warm = None
        for psi in PSI_GRID:
            d, warm = val(psi, lb1, lp1, warm)
            if best_psi is None or d < best_psi[0]:
                best_psi = (d, psi)
        psi_hat = best_psi[1]
        # stage 3: re-select lambda at psi_hat
        best_ao = None
        for lb in LAM_GRID:
            warm = None
            for lp in LAM_GRID:
                d, warm = val(psi_hat, lb, lp, warm)
                if best_ao is None or d < best_ao[0]:
                    best_ao = (d, lb, lp)
        _, lb2, lp2 = best_ao

        d_sh = node_dev(fit(*TR, n_bat, n_pit, 1.0, best_sh[1], best_sh[1]),
                        *TE, n_bat, n_pit, 1.0)
        d_ls = node_dev(fit(*TR, n_bat, n_pit, 1.0, lb1, lp1), *TE, n_bat, n_pit, 1.0)
        d_ao = node_dev(fit(*TR, n_bat, n_pit, psi_hat, lb2, lp2),
                        *TE, n_bat, n_pit, psi_hat)
        tot["logit_shared"] += d_sh
        tot["logit_side"] += d_ls
        tot["ao_side"] += d_ao

        results.append(dict(node=name, n=int(len(TR[3])), rate=float(TR[3].mean()),
                            lam_shared=best_sh[1], lam_bat=lb2, lam_pit=lp2,
                            psi=psi_hat, dev_shared=d_sh / n_test,
                            dev_side=d_ls / n_test, dev_ao=d_ao / n_test))
        log(f"{name:9} n={len(TR[3]):>6} rate={TR[3].mean():.4f}  "
            f"lam_shared={best_sh[1]:<6g} lam_bat={lb2:<6g} lam_pit={lp2:<6g} "
            f"psi={psi_hat:<5g} | shared={d_sh/n_test:.5f} side={d_ls/n_test:.5f} "
            f"ao={d_ao/n_test:.5f}")

    log("")
    log(f"{'per-node lambda, logit (= link.py)':42} {tot['logit_shared']/n_test:.5f}")
    log(f"{'per-node x per-SIDE lambda, logit':42} {tot['logit_side']/n_test:.5f} "
        f"({(tot['logit_side']-tot['logit_shared'])/n_test:+.5f})")
    log(f"{'per-node x per-side lambda + fitted psi':42} {tot['ao_side']/n_test:.5f} "
        f"({(tot['ao_side']-tot['logit_side'])/n_test:+.5f})")
    log("")
    log("reference (frozen test): NULL 4.01172 | flat ridge 3.95550 | NPMR 3.95424")
    log("                         nested_sep 3.94846 | nested_sep+OOF cal 3.94603")

    out = {"nodes": results, **{k: v / n_test for k, v in tot.items()}}
    with open(os.path.join(os.path.dirname(__file__), "step1_result.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    log("wrote step1_result.json")


if __name__ == "__main__":
    main()
