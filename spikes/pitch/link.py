"""Is the logit link right? Fit the link's SHAPE instead of assuming it.

Design
------
The gate tree is re-expressed as NINE binary splits. That is still an exact
factorisation of the joint -- the chain rule does not care how the tree is
drawn -- but every node is now binary, which is what a binary link family
needs. Coverage check below asserts all ten categories are reached exactly.

    root     all            -> {K,BB,HBP}   vs contact
    tto_K    {K,BB,HBP}     -> {K}
    tto_BB   {BB,HBP}       -> {BB}
    con_OTH  contact        -> {OTHER}
    con_OUT  non-OTHER      -> {F,G}
    out_F    {F,G}          -> {F}
    hit_1B   {1B,2B,3B,HR}  -> {1B}
    hit_2B   {2B,3B,HR}     -> {2B}
    hit_3B   {3B,HR}        -> {3B}

At each node, an additive ridge-penalised model
    eta = alpha + b[batter] + p[pitcher]
is fitted under the Aranda-Ordaz link family

    P(y=1) = 1 - (1 + psi*exp(eta))^(-1/psi)

    psi = 1  ->  logit          (symmetric; what every model here has assumed)
    psi -> 0 ->  cloglog        (asymmetric; the "at least one event" link)

psi is PROFILED over a grid, and the coefficients are REFITTED at every psi.
That matters: link shape and coefficient scale are partially confounded, so
holding coefficients fixed while bending the link would understate the gain
and can flip its sign. Selection is on an inner validation split; the reported
number is on the frozen test games, which no fit ever sees.
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import minimize
import common

CI = common.CAT_INDEX
S = lambda *cs: frozenset(CI[c] for c in cs)
ALL = S(*common.CATEGORIES)

NODES = [
    ("root",    ALL,                          S("K", "BB", "HBP")),
    ("tto_K",   S("K", "BB", "HBP"),          S("K")),
    ("tto_BB",  S("BB", "HBP"),               S("BB")),
    ("con_OTH", ALL - S("K", "BB", "HBP"),    S("OTHER")),
    ("con_OUT", ALL - S("K", "BB", "HBP", "OTHER"), S("F", "G")),
    ("out_F",   S("F", "G"),                  S("F")),
    ("hit_1B",  S("1B", "2B", "3B", "HR"),    S("1B")),
    ("hit_2B",  S("2B", "3B", "HR"),          S("2B")),
    ("hit_3B",  S("3B", "HR"),                S("3B")),
]

PSI_GRID = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0,
            3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0]
LAM_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
EPS = 1e-12


def coverage_check():
    """Every category must be produced by exactly one root-to-leaf path."""
    paths = {}
    for cat in common.CATEGORIES:
        c = CI[cat]
        path = []
        for name, reach, pos in NODES:
            # No early exit: taking a node's positive branch can CONTINUE to a
            # child (con_OUT -> out_F), so `reach` alone defines the path.
            if c in reach:
                path.append((name, c in pos))
        paths[cat] = path
    seen = [c for c in common.CATEGORIES if paths[c]]
    assert len(seen) == len(common.CATEGORIES), "a category reaches no node"
    sigs = {tuple(v) for v in paths.values()}
    assert len(sigs) == len(common.CATEGORIES), "two categories share a path"
    return paths


def ao_prob(eta, psi):
    """Aranda-Ordaz inverse link, computed in log space for stability."""
    e = np.exp(np.clip(eta, -30, 30))
    u = 1.0 + psi * e
    log1mp = -np.log(u) / psi
    p = -np.expm1(log1mp)          # 1 - exp(log1mp), accurate as p -> 0
    return np.clip(p, EPS, 1 - EPS), np.exp(log1mp), e, u


def nll_grad(theta, bi, pj, y, n_bat, n_pit, psi, lam):
    alpha = theta[0]
    b = theta[1:1 + n_bat]
    q = theta[1 + n_bat:]
    eta = alpha + b[bi] + q[pj]
    p, omp, e, u = ao_prob(eta, psi)

    nll = -np.sum(y * np.log(p) + (1 - y) * np.log(np.clip(omp, EPS, 1.0)))
    nll += 0.5 * lam * (b @ b + q @ q)

    # d(-loglik)/d(eta) = -(y - p) * rho / p,  rho = e/u = p'(eta)/(1-p)
    g_eta = -(y - p) * (e / u) / p
    ga = g_eta.sum()
    gb = np.bincount(bi, weights=g_eta, minlength=n_bat) + lam * b
    gq = np.bincount(pj, weights=g_eta, minlength=n_pit) + lam * q
    return nll, np.concatenate(([ga], gb, gq))


def fit_node(bi, pj, y, n_bat, n_pit, psi, lam, x0=None):
    n = 1 + n_bat + n_pit
    x0 = np.zeros(n) if x0 is None else x0
    res = minimize(nll_grad, x0, args=(bi, pj, y, n_bat, n_pit, psi, lam),
                   jac=True, method="L-BFGS-B",
                   options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-8})
    return res.x


def node_dev(theta, bi, pj, y, n_bat, n_pit, psi):
    alpha = theta[0]
    b = theta[1:1 + n_bat]
    q = theta[1 + n_bat:]
    eta = alpha + b[bi] + q[pj]
    p, omp, _, _ = ao_prob(eta, psi)
    return -2.0 * np.sum(y * np.log(p) + (1 - y) * np.log(np.clip(omp, EPS, 1.0)))


def main():
    paths = coverage_check()
    print("binarised tree covers all 10 categories, one unique path each\n")

    rows = common.load_pa(with_handedness=False)
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]

    # inner split for selecting (psi, lam) -- by game, never by row
    tr_games = sorted({r["game_id"] for r in tr})
    rng = np.random.default_rng(90210)
    rng.shuffle(tr_games)
    cut = int(0.8 * len(tr_games))
    inner_fit_g = set(tr_games[:cut])
    inner_val_g = set(tr_games[cut:])

    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)
    print(f"rows: train {len(tr)}  test {len(te)}   batters {n_bat}  pitchers {n_pit}")
    print(f"inner: fit {len(inner_fit_g)} games / val {len(inner_val_g)} games\n")

    def arrays(rs, reach, pos):
        rs = [r for r in rs if r["y"] in reach]
        bi = np.fromiter((BI[r["batter"]] for r in rs), int, len(rs))
        pj = np.fromiter((PI[r["pitcher"]] for r in rs), int, len(rs))
        y = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in rs), float, len(rs))
        return bi, pj, y

    n_test = len(te)
    total_logit = 0.0
    total_fit = 0.0
    results = []

    for name, reach, pos in NODES:
        f_bi, f_pj, f_y = arrays([r for r in tr if r["game_id"] in inner_fit_g], reach, pos)
        v_bi, v_pj, v_y = arrays([r for r in tr if r["game_id"] in inner_val_g], reach, pos)
        t_bi, t_pj, t_y = arrays(tr, reach, pos)
        e_bi, e_pj, e_y = arrays(te, reach, pos)

        # Select (psi, lam) jointly for AO, but lam SEPARATELY for logit.
        # Sharing one lam across both hands the logit reference a penalty tuned
        # for a different link, which manufactures the gain being measured.
        best = None
        best_logit = None
        for lam in LAM_GRID:
            warm = None
            for psi in PSI_GRID:
                th = fit_node(f_bi, f_pj, f_y, n_bat, n_pit, psi, lam, warm)
                warm = th
                d = node_dev(th, v_bi, v_pj, v_y, n_bat, n_pit, psi) / max(1, len(v_y))
                if best is None or d < best[0]:
                    best = (d, psi, lam)
                if psi == 1.0 and (best_logit is None or d < best_logit[0]):
                    best_logit = (d, lam)
        _, psi_hat, lam_hat = best
        _, lam_logit = best_logit

        # refit both links on the FULL training set, score on frozen test
        th_l = fit_node(t_bi, t_pj, t_y, n_bat, n_pit, 1.0, lam_logit)
        th_a = fit_node(t_bi, t_pj, t_y, n_bat, n_pit, psi_hat, lam_hat)
        d_l = node_dev(th_l, e_bi, e_pj, e_y, n_bat, n_pit, 1.0)
        d_a = node_dev(th_a, e_bi, e_pj, e_y, n_bat, n_pit, psi_hat)
        total_logit += d_l
        total_fit += d_a

        results.append({"node": name, "n_train": int(len(t_y)), "n_test": int(len(e_y)),
                        "rate": float(t_y.mean()), "psi_hat": psi_hat,
                        "lam_ao": lam_hat, "lam_logit": lam_logit,
                        "test_dev_logit": d_l / n_test, "test_dev_ao": d_a / n_test,
                        "delta": (d_a - d_l) / n_test})
        r = results[-1]
        flag = "  <-- not logit" if psi_hat != 1.0 else ""
        print(f"{name:9} n={len(t_y):>6} rate={t_y.mean():.4f}  psi_hat={psi_hat:<5} "
              f"lam={lam_hat:<5}/{lam_logit:<5} dev(logit)={r['test_dev_logit']:.5f} "
              f"dev(AO)={r['test_dev_ao']:.5f}  delta={r['delta']:+.5f}{flag}")

    print(f"\n{'TOTAL joint test deviance':32} logit = {total_logit / n_test:.5f}")
    print(f"{'':32}    AO = {total_fit / n_test:.5f}")
    print(f"{'':32} delta = {(total_fit - total_logit) / n_test:+.5f}")
    print("\nreference: NULL 4.01172 | flat ridge 3.95550 | NPMR 3.95424 | "
          "flat GLLVM 3.95563")

    out = {"nodes": results,
           "test_dev_logit": total_logit / n_test,
           "test_dev_ao": total_fit / n_test,
           "delta": (total_fit - total_logit) / n_test,
           "psi_grid": PSI_GRID, "lam_grid": LAM_GRID}
    with open(os.path.join(os.path.dirname(__file__), "link_result.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote link_result.json")


if __name__ == "__main__":
    main()
