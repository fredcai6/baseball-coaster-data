"""Pitching style measured from PITCH SEQUENCES, not from results.

Why this is not just the clustering spike again
-----------------------------------------------
Every style axis we have tried so far was derived from outcomes, so "style"
and "skill" were the same measurement wearing two hats -- which is part of why
the cluster analysis found a continuum. The 2026 pitch strings are a genuinely
different instrument: they record HOW a pitcher works a count, and two
pitchers with identical strikeout rates can reach them differently.

Three questions, in increasing order of interest:

  1. How much of a pitcher's fitted skill does pitch shape explain? (R^2)
  2. Are there clusters in pitch-shape space? (the original question, better
     instrumented)
  3. Does pitch shape INTERACT with batter type?

(3) is the one that matters. A pitcher-level covariate is a function of
pitcher identity, so it is perfectly collinear with the pitcher main effect
and can add exactly nothing to an additive model -- there is no point testing
that. But `batter_effect x style` is NOT collinear with either main effect,
and it is precisely the original question: does a batter's strikeout-proneness
matter more against some pitching styles than others?

Scope: 2026 only. 2024 and 2025 publish no pitch strings at all.
"""
import sys, os, json, math, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import minimize
import common

EPS = 1e-12
K_CAT = common.CAT_INDEX["K"]

#: Per-pitch shape descriptors. Deliberately RATES, not counts, so a pitcher's
#: workload does not masquerade as a style.
FEATS = ["pitches_per_pa", "ball_rate", "called_share", "swing_miss_rate",
         "foul_rate", "first_pitch_strike"]


def pitcher_style(rows):
    """Aggregate pitch-shape rates per pitcher, and per (pitcher, game).

    Returns (totals, by_game) so a caller can build leave-one-game-out
    features -- a pitcher's style must never be measured on the PA it is
    being used to predict.
    """
    tot = collections.defaultdict(lambda: collections.Counter())
    by_game = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        seq = r.get("pitch_seq")
        n = r.get("n_pitches")
        if n is None:
            continue
        seq = seq or ""
        c = collections.Counter(seq)
        acc = {"pa": 1, "pitches": n, "B": c["B"], "K": c["K"], "S": c["S"],
               "F": c["F"], "fps": 1 if (seq and seq[0] in "KSF") else 0,
               "has_first": 1 if seq else 0}
        tot[r["pitcher"]].update(acc)
        by_game[(r["pitcher"], r["game_id"])].update(acc)
    return tot, by_game


def rates(c, min_pitches=150):
    """Turn raw counts into the FEATS vector, or None when too thin to trust."""
    if c["pitches"] < min_pitches:
        return None
    p = float(c["pitches"])
    strikes = c["K"] + c["S"]
    return np.array([
        c["pitches"] / c["pa"],
        c["B"] / p,
        c["K"] / strikes if strikes else 0.5,
        c["S"] / p,
        c["F"] / p,
        c["fps"] / c["has_first"] if c["has_first"] else 0.5,
    ])


def fit_binary(bi, pj, y, n_bat, n_pit, lam, extra=None, lam_x=1.0):
    """Ridge logistic: alpha + b[bi] + q[pj] (+ extra @ gamma)."""
    m = 0 if extra is None else extra.shape[1]
    n = 1 + n_bat + n_pit + m

    def f(th):
        a = th[0]
        b = th[1:1 + n_bat]
        q = th[1 + n_bat:1 + n_bat + n_pit]
        eta = a + b[bi] + q[pj]
        if m:
            g = th[1 + n_bat + n_pit:]
            eta = eta + extra @ g
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        p = np.clip(p, EPS, 1 - EPS)
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        nll += 0.5 * lam * (b @ b + q @ q)
        ge = p - y
        grad = np.concatenate((
            [ge.sum()],
            np.bincount(bi, weights=ge, minlength=n_bat) + lam * b,
            np.bincount(pj, weights=ge, minlength=n_pit) + lam * q,
        ))
        if m:
            g = th[1 + n_bat + n_pit:]
            nll += 0.5 * lam_x * (g @ g)
            grad = np.concatenate((grad, extra.T @ ge + lam_x * g))
        return nll, grad

    res = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-11})
    return res.x


def dev_binary(th, bi, pj, y, n_bat, n_pit, extra=None):
    a = th[0]
    b = th[1:1 + n_bat]
    q = th[1 + n_bat:1 + n_bat + n_pit]
    eta = a + b[bi] + q[pj]
    if extra is not None:
        eta = eta + extra @ th[1 + n_bat + n_pit:]
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30))), EPS, 1 - EPS)
    return -2.0 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y)


def main():
    rows = [r for r in common.load_pa(with_handedness=False)
            if r["season"] == 2026 and r.get("n_pitches") is not None]
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    print(f"2026 PAs with pitch strings: {len(rows)}   train {len(tr)}  test {len(te)}")

    tot, by_game = pitcher_style(tr)
    style = {p: rates(c) for p, c in tot.items()}
    style = {p: v for p, v in style.items() if v is not None}
    print(f"pitchers with >=150 train pitches: {len(style)} of {len(tot)}")

    Z = np.array([style[p] for p in sorted(style)])
    mu, sd = Z.mean(0), Z.std(0) + 1e-9
    print("\n=== pitch-shape descriptors (train, pitchers >=150 pitches) ===")
    for i, f in enumerate(FEATS):
        print(f"  {f:20} mean={mu[i]:.4f}  sd={sd[i]:.4f}  "
              f"range=[{Z[:,i].min():.3f}, {Z[:,i].max():.3f}]")

    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)

    def arr(rs):
        bi = np.fromiter((BI[r["batter"]] for r in rs), int, len(rs))
        pj = np.fromiter((PI[r["pitcher"]] for r in rs), int, len(rs))
        y = np.fromiter((1.0 if r["y"] == K_CAT else 0.0 for r in rs), float, len(rs))
        return bi, pj, y

    t_bi, t_pj, t_y = arr(tr)
    e_bi, e_pj, e_y = arr(te)

    LAM = 20.0
    base = fit_binary(t_bi, t_pj, t_y, n_bat, n_pit, LAM)
    d_base = dev_binary(base, e_bi, e_pj, e_y, n_bat, n_pit)
    print(f"\n=== Q1/Q3: strikeout node, 2026 ===")
    print(f"additive baseline (batter + pitcher)   test deviance = {d_base:.5f}")

    # --- Q1: how much of fitted pitcher K-skill does pitch shape explain?
    q_hat = base[1 + n_bat:1 + n_bat + n_pit]
    ids = [p for p in sorted(style) if tot[p]["pa"] >= 40]
    A = np.array([(style[p] - mu) / sd for p in ids])
    yv = np.array([q_hat[PI[p]] for p in ids])
    A1 = np.hstack([np.ones((len(A), 1)), A])
    coef, *_ = np.linalg.lstsq(A1, yv, rcond=None)
    resid = yv - A1 @ coef
    r2 = 1 - resid.var() / yv.var()
    print(f"\nQ1  pitcher K-effect explained by pitch shape:  R^2 = {r2:.3f}  "
          f"(n={len(ids)} pitchers)")
    for f, c in zip(FEATS, coef[1:]):
        print(f"      {f:20} {c:+.4f}")

    # --- Q2: clusters in pitch-shape space, vs a matched-Gaussian null
    from scipy.spatial.distance import cdist
    Zs = (Z - mu) / sd
    rng = np.random.default_rng(7)
    def hopkins(X, m=None):
        n, d = X.shape
        m = m or max(5, n // 10)
        idx = rng.choice(n, m, replace=False)
        lo, hi = X.min(0), X.max(0)
        U = rng.uniform(lo, hi, (m, d))
        du = np.sort(cdist(U, X))[:, 0]
        dw = np.sort(cdist(X[idx], X))[:, 1]
        return du.sum() / (du.sum() + dw.sum())
    h_real = np.mean([hopkins(Zs) for _ in range(30)])
    null = np.array([np.mean([hopkins(rng.multivariate_normal(
        np.zeros(Zs.shape[1]), np.cov(Zs.T), len(Zs))) for _ in range(5)])
        for _ in range(20)])
    print(f"\nQ2  Hopkins statistic  real = {h_real:.4f}   "
          f"matched-Gaussian null = {null.mean():.4f} +/- {null.std():.4f}")
    print("      (0.5 = no cluster structure beyond the null's own)")

    # --- Q3: does pitch shape INTERACT with batter strikeout-proneness?
    b_hat = base[1:1 + n_bat]
    def inter(rs, bi, pj):
        M = np.zeros((len(rs), len(FEATS)))
        keep = np.zeros(len(rs), bool)
        for i, r in enumerate(rs):
            s = style.get(r["pitcher"])
            if s is None:
                continue
            keep[i] = True
            M[i] = b_hat[bi[i]] * ((s - mu) / sd)
        return M, keep

    M_tr, k_tr = inter(tr, t_bi, t_pj)
    M_te, k_te = inter(te, e_bi, e_pj)
    print(f"\nQ3  rows with a styled pitcher: train {k_tr.sum()}  test {k_te.sum()}")

    b2 = fit_binary(t_bi[k_tr], t_pj[k_tr], t_y[k_tr], n_bat, n_pit, LAM)
    d2 = dev_binary(b2, e_bi[k_te], e_pj[k_te], e_y[k_te], n_bat, n_pit)
    b3 = fit_binary(t_bi[k_tr], t_pj[k_tr], t_y[k_tr], n_bat, n_pit, LAM,
                    extra=M_tr[k_tr], lam_x=1.0)
    d3 = dev_binary(b3, e_bi[k_te], e_pj[k_te], e_y[k_te], n_bat, n_pit,
                    extra=M_te[k_te])
    print(f"      additive only            test deviance = {d2:.5f}")
    print(f"      + batter x style         test deviance = {d3:.5f}   "
          f"({d3 - d2:+.5f})")
    print("      interaction coefficients:")
    for f, c in zip(FEATS, b3[1 + n_bat + n_pit:]):
        print(f"        {f:20} {c:+.4f}")

    out = {"n_rows": len(rows), "n_pitchers_styled": len(style),
           "d_base": d_base, "q1_r2": float(r2),
           "q1_coef": dict(zip(FEATS, map(float, coef[1:]))),
           "q2_hopkins_real": float(h_real),
           "q2_hopkins_null_mean": float(null.mean()),
           "q2_hopkins_null_sd": float(null.std()),
           "q3_additive": float(d2), "q3_interaction": float(d3),
           "q3_delta": float(d3 - d2),
           "q3_coef": dict(zip(FEATS, map(float, b3[1 + n_bat + n_pit:])))}
    with open(os.path.join(os.path.dirname(__file__), "style_result.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote style_result.json")


if __name__ == "__main__":
    main()
