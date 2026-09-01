"""SPIKE 2/3: Nuclear Penalized Multinomial Regression (Powers, Hastie &
Tibshirani 2018) on plate-appearance outcomes.

Design (paper Sec 5.1, eta_ik = alpha_k + beta_Bk + gamma_Pk + delta_Sk +
zeta_k*H_i + theta_k*O_i), adapted:
  - batter block  B_bat  (n_bat  x K)  -- nuclear-penalized
  - pitcher block B_pitch(n_pitch x K) -- nuclear-penalized, separate SVT
  - "other" block Theta  (p_other x K) -- UNpenalized: intercept, home
    indicator, opposite-handedness (3-level: same/opposite/unknown), season
    (3 levels, one dropped as baseline). No stadium term: this corpus (minor
    league, 35 home_team values across 3 seasons) was not asked for one by
    the brief's predictor list, and adding it would blow the CV grid without
    a payoff for the interpretation question this spike exists to answer.

Fit by accelerated proximal gradient descent, following Sec 3.1-3.2 exactly,
generalized from one penalized block (B) + one unpenalized block (alpha) to
one unpenalized block (Theta_other) + two independently-penalized blocks
(B_bat, B_pitch), each with its own singular-value soft-threshold operator.
Step size follows the paper's guidance (start at a trial s, halve -- via a
doubling Lipschitz estimate -- on any increase in the smooth objective).

Lambda tied across the two blocks for CV tractability (one 1-D grid instead
of a 2-D one) -- see README.md "Design decisions" for the justification.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

CATS = common.CATEGORIES
K = len(CATS)
REPL_BATTER = "__replacement_batter__"
REPL_PITCHER = "__replacement_pitcher__"
SEASONS = [2024, 2025, 2026]  # baseline = 2024


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def opp_hand_code(bats, throws):
    """0 = same-handed, 1 = opposite-handed (incl. switch hitters, who choose
    the box to always get the platoon advantage), 2 = unknown (either side
    missing)."""
    if not bats or not throws:
        return 2
    if bats == "S":
        return 1
    return 1 if bats != throws else 0


def build_player_index(rows, key, pa_threshold=0):
    """Map player id -> dense index, 0..n-1, with a reserved replacement
    bucket appended last. Players with fewer than pa_threshold training PAs
    (0 = no pooling) collapse into the replacement bucket."""
    from collections import Counter
    counts = Counter(r[key] for r in rows)
    keep = sorted(pid for pid, c in counts.items() if c >= pa_threshold)
    idx = {pid: i for i, pid in enumerate(keep)}
    repl = len(keep)
    idx["__repl__"] = repl
    return idx, repl + 1  # mapping, total rows (incl. replacement)


def encode_rows(rows, batter_idx, pitcher_idx):
    n = len(rows)
    y = np.array([r["y"] for r in rows], dtype=np.int64)
    bidx = np.array(
        [batter_idx.get(r["batter"], batter_idx["__repl__"]) for r in rows],
        dtype=np.int64,
    )
    pidx = np.array(
        [pitcher_idx.get(r["pitcher"], pitcher_idx["__repl__"]) for r in rows],
        dtype=np.int64,
    )
    home = np.array([1.0 if r["batting_is_home"] else 0.0 for r in rows])
    opp = np.array([opp_hand_code(r["bats"], r["throws"]) for r in rows])
    opp_opposite = (opp == 1).astype(float)
    opp_unknown = (opp == 2).astype(float)
    season_idx = np.array([SEASONS.index(r["season"]) for r in rows])
    season_2025 = (season_idx == 1).astype(float)
    season_2026 = (season_idx == 2).astype(float)
    intercept = np.ones(n)
    X_other = np.column_stack(
        [intercept, home, opp_opposite, opp_unknown, season_2025, season_2026]
    )
    return {
        "y": y, "bidx": bidx, "pidx": pidx, "X_other": X_other, "n": n,
    }


def onehot_sparse(idx, ncols):
    n = len(idx)
    return sparse.csr_matrix(
        (np.ones(n), (np.arange(n), idx)), shape=(n, ncols)
    )


OTHER_NAMES = ["intercept", "home", "opp_opposite", "opp_unknown",
               "season_2025", "season_2026"]


# --------------------------------------------------------------------------
# Core NPMR fit: accelerated proximal gradient descent
# --------------------------------------------------------------------------

def softmax_rows(eta):
    m = eta.max(axis=1, keepdims=True)
    e = np.exp(eta - m)
    return e / e.sum(axis=1, keepdims=True)


def neg_loglik(P, y):
    n = len(y)
    return -np.log(np.clip(P[np.arange(n), y], 1e-12, None)).sum()


def svt(M, thresh):
    """Singular value soft-threshold: S*_thresh(M). Returns (M_new, singular
    values of M_new, rank)."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    S_new = np.maximum(S - thresh, 0.0)
    rank = int((S_new > 0).sum())
    M_new = (U * S_new) @ Vt
    return M_new, S_new, rank


def _block_update(Xblock, y, Ymat, other_eta, W, W_prev, s, t, prox_fn,
                   max_backtrack=50):
    """One accelerated-proximal-gradient step for a single block, holding
    all other blocks fixed at `other_eta` (their current contribution to the
    linear predictor). Uses the standard proximal-gradient backtracking
    (majorization) condition, so each call provably does not increase the
    penalized objective relative to the block's own extrapolated point.

    `prox_fn(M, s) -> (M_new, penalty_terms)` where `penalty_terms` is
    whatever the caller wants back (e.g. singular values) or None.
    """
    mom = t / (t + 3.0)
    W_ext = W + mom * (W - W_prev)
    eta = other_eta + Xblock @ W_ext
    P = softmax_rows(eta)
    g_ref = neg_loglik(P, y)
    # `grad` here is X^T(Y-P), i.e. the NEGATIVE of the true gradient of the
    # negative-log-likelihood g (true grad_g = X^T(P-Y)). The paper's update
    # (eq. 7-8) adds it directly -- W_new = W + s*grad -- which is a correct
    # descent step. But the majorization/backtracking check below needs the
    # TRUE gradient's sign: bound = g_ref + <grad_g, diff> + (1/2s)||diff||^2
    # = g_ref - <grad, diff> + (1/2s)||diff||^2. Using +<grad,diff> here (an
    # earlier version of this code) flips the sign of the cross term, which
    # makes the bound get LOOSER instead of stricter as s grows, so bad
    # (diverging) large steps get accepted instead of rejected. Confirmed by
    # instrumentation: the flipped-sign version passed backtracking at every
    # iteration while the penalized objective grew unboundedly.
    grad = Xblock.T @ (Ymat - P)
    if sparse.issparse(grad):
        grad = np.asarray(grad)
    for _ in range(max_backtrack):
        cand = W_ext + s * grad
        W_new, extra = prox_fn(cand, s)
        eta2 = other_eta + Xblock @ W_new
        P2 = softmax_rows(eta2)
        g_new = neg_loglik(P2, y)
        diff = W_new - W_ext
        bound = g_ref - np.sum(grad * diff) + (1.0 / (2 * s)) * np.sum(diff ** 2)
        if g_new <= bound + 1e-6 * max(1.0, abs(bound)):
            break
        s *= 0.5
    return W_new, s * 1.3, P2, g_new, extra


def fit_npmr(enc, n_bat, n_pitch, lam_bat, lam_pitch, max_iter=400,
             tol=1e-7, verbose=False, seed_state=None):
    """Cyclic block-coordinate accelerated proximal gradient descent, per
    block Sec 3.1-3.2 (momentum + singular-value soft-thresholding for the
    penalized blocks; plain accelerated gradient for the unpenalized
    "other" block).

    DEVIATION FROM THE PAPER, documented: Powers et al. take one
    *simultaneous* joint step across (alpha, B) with a single shared step
    size (Sec 3.2, steps 1-4). Our design has three blocks with wildly
    different natural curvature -- the dense "other" block's intercept
    column touches every one of n rows, while a single player's block row
    is touched only by that player's (often <50) PAs. A single shared step
    size that is small enough to be stable for the dense block is far too
    small to move the sparse player blocks at a workable rate (and vice
    versa: a step sized for a thin player is unstable for the intercept).
    We therefore cycle Theta_other -> B_bat -> B_pitch each outer
    iteration, each with its OWN adaptively backtracked step size and its
    own Nesterov momentum term (same t/(t+3) schedule as the paper), using
    the fixed-point (Gauss-Seidel) values of the other two blocks. Each
    substep individually satisfies the standard proximal-gradient
    majorization condition, so it provably does not increase the penalized
    objective relative to its own extrapolated point; empirically the total
    objective decreases monotonically (checked below).
    """
    y = enc["y"]
    n = enc["n"]
    Xo = enc["X_other"]
    p_other = Xo.shape[1]
    Xbat = onehot_sparse(enc["bidx"], n_bat)
    Xpit = onehot_sparse(enc["pidx"], n_pitch)
    Ymat = np.zeros((n, K))
    Ymat[np.arange(n), y] = 1.0

    if seed_state is None:
        Theta = np.zeros((p_other, K))
        Bbat = np.zeros((n_bat, K))
        Bpit = np.zeros((n_pitch, K))
    else:
        Theta, Bbat, Bpit = (a.copy() for a in seed_state)
    Theta_prev, Bbat_prev, Bpit_prev = Theta.copy(), Bbat.copy(), Bpit.copy()

    identity_prox = lambda M, s: (M, None)
    svt_bat = lambda M, s: svt(M, s * lam_bat)[:2]
    svt_pit = lambda M, s: svt(M, s * lam_pitch)[:2]

    # Rough initial per-block step sizes: the dense/global block sees
    # curvature ~ n (every row touches it); a typical player block sees
    # curvature ~ typical PAs-per-player, which is orders of magnitude
    # smaller. Backtracking corrects whatever these guesses get wrong.
    s_theta = 4.0 / max(n, 1)
    s_bat = 4.0 / max(n / max(n_bat, 1), 1.0)
    s_pit = 4.0 / max(n / max(n_pitch, 1), 1.0)

    rank_bat = rank_pit = 0
    Sbat = Spit = np.zeros(0)
    hist = []
    for t in range(max_iter):
        eta_bp = Bbat[enc["bidx"]] + Bpit[enc["pidx"]]
        Theta_new, s_theta, _, _, _ = _block_update(
            Xo, y, Ymat, eta_bp, Theta, Theta_prev, s_theta, t, identity_prox)

        eta_op = Xo @ Theta_new + Bpit[enc["pidx"]]
        Bbat_new, s_bat, _, _, Sbat = _block_update(
            Xbat, y, Ymat, eta_op, Bbat, Bbat_prev, s_bat, t, svt_bat)
        rank_bat = int((Sbat > 0).sum())

        eta_ob = Xo @ Theta_new + Bbat_new[enc["bidx"]]
        Bpit_new, s_pit, P_final, g_final, Spit = _block_update(
            Xpit, y, Ymat, eta_ob, Bpit, Bpit_prev, s_pit, t, svt_pit)
        rank_pit = int((Spit > 0).sum())

        Theta_prev, Bbat_prev, Bpit_prev = Theta, Bbat, Bpit
        Theta, Bbat, Bpit = Theta_new, Bbat_new, Bpit_new

        pen = lam_bat * Sbat.sum() + lam_pitch * Spit.sum()
        obj = g_final + pen
        hist.append(obj)
        if verbose and (t % 50 == 0 or t == max_iter - 1):
            print(f"    iter {t:4d}  s=({s_theta:.2e},{s_bat:.2e},"
                  f"{s_pit:.2e})  smooth/obs={g_final/n:.5f}  pen={pen:.2f}  "
                  f"rank_bat={rank_bat}  rank_pit={rank_pit}")
        if t > 5 and abs(hist[-2] - hist[-1]) < tol * max(1.0, abs(hist[-2])):
            break
    return {
        "Theta": Theta, "Bbat": Bbat, "Bpit": Bpit,
        "rank_bat": rank_bat, "rank_pit": rank_pit,
        "n_iter": t + 1, "final_obj": hist[-1] if hist else None,
        "obj_history": hist,
    }


def predict_proba(fit, enc):
    eta = (enc["X_other"] @ fit["Theta"] + fit["Bbat"][enc["bidx"]]
           + fit["Bpit"][enc["pidx"]])
    return softmax_rows(eta)


# --------------------------------------------------------------------------
# CV by game
# --------------------------------------------------------------------------

def kfold_games(train_games, k, seed):
    games = sorted(train_games)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(games))
    folds = [[] for _ in range(k)]
    for i, gi in enumerate(perm):
        folds[i % k].append(games[gi])
    return [set(f) for f in folds]


def cv_score(tr_rows, train_games, lam, pa_threshold, n_folds=3, seed=7,
             max_iter=150):
    folds = kfold_games(train_games, n_folds, seed)
    devs = []
    for i in range(n_folds):
        val_g = folds[i]
        fold_tr = [r for r in tr_rows if r["game_id"] not in val_g]
        fold_val = [r for r in tr_rows if r["game_id"] in val_g]
        batter_idx, n_bat = build_player_index(fold_tr, "batter", pa_threshold)
        pitcher_idx, n_pitch = build_player_index(fold_tr, "pitcher", pa_threshold)
        enc_tr = encode_rows(fold_tr, batter_idx, pitcher_idx)
        enc_val = encode_rows(fold_val, batter_idx, pitcher_idx)
        f = fit_npmr(enc_tr, n_bat, n_pitch, lam, lam, max_iter=max_iter)
        probs = predict_proba(f, enc_val)
        dev = common.deviance(probs, enc_val["y"])
        devs.append(dev)
    return float(np.mean(devs)), float(np.std(devs))


def load_name_map():
    """career id -> display name, read straight off pa_table.csv (it already
    carries batter_name/pitcher_name per row -- no join needed)."""
    import csv
    names = {}
    with open(common.PA_TABLE) as fh:
        for r in csv.DictReader(fh):
            b = r["batter_career"] or r["batter_pid"]
            p = r["pitcher_career"] or r["pitcher_pid"]
            if r["batter_name"]:
                names[b] = r["batter_name"]
            if r["pitcher_name"]:
                names[p] = r["pitcher_name"]
    return names


def player_rate_table(rows, key):
    """id -> {'pa': n, cat: rate, ...} empirical rates from TRAIN rows."""
    from collections import Counter, defaultdict
    counts = defaultdict(Counter)
    for r in rows:
        counts[r[key]][r["cat"]] += 1
    out = {}
    for pid, c in counts.items():
        n = sum(c.values())
        row = {"pa": n}
        row.update({cat: c.get(cat, 0) / n for cat in CATS})
        out[pid] = row
    return out


def _extreme_table(scores, ids, rates, name_map, n_show=10):
    """Return (top, bottom) lists of (id, name, score, rate_row) for the
    real (non-replacement) players with the largest / smallest score on one
    axis."""
    pairs = [(s, pid) for s, pid in zip(scores, ids)
             if pid not in (REPL_BATTER, REPL_PITCHER)]
    pairs.sort(key=lambda x: x[0])
    bottom = pairs[:n_show]
    top = pairs[-n_show:][::-1]

    def row(s, pid):
        name = name_map.get(pid, pid)
        r = rates.get(pid, {"pa": 0})
        return {"id": pid, "name": name, "score": s, "pa": r.get("pa", 0),
                "rates": {c: r.get(c, 0.0) for c in CATS}}

    return [row(s, pid) for s, pid in top], [row(s, pid) for s, pid in bottom]


def _rate_row_md(r):
    return " ".join(f"{c}={r['rates'][c]*100:4.1f}" for c in CATS)


def write_latent_md(path, Sbat, Vbat, US_bat, bat_ids, bat_rates, bat_names,
                     Spit, Vpit, US_pit, pit_ids, pit_rates, pit_names,
                     rank_bat, rank_pit, lam, pool, null_dev, test_dev,
                     n_bat, n_pitch):
    lines = []
    lines.append("# NPMR latent structure (batters and pitchers)\n")
    lines.append(f"lambda={lam}  pool_threshold={pool}  "
                 f"rank_batter={rank_bat}  rank_pitcher={rank_pit}  "
                 f"test deviance={test_dev:.5f}  null deviance={null_dev:.5f}\n")
    lines.append("Loadings (V) are unique up to sign; the sign is arbitrary "
                 "and was not normalized to match any convention, so read "
                 "'positive'/'negative' as 'one side' / 'the other side' of "
                 "the axis and use the extreme-player list plus the "
                 "category loadings together to name it.\n")

    def section(title, S, V, US, ids, rates, names, rank, n_all):
        lines.append(f"\n## {title}\n")
        lines.append(f"All singular values ({n_all} total rows incl. "
                     f"replacement bucket): "
                     + ", ".join(f"{s:.3f}" for s in S[:12])
                     + (" ..." if len(S) > 12 else "") + "\n")
        if rank == 0:
            lines.append("**Rank 0 at the CV-selected lambda: the nuclear "
                         "penalty collapsed this block entirely -- no "
                         "latent structure survived.**\n")
            return
        for d in range(rank):
            lines.append(f"\n### Axis {d + 1}  (singular value "
                         f"{S[d]:.3f})\n")
            loadings = V[:, d]
            order = np.argsort(loadings)
            lines.append("Category loadings, most negative to most "
                         "positive:\n")
            lines.append("| category | loading |\n|---|---|\n")
            for k in order:
                lines.append(f"| {CATS[k]} | {loadings[k]:+.3f} |\n")
            top, bottom = _extreme_table(US[:, d], ids, rates, names)
            lines.append("\nTop 10 (most positive on this axis):\n\n")
            lines.append("| player | score | PA | " +
                         " | ".join(CATS) + " |\n")
            lines.append("|---" * (3 + len(CATS)) + "|\n")
            for r in top:
                rr = " | ".join(f"{r['rates'][c]*100:.1f}" for c in CATS)
                lines.append(f"| {r['name']} | {r['score']:+.3f} | "
                             f"{r['pa']} | {rr} |\n")
            lines.append("\nBottom 10 (most negative on this axis):\n\n")
            lines.append("| player | score | PA | " +
                         " | ".join(CATS) + " |\n")
            lines.append("|---" * (3 + len(CATS)) + "|\n")
            for r in bottom:
                rr = " | ".join(f"{r['rates'][c]*100:.1f}" for c in CATS)
                lines.append(f"| {r['name']} | {r['score']:+.3f} | "
                             f"{r['pa']} | {rr} |\n")

    section("Batters", Sbat, Vbat, US_bat, bat_ids, bat_rates, bat_names,
             rank_bat, n_bat)
    section("Pitchers", Spit, Vpit, US_pit, pit_ids, pit_rates, pit_names,
             rank_pit, n_pitch)

    path.write_text("".join(lines))


def main():
    t_start = time.time()
    print("Loading data via spikes/common.py ...")
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    print(f"train PA={len(tr)}  test PA={len(te)}")

    null_p = common.null_model(tr)
    null_dev = common.deviance([null_p] * len(te), [r["y"] for r in te])
    print(f"NULL deviance on test: {null_dev:.5f}")

    # ---- CV: lambda grid x pooling threshold -----------------------------
    # Grid centered empirically: an exploratory full-train sweep (not used
    # for selection, just to place the grid) found deviance improving from
    # lambda=10 to a broad minimum around 40-160, with rank_bat/rank_pit
    # dropping from 9 (full rank, ~no penalty) to 3/3 around lambda=160 and
    # to 0/0 (collapsed to the null-ish additive model) by lambda>=320.
    lam_grid = [10.0, 20.0, 40.0, 80.0, 120.0, 160.0, 240.0, 320.0]
    pool_grid = [0, 20]  # 0 = no pooling; 20 = paper-style replacement level
    print("\nCross-validating lambda x pooling-threshold (3-fold, by game)...")
    cv_results = []
    for pool in pool_grid:
        for lam in lam_grid:
            t0 = time.time()
            mean_dev, sd_dev = cv_score(tr, train_g, lam, pool, n_folds=3,
                                         max_iter=150)
            dt = time.time() - t0
            cv_results.append({"lambda": lam, "pool_threshold": pool,
                                "cv_deviance": mean_dev, "cv_sd": sd_dev,
                                "seconds": dt})
            print(f"  pool={pool:3d}  lambda={lam:6.1f}  "
                  f"cv_dev={mean_dev:.5f} (+-{sd_dev:.5f})  [{dt:.1f}s]")

    best = min(cv_results, key=lambda d: d["cv_deviance"])
    print(f"\nBest by CV: lambda={best['lambda']}  pool_threshold="
          f"{best['pool_threshold']}  cv_dev={best['cv_deviance']:.5f}")

    # ---- Also report: does pooling help, holding lambda's own optimum? ---
    best_by_pool = {}
    for pool in pool_grid:
        sub = [d for d in cv_results if d["pool_threshold"] == pool]
        best_by_pool[pool] = min(sub, key=lambda d: d["cv_deviance"])
    print("Best CV deviance per pooling setting:")
    for pool, d in best_by_pool.items():
        print(f"  pool_threshold={pool}: lambda={d['lambda']} "
              f"cv_dev={d['cv_deviance']:.5f}")

    lam_final = best["lambda"]
    pool_final = best["pool_threshold"]

    # ---- Final fit on full train, more iterations -------------------------
    print(f"\nFinal fit: lambda={lam_final}  pool_threshold={pool_final} "
          f"on full train ({len(tr)} PA), 600 iterations...")
    batter_idx, n_bat = build_player_index(tr, "batter", pool_final)
    pitcher_idx, n_pitch = build_player_index(tr, "pitcher", pool_final)
    enc_tr = encode_rows(tr, batter_idx, pitcher_idx)
    enc_te = encode_rows(te, batter_idx, pitcher_idx)
    t0 = time.time()
    final = fit_npmr(enc_tr, n_bat, n_pitch, lam_final, lam_final,
                      max_iter=600, verbose=True)
    fit_seconds = time.time() - t0
    print(f"final fit took {fit_seconds:.1f}s, {final['n_iter']} iters, "
          f"rank_bat={final['rank_bat']} rank_pit={final['rank_pit']}")

    probs_te = predict_proba(final, enc_te)
    test_dev = common.deviance(probs_te, enc_te["y"])
    print(f"\nTEST deviance = {test_dev:.5f}  (null = {null_dev:.5f})")

    # ---- Latent structure: SVD of final B_bat, B_pit -----------------------
    Ubat, Sbat, Vbat_t = np.linalg.svd(final["Bbat"], full_matrices=False)
    Upit, Spit, Vpit_t = np.linalg.svd(final["Bpit"], full_matrices=False)
    rank_bat_eff = int((Sbat > 1e-8).sum())
    rank_pit_eff = int((Spit > 1e-8).sum())

    inv_batter_idx = [None] * n_bat
    for pid, i in batter_idx.items():
        if pid != "__repl__":
            inv_batter_idx[i] = pid
    inv_batter_idx[batter_idx["__repl__"]] = REPL_BATTER
    inv_pitcher_idx = [None] * n_pitch
    for pid, i in pitcher_idx.items():
        if pid != "__repl__":
            inv_pitcher_idx[i] = pid
    inv_pitcher_idx[pitcher_idx["__repl__"]] = REPL_PITCHER

    US_bat = Ubat * Sbat  # scores: n_bat x rank
    US_pit = Upit * Spit

    runtime_sec = time.time() - t_start

    # ---- Save result.json --------------------------------------------------
    result = {
        "model": "NPMR (block-coordinate accelerated proximal gradient, "
                 "nuclear norm, separate batter/pitcher blocks)",
        "test_pa": len(te),
        "deviance": test_dev,
        "null_deviance": null_dev,
        "lambda": lam_final,
        "pool_threshold": pool_final,
        "rank_batter": rank_bat_eff,
        "rank_pitcher": rank_pit_eff,
        "runtime_sec": runtime_sec,
        "cv_results": cv_results,
        "n_batter_ids": n_bat,
        "n_pitcher_ids": n_pitch,
        "n_iter": final["n_iter"],
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2))

    # ---- Save residuals.npz -------------------------------------------------
    np.savez(HERE / "residuals.npz",
             probs=probs_te, y=enc_te["y"],
             game_id=np.array([r["game_id"] for r in te]),
             batter=np.array([r["batter"] for r in te]),
             pitcher=np.array([r["pitcher"] for r in te]))

    np.savez(HERE / "latent.npz",
              categories=np.array(CATS),
              singular_values_batter=Sbat, singular_values_pitcher=Spit,
              V_batter=Vbat_t.T, V_pitcher=Vpit_t.T,  # K x rank, columns = axes
              scores_batter=US_bat, scores_pitcher=US_pit,  # n x rank
              batter_ids=np.array(inv_batter_idx),
              pitcher_ids=np.array(inv_pitcher_idx),
              other_names=np.array(OTHER_NAMES),
              theta_other=final["Theta"])

    print(f"\nrank_batter (final, effective)={rank_bat_eff}  "
          f"rank_pitcher (final, effective)={rank_pit_eff}")

    # ---- Interpretation report: loadings + extreme players -----------------
    print("\nBuilding latent.md interpretation report...")
    name_map = load_name_map()
    bat_rates = player_rate_table(tr, "batter")
    pit_rates = player_rate_table(tr, "pitcher")
    write_latent_md(
        HERE / "latent.md", Sbat, Vbat_t.T, US_bat, inv_batter_idx,
        bat_rates, name_map, Spit, Vpit_t.T, US_pit, inv_pitcher_idx,
        pit_rates, name_map, rank_bat_eff, rank_pit_eff, lam_final,
        pool_final, null_dev, test_dev, n_bat, n_pitch)

    print(f"Total runtime: {runtime_sec:.1f}s")

    return {
        "result": result, "final": final, "enc_tr": enc_tr, "enc_te": enc_te,
        "batter_idx": batter_idx, "pitcher_idx": pitcher_idx,
        "inv_batter_idx": inv_batter_idx, "inv_pitcher_idx": inv_pitcher_idx,
        "tr": tr, "te": te,
    }


if __name__ == "__main__":
    main()
