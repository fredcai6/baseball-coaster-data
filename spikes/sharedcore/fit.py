"""SPIKE 4: shared-latent-space interaction (Tucker core) on top of GLLVM.

The three additive spikes all answer "how good is this player". None of them
can express "does THIS KIND of batter do unusually well against THAT KIND of
pitcher", because every one of them is additive in batter and pitcher:

    eta_ijk = alpha_k + <batter_i, cat_k> + <pitcher_j, cat_k> + structural

The (i,j) slice is a sum of a function of i and a function of j -- no coupling.
This spike adds the coupling term, as a Tucker core over the SHARED latent
axes the additive fit already found:

    + sum_{a,b} P[i,a] * Q[j,b] * G[a,b,k]

P and Q are the batter/pitcher latent coordinates, reused rather than
re-estimated. That reuse is the whole point. The recommender-systems
literature (Factorization Machines, biased MF, Hoff's AME) uniformly gives
each entity a SEPARATE vector for its interaction role, which costs ~2,000 new
parameters at rank 1 and ~6,000 at rank 3. Sharing costs at most d*d*K = 250,
and 15 for the leanest variant. In a corpus whose entire additive signal is
0.056 deviance, that difference decides whether the question is answerable at
all. The cost of sharing is an assumption: that the axes explaining a player's
overall quality are also the axes along which matchups play out.

Two properties make this cheap and safe:

  * **Convex.** With the additive fit held fixed as a per-row offset, eta is
    LINEAR in G, so this is plain multinomial logistic regression with an
    offset. No local minima, no restart lottery -- unlike the GLLVM fit that
    produced P and Q, where a naive joint solve collapsed to zero.
  * **Centered.** P and Q columns are standardised (PA-weighted, on train) so
    the interaction has ~zero mean over each margin and cannot impersonate a
    main effect. This is exact only on a balanced grid; ours is wildly
    unbalanced (median 2 PA per pair), so decoupling is approximate -- which
    is why the permutation null below is not optional.

The ladder is nested, so one CV sweep adjudicates it:

    null (G=0)  ->  diagonal core (d*K)  ->  full core (d*d*K)

and a PERMUTATION NULL refits the same core after shuffling which pitcher each
batter faced (preserving both margins, destroying only the pairing). If the
real core does not beat the permuted core, there is no interaction here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

GLLVM = HERE.parent / "gllvm"
K = len(common.CATEGORIES)
RIDGE_GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
RANKS = [1, 2, 3, 5]
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def hand_state(bats, throws):
    if not bats or not throws:
        return "unknown"
    if bats == "S":
        return "opposite"
    return "same" if bats == throws else "opposite"


def load_additive():
    """The fitted GLLVM additive model: parameters plus its own index maps."""
    z = np.load(GLLVM / "latent.npz", allow_pickle=True)
    res = json.loads((GLLVM / "result.json").read_text())
    return z, res


def build(rows, bat_idx, pit_idx, season_idx, n_bat, n_pit, z):
    """Per-row additive offset (fixed) plus latent coords for both sides."""
    d = z["Lbat"].shape[1]
    bi = np.array([bat_idx.get(r["batter"], n_bat) for r in rows])
    pi = np.array([pit_idx.get(r["pitcher"], n_pit) for r in rows])
    n_seasons = len(season_idx)
    p = 3 + (n_seasons - 1)
    Xs = np.zeros((len(rows), p))
    Xs[:, 0] = [1.0 if r["batting_is_home"] else 0.0 for r in rows]
    hs = [hand_state(r["bats"], r["throws"]) for r in rows]
    Xs[:, 1] = [1.0 if h == "opposite" else 0.0 for h in hs]
    Xs[:, 2] = [1.0 if h == "unknown" else 0.0 for h in hs]
    for i, r in enumerate(rows):
        si = season_idx[r["season"]]
        if si > 0:
            Xs[i, 3 + si - 1] = 1.0

    Lb = np.vstack([z["Lbat"], np.zeros((1, d))])[bi]
    Lp = np.vstack([z["Lpit"], np.zeros((1, d))])[pi]
    offset = z["alpha"][None, :] + Xs @ z["beta"] + Lb @ z["Fbat"].T + Lp @ z["Fpit"].T
    y = np.array([r["y"] for r in rows])
    return dict(offset=offset, Lb=Lb, Lp=Lp, y=y, bi=bi, pi=pi, n=len(rows))


def orthonormal_axes(z):
    """Ordered, orthogonal latent axes from the fitted additive effect matrices.

    GLLVM's raw L/F are only identified up to an invertible mixing, so the raw
    columns are not comparable between the two sides. SVD of the effect matrix
    B = L @ F.T gives axes ordered by how much of the fitted effect they carry
    -- the same basis NPMR reports, which is what makes 'axis 1' mean the same
    thing for batters and pitchers.
    """
    out = {}
    for side in ("bat", "pit"):
        B = z[f"L{side}"] @ z[f"F{side}"].T
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        out[side] = (U * S, Vt, S)
    return out


def make_features(Lb_ax, Lp_ax, r, diagonal):
    """Interaction design: outer product of the two sides' latent coords."""
    P, Q = Lb_ax[:, :r], Lp_ax[:, :r]
    if diagonal:
        return P * Q                                   # (n, r)
    return (P[:, :, None] * Q[:, None, :]).reshape(len(P), r * r)


def fit_core(Z, offset, y, ridge, maxiter=400):
    """Multinomial logistic on Z with a FIXED offset. Convex in G."""
    m = Z.shape[1]

    def obj(theta):
        G = theta.reshape(m, K)
        eta = offset + Z @ G
        eta = eta - eta.max(axis=1, keepdims=True)
        ex = np.exp(eta)
        P = ex / ex.sum(axis=1, keepdims=True)
        nll = -np.sum(np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)))
        dEta = P.copy()
        dEta[np.arange(len(y)), y] -= 1.0
        return nll + ridge * np.sum(G**2), (Z.T @ dEta + 2 * ridge * G).ravel()

    res = optimize.minimize(obj, np.zeros(m * K), jac=True, method="L-BFGS-B",
                            options=dict(maxiter=maxiter))
    return res.x.reshape(m, K)


def deviance_with(offset, Z, G, y):
    eta = offset + (Z @ G if G is not None else 0.0)
    eta = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(eta)
    P = ex / ex.sum(axis=1, keepdims=True)
    return -2.0 * np.mean(np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)))


def standardize(train_ax, *others):
    """PA-weighted centering+scaling on TRAIN, applied to all splits.

    Centering is what stops the interaction impersonating a main effect: with
    mean-zero columns the interaction averages to zero over each margin.
    """
    mu, sd = train_ax.mean(axis=0), train_ax.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return [(a - mu) / sd for a in (train_ax,) + others]


def main():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    log(f"train={len(tr)} test={len(te)}")

    z, gres = load_additive()
    bat_ids, pit_ids = list(z["bat_ids"]), list(z["pit_ids"])
    bat_idx = {b: i for i, b in enumerate(bat_ids)}
    pit_idx = {p: i for i, p in enumerate(pit_ids)}
    season_idx = {s: i for i, s in enumerate([2024, 2025, 2026])}
    n_bat, n_pit = len(bat_ids), len(pit_ids)
    d = z["Lbat"].shape[1]
    log(f"additive GLLVM loaded: d={d} n_bat={n_bat} n_pit={n_pit}")

    ax = orthonormal_axes(z)
    log(f"batter axis singular values : {np.round(ax['bat'][2], 3).tolist()}")
    log(f"pitcher axis singular values: {np.round(ax['pit'][2], 3).tolist()}")

    Dtr = build(tr, bat_idx, pit_idx, season_idx, n_bat, n_pit, z)
    Dte = build(te, bat_idx, pit_idx, season_idx, n_bat, n_pit, z)

    # Per-row latent coords in the ordered SVD basis (zero row for unseen players)
    def coords(D, side, idx_key):
        base = ax[side][0]
        A = np.vstack([base, np.zeros((1, base.shape[1]))])
        return A[D[idx_key]]

    Btr, Bte = coords(Dtr, "bat", "bi"), coords(Dte, "bat", "bi")
    Ptr, Pte = coords(Dtr, "pit", "pi"), coords(Dte, "pit", "pi")
    Btr, Bte = standardize(Btr, Bte)
    Ptr, Pte = standardize(Ptr, Pte)

    base_test = deviance_with(Dte["offset"], None, None, Dte["y"])
    log(f"ADDITIVE GLLVM test deviance (offset only) = {base_test:.5f}")
    log(f"NULL test deviance = {common.deviance([common.null_model(tr)]*len(te), Dte['y']):.5f}")

    # ---- inner CV by game, train only -----------------------------------
    inner_games = sorted(train_g)
    rng = np.random.RandomState(7)
    rng.shuffle(inner_games)
    cut = int(len(inner_games) * 0.8)
    itr_g, ival_g = set(inner_games[:cut]), set(inner_games[cut:])
    m_itr = np.array([r["game_id"] in itr_g for r in tr])
    m_ival = ~m_itr
    log(f"inner split: itrain={m_itr.sum()} ival={m_ival.sum()}")

    grid = []
    for diagonal in (True, False):
        for r in RANKS:
            if not diagonal and r * r * K > 400:
                continue
            Ztr_all = make_features(Btr, Ptr, r, diagonal)
            for ridge in RIDGE_GRID:
                t = time.time()
                G = fit_core(Ztr_all[m_itr], Dtr["offset"][m_itr], Dtr["y"][m_itr], ridge)
                dv = deviance_with(Dtr["offset"][m_ival], Ztr_all[m_ival], G, Dtr["y"][m_ival])
                grid.append(dict(diagonal=diagonal, rank=r, ridge=ridge,
                                 val_deviance=float(dv), n_params=int(G.size)))
                log(f"[CV] {'diag' if diagonal else 'full'} r={r} ridge={ridge:6g} "
                    f"val={dv:.5f} params={G.size} ({time.time()-t:.1f}s)")

    base_val = deviance_with(Dtr["offset"][m_ival], None, None, Dtr["y"][m_ival])
    log(f"[CV] additive-only val deviance = {base_val:.5f}  <-- the bar")

    best = min(grid, key=lambda g: g["val_deviance"])
    log(f"selected: {best}")
    beats_bar = best["val_deviance"] < base_val

    # ---- final fit on full train, evaluate on the frozen test split ------
    Ztr = make_features(Btr, Ptr, best["rank"], best["diagonal"])
    Zte = make_features(Bte, Pte, best["rank"], best["diagonal"])
    G = fit_core(Ztr, Dtr["offset"], Dtr["y"], best["ridge"])
    test_dev = deviance_with(Dte["offset"], Zte, G, Dte["y"])
    log(f"FINAL shared-core test deviance = {test_dev:.5f}  "
        f"(additive {base_test:.5f}, delta {test_dev-base_test:+.5f})")

    # ---- permutation null -----------------------------------------------
    # Shuffle WHICH PITCHER each plate appearance faced, within season, so both
    # margins survive and only the pairing dies. Any "interaction" the core
    # finds here is what the method invents on data with none.
    log("permutation null: refitting the same core on shuffled pairings...")
    perm_deltas = []
    seasons_tr = np.array([r["season"] for r in tr])
    for rep in range(10):
        rs = np.random.RandomState(1000 + rep)
        Pperm = Ptr.copy()
        for s in (2024, 2025, 2026):
            m = seasons_tr == s
            idx = np.where(m)[0]
            Pperm[idx] = Ptr[rs.permutation(idx)]
        Zp = make_features(Btr, Pperm, best["rank"], best["diagonal"])
        Gp = fit_core(Zp[m_itr], Dtr["offset"][m_itr], Dtr["y"][m_itr], best["ridge"])
        dvp = deviance_with(Dtr["offset"][m_ival], Zp[m_ival], Gp, Dtr["y"][m_ival])
        perm_deltas.append(float(dvp - base_val))
        log(f"  perm {rep}: val delta = {dvp - base_val:+.5f}")

    real_delta = best["val_deviance"] - base_val
    perm = np.array(perm_deltas)
    log(f"real val delta = {real_delta:+.5f}")
    log(f"perm val delta = mean {perm.mean():+.5f}  sd {perm.std():.5f}  min {perm.min():+.5f}")
    z_score = (real_delta - perm.mean()) / max(perm.std(), 1e-12)
    log(f"z vs permutation null = {z_score:+.2f}  (negative delta = improvement)")

    out = dict(
        model="shared_core_tucker_interaction",
        test_pa=len(te), null_deviance=common.deviance(
            [common.null_model(tr)] * len(te), Dte["y"]),
        additive_gllvm_test_deviance=float(base_test),
        shared_core_test_deviance=float(test_dev),
        test_delta=float(test_dev - base_test),
        selected=best, beats_additive_on_val=bool(beats_bar),
        val_additive=float(base_val), val_real_delta=float(real_delta),
        perm_deltas=perm_deltas, perm_mean=float(perm.mean()),
        perm_sd=float(perm.std()), z_vs_perm=float(z_score),
        cv_grid=grid, runtime_sec=time.time() - T0,
    )
    (HERE / "result.json").write_text(json.dumps(out, indent=1) + "\n")
    np.savez(HERE / "core.npz", G=G, bat_axes=ax["bat"][0], pit_axes=ax["pit"][0],
             bat_loadings=ax["bat"][1], pit_loadings=ax["pit"][1],
             categories=np.array(common.CATEGORIES), bat_ids=z["bat_ids"], pit_ids=z["pit_ids"])
    log("wrote result.json and core.npz")


if __name__ == "__main__":
    main()
