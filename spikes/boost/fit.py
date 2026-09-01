"""Drop the functional form: boost the residuals of the additive fit.

The bilinear core and the cluster analysis each assumed a SHAPE for the
interaction -- low-rank and multiplicative, then piecewise-constant on a
partition. Both came back empty, which rules out those two forms and nothing
else. This drops the assumption: gradient-boosted trees on the latent
coordinates of both sides, free to carve the space however they like, with the
additive fit held as a per-row offset so anything they find is by construction
what additivity missed.

Implemented as honest multinomial gradient boosting rather than an off-the-
shelf classifier, because the offset is the whole point and sklearn's
classifiers will not take one: start at eta = additive fit, and at each round
fit one shallow regression tree per category to the multinomial gradient
(y_onehot - p), then step eta by lr * tree.

The control that makes it a test rather than a demonstration is the same
permutation null as before. Trees WILL reduce training loss on any input; the
question is whether they reduce HELD-OUT loss more on real pairings than on
pairings shuffled within season. Shuffling preserves both margins and destroys
only who-faced-whom, so anything above the permuted band is interaction and
anything inside it is the learner fitting noise.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

GLLVM = HERE.parent / "gllvm"
K = len(common.CATEGORIES)
ROUNDS = 300
LR = 0.10
MAX_DEPTH = 4
MIN_LEAF = 200
PATIENCE = 20
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def hand_state(bats, throws):
    if not bats or not throws:
        return "unknown"
    if bats == "S":
        return "opposite"
    return "same" if bats == throws else "opposite"


def additive_offset_and_coords(rows, z, bat_idx, pit_idx, season_idx, n_bat, n_pit, axes):
    d = z["Lbat"].shape[1]
    bi = np.array([bat_idx.get(r["batter"], n_bat) for r in rows])
    pi = np.array([pit_idx.get(r["pitcher"], n_pit) for r in rows])
    p = 3 + (len(season_idx) - 1)
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
    Ab = np.vstack([axes["bat"], np.zeros((1, axes["bat"].shape[1]))])[bi]
    Ap = np.vstack([axes["pit"], np.zeros((1, axes["pit"].shape[1]))])[pi]
    y = np.array([r["y"] for r in rows])
    return offset, Ab, Ap, y, Xs


def softmax(eta):
    e = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(e)
    return ex / ex.sum(axis=1, keepdims=True)


def deviance(eta, y):
    P = softmax(eta)
    return -2.0 * np.mean(np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)))


def boost(Xtr, offtr, ytr, Xva, offva, yva, rounds=ROUNDS, seed=0):
    """Multinomial gradient boosting from a fixed offset. Returns best val deviance."""
    eta_tr, eta_va = offtr.copy(), offva.copy()
    Y = np.zeros((len(ytr), K))
    Y[np.arange(len(ytr)), ytr] = 1.0
    best, best_round, since = deviance(eta_va, yva), 0, 0
    for m in range(1, rounds + 1):
        P = softmax(eta_tr)
        G = Y - P
        for k in range(K):
            t = DecisionTreeRegressor(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                                      random_state=seed + k)
            t.fit(Xtr, G[:, k])
            eta_tr[:, k] += LR * t.predict(Xtr)
            eta_va[:, k] += LR * t.predict(Xva)
        dv = deviance(eta_va, yva)
        if dv < best - 1e-7:
            best, best_round, since = dv, m, 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    return best, best_round


def main():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    log(f"train={len(tr)}")

    z = np.load(GLLVM / "latent.npz", allow_pickle=True)
    bat_ids, pit_ids = list(z["bat_ids"]), list(z["pit_ids"])
    bat_idx = {b: i for i, b in enumerate(bat_ids)}
    pit_idx = {p: i for i, p in enumerate(pit_ids)}
    season_idx = {s: i for i, s in enumerate([2024, 2025, 2026])}
    axes = {}
    for side in ("bat", "pit"):
        B = z[f"L{side}"] @ z[f"F{side}"].T
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        axes[side] = (U * S)[:, :5]

    off, Ab, Ap, y, Xs = additive_offset_and_coords(
        tr, z, bat_idx, pit_idx, season_idx, len(bat_ids), len(pit_ids), axes)

    # inner split by game -- boosting needs a validation set for early stopping
    g = sorted(train_g)
    rs = np.random.RandomState(11)
    rs.shuffle(g)
    itr_g = set(g[: int(len(g) * 0.8)])
    m = np.array([r["game_id"] in itr_g for r in tr])
    log(f"inner: fit={m.sum()} val={(~m).sum()}")

    X = np.hstack([Ab, Ap])          # 5 batter axes + 5 pitcher axes
    base_val = deviance(off[~m], y[~m])
    log(f"additive-only val deviance = {base_val:.5f}   <-- the bar")

    real, rnd = boost(X[m], off[m], y[m], X[~m], off[~m], y[~m])
    log(f"REAL boosted val deviance = {real:.5f}  delta {real-base_val:+.5f}  "
        f"(best round {rnd})")

    seasons = np.array([r["season"] for r in tr])
    perms = []
    for rep in range(5):
        rr = np.random.RandomState(500 + rep)
        Ap_p = Ap.copy()
        for s in (2024, 2025, 2026):
            idx = np.where(seasons == s)[0]
            Ap_p[idx] = Ap[rr.permutation(idx)]
        Xp = np.hstack([Ab, Ap_p])
        dv, r_ = boost(Xp[m], off[m], y[m], Xp[~m], off[~m], y[~m], seed=100 * rep)
        perms.append(float(dv - base_val))
        log(f"  perm {rep}: delta {dv-base_val:+.5f} (round {r_})")

    P = np.array(perms)
    zsc = ((real - base_val) - P.mean()) / max(P.std(), 1e-12)
    log(f"real delta {real-base_val:+.5f}   perm mean {P.mean():+.5f} sd {P.std():.5f}   z={zsc:+.2f}")
    log("(negative delta = improvement; real must be BELOW the permuted band to mean anything)")

    (HERE / "result.json").write_text(json.dumps(dict(
        model="multinomial_gradient_boosting_on_additive_offset",
        val_additive=float(base_val), val_real=float(real),
        real_delta=float(real - base_val), best_round=int(rnd),
        perm_deltas=perms, perm_mean=float(P.mean()), perm_sd=float(P.std()),
        z_vs_perm=float(zsc), rounds=ROUNDS, lr=LR, max_depth=MAX_DEPTH,
        runtime_sec=time.time() - T0), indent=1) + "\n")
    log("wrote result.json")


if __name__ == "__main__":
    main()
