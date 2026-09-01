"""The control the permutation test should have been.

`fit.py` compared boosting on real pairings against boosting on pitcher
coordinates shuffled within season. That is confounded: shuffling destroys the
pairing AND the pitcher's own residual main effect, so the real run could
exploit both sides' un-shrunk main effects while the permuted run could only
use the batter's. The gap therefore mixes pitcher main effects with
interaction and cannot be read as interaction alone.

The correct control is a learner with the same capacity that is ADDITIVE BY
CONSTRUCTION. Boost on batter coordinates alone, then on pitcher coordinates
alone, alternating to convergence: trees never see both sides at once, so the
fitted function is exactly f(batter) + g(pitcher) with no cross terms, while
still absorbing every bit of residual main effect the shrunken GLLVM left
behind. Then boost on the two sides JOINTLY, where a tree can split on batter
depth 1 and pitcher depth 2 and thereby represent interaction.

The difference between the two is interaction, with residual main effects
already spent in both arms.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
from fit import (K, additive_offset_and_coords, boost, deviance, softmax,  # noqa: E402
                 LR, MAX_DEPTH, MIN_LEAF, PATIENCE)
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

GLLVM = HERE.parent / "gllvm"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def boost_additive(Ab_tr, Ap_tr, off_tr, y_tr, Ab_va, Ap_va, off_va, y_va,
                   passes=6, rounds_per=60):
    """Alternate batter-only / pitcher-only boosting. Cannot represent interaction."""
    eta_tr, eta_va = off_tr.copy(), off_va.copy()
    Y = np.zeros((len(y_tr), K))
    Y[np.arange(len(y_tr)), y_tr] = 1.0
    best, since = deviance(eta_va, y_va), 0
    best_eta_va = eta_va.copy()
    for p in range(passes):
        for side, (Xt, Xv) in enumerate(((Ab_tr, Ab_va), (Ap_tr, Ap_va))):
            for _ in range(rounds_per):
                P = softmax(eta_tr)
                G = Y - P
                for k in range(K):
                    t = DecisionTreeRegressor(max_depth=MAX_DEPTH,
                                              min_samples_leaf=MIN_LEAF,
                                              random_state=p * 100 + side * 10 + k)
                    t.fit(Xt, G[:, k])
                    eta_tr[:, k] += LR * t.predict(Xt)
                    eta_va[:, k] += LR * t.predict(Xv)
                dv = deviance(eta_va, y_va)
                if dv < best - 1e-7:
                    best, since = dv, 0
                    best_eta_va = eta_va.copy()
                else:
                    since += 1
                    if since >= PATIENCE:
                        log(f"  additive control converged: pass {p} side {side} val={best:.5f}")
                        return best
    return best


def main():
    rows = common.load_pa()
    train_g, _ = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
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
    off, Ab, Ap, y, _ = additive_offset_and_coords(
        tr, z, bat_idx, pit_idx, season_idx, len(bat_ids), len(pit_ids), axes)

    g = sorted(train_g)
    rs = np.random.RandomState(11)          # SAME inner split as fit.py
    rs.shuffle(g)
    itr_g = set(g[: int(len(g) * 0.8)])
    m = np.array([r["game_id"] in itr_g for r in tr])

    base = deviance(off[~m], y[~m])
    log(f"additive-GLLVM-only val deviance = {base:.5f}")

    add = boost_additive(Ab[m], Ap[m], off[m], y[m], Ab[~m], Ap[~m], off[~m], y[~m])
    log(f"ADDITIVE-BY-CONSTRUCTION boost val = {add:.5f}  delta vs offset {add-base:+.5f}")

    X = np.hstack([Ab, Ap])
    full, rnd = boost(X[m], off[m], y[m], X[~m], off[~m], y[~m])
    log(f"JOINT (interaction-capable) boost val = {full:.5f}  delta vs offset {full-base:+.5f}"
        f"  (round {rnd})")
    log(f"INTERACTION = joint - additive = {full-add:+.5f}   "
        f"{'<-- negative means real interaction' if full < add else '<-- no interaction'}")

    (HERE / "control_result.json").write_text(json.dumps(dict(
        val_offset_only=float(base), val_additive_boost=float(add),
        val_joint_boost=float(full), interaction_delta=float(full - add),
        runtime_sec=time.time() - T0), indent=1) + "\n")
    log("wrote control_result.json")


if __name__ == "__main__":
    main()
