"""Parametric bootstrap: is the -0.00748 'interaction' real, or procedural?

`control.py` compared an additive-by-construction booster against an
interaction-capable one and found the joint arm 0.00748 better on held-out
data. That is the first positive signal in this project, which is exactly why
it needs a null before it is believed.

The worry is specific and mundane: the additive arm stops on early-stopping
patience. If it quit on a plateau rather than at a true optimum, part of that
gap is the CONTROL being undertrained, not interaction in the data.

So: simulate outcomes FROM the fitted additive model. Draw each plate
appearance's category from that model's own predicted probabilities. In the
simulated world there is provably NO interaction -- the generating process is
additive by construction. Then run the identical additive-vs-joint comparison.

  * simulated gap ~= -0.0075  ->  the gap is an artifact of the procedure
  * simulated gap ~= 0        ->  the real gap is real

Same split, same features, same hyperparameters, same early stopping. Only the
outcomes change.
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
from fit import additive_offset_and_coords, boost, deviance, softmax  # noqa: E402
from control import boost_additive  # noqa: E402

GLLVM = HERE.parent / "gllvm"
REPS = 4
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


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
    off, Ab, Ap, y_real, _ = additive_offset_and_coords(
        tr, z, bat_idx, pit_idx, season_idx, len(bat_ids), len(pit_ids), axes)

    g = sorted(train_g)
    rs = np.random.RandomState(11)          # SAME inner split as control.py
    rs.shuffle(g)
    itr_g = set(g[: int(len(g) * 0.8)])
    m = np.array([r["game_id"] in itr_g for r in tr])
    X = np.hstack([Ab, Ap])

    P = softmax(off)                        # the additive model's own beliefs
    cum = P.cumsum(axis=1)
    gaps = []
    for rep in range(REPS):
        rr = np.random.RandomState(9000 + rep)
        u = rr.random_sample(len(P))[:, None]
        y = (u > cum).sum(axis=1)           # inverse-CDF multinomial draw
        y = np.clip(y, 0, P.shape[1] - 1)
        base = deviance(off[~m], y[~m])
        add = boost_additive(Ab[m], Ap[m], off[m], y[m], Ab[~m], Ap[~m], off[~m], y[~m])
        full, rnd = boost(X[m], off[m], y[m], X[~m], off[~m], y[~m])
        gap = full - add
        gaps.append(float(gap))
        log(f"rep {rep}: base={base:.5f} additive={add:.5f} joint={full:.5f} "
            f"GAP={gap:+.5f} (round {rnd})")

    G = np.array(gaps)
    real_gap = -0.00748
    zsc = (real_gap - G.mean()) / max(G.std(), 1e-12)
    log(f"simulated gaps: mean {G.mean():+.5f}  sd {G.std():.5f}  "
        f"min {G.min():+.5f} max {G.max():+.5f}")
    log(f"REAL gap {real_gap:+.5f}   z vs simulated null = {zsc:+.2f}")
    log("(real gap must be MORE negative than the simulated band to be real)")

    (HERE / "bootstrap_result.json").write_text(json.dumps(dict(
        model="parametric_bootstrap_under_additive_null",
        simulated_gaps=gaps, sim_mean=float(G.mean()), sim_sd=float(G.std()),
        real_gap=real_gap, z_vs_sim=float(zsc), reps=REPS,
        runtime_sec=time.time() - T0), indent=1) + "\n")
    log("wrote bootstrap_result.json")


if __name__ == "__main__":
    main()
