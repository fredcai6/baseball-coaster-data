"""Two closing questions: does the interaction generalise, and is it a link effect?

PHASE 1 -- HELD-OUT PAIRS. Every split so far has been by GAME, so the booster
has usually seen this batter against this pitcher before. Split by PAIR
instead: train the interaction on one set of batter-pitcher matchups and
evaluate on matchups it has never seen, though it has seen both players
extensively against everyone else.

  * gap persists on unseen pairs  -> structural, predictable from who the
    players are, and real even though it is not style-shaped
  * gap vanishes                  -> pair-level memorisation, effect dissolves

The additive offset may safely have seen every pair: it is additive, so a pair
tells it nothing pair-specific. Only the BOOSTER needs the pair firewall.

PHASE 2 -- LATENT NONLINEARITY. If the truth is eta = g(additive predictor) for
some nonlinear g rather than the additive predictor itself, then on our scale
the curvature of g appears as interaction, and a tree will find it. That
hypothesis makes a sharp prediction: the interaction should be a function of
the PREDICTION, not of the two players' coordinates separately.

So boost using ONLY the offset values as features -- no player coordinates at
all. Such a model can represent nothing except a recalibration map g(eta). If
recalibration alone captures most of the -0.01410, the "interaction" is a link
mismatch, not matchup structure.

This also explains a pattern the axis sweep found: the gap grew nearly linearly
with the number of latent axes included (top-1 -0.0005, top-5 -0.0141). More
axes means a better-resolved eta, hence more of g's curvature recoverable --
exactly what a link nonlinearity predicts, and NOT what a low-dimensional style
interaction predicts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "boost"))
sys.path.insert(0, str(HERE.parent / "fuse"))
import common  # noqa: E402
import fit as BF  # noqa: E402
import analyze as FZ  # noqa: E402
from control import boost_additive  # noqa: E402

SEP = HERE.parent / "nested_sep"
GLLVM = HERE.parent / "gllvm"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def setup():
    rows = common.load_pa()
    train_g, _ = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    season_idx = {s: i for i, s in enumerate([2024, 2025, 2026])}
    y = np.array([r["y"] for r in tr])
    z = np.load(GLLVM / "latent.npz", allow_pickle=True)
    axes = {}
    for side in ("bat", "pit"):
        B = z[f"L{side}"] @ z[f"F{side}"].T
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        axes[side] = (U * S)[:, :5]
    bat_ids, pit_ids = list(z["bat_ids"]), list(z["pit_ids"])
    _, Ab, Ap, _, _ = BF.additive_offset_and_coords(
        tr, z, {b: i for i, b in enumerate(bat_ids)},
        {p: i for i, p in enumerate(pit_ids)}, season_idx,
        len(bat_ids), len(pit_ids), axes)
    zA = np.load(SEP / "latent.npz", allow_pickle=True)
    PA = FZ.nested_category_probs(tr, zA, False, season_idx)
    off = np.log(np.maximum(PA, 1e-300))
    return tr, y, Ab, Ap, off


def main():
    tr, y, Ab, Ap, off = setup()
    X = np.hstack([Ab, Ap])
    out = {}

    # ---------------- PHASE 1: held-out PAIRS ---------------------------
    pairs = np.array([f"{r['batter']}|{r['pitcher']}" for r in tr])
    uniq = np.unique(pairs)
    rs = np.random.RandomState(4242)
    held = set(rs.choice(uniq, size=int(0.2 * len(uniq)), replace=False))
    m = np.array([p not in held for p in pairs])       # True = booster trains here
    log(f"PHASE 1: {len(uniq)} distinct pairs, {len(held)} held out; "
        f"rows fit={m.sum()} eval={(~m).sum()}")

    base = BF.deviance(off[~m], y[~m])
    add = boost_additive(Ab[m], Ap[m], off[m], y[m], Ab[~m], Ap[~m], off[~m], y[~m])
    joint, rnd = BF.boost(X[m], off[m], y[m], X[~m], off[~m], y[~m])
    gap_pairs = joint - add
    log(f"  offset-only={base:.5f} additive={add:.5f} joint={joint:.5f}")
    log(f"  GAP ON UNSEEN PAIRS = {gap_pairs:+.5f}   (game-split gap was -0.01410)")
    log(f"  -> {'GENERALISES: structural' if gap_pairs < -0.004 else 'DISSOLVES: pair memorisation'}")
    out["phase1_pairs"] = dict(n_pairs=int(len(uniq)), n_held=int(len(held)),
                               base=float(base), additive=float(add),
                               joint=float(joint), gap=float(gap_pairs))

    # ---------------- PHASE 2: is it a recalibration? -------------------
    g = sorted({r["game_id"] for r in tr})
    rs2 = np.random.RandomState(11)                   # SAME inner split as before
    rs2.shuffle(g)
    itr = set(g[: int(len(g) * 0.8)])
    mg = np.array([r["game_id"] in itr for r in tr])

    base2 = BF.deviance(off[~mg], y[~mg])
    log(f"PHASE 2: offset-only val = {base2:.5f}")

    # features = the model's OWN predictions. Can only learn g(eta).
    Xcal = off - off.mean(axis=1, keepdims=True)
    cal, rc = BF.boost(Xcal[mg], off[mg], y[mg], Xcal[~mg], off[~mg], y[~mg])
    log(f"  recalibration-only (offset as features) val = {cal:.5f} "
        f"({cal-base2:+.5f}) round {rc}")

    # players + calibration together: does anything remain beyond g(eta)?
    Xboth = np.hstack([X, Xcal])
    both, rb = BF.boost(Xboth[mg], off[mg], y[mg], Xboth[~mg], off[~mg], y[~mg])
    log(f"  players + calibration val = {both:.5f} ({both-base2:+.5f}) round {rb}")

    joint_only, _ = BF.boost(X[mg], off[mg], y[mg], X[~mg], off[~mg], y[~mg])
    log(f"  players only (reference)  val = {joint_only:.5f} ({joint_only-base2:+.5f})")
    log(f"  recalibration explains {100*(cal-base2)/max(1e-12,(joint_only-base2)):.0f}% "
        f"of the players-only gain")
    out["phase2_link"] = dict(base=float(base2), recalibration_only=float(cal),
                              players_only=float(joint_only), both=float(both),
                              recal_share=float((cal - base2) / (joint_only - base2)))

    (HERE / "final_result.json").write_text(json.dumps(out, indent=1) + "\n")
    log("wrote final_result.json")


if __name__ == "__main__":
    main()
