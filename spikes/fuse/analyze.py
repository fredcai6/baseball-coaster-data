"""Is the 'interaction' actually a SCALE artifact? And do A and B differ?

Two questions, one script.

QUESTION 1 -- the link-function question, made decisive.
The boosting test found -0.00748 of interaction in the residual of the FLAT
GLLVM. Variant A (nested, separate latents per gate) beats that same flat model
by 3.95563 - 3.94846 = 0.00717. Those numbers are suspiciously close.

If a process is additive on one scale, then on a DIFFERENT scale it appears as
interaction -- and a tree will happily find it. The flat softmax over 10
categories may simply be the wrong scale, and the sequential gate decomposition
the right one. In that case the "interaction" the trees found is not a matchup
effect at all: it is the sequential structure, showing up as curvature because
we asked the question in the wrong coordinates.

Decisive test: rerun the IDENTICAL additive-vs-joint boosting comparison, but
starting from Variant A's predictions instead of the flat model's. If the gap
collapses toward zero, the interaction was a scale artifact. If it survives,
there is genuine matchup structure that neither scale explains.

QUESTION 2 -- do A and B carry different information?
A wins on deviance; B found a cross-gate axis A structurally cannot express
(one shared dimension reading as fly-ball/power at BOTH the out gate and the
hit gate, fit on non-overlapping rows). If they are making the SAME errors they
are the same model wearing different clothes and fusion buys nothing. If their
errors differ, a blend should beat both -- which is the cheap, atheoretical
evidence that a proper fusion (a correctly-conditioned hierarchical model) is
worth building.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "boost"))   # so control.py finds ITS fit.py
import common  # noqa: E402
import gates as GT  # noqa: E402
import fit as BF  # noqa: E402  (spikes/boost/fit.py)
from control import boost_additive  # noqa: E402

SEP = HERE.parent / "nested_sep"
SHR = HERE.parent / "nested_shared"
GLLVM = HERE.parent / "gllvm"
CATS = common.CATEGORIES
T0 = time.time()

# category -> [(gate, branch_index), ...] path through the tree
PATHS = {}
for ci, cat in enumerate(CATS):
    path = []
    g = "root"
    while True:
        bi = GT.LEAF_TO_BRANCH[g][ci]
        path.append((g, bi))
        bname = GT.GATES[g][bi][0]
        nxt = GT.NEXT.get((g, bname))
        if nxt is None:
            break
        g = nxt
    PATHS[cat] = path


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def hand_state(bats, throws):
    if not bats or not throws:
        return "unknown"
    if bats == "S":
        return "opposite"
    return "same" if bats == throws else "opposite"


def structural(rows, season_idx):
    p = 3 + (len(season_idx) - 1)
    Xs = np.zeros((len(rows), p))
    # Home-field advantage is "was the batting team in its OWN ballpark",
    # not "did it bat last". Those differ on 2,345 PAs (1.86%): the 2025
    # Colorado Springs Sky Sox were the designated home team for 26 games
    # played in the opponent's park, and in every one of them BOTH sides were
    # mislabelled -- the nominal away team really was at home. Using
    # batting_is_home here was a real defect, corrected in schema 1.13.0 once
    # the corpus could finally tell the two apart.
    Xs[:, 0] = [1.0 if r["batting_at_home_park"] else 0.0 for r in rows]
    hs = [hand_state(r["bats"], r["throws"]) for r in rows]
    Xs[:, 1] = [1.0 if h == "opposite" else 0.0 for h in hs]
    Xs[:, 2] = [1.0 if h == "unknown" else 0.0 for h in hs]
    for i, r in enumerate(rows):
        si = season_idx[r["season"]]
        if si > 0:
            Xs[i, 3 + si - 1] = 1.0
    return Xs


def gate_probs(eta):
    e = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(e)
    return ex / ex.sum(axis=1, keepdims=True)


def refit_beta(rows, z, season_idx):
    """Variant B did not save its structural coefficients. Given the latents,
    loadings and intercepts fixed, beta is a small convex fit -- recover it on
    TRAIN rows only. This is the conditional MLE given the rest, which is what
    the block-coordinate fit was solving for anyway."""
    from scipy import optimize
    Xs = structural(rows, season_idx)
    y = np.array([r["y"] for r in rows])
    bidx = {b: i for i, b in enumerate(list(z["bat_ids"]))}
    pidx = {q: i for i, q in enumerate(list(z["pit_ids"]))}
    bi = np.array([bidx.get(r["batter"], len(bidx)) for r in rows])
    pi = np.array([pidx.get(r["pitcher"], len(pidx)) for r in rows])
    assign = GT.assign(y)
    betas = {}
    for g in GT.GATE_ORDER:
        idx, br = assign[g]
        Fb, Fp, al = z[f"Fbat_{g}"], z[f"Fpit_{g}"], z[f"alpha_{g}"]
        Kg = len(al)
        Lbf = np.vstack([z["Lbat"], np.zeros((1, z["Lbat"].shape[1]))])[bi[idx]]
        Lpf = np.vstack([z["Lpit"], np.zeros((1, z["Lpit"].shape[1]))])[pi[idx]]
        fixed = al[None, :] + Lbf @ Fb.T + Lpf @ Fp.T
        Xg, n = Xs[idx], len(idx)

        def og(th):
            B = th.reshape(Xg.shape[1], Kg)
            eta = fixed + Xg @ B
            eta = eta - eta.max(axis=1, keepdims=True)
            ex = np.exp(eta); P = ex / ex.sum(axis=1, keepdims=True)
            nll = -np.sum(np.log(np.maximum(P[np.arange(n), br], 1e-300)))
            D = P.copy(); D[np.arange(n), br] -= 1.0
            return nll + 1e-3 * np.sum(B**2), (Xg.T @ D + 2e-3 * B).ravel()

        r = optimize.minimize(og, np.zeros(Xg.shape[1] * Kg), jac=True,
                              method="L-BFGS-B", options=dict(maxiter=300))
        betas[g] = r.x.reshape(Xg.shape[1], Kg)
    return betas


def nested_category_probs(rows, z, shared, season_idx, betas=None):
    """Evaluate every gate on every row, then multiply along each path."""
    Xs = structural(rows, season_idx)
    batters = [r["batter"] for r in rows]
    pitchers = [r["pitcher"] for r in rows]

    def index_for(bat_ids, pit_ids):
        bidx = {b: i for i, b in enumerate(bat_ids)}
        pidx = {p: i for i, p in enumerate(pit_ids)}
        return (np.array([bidx.get(b, len(bat_ids)) for b in batters]),
                np.array([pidx.get(p, len(pit_ids)) for p in pitchers]))

    if shared:
        bi, pi = index_for(list(z["bat_ids"]), list(z["pit_ids"]))

    P = {}
    for g in GT.GATE_ORDER:
        if not shared:
            # Variant A fit each gate independently, so each gate has its OWN
            # player roster -- a player who never reached a gate is absent from it.
            bi, pi = index_for(list(z[f"{g}_bat_ids"]), list(z[f"{g}_pit_ids"]))
        if shared:
            Lb, Lp = z["Lbat"], z["Lpit"]
            Fb, Fp = z[f"Fbat_{g}"], z[f"Fpit_{g}"]
            al = z[f"alpha_{g}"]
            be = betas[g] if betas is not None else np.zeros((Xs.shape[1], len(al)))
        else:
            Lb, Lp = z[f"{g}_Lbat"], z[f"{g}_Lpit"]
            Fb, Fp = z[f"{g}_Fbat"], z[f"{g}_Fpit"]
            al, be = z[f"{g}_alpha"], z[f"{g}_beta"]
        Lbf = np.vstack([Lb, np.zeros((1, Lb.shape[1]))])[bi]
        Lpf = np.vstack([Lp, np.zeros((1, Lp.shape[1]))])[pi]
        eta = al[None, :] + Xs @ be + Lbf @ Fb.T + Lpf @ Fp.T
        P[g] = gate_probs(eta)

    out = np.ones((len(rows), len(CATS)))
    for ci, cat in enumerate(CATS):
        for (g, b) in PATHS[cat]:
            out[:, ci] *= P[g][:, b]
    return out / out.sum(axis=1, keepdims=True)


def dev(P, y):
    return -2.0 * np.mean(np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)))


def main():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    season_idx = {s: i for i, s in enumerate([2024, 2025, 2026])}
    ytr = np.array([r["y"] for r in tr])
    yte = np.array([r["y"] for r in te])

    zA = np.load(SEP / "latent.npz", allow_pickle=True)
    zB = np.load(SHR / "latent.npz", allow_pickle=True)
    PA_te = nested_category_probs(te, zA, False, season_idx)
    betasB = refit_beta(tr, zB, season_idx)
    log("refit Variant B structural coefficients (it did not save them)")
    PB_te = nested_category_probs(te, zB, True, season_idx, betasB)
    log(f"reconstructed A test deviance = {dev(PA_te, yte):.5f}  (reported 3.94846)")
    log(f"reconstructed B test deviance = {dev(PB_te, yte):.5f}  (reported 3.95105)")

    # ---- Q2: do they carry different information? ----------------------
    corr = np.corrcoef(np.log(PA_te).ravel(), np.log(PB_te).ravel())[0, 1]
    log(f"corr(log P_A, log P_B) over all test cells = {corr:.5f}")
    best_w, best_d = None, 1e9
    for w in np.linspace(0, 1, 21):
        d = dev(w * PA_te + (1 - w) * PB_te, yte)
        if d < best_d:
            best_w, best_d = float(w), float(d)
    log(f"best blend w*A+(1-w)*B : w={best_w:.2f} deviance={best_d:.5f}   "
        f"(A {dev(PA_te,yte):.5f}, B {dev(PB_te,yte):.5f})")
    log(f"  blend gain over best single = {best_d - min(dev(PA_te,yte), dev(PB_te,yte)):+.5f}")

    # ---- Q1: is the boosted interaction a scale artifact? --------------
    PA_tr = nested_category_probs(tr, zA, False, season_idx)
    offA = np.log(np.maximum(PA_tr, 1e-300))

    z = np.load(GLLVM / "latent.npz", allow_pickle=True)
    bat_ids, pit_ids = list(z["bat_ids"]), list(z["pit_ids"])
    axes = {}
    for side in ("bat", "pit"):
        B = z[f"L{side}"] @ z[f"F{side}"].T
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        axes[side] = (U * S)[:, :5]
    _, Ab, Ap, _, _ = BF.additive_offset_and_coords(
        tr, z, {b: i for i, b in enumerate(bat_ids)},
        {p: i for i, p in enumerate(pit_ids)}, season_idx,
        len(bat_ids), len(pit_ids), axes)

    g = sorted(train_g)
    rs = np.random.RandomState(11)          # SAME inner split as boost/control.py
    rs.shuffle(g)
    itr = set(g[: int(len(g) * 0.8)])
    m = np.array([r["game_id"] in itr for r in tr])
    X = np.hstack([Ab, Ap])

    base = BF.deviance(offA[~m], ytr[~m])
    log(f"[on A's offset] offset-only val = {base:.5f}")
    add = boost_additive(Ab[m], Ap[m], offA[m], ytr[m], Ab[~m], Ap[~m], offA[~m], ytr[~m])
    log(f"[on A's offset] additive-by-construction val = {add:.5f} ({add-base:+.5f})")
    full, rnd = BF.boost(X[m], offA[m], ytr[m], X[~m], offA[~m], ytr[~m])
    log(f"[on A's offset] joint (interaction-capable) val = {full:.5f} ({full-base:+.5f}) round {rnd}")
    gap = full - add
    log(f"[on A's offset] INTERACTION GAP = {gap:+.5f}      (on flat offset it was -0.00748)")
    log(f"  -> {'SCALE ARTIFACT: gap collapsed' if gap > -0.002 else 'SURVIVES: genuine matchup structure'}")

    (HERE / "result.json").write_text(json.dumps(dict(
        A_test=float(dev(PA_te, yte)), B_test=float(dev(PB_te, yte)),
        corr_logP=float(corr), blend_w=best_w, blend_deviance=best_d,
        blend_gain=float(best_d - min(dev(PA_te, yte), dev(PB_te, yte))),
        gap_on_A_offset=float(gap), gap_on_flat_offset=-0.00748,
        val_offset_only=float(base), val_additive=float(add), val_joint=float(full),
        runtime_sec=time.time() - T0), indent=1) + "\n")
    log("wrote result.json")


if __name__ == "__main__":
    main()
