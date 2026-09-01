"""Where is the interaction, and what shape is it?

We now know a real, non-bilinear, non-cluster interaction exists: boosting on
Variant A's sequential offset beats an additive-by-construction control by
-0.01410, about a quarter of the entire additive signal, and it survives both a
parametric bootstrap (simulated gap -0.00002) and a 5x-patience check.

But "a flexible learner found something" is not a finding. Two questions:

PART 1 -- WHERE. Run the identical additive-vs-joint comparison separately at
each gate of the sequential tree, on that gate's own rows and branches. The
interaction has to live somewhere: in whether the ball is put in play at all,
in whether contact falls in (the BABIP gate, where DIPS says nobody has
control), or in extra-base power. Each gate is its own scale with its own rows,
so this localises the effect instead of averaging it away.

PART 2 -- WHAT SHAPE. Interrogate the fitted trees. A tree that splits on a
batter axis and then a pitcher axis has carved out a genuine two-sided rule;
counting which axis PAIRS co-occur on root-to-leaf paths says which style
dimensions interact. Then evaluate the fitted ensemble over a grid of the top
pair to get the actual interaction surface, per outcome category.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "boost"))
sys.path.insert(0, str(HERE.parent / "fuse"))
import common  # noqa: E402
import gates as GT  # noqa: E402
import fit as BF  # noqa: E402
import analyze as FZ  # noqa: E402

SEP = HERE.parent / "nested_sep"
GLLVM = HERE.parent / "gllvm"
LR, DEPTH, LEAF, PATIENCE = 0.10, 4, 200, 20
T0 = time.time()
AXNAMES = [f"B{i+1}" for i in range(5)] + [f"P{i+1}" for i in range(5)]


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def gate_branch_probs(rows, z, season_idx):
    """Variant A's fitted conditional branch probabilities, every gate, every row."""
    Xs = FZ.structural(rows, season_idx)
    batters = [r["batter"] for r in rows]
    pitchers = [r["pitcher"] for r in rows]
    P = {}
    for g in GT.GATE_ORDER:
        bids, pids = list(z[f"{g}_bat_ids"]), list(z[f"{g}_pit_ids"])
        bidx = {b: i for i, b in enumerate(bids)}
        pidx = {q: i for i, q in enumerate(pids)}
        bi = np.array([bidx.get(b, len(bids)) for b in batters])
        pi = np.array([pidx.get(q, len(pids)) for q in pitchers])
        Lb, Lp = z[f"{g}_Lbat"], z[f"{g}_Lpit"]
        Lbf = np.vstack([Lb, np.zeros((1, Lb.shape[1]))])[bi]
        Lpf = np.vstack([Lp, np.zeros((1, Lp.shape[1]))])[pi]
        eta = z[f"{g}_alpha"][None, :] + Xs @ z[f"{g}_beta"] + Lbf @ z[f"{g}_Fbat"].T \
            + Lpf @ z[f"{g}_Fpit"].T
        P[g] = FZ.gate_probs(eta)
    return P


def dev_k(eta, y):
    e = eta - eta.max(axis=1, keepdims=True)
    ex = np.exp(e); Pp = ex / ex.sum(axis=1, keepdims=True)
    return -2.0 * np.mean(np.log(np.maximum(Pp[np.arange(len(y)), y], 1e-300)))


def boost_gate(Xf, Xa_b, Xa_p, off, y, mfit, additive_only, rounds=300, keep=False):
    """Boost one gate. additive_only alternates single-side features."""
    K = off.shape[1]
    eta_t, eta_v = off[mfit].copy(), off[~mfit].copy()
    yt, yv = y[mfit], y[~mfit]
    Y = np.zeros((len(yt), K)); Y[np.arange(len(yt)), yt] = 1.0
    best, since, trees = dev_k(eta_v, yv), 0, []
    feats = [(Xa_b[mfit], Xa_b[~mfit]), (Xa_p[mfit], Xa_p[~mfit])] if additive_only \
        else [(Xf[mfit], Xf[~mfit])]
    r = 0
    while r < rounds:
        for (Ft, Fv) in feats:
            for _ in range(60 if additive_only else 1):
                P = np.exp(eta_t - eta_t.max(1, keepdims=True))
                P /= P.sum(1, keepdims=True)
                G = Y - P
                for k in range(K):
                    t = DecisionTreeRegressor(max_depth=DEPTH, min_samples_leaf=LEAF,
                                              random_state=r * 31 + k)
                    t.fit(Ft, G[:, k])
                    eta_t[:, k] += LR * t.predict(Ft)
                    eta_v[:, k] += LR * t.predict(Fv)
                    if keep:
                        trees.append((k, t))
                r += 1
                d = dev_k(eta_v, yv)
                if d < best - 1e-7:
                    best, since = d, 0
                else:
                    since += 1
                    if since >= PATIENCE:
                        return (best, trees) if keep else best
                if r >= rounds:
                    break
    return (best, trees) if keep else best


def path_feature_pairs(tree):
    """Which feature pairs co-occur on a root-to-leaf path? That IS interaction."""
    t = tree.tree_
    pairs, solo = Counter(), Counter()

    def walk(node, seen):
        if t.children_left[node] == -1:
            f = sorted(seen)
            for a in f:
                solo[a] += 1
            for i in range(len(f)):
                for j in range(i + 1, len(f)):
                    pairs[(f[i], f[j])] += 1
            return
        nf = t.feature[node]
        walk(t.children_left[node], seen | {nf})
        walk(t.children_right[node], seen | {nf})

    walk(0, set())
    return pairs, solo


def main():
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
    Xf = np.hstack([Ab, Ap])

    g = sorted(train_g)
    rs = np.random.RandomState(11)
    rs.shuffle(g)
    itr = set(g[: int(len(g) * 0.8)])
    m = np.array([r["game_id"] in itr for r in tr])

    zA = np.load(SEP / "latent.npz", allow_pickle=True)
    GP = gate_branch_probs(tr, zA, season_idx)
    assign = GT.assign(y)

    # ---------------- PART 1: where does the interaction live? -----------
    log("PART 1 -- interaction gap, gate by gate")
    per_gate, results = {}, {}
    for gname in GT.GATE_ORDER:
        idx, br = assign[gname]
        off = np.log(np.maximum(GP[gname][idx], 1e-300))
        mg = m[idx]
        if mg.sum() < 500 or (~mg).sum() < 200:
            continue
        base = dev_k(off[~mg], br[~mg])
        add = boost_gate(Xf[idx], Ab[idx], Ap[idx], off, br, mg, True)
        joint = boost_gate(Xf[idx], Ab[idx], Ap[idx], off, br, mg, False)
        gap = joint - add
        per_gate[gname] = dict(n_rows=int(len(idx)), n_branches=int(off.shape[1]),
                               base=float(base), additive=float(add),
                               joint=float(joint), gap=float(gap))
        log(f"  {gname:8s} rows={len(idx):6d} br={off.shape[1]}  base={base:.5f} "
            f"add={add:.5f} joint={joint:.5f}  GAP={gap:+.5f}")
    results["per_gate"] = per_gate
    tot = sum(v["gap"] * v["n_rows"] for v in per_gate.values()) / len(tr)
    log(f"  PA-weighted total gap across gates = {tot:+.5f}")

    # ---------------- PART 2: what shape is it? --------------------------
    log("PART 2 -- interrogating the trees fit on the FULL sequential offset")
    PA = FZ.nested_category_probs(tr, zA, False, season_idx)
    offA = np.log(np.maximum(PA, 1e-300))
    best, trees = boost_gate(Xf, Ab, Ap, offA, y, m, False, keep=True)
    log(f"  joint model val deviance = {best:.5f}, {len(trees)} trees kept")

    pairs, solo = Counter(), Counter()
    imp = np.zeros(10)
    for k, t in trees:
        p, s = path_feature_pairs(t)
        pairs.update(p); solo.update(s)
        imp += t.tree_.compute_feature_importances(normalize=False)
    imp = imp / imp.sum()
    log("  feature importance:")
    for i in np.argsort(-imp):
        log(f"    {AXNAMES[i]:4s} {imp[i]:.4f}")
    cross = {k: v for k, v in pairs.items() if (k[0] < 5) != (k[1] < 5)}
    within = {k: v for k, v in pairs.items() if (k[0] < 5) == (k[1] < 5)}
    log(f"  co-occurring pairs on a path: {sum(cross.values())} CROSS-side "
        f"(true interaction) vs {sum(within.values())} within-side")
    log("  top cross-side axis pairs:")
    top = sorted(cross.items(), key=lambda kv: -kv[1])[:8]
    for (a, b), c in top:
        log(f"    {AXNAMES[a]:4s} x {AXNAMES[b]:4s} : {c}")
    results["importance"] = {AXNAMES[i]: float(imp[i]) for i in range(10)}
    results["cross_pairs"] = {f"{AXNAMES[a]}x{AXNAMES[b]}": int(c)
                              for (a, b), c in sorted(cross.items(), key=lambda kv: -kv[1])}
    results["cross_total"] = int(sum(cross.values()))
    results["within_total"] = int(sum(within.values()))

    # interaction surface over the top cross pair
    if top:
        (fa, fb), _ = top[0]
        qa = np.quantile(Xf[:, fa], np.linspace(0.05, 0.95, 7))
        qb = np.quantile(Xf[:, fb], np.linspace(0.05, 0.95, 7))
        base_row = Xf.mean(axis=0)
        surf = np.zeros((7, 7, 10))
        for i, va in enumerate(qa):
            for j, vb in enumerate(qb):
                x = base_row.copy(); x[fa] = va; x[fb] = vb
                e = np.zeros(10)
                for k, t in trees:
                    e[k] += LR * t.predict(x[None, :])[0]
                surf[i, j] = e
        results["surface"] = dict(axis_a=AXNAMES[fa], axis_b=AXNAMES[fb],
                                  grid_a=qa.tolist(), grid_b=qb.tolist(),
                                  eta_delta=surf.tolist(), categories=common.CATEGORIES)
        log(f"  interaction surface over {AXNAMES[fa]} x {AXNAMES[fb]}:")
        for ci, cat in enumerate(common.CATEGORIES):
            rng = surf[:, :, ci].max() - surf[:, :, ci].min()
            corners = (surf[0, 0, ci], surf[0, -1, ci], surf[-1, 0, ci], surf[-1, -1, ci])
            if rng > 0.02:
                log(f"    {cat:5s} range {rng:.3f}  corners(lo/lo,lo/hi,hi/lo,hi/hi) = "
                    + " ".join(f"{c:+.3f}" for c in corners))

    (HERE / "result.json").write_text(json.dumps(results, indent=1) + "\n")
    log("wrote result.json")


if __name__ == "__main__":
    main()
