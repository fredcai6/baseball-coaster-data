"""Test 1 -- split-half reliability. Loads full.npz / halfA.npz / halfB.npz,
aligns half B onto half A by orthogonal Procrustes (batters and pitchers
separately, players with >=30 PA/BF in BOTH halves only), reports per-axis
Pearson r with Spearman-Brown correction, and the alignment-free per-node
predicted-effect correlation (L @ f.T). Fast (seconds) -- no fitting here.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_style as CS

log = CS.log
FLOOR = 30


def analyze_side(idA, coordsA, idB, coordsB, predA_full, predB_full, paA, paB,
                  floor, label):
    """idA/idB: full id arrays for that side in half A / half B fits (same
    universe, since BI/PI are shared -- so idA == idB elementwise, but we
    keep both for clarity). coordsA/coordsB: (n, d) latent coords. predA/predB:
    (n, NNODE) predicted node-effect (L @ f.T or M @ g.T), alignment-free.
    paA/paB: per-player PA/BF counts in that half.
    """
    assert list(idA) == list(idB), "player universe must match between halves"
    ids = np.array(idA)
    mask = (paA >= floor) & (paB >= floor)
    n_survive = int(mask.sum())
    log(f"  [{label}] players with >={floor} in BOTH halves: {n_survive} / {len(ids)}")

    A = coordsA[mask]; B = coordsB[mask]
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    R, B_aligned = CS.orthogonal_procrustes_align(Ac, Bc)

    d = A.shape[1]
    axis_r = []
    for k in range(d):
        r = CS.pearson(Ac[:, k], B_aligned[:, k])
        sb = CS.spearman_brown(r) if r == r else float("nan")
        axis_r.append(dict(axis=k, r_half=r, r_spearman_brown=sb))
        log(f"  [{label}] axis {k}: split-half r={r:.3f}  Spearman-Brown r_full={sb:.3f}")

    # alignment-free per-node predicted-effect correlation
    predA = predA_full[mask]; predB = predB_full[mask]
    node_r = []
    for n, nm in enumerate(CS.NODE_NAMES):
        r = CS.pearson(predA[:, n], predB[:, n])
        sb = CS.spearman_brown(r) if r == r else float("nan")
        node_r.append(dict(node=nm, r_half=r, r_spearman_brown=sb))
        log(f"  [{label}] node {nm:9} predicted-effect split-half r={r:.3f}  "
            f"Spearman-Brown r_full={sb:.3f}")

    return dict(n_survive=n_survive, n_total=len(ids), floor=floor,
                rotation=R.tolist(), axis_r=axis_r, node_r=node_r)


def main():
    log("=" * 70)
    log("TEST 1: split-half reliability")
    full = np.load(os.path.join(HERE, "full.npz"), allow_pickle=True)
    A = np.load(os.path.join(HERE, "halfA.npz"), allow_pickle=True)
    B = np.load(os.path.join(HERE, "halfB.npz"), allow_pickle=True)

    bat_ids = full["bat_ids"]; pit_ids = full["pit_ids"]
    assert list(A["bat_ids"]) == list(bat_ids) == list(B["bat_ids"])
    assert list(A["pit_ids"]) == list(pit_ids) == list(B["pit_ids"])

    predA_bat = A["L"] @ A["f"].T   # (n_bat, NNODE)
    predB_bat = B["L"] @ B["f"].T
    predA_pit = A["M"] @ A["g"].T
    predB_pit = B["M"] @ B["g"].T

    out = CS.load_result()
    out["test1"] = {}
    out["test1"]["batters"] = analyze_side(
        bat_ids, A["L"], bat_ids, B["L"], predA_bat, predB_bat,
        A["bat_pa"], B["bat_pa"], FLOOR, "batters")
    out["test1"]["pitchers"] = analyze_side(
        pit_ids, A["M"], pit_ids, B["M"], predA_pit, predB_pit,
        A["pit_pa"], B["pit_pa"], FLOOR, "pitchers")
    out["test1"]["floor_pa"] = FLOOR
    out["test1"]["procrustes"] = "orthogonal (rotation/reflection only, no scaling) " \
        "-- Pearson r is scale-invariant so the optimal rotation from " \
        "SVD(B^T A) is identical whether or not a scale factor is also fit; " \
        "only rotation resolves axis-mixing/reflection, which is what matters " \
        "for per-axis correlation."
    CS.save_result(out)


if __name__ == "__main__":
    main()
