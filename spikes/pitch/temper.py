"""Is the recalibration CURVATURE, or just an overconfident fit being shrunk?

Phase 2 found that boosting on the offset alone -- features that know nothing
about who is batting or pitching -- beats boosting on the player coordinates
(recal_share = 1.449). That is consistent with a latent nonlinearity, but it
is ALSO consistent with something far more boring: Variant A's held-out
probabilities being systematically too extreme, with the booster simply
learning to shrink them back.

The two have very different consequences and a cheap test tells them apart.
Temperature scaling can express shrinkage and NOTHING else:

    p ~ exp(eta / T)        T > 1 shrinks toward uniform, T < 1 sharpens

If a single scalar T recovers most of the -0.02346, the finding is
calibration repair of one particular fit, and says nothing about the link.
Ladder, in increasing capacity:

    scalar T          1 param    pure shrinkage
    vector a_k        10 params  per-category shrinkage
    affine a_k, b_k   20 params  + per-category intercept (prior correction)
    boosted g         nonparam   whatever is left is real curvature

Uses the SAME inner split as final.py Phase 2 (RandomState(11), 80/20 by game)
so the numbers are directly comparable to base=3.85776, recal=3.83430.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "boost"))
sys.path.insert(0, str(HERE.parent / "fuse"))
import common
import fit as BF
import analyze as FZ

SEP = HERE.parent / "nested_sep"


def dev(logits, y):
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    return -2.0 * np.mean(logits[np.arange(len(y)), y] - lse)


def main():
    rows = common.load_pa()
    train_g, _ = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    season_idx = {s: i for i, s in enumerate([2024, 2025, 2026])}
    y = np.array([r["y"] for r in tr])
    zA = np.load(SEP / "latent.npz", allow_pickle=True)
    off = np.log(np.maximum(FZ.nested_category_probs(tr, zA, False, season_idx), 1e-300))

    g = sorted({r["game_id"] for r in tr})
    rs2 = np.random.RandomState(11)
    rs2.shuffle(g)
    itr = set(g[: int(len(g) * 0.8)])
    mg = np.array([r["game_id"] in itr for r in tr])

    A, B = off[mg], off[~mg]
    ya, yb = y[mg], y[~mg]
    base = dev(B, yb)
    print(f"offset-only val            = {base:.5f}   (final.py: 3.85776)")

    # --- scalar temperature
    def f_T(t):
        return dev(A / np.exp(t[0]), ya)
    r = minimize(f_T, [0.0], method="Nelder-Mead")
    T = float(np.exp(r.x[0]))
    d_T = dev(B / T, yb)
    print(f"scalar temperature  T={T:.4f}   val = {d_T:.5f} ({d_T-base:+.5f})")

    # --- per-category scale
    def f_a(a):
        return dev(A * a, ya)
    r = minimize(f_a, np.ones(off.shape[1]), method="L-BFGS-B")
    a = r.x
    d_a = dev(B * a, yb)
    print(f"per-category scale         val = {d_a:.5f} ({d_a-base:+.5f})")

    # --- per-category affine
    def f_ab(p):
        k = off.shape[1]
        return dev(A * p[:k] + p[k:], ya)
    r = minimize(f_ab, np.concatenate([np.ones(off.shape[1]), np.zeros(off.shape[1])]),
                 method="L-BFGS-B")
    k = off.shape[1]
    d_ab = dev(B * r.x[:k] + r.x[k:], yb)
    print(f"per-category affine        val = {d_ab:.5f} ({d_ab-base:+.5f})")

    recal = 3.8342972907055923
    tot = recal - base
    print(f"\nboosted recalibration      val = {recal:.5f} ({tot:+.5f})   [from final.py]")
    print("\nshare of the boosted recalibration each rung explains:")
    for name, d in [("scalar temperature", d_T), ("per-category scale", d_a),
                    ("per-category affine", d_ab)]:
        print(f"  {name:22} {100*(d-base)/tot:6.1f}%")

    out = {"base": float(base), "T": T, "scalar": float(d_T),
           "vector_scale": float(d_a), "affine": float(d_ab),
           "boosted_recal": recal,
           "share_scalar": float((d_T - base) / tot),
           "share_scale": float((d_a - base) / tot),
           "share_affine": float((d_ab - base) / tot)}
    (HERE / "temper_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\nwrote temper_result.json")


if __name__ == "__main__":
    main()
