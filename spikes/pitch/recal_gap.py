"""The interaction test, re-run above a CORRECTLY CALIBRATED baseline.

temper.py showed that a 20-parameter affine recalibration of Variant A's
log-probability vector explains 105% of what a nonparametric booster extracts
from the offset alone. So Variant A is miscalibrated per category, and the
"interaction" we have been measuring was, in large part, a booster repairing
that miscalibration using player coordinates as a proxy for it.

That makes every earlier gap uninterpretable, because they were all measured
against a broken baseline. This re-runs the held-out-PAIRS test with the
offset recalibrated first:

    off_cal = log_softmax(a * off + b),  (a, b) fitted on in-sample pairs only

If the gap collapses, the interaction was calibration repair and nothing else.
Whatever survives is interaction that a correctly calibrated additive model
cannot produce -- which is the number we actually wanted all along.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "boost"))
sys.path.insert(0, str(HERE.parent / "fuse"))
import common
import fit as BF
import analyze as FZ
from control import boost_additive

SEP = HERE.parent / "nested_sep"
GLLVM = HERE.parent / "gllvm"


def dev(logits, y):
    return -2.0 * np.mean(logits[np.arange(len(y)), y] - logsumexp(logits, axis=1))


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
    X = np.hstack([Ab, Ap])

    zA = np.load(SEP / "latent.npz", allow_pickle=True)
    off = np.log(np.maximum(FZ.nested_category_probs(tr, zA, False, season_idx), 1e-300))

    # SAME pair split as final.py Phase 1
    pairs = np.array([f"{r['batter']}|{r['pitcher']}" for r in tr])
    uniq = np.unique(pairs)
    rs = np.random.RandomState(4242)
    held = set(rs.choice(uniq, size=int(0.2 * len(uniq)), replace=False))
    m = np.array([p not in held for p in pairs])
    print(f"{len(uniq)} pairs, {len(held)} held out; fit={m.sum()} eval={(~m).sum()}")

    # affine recalibration fitted ONLY on rows whose pair is in-sample
    k = off.shape[1]

    def f(p):
        return dev(off[m] * p[:k] + p[k:], y[m])

    r = minimize(f, np.concatenate([np.ones(k), np.zeros(k)]), method="L-BFGS-B")
    a, b = r.x[:k], r.x[k:]
    raw = off * a + b
    off_cal = raw - logsumexp(raw, axis=1, keepdims=True)

    print("\nper-category recalibration (a = scale, b = intercept):")
    for i, c in enumerate(common.CATEGORIES):
        print(f"  {c:6} a={a[i]:+.4f}  b={b[i]:+.4f}")

    base_raw = dev(off[~m], y[~m])
    base_cal = dev(off_cal[~m], y[~m])
    print(f"\nbaseline on unseen pairs   raw = {base_raw:.5f}")
    print(f"                    recalibrated = {base_cal:.5f}  ({base_cal-base_raw:+.5f})")

    add = boost_additive(Ab[m], Ap[m], off_cal[m], y[m],
                         Ab[~m], Ap[~m], off_cal[~m], y[~m])
    joint, rnd = BF.boost(X[m], off_cal[m], y[m], X[~m], off_cal[~m], y[~m])
    gap = joint - add
    print(f"\nadditive control  = {add:.5f}")
    print(f"joint booster     = {joint:.5f}   (round {rnd})")
    print(f"GAP               = {gap:+.5f}")
    print(f"  raw-offset gap on the same split was -0.00589")
    print(f"  game-split gap was                   -0.01410")
    verdict = ("SURVIVES: interaction above a calibrated baseline"
               if gap < -0.002 else
               "COLLAPSES: the gap was calibration repair")
    print(f"  -> {verdict}")

    out = {"base_raw": float(base_raw), "base_cal": float(base_cal),
           "additive": float(add), "joint": float(joint), "gap": float(gap),
           "a": a.tolist(), "b": b.tolist(), "verdict": verdict}
    (HERE / "recal_gap_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\nwrote recal_gap_result.json")


if __name__ == "__main__":
    main()
