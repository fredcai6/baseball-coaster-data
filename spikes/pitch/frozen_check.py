"""Does the recalibration hold on the FROZEN test games?

Every recalibration number so far was measured inside the training games --
an inner 80/20 game split for temper.py, a held-out-PAIRS split for
recal_gap.py. Neither touched the frozen test set, so none of them is
comparable to the 3.94846 / 3.95424 / 4.01172 ladder every model in this
project is quoted against.

This fits the 20-parameter affine map on ALL training rows and scores it on
the frozen test games. If the gain survives here, the recalibration is a real
improvement to the best model we have. If it shrinks, it was partly an
artifact of fitting and evaluating inside the same season-stratified pool.
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "fuse"))
import common
import analyze as FZ

SEP = HERE.parent / "nested_sep"


def dev(logits, y):
    return -2.0 * np.mean(logits[np.arange(len(y)), y] - logsumexp(logits, axis=1))


def main():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    season_idx = {s: i for i, s in enumerate([2024, 2025, 2026])}
    ytr = np.array([r["y"] for r in tr])
    yte = np.array([r["y"] for r in te])
    print(f"train {len(tr)}  test {len(te)}")

    zA = np.load(SEP / "latent.npz", allow_pickle=True)
    off_tr = np.log(np.maximum(FZ.nested_category_probs(tr, zA, False, season_idx), 1e-300))
    off_te = np.log(np.maximum(FZ.nested_category_probs(te, zA, False, season_idx), 1e-300))

    k = off_tr.shape[1]
    raw_tr = dev(off_tr, ytr)
    raw_te = dev(off_te, yte)
    print(f"\nraw offset      train = {raw_tr:.5f}   test = {raw_te:.5f}")

    def fit(p):
        return dev(off_tr * p[:k] + p[k:], ytr)

    r = minimize(fit, np.concatenate([np.ones(k), np.zeros(k)]), method="L-BFGS-B")
    a, b = r.x[:k], r.x[k:]
    cal_tr = dev(off_tr * a + b, ytr)
    cal_te = dev(off_te * a + b, yte)
    print(f"affine recal    train = {cal_tr:.5f}   test = {cal_te:.5f}  "
          f"({cal_te - raw_te:+.5f})")

    # scale-only, for the ladder
    def fit_a(p):
        return dev(off_tr * p, ytr)
    ra = minimize(fit_a, np.ones(k), method="L-BFGS-B")
    sc_te = dev(off_te * ra.x, yte)
    # single global temperature, for the ladder
    def fit_T(t):
        return dev(off_tr / np.exp(t[0]), ytr)
    rt = minimize(fit_T, [0.0], method="Nelder-Mead")
    T = float(np.exp(rt.x[0]))
    T_te = dev(off_te / T, yte)

    print(f"\n--- frozen-test ladder ---")
    print(f"  raw (nested_sep)          {raw_te:.5f}")
    print(f"  + global temperature      {T_te:.5f} ({T_te-raw_te:+.5f})  T={T:.4f}")
    print(f"  + per-category scale      {sc_te:.5f} ({sc_te-raw_te:+.5f})")
    print(f"  + per-category affine     {cal_te:.5f} ({cal_te-raw_te:+.5f})")
    print(f"\nreference: NULL 4.01172 | flat ridge 3.95550 | NPMR 3.95424 | "
          f"nested_sep 3.94846 | binarised additive+AO 3.94939")
    print(f"\nper-category scales a (all >1 = the fit was over-shrunk):")
    for i, c in enumerate(common.CATEGORIES):
        print(f"  {c:6} a={a[i]:+.4f}  b={b[i]:+.4f}")

    (HERE / "frozen_check_result.json").write_text(json.dumps(
        {"raw_test": float(raw_te), "affine_test": float(cal_te),
         "scale_test": float(sc_te), "temp_test": float(T_te), "T": T,
         "delta": float(cal_te - raw_te),
         "a": a.tolist(), "b": b.tolist()}, indent=1) + "\n")
    print("\nwrote frozen_check_result.json")


if __name__ == "__main__":
    main()
