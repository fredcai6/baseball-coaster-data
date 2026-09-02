"""Test 2 -- external validity. Using the full fit's canonical latent
coordinates (L, M) and the free-9 per-node effects (B_free, Q_free, refit
already saved in full.npz), test whether they predict per-player descriptive
variables the model never saw as a target: spray/pull, bb_type (partially
in-target, see note), 2026 pitch-sequence rates, position, height/weight,
handedness.

For each external variable and side, fit three ridge regressions with
nested cross-validation (outer 5-fold for the R^2 estimate, RidgeCV's own
internal CV for the alpha, generalized-CV so no separate inner loop needed):
  (a) 3 latent coordinates
  (b) 9 free per-node effects (shape D's own per-node ridge, pinned lambdas)
  (c) top-3 PCA of the 9 free effects (SVD of the full 772x9 / 1220x9 matrix)
Qualified players: >=100 training PA/BF. Continuous targets use plain R^2;
categorical targets (position, bats, throws) are one-hot encoded and scored
by uniform-average multi-output R^2 -- documented, not a standard multiclass
metric, but consistent with "fit a ridge regression, report R^2" as asked.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_style as CS
import external_vars as EV

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

log = CS.log
QUAL_FLOOR = 100
ALPHAS = np.logspace(-2, 5, 30)
N_OUTER = 5
SEED = 4242


def top3_pca(mat_all):
    """mat_all: (n_all, 9) free-effect matrix, ALL players of that side.
    Column-centre then SVD; return the (n_all, 3) PC-score matrix."""
    mu = mat_all.mean(axis=0, keepdims=True)
    Xc = mat_all - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :3] * S[:3]


def fit_r2(X, y, is_multi):
    """Nested CV: outer KFold for the R^2 estimate, RidgeCV's built-in
    generalized CV picks alpha inside each outer training fold."""
    n = X.shape[0]
    n_splits = min(N_OUTER, n) if n >= 2 else 1
    if n_splits < 2:
        return float("nan")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    yp = cross_val_predict(pipe, X, y, cv=kf)
    if is_multi:
        return float(r2_score(y, yp, multioutput="uniform_average"))
    return float(r2_score(y, yp))


def one_hot(labels, categories):
    idx = {c: i for i, c in enumerate(categories)}
    Y = np.zeros((len(labels), len(categories)))
    for i, lab in enumerate(labels):
        Y[i, idx[lab]] = 1.0
    return Y


def run_side(label, ids, pa, L_latent, free9, ext, qual_floor):
    qual = pa >= qual_floor
    log(f"  [{label}] qualified (>= {qual_floor} training PA/BF): {int(qual.sum())} / {len(ids)}")
    pca3_all = top3_pca(free9)

    feats = dict(latent3=L_latent, free9=free9, pca3=pca3_all)
    results = {}
    for varname, vals in ext.items():
        vals = np.asarray(vals)
        is_cat = vals.dtype == object
        if is_cat:
            mask = qual & np.array([v is not None for v in vals])
        else:
            valsf = vals.astype(float)
            mask = qual & ~np.isnan(valsf)
        n = int(mask.sum())
        if n < 20:
            log(f"    {varname:24} n={n} -- too few qualified & non-null, skipping")
            continue
        if is_cat:
            cats = sorted(set(vals[mask]))
            if len(cats) < 2:
                log(f"    {varname:24} n={n} only one category present, skipping")
                continue
            y = one_hot(vals[mask], cats)
            is_multi = True
        else:
            y = vals[mask].astype(float)
            is_multi = False
            cats = None

        r2s = {}
        for fname, Xall in feats.items():
            X = Xall[mask]
            r2s[fname] = fit_r2(X, y, is_multi)
        results[varname] = dict(n=n, categories=cats, r2=r2s)
        log(f"    {varname:24} n={n:4}  latent3 R2={r2s['latent3']:+.3f}  "
            f"free9 R2={r2s['free9']:+.3f}  pca3 R2={r2s['pca3']:+.3f}")
    return results


def main():
    log("=" * 70)
    log("TEST 2: external validity")
    full = np.load(os.path.join(HERE, "full.npz"), allow_pickle=True)
    base = CS.base_universe()
    rows, train_g, BI, PI = base["rows"], base["train_g"], base["BI"], base["PI"]

    bat_ext, pit_ext = EV.build(rows, train_g, BI, PI)

    out = CS.load_result()
    out["test2"] = {}
    out["test2"]["qual_floor_pa"] = QUAL_FLOOR
    out["test2"]["method"] = ("nested CV: outer 5-fold KFold(shuffle, seed=4242) for R^2, "
                               "RidgeCV(alphas=logspace(-2,5,30)) inner generalized-CV for alpha, "
                               "features standardized. Categorical targets one-hot + multi-output "
                               "R^2 (uniform average).")

    log("-- batters --")
    bat_ext_cont = {k: v for k, v in bat_ext.items() if k not in ("position", "bats")}
    bat_ext_cat = {k: v for k, v in bat_ext.items() if k in ("position", "bats")}
    bat_results = run_side("batters", full["bat_ids"], full["bat_pa"], full["L"], full["B_free"],
                            {**bat_ext_cont, **bat_ext_cat}, QUAL_FLOOR)
    out["test2"]["batters"] = bat_results

    log("-- pitchers --")
    pit_ext_cont = {k: v for k, v in pit_ext.items() if k != "throws"}
    pit_ext_cat = {k: v for k, v in pit_ext.items() if k == "throws"}
    pit_results = run_side("pitchers", full["pit_ids"], full["pit_pa"], full["M"], full["Q_free"],
                            {**pit_ext_cont, **pit_ext_cat}, QUAL_FLOOR)
    out["test2"]["pitchers"] = pit_results

    CS.save_result(out)


if __name__ == "__main__":
    main()
