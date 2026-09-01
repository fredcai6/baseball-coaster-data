"""Do the GLLVM latent spaces actually contain CLUSTERS, or a continuum?

This is the precondition for a cluster-augmented interaction model, and it is
a real question rather than a formality. Player ability is a priori a
continuum: nothing about baseball says pitchers come in discrete kinds. Every
clustering algorithm will happily return k clusters from a single Gaussian
blob, so "k-means found 4 groups" is not evidence of anything. What matters is
whether the structure is MORE clustered than a unimodal cloud with the same
mean and covariance.

So every statistic here is reported against a matched-Gaussian null: synthetic
data drawn from one multivariate normal fitted to the observed coordinates,
pushed through the identical pipeline. If the real data scores inside the
null band, the space is a continuum and the two-stage cluster model has no
foundation to stand on.

One artifact this must avoid. Latent coordinates are ridge-shrunk, so a player
with 20 PA sits near the origin regardless of talent -- not because he is
average but because the model knows nothing about him. Include those and you
manufacture a dense central "cluster" out of pure ignorance. Hence the PA
floor, swept rather than fixed so the conclusion can be checked for
sensitivity to it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

GLLVM = HERE.parent / "gllvm"
KS = list(range(2, 11))
PA_FLOORS = [50, 100, 200]
NULL_REPS = 25
SEED = 20260830


def axes_from_gllvm():
    z = np.load(GLLVM / "latent.npz", allow_pickle=True)
    out = {}
    for side in ("bat", "pit"):
        B = z[f"L{side}"] @ z[f"F{side}"].T
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        out[side] = dict(coords=U * S, sv=S, loadings=Vt,
                         ids=[str(x) for x in z[f"{side}_ids"]])
    return out


def pa_counts(rows):
    b, p = Counter(), Counter()
    for r in rows:
        b[r["batter"]] += 1
        p[r["pitcher"]] += 1
    return b, p


def hopkins(X, rng, m_frac=0.1):
    """Hopkins statistic. ~0.5 = uniformly random (no cluster tendency);
    values well above 0.5 indicate clusterable structure."""
    n, d = X.shape
    m = max(5, int(m_frac * n))
    idx = rng.choice(n, m, replace=False)
    lo, hi = X.min(axis=0), X.max(axis=0)
    synth = rng.uniform(lo, hi, size=(m, d))

    def nn_dist(pts, exclude_self):
        D = np.linalg.norm(pts[:, None, :] - X[None, :, :], axis=2)
        if exclude_self:
            D[np.arange(len(pts)), idx] = np.inf
        return D.min(axis=1)

    u = nn_dist(synth, False).sum()
    w = nn_dist(X[idx], True).sum()
    return u / (u + w)


def profile(X, rng):
    """Cluster-structure statistics for one coordinate matrix."""
    out = {"n": len(X)}
    # GMM BIC: does any k>1 beat k=1?
    bics = {}
    for k in [1] + KS:
        if k >= len(X):
            continue
        gm = GaussianMixture(k, covariance_type="full", random_state=SEED,
                             n_init=3, reg_covar=1e-4).fit(X)
        bics[k] = float(gm.bic(X))
    out["bic"] = bics
    out["bic_best_k"] = int(min(bics, key=bics.get))
    out["bic_gain_over_k1"] = float(bics[1] - bics[out["bic_best_k"]])
    # silhouette
    sil = {}
    for k in KS:
        if k >= len(X):
            continue
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
        sil[k] = float(silhouette_score(X, lab))
    out["silhouette"] = sil
    out["silhouette_best_k"] = int(max(sil, key=sil.get))
    out["silhouette_best"] = float(max(sil.values()))
    # within-cluster dispersion, for the gap statistic
    out["logW"] = {}
    for k in KS:
        km = KMeans(k, n_init=10, random_state=SEED).fit(X)
        out["logW"][k] = float(np.log(max(km.inertia_, 1e-12)))
    out["hopkins"] = float(hopkins(X, rng))
    return out


def main():
    rows = common.load_pa()
    bpa, ppa = pa_counts(rows)
    ax = axes_from_gllvm()
    rng = np.random.RandomState(SEED)
    report = {}

    for side, counts, label in (("bat", bpa, "batters"), ("pit", ppa, "pitchers")):
        sv = ax[side]["sv"]
        r = int((sv > 1e-6).sum())
        print(f"\n{'='*66}\n{label.upper()}  (using {r} non-degenerate axes; sv={np.round(sv[:r],3).tolist()})")
        report[side] = {"n_axes": r, "singular_values": sv[:r].tolist(), "floors": {}}

        for floor in PA_FLOORS:
            keep = [i for i, pid in enumerate(ax[side]["ids"]) if counts.get(pid, 0) >= floor]
            if len(keep) < 40:
                continue
            X = ax[side]["coords"][keep][:, :r]
            X = (X - X.mean(0)) / np.where(X.std(0) < 1e-12, 1.0, X.std(0))

            real = profile(X, rng)
            # matched-Gaussian continuum null: same mean+covariance, one blob
            mu, cov = X.mean(0), np.cov(X.T)
            nulls = []
            for rep in range(NULL_REPS):
                Xn = np.random.RandomState(SEED + rep).multivariate_normal(mu, cov, size=len(X))
                Xn = (Xn - Xn.mean(0)) / np.where(Xn.std(0) < 1e-12, 1.0, Xn.std(0))
                nulls.append(profile(Xn, np.random.RandomState(SEED + rep)))

            def band(getter):
                v = np.array([getter(nn) for nn in nulls])
                return float(v.mean()), float(v.std())

            sil_m, sil_s = band(lambda p: p["silhouette_best"])
            hop_m, hop_s = band(lambda p: p["hopkins"])
            bic_m, bic_s = band(lambda p: p["bic_gain_over_k1"])
            k1_frac = float(np.mean([p["bic_best_k"] == 1 for p in nulls]))
            # gap statistic at each k
            gaps = {}
            for k in KS:
                lw = np.array([nn["logW"][k] for nn in nulls])
                gaps[k] = float(lw.mean() - real["logW"][k])

            z_sil = (real["silhouette_best"] - sil_m) / max(sil_s, 1e-12)
            z_hop = (real["hopkins"] - hop_m) / max(hop_s, 1e-12)
            z_bic = (real["bic_gain_over_k1"] - bic_m) / max(bic_s, 1e-12)

            print(f"\n  PA >= {floor}:  n={real['n']}")
            print(f"    GMM BIC best k        : {real['bic_best_k']}   "
                  f"(null picks k=1 in {k1_frac*100:.0f}% of reps)")
            print(f"    BIC gain over k=1     : {real['bic_gain_over_k1']:.1f}   "
                  f"null {bic_m:.1f}+-{bic_s:.1f}   z={z_bic:+.2f}")
            print(f"    best silhouette       : {real['silhouette_best']:.4f} at k={real['silhouette_best_k']}   "
                  f"null {sil_m:.4f}+-{sil_s:.4f}   z={z_sil:+.2f}")
            print(f"    Hopkins               : {real['hopkins']:.4f}   "
                  f"null {hop_m:.4f}+-{hop_s:.4f}   z={z_hop:+.2f}")
            print(f"    gap statistic by k    : " +
                  "  ".join(f"k{k}:{gaps[k]:+.3f}" for k in KS[:6]))
            report[side]["floors"][floor] = dict(
                real=real, z_silhouette=z_sil, z_hopkins=z_hop, z_bic=z_bic,
                null_picks_k1_frac=k1_frac, gaps=gaps,
                null_silhouette=[sil_m, sil_s], null_hopkins=[hop_m, hop_s],
                null_bic_gain=[bic_m, bic_s])

    (HERE / "result.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {HERE/'result.json'}")


if __name__ == "__main__":
    main()
