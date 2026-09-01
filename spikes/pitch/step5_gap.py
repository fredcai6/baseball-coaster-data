"""Task 5: remove the last contamination from the batter-pitcher interaction
measurement -- cross-fit the COORDINATES, not just the offset.

Lineage
-------
-0.01410  offset and coordinates both fitted on all training data
-0.00816  offset cross-fitted (step3_crossfit / pitch/crossfit.py), but the
          GLLVM coordinates fed to the booster were still fitted on all of
          train (spikes/gllvm/latent.npz -- one fit, never refit per fold)

The key insight this script exploits: step3_crossfit.py already refits all
nine step1a nodes per fold. Each fold's theta for a node contains a per-batter
effect vector and a per-pitcher effect vector (that node's contribution to
eta is `b[batter_idx] + q[pitcher_idx]`, nothing else player-specific). Across
the 9 nodes those ARE a 9-dimensional coordinate per batter and per pitcher --
no separate GLLVM latent fit is needed. So for every held-out row we can pull
BOTH its offset (10-category log-probability) AND its coordinates (9+9
numbers) from the exact fold model that never saw that row.

This script reuses step3_crossfit.py's fold construction VERBATIM (imported,
not re-implemented) so offsets and coordinates align row-for-row with
step3_oof.npz, then runs the identical boost/control comparison that
pitch/crossfit.py ran for nested_sep, twice:
  1. fully cross-fit: OOF offset + OOF coordinates
  2. contaminated coordinates: OOF offset + coordinates from a single fit on
     ALL training rows (the same kind of contamination -0.00816 had, just
     transplanted onto this simpler node-effect notion of "coordinate")

GAP = joint - additive in each case; the difference between the two GAPs is
exactly the contamination the coordinates were contributing.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # step3_crossfit, step1
sys.path.insert(0, str(HERE.parent))             # common
sys.path.insert(0, str(HERE.parent / "boost"))   # boost/fit.py, boost/control.py

import common               # noqa: E402
import step3_crossfit as S3  # noqa: E402  -- reuse its fold construction VERBATIM
import fit as BF             # noqa: E402  -- spikes/boost/fit.py
import control as CTRL       # noqa: E402  -- spikes/boost/control.py

S1 = S3.S1
CATS = S3.CATS
K = S3.K
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def node_coords(rows, thetas, BI, PI, n_bat, n_pit):
    """Pull the per-batter / per-pitcher effect scalar out of each of the 9
    node thetas for every row -> two (n, 9) coordinate arrays."""
    bi = np.fromiter((BI[r["batter"]] for r in rows), int, len(rows))
    pj = np.fromiter((PI[r["pitcher"]] for r in rows), int, len(rows))
    n_nodes = len(S1.NODES)
    Ab = np.zeros((len(rows), n_nodes))
    Ap = np.zeros((len(rows), n_nodes))
    for ni, (name, reach, pos) in enumerate(S1.NODES):
        th = thetas[name]
        ps = len(th) - 1 - n_bat - n_pit
        b = th[1 + ps: 1 + ps + n_bat]
        q = th[1 + ps + n_bat:]
        Ab[:, ni] = b[bi]
        Ap[:, ni] = q[pj]
    return Ab, Ap


def run_gap(tag, off, Ab, Ap, y, mg):
    """Same comparison pitch/crossfit.py ran for nested_sep's GLLVM axes, now
    on the 9-dim node-effect coordinates. `mg` selects the inner-fit rows."""
    X = np.hstack([Ab, Ap])
    base = BF.deviance(off[~mg], y[~mg])
    add = CTRL.boost_additive(Ab[mg], Ap[mg], off[mg], y[mg],
                              Ab[~mg], Ap[~mg], off[~mg], y[~mg])
    joint, rnd = BF.boost(X[mg], off[mg], y[mg], X[~mg], off[~mg], y[~mg])
    gap = joint - add
    log(f"=== {tag} ===")
    log(f"  base={base:.5f}  additive={add:.5f}  joint={joint:.5f}  (round {rnd})")
    log(f"  GAP = joint - additive = {gap:+.5f}")
    return dict(base=float(base), additive=float(add), joint=float(joint),
                gap=float(gap), best_round=int(rnd))


def main():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    y_tr = np.array([r["y"] for r in tr])
    y_te = np.array([r["y"] for r in te])
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}
    BI, PI, n_bat, n_pit = S3.build_index(rows)
    log(f"train={len(tr)} test={len(te)} batters={n_bat} pitchers={n_pit}")

    res = json.loads((HERE / "step1_result.json").read_text())
    node_hp = {d["node"]: d for d in res["nodes"]}

    # ---- CONTAMINATED coordinates: single fit on ALL training rows ----------
    log("fitting full-train model on all TRAIN rows (contaminated coordinates) ...")
    thetas_full = S3.fit_all_nodes(tr, node_hp, BI, PI, season_idx)
    P_full = S3.category_probs(tr, thetas_full, node_hp, BI, PI, season_idx)
    P_te = S3.category_probs(te, thetas_full, node_hp, BI, PI, season_idx)
    logP_full = np.log(np.maximum(P_full, 1e-300))
    logP_te = np.log(np.maximum(P_te, 1e-300))
    d_in = S3.dev(logP_full, y_tr)
    d_te = S3.dev(logP_te, y_te)
    Ab_full, Ap_full = node_coords(tr, thetas_full, BI, PI, n_bat, n_pit)
    log(f"in-sample deviance = {d_in:.5f}   frozen-test deviance = {d_te:.5f}")

    # ---- CROSS-FIT: identical folds to step3_crossfit.py --------------------
    g = sorted(train_g)
    rs = np.random.RandomState(S3.FOLD_SEED)
    rs.shuffle(g)
    folds = [set(g[i::S3.N_FOLDS]) for i in range(S3.N_FOLDS)]
    log(f"fold seed={S3.FOLD_SEED}  fold sizes={[len(f) for f in folds]} games "
        f"(identical construction to step3_crossfit.py)")

    oof = np.zeros((len(tr), K))
    Ab_oof = np.zeros((len(tr), len(S1.NODES)))
    Ap_oof = np.zeros((len(tr), len(S1.NODES)))
    for k, fold in enumerate(folds):
        mask = np.array([r["game_id"] in fold for r in tr])
        fit_rows = [r for r, m in zip(tr, mask) if not m]
        pred_rows = [r for r, m in zip(tr, mask) if m]
        thetas_k = S3.fit_all_nodes(fit_rows, node_hp, BI, PI, season_idx)
        Pk = S3.category_probs(pred_rows, thetas_k, node_hp, BI, PI, season_idx)
        oof[mask] = Pk
        Abk, Apk = node_coords(pred_rows, thetas_k, BI, PI, n_bat, n_pit)
        Ab_oof[mask] = Abk
        Ap_oof[mask] = Apk
        d_fold = S3.dev(np.log(np.maximum(Pk, 1e-300)), y_tr[mask])
        log(f"fold {k}: fit={len(fit_rows)} pred={len(pred_rows)}  "
            f"out-of-fold deviance = {d_fold:.5f}")

    logP_oof = np.log(np.maximum(oof, 1e-300))
    d_oof = S3.dev(logP_oof, y_tr)

    log("")
    log(f"SANITY  in-sample (full-train fit) = {d_in:.5f}")
    log(f"SANITY  out-of-fold                = {d_oof:.5f}")
    log(f"SANITY  frozen test                = {d_te:.5f}")
    ok = abs(d_oof - d_te) < abs(d_oof - d_in)
    log(f"  -> OOF sits {'NEAR TEST (clean)' if ok else 'NEAR IN-SAMPLE (LEAK)'}")

    # cross-check against step3_oof.npz -- same fold seed/construction, should match
    prior = np.load(HERE / "step3_oof.npz")
    same_oof = np.allclose(prior["oof"], oof, atol=1e-8)
    log(f"cross-check vs step3_oof.npz: identical offsets = {same_oof}")

    if not ok:
        log("ABORT: cross-fit leaked; nothing below would be trustworthy")
        out = dict(sanity=dict(in_sample=float(d_in), oof=float(d_oof),
                               frozen_test=float(d_te), passed=False))
        (HERE / "step5_result.json").write_text(json.dumps(out, indent=1) + "\n")
        return

    # ---- inner fit/eval split by game -- IDENTICAL to boost/fit.py & control.py
    gg = sorted(train_g)
    rs2 = np.random.RandomState(11)
    rs2.shuffle(gg)
    itr_g = set(gg[: int(0.8 * len(gg))])
    mg = np.array([r["game_id"] in itr_g for r in tr])
    log(f"inner split: fit={mg.sum()} eval={(~mg).sum()}")

    log("")
    log("##### RUN 1: fully cross-fit (OOF offset + OOF coordinates) #####")
    run1 = run_gap("fully cross-fit", logP_oof, Ab_oof, Ap_oof, y_tr, mg)

    log("")
    log("##### RUN 2: OOF offset + IN-SAMPLE coordinates (contaminated) #####")
    run2 = run_gap("OOF offset + in-sample coordinates", logP_oof, Ab_full, Ap_full, y_tr, mg)

    contamination = run2["gap"] - run1["gap"]
    log("")
    log(f"contamination (run2.gap - run1.gap) = {contamination:+.5f}")
    if abs(run1["gap"]) < 0.002:
        verdict = ("fully cross-fit gap is near zero: the interaction was "
                    "contamination all the way down")
    else:
        verdict = ("fully cross-fit gap stays clearly negative: a real "
                    "interaction survives every firewall built so far")
    log(f"VERDICT: {verdict}")

    out = dict(
        fold_seed=int(S3.FOLD_SEED), n_folds=int(S3.N_FOLDS),
        sanity=dict(in_sample=float(d_in), oof=float(d_oof),
                   frozen_test=float(d_te), passed=True,
                   matches_step3_oof=bool(same_oof)),
        fully_crossfit=run1,
        oof_offset_insample_coords=run2,
        contamination_delta=float(contamination),
        verdict=verdict,
        runtime_sec=time.time() - T0,
    )
    (HERE / "step5_result.json").write_text(json.dumps(out, indent=1) + "\n")
    log("wrote step5_result.json")


if __name__ == "__main__":
    main()
