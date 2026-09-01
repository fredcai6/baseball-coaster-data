"""Standardized opponent-population layer for the step1.py binary-tree model,
and a measurement of the evaluate-at-mean vs average-over-population gap.

TASK 1 -- standardized populations
-----------------------------------
step1.py fits, per node, a scalar batter effect and a scalar pitcher effect
under ridge shrinkage (lam_bat, lam_pit) and a per-node Aranda-Ordaz psi. It
never SAVES those per-player vectors -- step1_result.json only has the
hyperparameters. This script refits each node ONCE on the full training set
using the frozen (lam_bat, lam_pit, psi) triple for that node, which is
exactly what step1.py itself does internally to produce its reported test
deviance (the `TR -> TE` fit at the bottom of its per-node loop). That gives a
per-node batter vector `b` (length n_bat) and pitcher vector `q` (length
n_pit), plus alpha/beta.

The "population" is then the PA-weighted empirical distribution of `b`
across batters and `q` across pitchers -- weighted by each player's share of
plate appearances IN THE TRAINING SET (the set the model was fit on).

TASK 2 -- evaluate-at-mean vs average-over-population
-------------------------------------------------------
For a fixed batter, two ways to summarize "vs league-average pitching":
  (a) evaluate-at-mean: plug the PA-weighted MEAN pitcher effect into each
      node and compute one 10-category probability vector.
  (b) average-over-population: compute a probability vector against EVERY
      pitcher, then take the PA-weighted average of those vectors.
Because the Aranda-Ordaz link is nonlinear, (a) != (b) in general (Jensen's
gap). This script measures the gap directly, by category, and asks whether it
reorders a leaderboard (rank correlation + max rank displacement) or is a
constant offset.

TASK 3 -- handedness
---------------------
structural() (spikes/fuse/analyze.py) encodes home/platoon/season as binary
covariates PER PA, using the two real people's actual bats/throws. For a
STANDARDIZED population there is no "the batter's own" platoon mix to use --
every batter must be evaluated against the SAME fixed covariate context, or
the population comparison is contaminated by scheduling/matchup differences
that have nothing to do with skill. This script fixes X to the single
PA-weighted MEAN structural covariate vector over the training set (mean
home-indicator, mean platoon-opposite/unknown indicators, mean season
dummies) and uses that same X for every batter x pitcher combination. This is
an average of the COVARIATE, not of a probability, so it does not reintroduce
the Task-2 nonlinearity gap -- it is a modeling convention, stated explicitly
in the output, not a defensible-only-in-expectation approximation.

Run: PYTHONPATH=pipeline spikes/../.venv/bin/python spikes/value/populations.py
(see repo root README / task note: use the repo .venv python, not system
python3, which has no numpy)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SPIKES = HERE.parent
PITCH = SPIKES / "pitch"
FUSE = SPIKES / "fuse"

# step1.py itself inserts SPIKES and FUSE onto sys.path when imported, but we
# need PITCH on the path *first* in order to import step1 at all.
sys.path.insert(0, str(PITCH))
sys.path.insert(0, str(SPIKES))
sys.path.insert(0, str(FUSE))

import common  # noqa: E402
import step1 as S1  # noqa: E402  (reuses fit/ao_prob/NODES -- the frozen model)
from analyze import structural  # noqa: E402

CATS = common.CATEGORIES
CI = common.CAT_INDEX
NODES = S1.NODES
STEP1_RESULT = PITCH / "step1_result.json"
OUT_JSON = HERE / "populations_result.json"

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# category -> ordered list of (node_index, is_positive_branch)
# NODES is already in topological (parent-before-child) order, and a node's
# `reach` set only contains a category if that category's path passes through
# it, so a single linear scan reproduces the tree walk exactly. No early
# exit: a category can pick up factors from every node whose reach contains
# it, per the task's explicit note.
# --------------------------------------------------------------------------
def build_paths():
    paths = {ci: [] for ci in range(len(CATS))}
    for ni, (name, reach, pos) in enumerate(NODES):
        for ci in range(len(CATS)):
            if ci in reach:
                paths[ci].append((ni, ci in pos))
    return paths


PATHS = build_paths()


def load_hparams():
    d = json.loads(STEP1_RESULT.read_text())
    hp = {}
    for nd in d["nodes"]:
        hp[nd["node"]] = dict(lam_bat=nd["lam_bat"], lam_pit=nd["lam_pit"], psi=nd["psi"])
    return hp


def fit_node_params(tr, te, bats, pits, BI, PI, season_idx, hp):
    """Refit every node on the full training set at its frozen hyperparameters.
    Returns per-node alpha, beta, b (n_bat,), q (n_pit,), psi -- and, as a
    sanity check, the reconstructed held-out deviance (should reproduce the
    frozen 3.94729 from step1.py's ao_side row).
    """
    n_bat, n_pit = len(bats), len(pits)
    node_params = []
    total_test_dev = 0.0
    n_test = len(te)
    for name, reach, pos in NODES:
        sub_tr = [r for r in tr if r["y"] in reach]
        Xs_tr = structural(sub_tr, season_idx)
        bi_tr = np.fromiter((BI[r["batter"]] for r in sub_tr), int, len(sub_tr))
        pj_tr = np.fromiter((PI[r["pitcher"]] for r in sub_tr), int, len(sub_tr))
        yv_tr = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub_tr), float, len(sub_tr))

        h = hp[name]
        th = S1.fit(Xs_tr, bi_tr, pj_tr, yv_tr, n_bat, n_pit, h["psi"], h["lam_bat"], h["lam_pit"])
        ps = Xs_tr.shape[1]
        alpha = float(th[0])
        beta = th[1:1 + ps].copy()
        b = th[1 + ps:1 + ps + n_bat].copy()
        q = th[1 + ps + n_bat:].copy()

        sub_te = [r for r in te if r["y"] in reach]
        Xs_te = structural(sub_te, season_idx)
        bi_te = np.fromiter((BI[r["batter"]] for r in sub_te), int, len(sub_te))
        pj_te = np.fromiter((PI[r["pitcher"]] for r in sub_te), int, len(sub_te))
        yv_te = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub_te), float, len(sub_te))
        d = S1.node_dev(th, Xs_te, bi_te, pj_te, yv_te, n_bat, n_pit, h["psi"])
        total_test_dev += d

        node_params.append(dict(name=name, alpha=alpha, beta=beta, b=b, q=q, psi=h["psi"]))
        log(f"fit {name:9} n_tr={len(sub_tr):>6} alpha={alpha:+.3f} psi={h['psi']:g}")

    frozen_dev = total_test_dev / n_test
    log(f"reconstructed test deviance = {frozen_dev:.5f}  (step1_result.json ao_side = 3.947290)")
    return node_params, frozen_dev


def kish_ess(w):
    """Kish effective sample size for weights summing to 1: (sum w)^2/sum w^2 == 1/sum w^2."""
    return float(1.0 / np.sum(w ** 2))


def main():
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}

    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)
    log(f"rows={len(rows)} train={len(tr)} test={len(te)} batters={n_bat} pitchers={n_pit}")

    hp = load_hparams()
    node_params, frozen_dev = fit_node_params(tr, te, bats, pits, BI, PI, season_idx, hp)

    # ---------------- TASK 1: PA-weighted populations ---------------------
    bat_pa = Counter(r["batter"] for r in tr)
    pit_pa = Counter(r["pitcher"] for r in tr)
    w_bat_raw = np.array([bat_pa.get(b, 0) for b in bats], dtype=float)
    w_pit_raw = np.array([pit_pa.get(p, 0) for p in pits], dtype=float)
    n_bat_seen = int(np.sum(w_bat_raw > 0))
    n_pit_seen = int(np.sum(w_pit_raw > 0))
    w_bat = w_bat_raw / w_bat_raw.sum()
    w_pit = w_pit_raw / w_pit_raw.sum()

    ess_bat = kish_ess(w_bat)
    ess_pit = kish_ess(w_pit)
    log(f"batters:  seen={n_bat_seen}  roster={n_bat}  PA-weighted ESS={ess_bat:.1f}")
    log(f"pitchers: seen={n_pit_seen}  roster={n_pit}  PA-weighted ESS={ess_pit:.1f}")

    # Unweighted-vs-weighted shift: root-node effect (biggest, most legible
    # single number) mean under an unweighted average over IDs vs the
    # PA-weighted mean.
    root_b = node_params[0]["b"]
    root_q = node_params[0]["q"]
    unweighted_mean_b = float(np.mean(root_b[w_bat_raw > 0]))
    weighted_mean_b = float(np.sum(w_bat * root_b))
    unweighted_mean_q = float(np.mean(root_q[w_pit_raw > 0]))
    weighted_mean_q = float(np.sum(w_pit * root_q))
    log(f"root-node batter effect: unweighted mean={unweighted_mean_b:+.4f}  "
        f"PA-weighted mean={weighted_mean_b:+.4f}")
    log(f"root-node pitcher effect: unweighted mean={unweighted_mean_q:+.4f}  "
        f"PA-weighted mean={weighted_mean_q:+.4f}")

    # ---------------- TASK 3: fixed structural covariate context -----------
    Xs_tr_all = structural(tr, season_idx)  # 1 row per training PA -> mean IS PA-weighted
    X_fixed = Xs_tr_all.mean(axis=0)
    home_frac = float(X_fixed[0])
    opp_frac = float(X_fixed[1])
    unk_frac = float(X_fixed[2])
    same_frac = 1.0 - opp_frac - unk_frac
    season_fracs = {}
    seasons_sorted = sorted(season_idx, key=lambda s: season_idx[s])
    season_fracs[seasons_sorted[0]] = 1.0 - float(np.sum(X_fixed[3:]))
    for i, s in enumerate(seasons_sorted[1:], start=0):
        season_fracs[s] = float(X_fixed[3 + i])
    log(f"league PA-weighted platoon mix: same={same_frac:.4f} opposite={opp_frac:.4f} "
        f"unknown={unk_frac:.4f}")
    log(f"league PA-weighted home-batting fraction: {home_frac:.4f}")
    log(f"league PA-weighted season mix: {season_fracs}")
    log("convention used for standardized eta: X_fixed = PA-weighted MEAN structural "
        "covariate vector (home, platoon-opposite, platoon-unknown, season dummies) "
        "over the full training set -- the SAME X_fixed for every batter x pitcher "
        "pair. This averages the covariate (linear in eta), not a probability, so it "
        "does not reintroduce the Task-2 nonlinearity gap.")

    C = np.array([np1["alpha"] + float(X_fixed @ np1["beta"]) for np1 in node_params])
    PSI = np.array([np1["psi"] for np1 in node_params])
    B = [np1["b"] for np1 in node_params]   # list of (n_bat,) arrays, one per node
    Q = [np1["q"] for np1 in node_params]   # list of (n_pit,) arrays, one per node
    n_nodes = len(NODES)

    q_mean = np.array([float(np.sum(w_pit * Q[ni])) for ni in range(n_nodes)])
    b_mean = np.array([float(np.sum(w_bat * B[ni])) for ni in range(n_nodes)])

    def category_matrix_from_node_p(node_p_list, n_rows):
        """node_p_list[ni] is either a scalar or an (n_rows,) array of P(positive
        branch) at node ni. Returns (n_rows, 10) category probability matrix
        (or a length-10 vector if n_rows is None, for the fully-scalar case)."""
        out = np.ones((n_rows, len(CATS))) if n_rows is not None else np.ones(len(CATS))
        for ci in range(len(CATS)):
            for ni, is_pos in PATHS[ci]:
                p = node_p_list[ni]
                out[..., ci] *= p if is_pos else (1.0 - p)
        return out

    def ao_p(eta, psi):
        p, _, _, _ = S1.ao_prob(eta, psi)
        return p

    # ---- batter side: (a) evaluate-at-mean-pitcher, (b) average-over-pitchers
    n_cat = len(CATS)
    at_mean_bat = np.zeros((n_bat, n_cat))
    avg_over_bat = np.zeros((n_bat, n_cat))
    for bi in range(n_bat):
        # (a) scalar eta per node
        node_p_scalar = [ao_p(C[ni] + B[ni][bi] + q_mean[ni], PSI[ni]) for ni in range(n_nodes)]
        at_mean_bat[bi] = category_matrix_from_node_p(node_p_scalar, None)
        # (b) vector eta per node, over every pitcher
        node_p_vec = [ao_p(C[ni] + B[ni][bi] + Q[ni], PSI[ni]) for ni in range(n_nodes)]
        probs_vs_all = category_matrix_from_node_p(node_p_vec, n_pit)
        avg_over_bat[bi] = w_pit @ probs_vs_all
        # sanity: category probs sum to 1
        assert abs(at_mean_bat[bi].sum() - 1.0) < 1e-10
        assert np.max(np.abs(probs_vs_all.sum(axis=1) - 1.0)) < 1e-10
    log(f"batter side: computed (a) and (b) for {n_bat} batters vs {n_pit} pitchers")

    # ---- pitcher side, symmetric: (a) evaluate-at-mean-batter,
    # (b) average-over-batters
    at_mean_pit = np.zeros((n_pit, n_cat))
    avg_over_pit = np.zeros((n_pit, n_cat))
    for pi in range(n_pit):
        node_p_scalar = [ao_p(C[ni] + b_mean[ni] + Q[ni][pi], PSI[ni]) for ni in range(n_nodes)]
        at_mean_pit[pi] = category_matrix_from_node_p(node_p_scalar, None)
        node_p_vec = [ao_p(C[ni] + B[ni] + Q[ni][pi], PSI[ni]) for ni in range(n_nodes)]
        probs_vs_all = category_matrix_from_node_p(node_p_vec, n_bat)
        avg_over_pit[pi] = w_bat @ probs_vs_all
        assert abs(at_mean_pit[pi].sum() - 1.0) < 1e-10
        assert np.max(np.abs(probs_vs_all.sum(axis=1) - 1.0)) < 1e-10
    log(f"pitcher side: computed (a) and (b) for {n_pit} pitchers vs {n_bat} batters")

    # ---------------- TASK 2 reporting --------------------------------
    diff_bat = avg_over_bat - at_mean_bat       # (n_bat, 10)
    diff_pit = avg_over_pit - at_mean_pit        # (n_pit, 10)

    per_cat = {}
    for ci, cat in enumerate(CATS):
        mean_prob_bat = float(np.mean(avg_over_bat[:, ci]))
        mean_prob_pit = float(np.mean(avg_over_pit[:, ci]))
        mean_abs_bat = float(np.mean(np.abs(diff_bat[:, ci])))
        mean_abs_pit = float(np.mean(np.abs(diff_pit[:, ci])))
        per_cat[cat] = dict(
            mean_prob_batter=mean_prob_bat,
            mean_prob_pitcher=mean_prob_pit,
            batter_mean_abs=mean_abs_bat,
            batter_max_abs=float(np.max(np.abs(diff_bat[:, ci]))),
            pitcher_mean_abs=mean_abs_pit,
            pitcher_max_abs=float(np.max(np.abs(diff_pit[:, ci]))),
            batter_mean_rel=mean_abs_bat / mean_prob_bat if mean_prob_bat > 0 else float("nan"),
            pitcher_mean_rel=mean_abs_pit / mean_prob_pit if mean_prob_pit > 0 else float("nan"),
        )
    log("(a)-vs-(b) abs diff by category (mean / max / mean-relative-to-prob), "
        "batter side then pitcher side:")
    for cat in CATS:
        d = per_cat[cat]
        log(f"  {cat:6} p~{d['mean_prob_batter']:.4f}  bat mean={d['batter_mean_abs']:.6f} "
            f"max={d['batter_max_abs']:.6f} rel={d['batter_mean_rel']:.4f}  "
            f"| pit mean={d['pitcher_mean_abs']:.6f} max={d['pitcher_max_abs']:.6f} "
            f"rel={d['pitcher_mean_rel']:.4f}")

    rare = ["HR", "3B", "2B"]
    common_cats = ["K", "G", "F"]
    rare_mean_bat = float(np.mean([per_cat[c]["batter_mean_abs"] for c in rare]))
    common_mean_bat = float(np.mean([per_cat[c]["batter_mean_abs"] for c in common_cats]))
    rare_mean_pit = float(np.mean([per_cat[c]["pitcher_mean_abs"] for c in rare]))
    common_mean_pit = float(np.mean([per_cat[c]["pitcher_mean_abs"] for c in common_cats]))
    rare_rel_bat = float(np.mean([per_cat[c]["batter_mean_rel"] for c in rare]))
    common_rel_bat = float(np.mean([per_cat[c]["batter_mean_rel"] for c in common_cats]))
    rare_rel_pit = float(np.mean([per_cat[c]["pitcher_mean_rel"] for c in rare]))
    common_rel_pit = float(np.mean([per_cat[c]["pitcher_mean_rel"] for c in common_cats]))
    log(f"rare (HR,3B,2B) mean|diff| ABSOLUTE  bat={rare_mean_bat:.6f} pit={rare_mean_pit:.6f}")
    log(f"common (K,G,F) mean|diff| ABSOLUTE  bat={common_mean_bat:.6f} pit={common_mean_pit:.6f}")
    log(f"ratio rare/common ABSOLUTE  bat={rare_mean_bat / common_mean_bat:.2f}x  "
        f"pit={rare_mean_pit / common_mean_pit:.2f}x")
    log(f"rare (HR,3B,2B) mean|diff| RELATIVE-to-own-prob  bat={rare_rel_bat:.4f} pit={rare_rel_pit:.4f}")
    log(f"common (K,G,F) mean|diff| RELATIVE-to-own-prob  bat={common_rel_bat:.4f} pit={common_rel_pit:.4f}")
    log(f"ratio rare/common RELATIVE  bat={rare_rel_bat / common_rel_bat:.2f}x  "
        f"pit={rare_rel_pit / common_rel_pit:.2f}x")

    # ---------------- rank correlations (batters) ----------------------
    hr_idx = CI["HR"]
    hr_a = at_mean_bat[:, hr_idx]
    hr_b = avg_over_bat[:, hr_idx]
    rho_hr, pval_hr = spearmanr(hr_a, hr_b)

    def ranks_desc(x):
        # rank 1 = highest value
        order = np.argsort(-x, kind="stable")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(1, len(x) + 1)
        return r

    rank_a_hr = ranks_desc(hr_a)
    rank_b_hr = ranks_desc(hr_b)
    max_disp_hr = float(np.max(np.abs(rank_a_hr - rank_b_hr)))
    top10_a = set(np.argsort(-hr_a)[:10].tolist())
    top10_b = set(np.argsort(-hr_b)[:10].tolist())
    top10_overlap = len(top10_a & top10_b)

    i1b, i2b, i3b, ihr, ik = CI["1B"], CI["2B"], CI["3B"], CI["HR"], CI["K"]

    def value_proxy(P):
        return 1 * P[:, i1b] + 2 * P[:, i2b] + 3 * P[:, i3b] + 4 * P[:, ihr] - P[:, ik]

    val_a = value_proxy(at_mean_bat)
    val_b = value_proxy(avg_over_bat)
    rho_val, pval_val = spearmanr(val_a, val_b)
    rank_a_val = ranks_desc(val_a)
    rank_b_val = ranks_desc(val_b)
    max_disp_val = float(np.max(np.abs(rank_a_val - rank_b_val)))
    top10_a_val = set(np.argsort(-val_a)[:10].tolist())
    top10_b_val = set(np.argsort(-val_b)[:10].tolist())
    top10_overlap_val = len(top10_a_val & top10_b_val)

    log(f"HR-prob rank: spearman rho={rho_hr:.6f} (p={pval_hr:.2g})  "
        f"max rank displacement={max_disp_hr:.0f} / {n_bat}  "
        f"top-10 overlap={top10_overlap}/10")
    log(f"value-proxy rank: spearman rho={rho_val:.6f} (p={pval_val:.2g})  "
        f"max rank displacement={max_disp_val:.0f} / {n_bat}  "
        f"top-10 overlap={top10_overlap_val}/10")

    max_overall_diff = float(np.max(np.abs(diff_bat)))
    verdict = ("CONSTANT OFFSET -- ranking is preserved" if rho_hr > 0.999 and max_disp_hr <= 2
               else "REORDERS the leaderboard" if rho_hr < 0.999 or max_disp_hr > 2
               else "borderline")

    out = dict(
        frozen_test_deviance_reconstructed=frozen_dev,
        n_bat_roster=n_bat, n_pit_roster=n_pit,
        n_bat_seen_train=n_bat_seen, n_pit_seen_train=n_pit_seen,
        ess_bat_pa_weighted=ess_bat, ess_pit_pa_weighted=ess_pit,
        root_node_batter_effect=dict(unweighted_mean=unweighted_mean_b,
                                       pa_weighted_mean=weighted_mean_b),
        root_node_pitcher_effect=dict(unweighted_mean=unweighted_mean_q,
                                        pa_weighted_mean=weighted_mean_q),
        platoon_mix=dict(same=same_frac, opposite=opp_frac, unknown=unk_frac),
        home_batting_fraction=home_frac,
        season_mix={str(k): v for k, v in season_fracs.items()},
        x_fixed_convention=("PA-weighted MEAN structural covariate vector over the "
                             "training set, held fixed for every batter x pitcher pair "
                             "in the standardized-population comparisons; this averages "
                             "the linear covariate, not a probability."),
        category_diff_a_vs_b=per_cat,
        rare_vs_common=dict(rare_categories=rare, common_categories=common_cats,
                             rare_mean_abs_diff_batter=rare_mean_bat,
                             common_mean_abs_diff_batter=common_mean_bat,
                             rare_mean_abs_diff_pitcher=rare_mean_pit,
                             common_mean_abs_diff_pitcher=common_mean_pit,
                             ratio_batter_absolute=rare_mean_bat / common_mean_bat,
                             ratio_pitcher_absolute=rare_mean_pit / common_mean_pit,
                             rare_mean_rel_diff_batter=rare_rel_bat,
                             common_mean_rel_diff_batter=common_rel_bat,
                             rare_mean_rel_diff_pitcher=rare_rel_pit,
                             common_mean_rel_diff_pitcher=common_rel_pit,
                             ratio_batter_relative=rare_rel_bat / common_rel_bat,
                             ratio_pitcher_relative=rare_rel_pit / common_rel_pit),
        hr_rank=dict(spearman_rho=float(rho_hr), pvalue=float(pval_hr),
                     max_rank_displacement=max_disp_hr, n=n_bat,
                     top10_overlap=top10_overlap),
        value_proxy_rank=dict(spearman_rho=float(rho_val), pvalue=float(pval_val),
                               max_rank_displacement=max_disp_val, n=n_bat,
                               top10_overlap=top10_overlap_val),
        max_overall_category_diff_batter_side=max_overall_diff,
        verdict=verdict,
    )
    OUT_JSON.write_text(json.dumps(out, indent=1))
    log(f"wrote {OUT_JSON}")
    log(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
