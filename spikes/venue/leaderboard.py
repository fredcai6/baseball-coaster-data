"""Item 5: does the venue term move the batter leaderboard?

Reuses spikes/value/player_value.py's STEPS 2-3 convention verbatim (fits on
ALL data -- train+test -- as a descriptive leaderboard fit, applies the
STORED step-0 calibration from value_result.json rather than refitting it,
same QUAL_BAT_PA=150 qualifier, same per-season linear weights and baseline_a
loaded from value_result.json rather than re-derived).

Two models are fit on ALL PA rows at that convention:
  BASELINE   -- shape D exactly as player_value.py does it (sanity-checked
                against value_result.json's stored top15/bottom5 -- if this
                doesn't reproduce closely, the venue-side comparison below is
                not trustworthy and the script says so).
  VENUE      -- shape D + venue main effect + venue x stance deviation, at
                the (lam_bat, lam_pit, psi) from shape D and the (lam_ven,
                lam_vs) selected in spikes/venue/result.json (train-only
                selection, per the task spec), refit on ALL data.

"Venue set to the population mix, not the player's own mix" is implemented
the same way populations.py's Task 3 fixes the structural covariate: the
venue and venue x stance terms are LINEAR in eta (a categorical selector, no
interaction with the AO link), so their population-mean contribution is a
well-defined scalar -- the PA-weighted average of v_pad[venue_i] (and
w_pad[venue_stance_i]) over ALL PA rows, added once to the node's constant
term C[node], the same way X_fixed (PA-weighted mean structural covariate) is
added once in player_value.py. This is the SAME convention already used in
this repo (average the covariate, not a probability -- avoids reintroducing
the Task-2 Jensen's-gap issue from populations.py), just extended to the two
new terms. Every batter is then evaluated against the SAME population venue
mix instead of whatever their own b[batter] baked in.
"""
import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fuse"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pitch"))

import common  # noqa: E402
from analyze import structural  # noqa: E402
import step1 as S1  # noqa: E402
import venue_common as VC  # noqa: E402
import venue_model as VM  # noqa: E402
import fit_venue as FV  # noqa: E402  (SHAPE_D, load_shape_d_hp, pack)

CATS = common.CATEGORIES
CI = common.CAT_INDEX
N_CAT = len(CATS)
N_NODES = len(FV.SHAPE_D)
QUAL_BAT_PA = 150

VALUE_RESULT = os.path.join(os.path.dirname(HERE), "value", "value_result.json")
VENUE_RESULT = os.path.join(HERE, "result.json")
OUT_JSON = os.path.join(HERE, "leaderboard_result.json")

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def build_paths(nodes):
    paths = {ci: [] for ci in range(N_CAT)}
    for ni, (name, reach, pos) in enumerate(nodes):
        for ci in range(N_CAT):
            if ci in reach:
                paths[ci].append((ni, ci in pos))
    return paths


PATHS_D = build_paths(FV.SHAPE_D)


def ao_p(eta, psi):
    p, _, _, _ = S1.ao_prob(eta, psi)
    return p


def cat_matrix_from_node_p(node_p_list, n_rows):
    out = np.ones((n_rows, N_CAT)) if n_rows is not None else np.ones(N_CAT)
    for ci in range(N_CAT):
        for ni, is_pos in PATHS_D[ci]:
            p = node_p_list[ni]
            out[..., ci] *= p if is_pos else (1.0 - p)
    return out


def apply_calibration(P, a_hat, b_hat):
    logP = np.log(np.maximum(P, 1e-300))
    z = logP * a_hat + b_hat
    z = z - logsumexp(z, axis=1, keepdims=True)
    return np.exp(z)


# ---------------------------------------------------------------- baseline --
def fit_baseline_all(rows, hp, BI, PI, n_bat, n_pit, season_idx):
    node_params = []
    for name, reach, pos in FV.SHAPE_D:
        sub = [r for r in rows if r["y"] in reach]
        Xs = structural(sub, season_idx)
        bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
        pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
        yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
        h = hp[name]
        n = 1 + Xs.shape[1] + n_bat + n_pit
        r_ = minimize(S1.nll_grad, np.zeros(n),
                      args=(Xs, bi, pj, yv, n_bat, n_pit, h["psi"], h["lam_bat"], h["lam_pit"]),
                      jac=True, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-8})
        th = r_.x
        ps = Xs.shape[1]
        node_params.append(dict(name=name, alpha=float(th[0]), beta=th[1:1 + ps].copy(),
                                 b=th[1 + ps:1 + ps + n_bat].copy(), q=th[1 + ps + n_bat:].copy(),
                                 psi=h["psi"], converged=bool(r_.success)))
        log(f"  baseline fit {name:9} n={len(sub):>6} converged={r_.success}")
    return node_params


def fit_venue_all(rows, hp, venue_hp, BI, PI, VI, VSI, n_bat, n_pit, n_ven, n_vs, season_idx):
    node_params = []
    for name, reach, pos in FV.SHAPE_D:
        sub = [r for r in rows if r["y"] in reach]
        Xs = structural(sub, season_idx)
        bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
        pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
        vi = np.fromiter((VC.venue_index(r, VI) for r in sub), int, len(sub))
        vsi = np.fromiter((VC.venue_stance_index(r, VSI) for r in sub), int, len(sub))
        yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
        h = hp[name]
        vh = venue_hp[name]
        th = VM.fit(Xs, bi, pj, vi, vsi, yv, n_bat, n_pit, n_ven, n_vs,
                    h["psi"], h["lam_bat"], h["lam_pit"], vh["lam_ven"], vh["lam_vs"])
        ps = Xs.shape[1]
        _, _, _, _, v, w = VM.unpack(th, ps, n_bat, n_pit, n_ven, n_vs)
        node_params.append(dict(name=name, alpha=float(th[0]), beta=th[1:1 + ps].copy(),
                                 b=th[1 + ps:1 + ps + n_bat].copy(),
                                 q=th[1 + ps + n_bat:1 + ps + n_bat + n_pit].copy(),
                                 v=v.copy(), w=w.copy(), psi=h["psi"]))
        log(f"  venue fit    {name:9} n={len(sub):>6} lam_ven={vh['lam_ven']:g} lam_vs={vh['lam_vs']:g}")
    return node_params


def compute_avg_over_bat(B, Q, C, PSI, w_pit, n_bat, n_pit):
    out = np.zeros((n_bat, N_CAT))
    for bi in range(n_bat):
        node_p_vec = [ao_p(C[ni] + B[ni][bi] + Q[ni], PSI[ni]) for ni in range(N_NODES)]
        probs_vs_all = cat_matrix_from_node_p(node_p_vec, n_pit)
        out[bi] = w_pit @ probs_vs_all
    return out


def main():
    log("loading value_result.json (baseline_a, season weights, calibration) and "
        "spikes/venue/result.json (selected lam_ven/lam_vs per node)")
    vres = json.loads(open(VALUE_RESULT).read())
    calib = vres["step0_calibration"]
    assert calib["decision"] == "CALIBRATED"
    a_hat = np.array([calib["category_scale_a"][c] for c in CATS])
    b_hat = np.array([calib["category_offset_b"][c] for c in CATS])
    season_weights = vres["step1_season_weights"]
    baseline_a = {int(s): v for s, v in vres["baselines"]["all_pa"].items()}
    seasons_sorted = sorted(baseline_a)
    weight_vec = {s: np.array([season_weights[str(s)][c] for c in CATS]) for s in seasons_sorted}

    vres_venue = json.loads(open(VENUE_RESULT).read())
    hp, ref_dev = FV.load_shape_d_hp()
    venue_hp = {n: dict(lam_ven=vres_venue["nodes"][n]["lam_ven"],
                        lam_vs=vres_venue["nodes"][n]["lam_vs"]) for n in FV.NODE_NAMES}

    rows = VC.load_rows()
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}
    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)
    venues, VI, vs_keys, VSI = VC.build_venue_indices(rows)
    n_ven, n_vs = len(venues), len(vs_keys)
    log(f"rows={len(rows)} batters={n_bat} pitchers={n_pit} venues={n_ven}")

    bat_pa = Counter(r["batter"] for r in rows)
    pit_pa = Counter(r["pitcher"] for r in rows)
    w_bat_raw = np.array([bat_pa.get(b, 0) for b in bats], dtype=float)
    w_pit_raw = np.array([pit_pa.get(p, 0) for p in pits], dtype=float)
    w_pit = w_pit_raw / w_pit_raw.sum()

    X_fixed = structural(rows, season_idx).mean(axis=0)

    # ---- names ----
    name_bat = {}
    with open(common.PA_TABLE) as fh:
        for r in csv.DictReader(fh):
            bkey = r["batter_career"] or r["batter_pid"]
            name_bat.setdefault(bkey, r["batter_name"])

    batter_season_pa = Counter((r["batter"], r["season"]) for r in rows)
    qualified_batters = [(b, s) for (b, s), pa in batter_season_pa.items() if pa >= QUAL_BAT_PA]
    log(f"qualified batter-seasons: {len(qualified_batters)}")

    # ================= BASELINE (no venue) =================
    log("fitting BASELINE shape D on ALL data (sanity replication of player_value.py steps 2-3)")
    node_params_base = fit_baseline_all(rows, hp, BI, PI, n_bat, n_pit, season_idx)
    C_base = np.array([npm["alpha"] + float(X_fixed @ npm["beta"]) for npm in node_params_base])
    PSI = np.array([npm["psi"] for npm in node_params_base])
    B_base = [npm["b"] for npm in node_params_base]
    Q_base = [npm["q"] for npm in node_params_base]
    avg_bat_base = compute_avg_over_bat(B_base, Q_base, C_base, PSI, w_pit, n_bat, n_pit)
    avg_bat_base = apply_calibration(avg_bat_base, a_hat, b_hat)

    def raa(avg_bat, b, s):
        p = avg_bat[BI[b]]
        exp_rpa = float(p @ weight_vec[s])
        return (exp_rpa - baseline_a[s]) * 350.0

    base_rows = []
    for (b, s) in qualified_batters:
        base_rows.append(dict(batter_id=b, name=name_bat.get(b, b), season=s,
                              pa=batter_season_pa[(b, s)], raa_350=raa(avg_bat_base, b, s)))
    base_rows.sort(key=lambda d: -d["raa_350"])
    log("BASELINE top10 (sanity check vs value_result.json batter_leaderboard_top15):")
    for row in base_rows[:10]:
        log(f"  {row['name']:25} {row['season']}  PA={row['pa']:4d}  {row['raa_350']:+7.2f}")

    stored_top15 = vres["batter_leaderboard_top15"]
    stored_top10_names = [(d["name"], d["season"]) for d in stored_top15[:10]]
    my_top10_names = [(d["name"], d["season"]) for d in base_rows[:10]]
    sanity_match = stored_top10_names == my_top10_names
    log(f"SANITY: my baseline top10 == stored top10 order: {sanity_match}")
    if not sanity_match:
        log(f"  stored: {stored_top10_names}")
        log(f"  mine  : {my_top10_names}")

    # ================= VENUE model =================
    log("fitting VENUE-augmented shape D on ALL data at selected (lam_ven, lam_vs)")
    node_params_ven = fit_venue_all(rows, hp, venue_hp, BI, PI, VI, VSI, n_bat, n_pit,
                                    n_ven, n_vs, season_idx)
    C_struct = np.array([npm["alpha"] + float(X_fixed @ npm["beta"]) for npm in node_params_ven])
    B_ven = [npm["b"] for npm in node_params_ven]
    Q_ven = [npm["q"] for npm in node_params_ven]
    V_ven = [npm["v"] for npm in node_params_ven]
    W_ven = [npm["w"] for npm in node_params_ven]

    # population-mix venue offset: PA-weighted mean of v_pad[venue_i] / w_pad[venue_stance_i]
    # over ALL PA rows -- same "average the linear covariate" convention as X_fixed.
    vi_all = np.fromiter((VC.venue_index(r, VI) for r in rows), int, len(rows))
    vsi_all = np.fromiter((VC.venue_stance_index(r, VSI) for r in rows), int, len(rows))
    venue_offset = np.zeros(N_NODES)
    vs_offset = np.zeros(N_NODES)
    for ni in range(N_NODES):
        v_pad = np.concatenate([V_ven[ni], [0.0]])
        w_pad = np.concatenate([W_ven[ni], [0.0]])
        venue_offset[ni] = float(np.mean(v_pad[vi_all]))
        vs_offset[ni] = float(np.mean(w_pad[vsi_all]))
    C_venue = C_struct + venue_offset + vs_offset
    log(f"venue_offset by node: {dict(zip(FV.NODE_NAMES, venue_offset.round(5)))}")
    log(f"vs_offset by node:    {dict(zip(FV.NODE_NAMES, vs_offset.round(5)))}")

    avg_bat_venue = compute_avg_over_bat(B_ven, Q_ven, C_venue, PSI, w_pit, n_bat, n_pit)
    avg_bat_venue = apply_calibration(avg_bat_venue, a_hat, b_hat)

    venue_rows = []
    for (b, s) in qualified_batters:
        venue_rows.append(dict(batter_id=b, name=name_bat.get(b, b), season=s,
                               pa=batter_season_pa[(b, s)], raa_350=raa(avg_bat_venue, b, s)))
    venue_rows.sort(key=lambda d: -d["raa_350"])
    log("VENUE-adjusted top10:")
    for row in venue_rows[:10]:
        log(f"  {row['name']:25} {row['season']}  PA={row['pa']:4d}  {row['raa_350']:+7.2f}")

    # ================= comparison =================
    base_by_key = {(d["batter_id"], d["season"]): d["raa_350"] for d in base_rows}
    ven_by_key = {(d["batter_id"], d["season"]): d["raa_350"] for d in venue_rows}
    keys = qualified_batters
    base_vals = np.array([base_by_key[k] for k in keys])
    ven_vals = np.array([ven_by_key[k] for k in keys])
    delta = ven_vals - base_vals

    rho, pval = spearmanr(base_vals, ven_vals)
    sd_delta = float(np.std(delta))
    log(f"rank correlation (spearman) baseline vs venue-adjusted: rho={rho:.5f} (p={pval:.2g})")
    log(f"sd of per-player-season change in raa_350: {sd_delta:.3f}  "
        f"(pre-registered expectation from raw within-player estimator: 6.65)")

    order = np.argsort(-delta)
    movers_up = [dict(name=name_bat.get(keys[i][0], keys[i][0]), season=keys[i][1],
                      base=float(base_vals[i]), venue=float(ven_vals[i]), delta=float(delta[i]))
                 for i in order[:10]]
    movers_down = [dict(name=name_bat.get(keys[i][0], keys[i][0]), season=keys[i][1],
                        base=float(base_vals[i]), venue=float(ven_vals[i]), delta=float(delta[i]))
                   for i in order[-10:][::-1]]
    log("top 10 movers UP under venue compensation:")
    for m in movers_up:
        log(f"  {m['name']:25} {m['season']}  base={m['base']:+7.2f} -> venue={m['venue']:+7.2f}  "
            f"delta={m['delta']:+6.2f}")
    log("top 10 movers DOWN under venue compensation:")
    for m in movers_down:
        log(f"  {m['name']:25} {m['season']}  base={m['base']:+7.2f} -> venue={m['venue']:+7.2f}  "
            f"delta={m['delta']:+6.2f}")

    out = dict(
        sanity_baseline_matches_stored_top10=sanity_match,
        n_qualified_batter_seasons=len(qualified_batters),
        venue_offset_by_node={n: float(v) for n, v in zip(FV.NODE_NAMES, venue_offset)},
        vs_offset_by_node={n: float(v) for n, v in zip(FV.NODE_NAMES, vs_offset)},
        baseline_top15=base_rows[:15],
        venue_top15=venue_rows[:15],
        rank_correlation_spearman=float(rho),
        rank_correlation_pvalue=float(pval),
        sd_delta_raa_350=sd_delta,
        pre_registered_raw_sd=6.65,
        movers_up=movers_up,
        movers_down=movers_down,
    )
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    log(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
