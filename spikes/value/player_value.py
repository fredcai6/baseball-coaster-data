"""Player-value layer: runs above average per standardized playing time.

Builds on three existing spikes without modifying them:
  - spikes/pitch/step6_shapes.py / step6_result.json  -- SHAPE D, the winning
    tree (frozen-test deviance 3.94535), with its per-node lam_bat/lam_pit/psi.
  - spikes/value/re_result.json -- run expectancy + linear weights, pooled and
    per-season RE tables.
  - spikes/value/populations.py / populations_result.json -- the PA-weighted
    standardized-population / fixed-structural-covariate convention, reused
    here (not re-derived from scratch) for the "average over population, not
    evaluate-at-mean" form.

STEP 0 measures whether shape D's structural fix to the HR path (depth 6 -> 2)
already absorbed what shape A needed an out-of-fold affine recalibration for.
Cross-fits shape D over the SAME 5 folds step3_crossfit.py used (fold seed
20260830), runs the OOF-vs-in-sample sanity gate, fits the 20-param affine
recalibration OOF, and applies it to the frozen test. The gain and the fitted
per-category scale vector decide whether the value layer (steps 2-3) scores
RAW or CALIBRATED model probabilities -- decided from the number, not assumed.

STEP 1 computes PER-SEASON linear weights. re_result.json only stores POOLED
linear weights (by raw outcome_type) plus PER-SEASON run-expectancy tables --
it does not store per-season weights directly. This script re-derives them
the same way run_expectancy.py derives the pooled ones (RE(after) - RE(before)
+ runs_on_play, averaged by outcome_type) but looks up RE in that season's OWN
table instead of the pooled table, reusing run_expectancy.py's game-parsing
helpers (iter_games/half_innings/check_clean/state_key/load_disposed_ids) --
not rebuilding the RE tables themselves, which are already correct and frozen
in re_result.json. Outcome types are then rolled up to the model's 10
categories via common.OUTCOME_MAP, PA-count-weighted, and expressed relative
to that season's own generic-out mean (out = 0).

STEPS 2-3 fit shape D on ALL data (train+test) as a descriptive leaderboard
fit -- NOT a performance claim; the honest number is the train-only 3.94535
from step 0. For each batter/pitcher, the "average over population" 10-category
probability vector is computed once (their skill is a single scalar in this
model -- there is no batter x season interaction term), then scored against
EACH season's own linear weights and baseline to produce a batter-season row.
This is the only way this model's structure can express "same skill, different
run environment."

STEP 4 reports two measured baselines (all-PA vs qualified-only) and their
offset -- both computed from EMPIRICAL (not model-predicted) category
frequencies, since this is a selection-bias question, not a model-bias one.

STEP 5 bootstraps over GAMES (resample with replacement, 20 replicates),
refitting shape D's 9 nodes at FIXED (lam_bat, lam_pit, psi) per replicate
(no hyperparameter re-search -- same optimism accepted by step3_crossfit.py
and populations.py), and reports rank stability of the batter leaderboard.

Run: PYTHONPATH=pipeline .venv/bin/python spikes/value/player_value.py
(repo venv only -- system python3 has no numpy)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
SPIKES = HERE.parent
PITCH = SPIKES / "pitch"
FUSE = SPIKES / "fuse"

sys.path.insert(0, str(PITCH))
sys.path.insert(0, str(SPIKES))
sys.path.insert(0, str(FUSE))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import step1 as S1  # noqa: E402  (reuses nll_grad / ao_prob -- the frozen link)
import run_expectancy as RE_MOD  # noqa: E402  (reuses game-parsing helpers only)
from analyze import structural  # noqa: E402

CATS = common.CATEGORIES
CI = common.CAT_INDEX
N_CAT = len(CATS)

STEP6_RESULT = PITCH / "step6_result.json"
SPLIT_PATH = SPIKES / "split.json"
RE_RESULT = HERE / "re_result.json"
OUT_JSON = HERE / "value_result.json"

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ============================================================================
# SHAPE D -- copied verbatim from spikes/pitch/step6_shapes.py (not imported,
# since step6_shapes.py has no __main__-guard-safe importable SHAPES without
# also pulling in its argparse-driven main()). The winner: con_HR promoted
# from depth 6 to depth 2, frozen-test deviance 3.94535.
# ============================================================================
S = lambda *cs: frozenset(CI[c] for c in cs)  # noqa: E731
ALL = S(*CATS)

SHAPE_D = [
    ("root",       ALL,                                          S("K", "BB", "HBP")),
    ("tto_K",      S("K", "BB", "HBP"),                          S("K")),
    ("tto_BB",     S("BB", "HBP"),                                S("BB")),
    ("con_HR",     ALL - S("K", "BB", "HBP"),                    S("HR")),
    ("con_OTH",    ALL - S("K", "BB", "HBP", "HR"),               S("OTHER")),
    ("con_OUT",    ALL - S("K", "BB", "HBP", "HR", "OTHER"),      S("F", "G")),
    ("out_F",      S("F", "G"),                                  S("F")),
    ("hit_1B",     S("1B", "2B", "3B"),                           S("1B")),
    ("hit_2B",     S("2B", "3B"),                                 S("2B")),
]
N_NODES = len(SHAPE_D)


def build_paths(nodes):
    paths = {ci: [] for ci in range(N_CAT)}
    for ni, (name, reach, pos) in enumerate(nodes):
        for ci in range(N_CAT):
            if ci in reach:
                paths[ci].append((ni, ci in pos))
    return paths


PATHS_D = build_paths(SHAPE_D)


def load_shape_d_hp():
    d = json.loads(STEP6_RESULT.read_text())
    hp = {}
    for nd in d["D"]["nodes"]:
        hp[nd["node"]] = dict(lam_bat=nd["lam_bat"], lam_pit=nd["lam_pit"], psi=nd["psi"])
    return hp, d["D"]["total_deviance"]


# ============================================================================
# generic node fit / tree-walk (shape-D specific via SHAPE_D/PATHS_D defaults)
# ============================================================================
def fit_with_status(Xs, bi, pj, yv, n_bat, n_pit, psi, lam_b, lam_p):
    n = 1 + Xs.shape[1] + n_bat + n_pit
    x0 = np.zeros(n)
    r = minimize(S1.nll_grad, x0, args=(Xs, bi, pj, yv, n_bat, n_pit, psi, lam_b, lam_p),
                 jac=True, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-11, "gtol": 1e-8})
    return r.x, bool(r.success)


def fit_all_nodes(fit_rows, hp, BI, PI, n_bat, n_pit, season_idx):
    node_params = []
    all_converged = True
    for name, reach, pos in SHAPE_D:
        sub = [r for r in fit_rows if r["y"] in reach]
        Xs = structural(sub, season_idx)
        bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
        pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
        yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
        h = hp[name]
        th, ok = fit_with_status(Xs, bi, pj, yv, n_bat, n_pit, h["psi"], h["lam_bat"], h["lam_pit"])
        all_converged = all_converged and ok
        ps = Xs.shape[1]
        node_params.append(dict(
            name=name, alpha=float(th[0]), beta=th[1:1 + ps].copy(),
            b=th[1 + ps:1 + ps + n_bat].copy(), q=th[1 + ps + n_bat:].copy(),
            psi=h["psi"], converged=ok,
        ))
    return node_params, all_converged


def category_probs(rows, node_params, season_idx, BI, PI):
    Xs = structural(rows, season_idx)
    ps = Xs.shape[1]
    bi = np.fromiter((BI[r["batter"]] for r in rows), int, len(rows))
    pj = np.fromiter((PI[r["pitcher"]] for r in rows), int, len(rows))
    node_p = []
    for npm in node_params:
        eta = npm["alpha"] + Xs @ npm["beta"] + npm["b"][bi] + npm["q"][pj]
        p, _, _, _ = S1.ao_prob(eta, npm["psi"])
        node_p.append(p)
    P = np.ones((len(rows), N_CAT))
    for ci in range(N_CAT):
        for ni, is_pos in PATHS_D[ci]:
            p = node_p[ni]
            P[:, ci] *= p if is_pos else (1.0 - p)
    return P


def deviance(P, y):
    return -2.0 * np.mean(np.log(np.maximum(P[np.arange(len(y)), y], 1e-300)))


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


# ============================================================================
# STEP 0 -- crossfit shape D, sanity gate, OOF affine recalibration
# ============================================================================
def step0_calibration(rows, train_g, test_g, hp, ref_dev, BI, PI, n_bat, n_pit, season_idx):
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    y_tr = np.array([r["y"] for r in tr])
    y_te = np.array([r["y"] for r in te])

    log("STEP 0: fitting shape D on full TRAIN (reference fit)...")
    node_params_full, ok_full = fit_all_nodes(tr, hp, BI, PI, n_bat, n_pit, season_idx)
    P_tr = category_probs(tr, node_params_full, season_idx, BI, PI)
    P_te = category_probs(te, node_params_full, season_idx, BI, PI)
    d_in = float(deviance(P_tr, y_tr))
    d_te = float(deviance(P_te, y_te))
    log(f"  in-sample dev={d_in:.5f}  frozen-test dev={d_te:.5f}  "
        f"(step6 reference {ref_dev:.5f}, converged={ok_full})")

    N_FOLDS, FOLD_SEED = 5, 20260830  # SAME as step3_crossfit.py
    g = sorted(train_g)
    rs = np.random.RandomState(FOLD_SEED)
    rs.shuffle(g)
    folds = [set(g[i::N_FOLDS]) for i in range(N_FOLDS)]

    oof = np.zeros((len(tr), N_CAT))
    fold_devs = []
    all_folds_converged = True
    for k, fold in enumerate(folds):
        mask = np.array([r["game_id"] in fold for r in tr])
        fit_rows = [r for r, m in zip(tr, mask) if not m]
        pred_rows = [r for r, m in zip(tr, mask) if m]
        node_params_k, ok_k = fit_all_nodes(fit_rows, hp, BI, PI, n_bat, n_pit, season_idx)
        all_folds_converged = all_folds_converged and ok_k
        Pk = category_probs(pred_rows, node_params_k, season_idx, BI, PI)
        oof[mask] = Pk
        d_fold = float(deviance(Pk, y_tr[mask]))
        fold_devs.append(d_fold)
        log(f"  fold {k}: fit={len(fit_rows)} pred={len(pred_rows)} "
            f"converged={ok_k} oof_dev={d_fold:.5f}")

    d_oof = float(deviance(oof, y_tr))
    near_test = abs(d_oof - d_te) < abs(d_oof - d_in)
    log(f"  SANITY  in-sample={d_in:.5f}  oof={d_oof:.5f}  frozen-test={d_te:.5f}")
    log(f"  -> OOF sits {'NEAR TEST (clean)' if near_test else 'NEAR IN-SAMPLE (LEAK)'}")

    result = dict(fold_seed=FOLD_SEED, n_folds=N_FOLDS,
                  in_sample=d_in, oof=d_oof, frozen_test=d_te,
                  fold_deviances=fold_devs, sanity_passed=bool(near_test),
                  all_folds_converged=bool(all_folds_converged))

    if not near_test:
        log("  ABORT: sanity gate FAILED -- calibration step not run.")
        result["decision"] = "ABORT"
        return result, None

    logP_oof = np.log(np.maximum(oof, 1e-300))
    logP_te = np.log(np.maximum(P_te, 1e-300))

    def f(params):
        a, b = params[:N_CAT], params[N_CAT:]
        z = logP_oof * a + b
        z = z - logsumexp(z, axis=1, keepdims=True)
        return -2.0 * np.mean(z[np.arange(len(y_tr)), y_tr])

    r = minimize(f, np.concatenate([np.ones(N_CAT), np.zeros(N_CAT)]),
                 method="L-BFGS-B", options={"maxiter": 2000})
    a_hat, b_hat = r.x[:N_CAT], r.x[N_CAT:]

    z_te = logP_te * a_hat + b_hat
    z_te = z_te - logsumexp(z_te, axis=1, keepdims=True)
    d_te_cal = float(-2.0 * np.mean(z_te[np.arange(len(y_te)), y_te]))
    gain = d_te_cal - d_te

    log(f"  calibration fit converged={r.success}")
    log(f"  raw frozen test      = {d_te:.5f}")
    log(f"  + OOF-fitted affine  = {d_te_cal:.5f}  ({gain:+.5f})")
    for i, c in enumerate(CATS):
        log(f"    {c:6} a={a_hat[i]:+.4f}  b={b_hat[i]:+.4f}")

    hr_a = float(a_hat[CI["HR"]])
    # Decision rule: shape A's OOF recalibration was worth -0.00129, driven by
    # HR (a=1.25) and 2B (a=0.61). Call shape D "fixed" if the gain is at
    # least an order of magnitude smaller than that AND the HR scale sits
    # close to 1.0 (no more amplification needed on the shortened HR path).
    decision = "RAW" if (abs(gain) < 0.0003 and abs(hr_a - 1.0) < 0.15) else "CALIBRATED"
    log(f"  DECISION: use {decision} probabilities for the value layer "
        f"(gain={gain:+.5f}, HR scale={hr_a:+.4f}; shape A reference: gain=-0.00129, HR scale=1.25)")

    result["decision"] = decision
    result["gain"] = gain
    result["cal_test"] = d_te_cal
    result["category_scale_a"] = {c: float(a_hat[i]) for i, c in enumerate(CATS)}
    result["category_offset_b"] = {c: float(b_hat[i]) for i, c in enumerate(CATS)}

    cal = (a_hat, b_hat) if decision == "CALIBRATED" else None
    return result, cal


# ============================================================================
# STEP 1 -- per-season linear weights, relative to that season's generic out
# ============================================================================
def step1_season_weights():
    log("STEP 1: deriving per-season linear weights from re_result.json's "
        "per-season RE tables (re-walking games via run_expectancy.py's own "
        "parsing helpers -- NOT rebuilding the RE tables).")
    re_result = json.loads(RE_RESULT.read_text())
    re_by_season = re_result["re_table_by_season"]

    def re_lookup(table, key):
        outs = int(key.split("|")[1])
        if outs >= 3:
            return 0.0
        cell = table[key]
        return cell["re"] if cell["re"] is not None else 0.0

    disposed = RE_MOD.load_disposed_ids()
    pa_by_season = defaultdict(list)  # season -> [(outcome_type, run_value)]
    for season in RE_MOD.SEASONS:
        for path, game in RE_MOD.iter_games(season):
            game_id = game.get("game_id") or path
            if game_id in disposed:
                continue
            groups = RE_MOD.half_innings(game["events"])
            for _, evs in groups.items():
                if not RE_MOD.check_clean(evs):
                    continue
                for e in evs:
                    if e["kind"] != "plate_appearance":
                        continue
                    d = e["_derived"]
                    table = re_by_season[season]
                    before_key = RE_MOD.state_key(d["bases_before"], d["outs_before"])
                    after_key = RE_MOD.state_key(d["bases_after"], d["outs_after"])
                    re_before = re_lookup(table, before_key)
                    re_after = 0.0 if d["outs_after"] >= 3 else re_lookup(table, after_key)
                    run_value = re_after - re_before + d["runs_on_play"]
                    pa_by_season[season].append((e["outcome"]["type"], run_value))

    season_weights = {}       # season(int) -> {cat: weight relative to out}
    season_weights_raw = {}   # season(int) -> {cat: raw mean run value}
    season_generic_out = {}
    for season in RE_MOD.SEASONS:
        by_type = defaultdict(list)
        for otype, rv in pa_by_season[season]:
            by_type[otype].append(rv)
        out_vals = []
        for t in RE_MOD.OUT_TYPES:
            out_vals.extend(by_type.get(t, []))
        go_mean = sum(out_vals) / len(out_vals)
        season_generic_out[int(season)] = go_mean

        cat_vals = defaultdict(list)
        for otype, rv in pa_by_season[season]:
            cat = common.OUTCOME_MAP[otype]
            cat_vals[cat].append(rv)
        w_raw, w_rel = {}, {}
        for cat in CATS:
            vals = cat_vals.get(cat, [])
            m = sum(vals) / len(vals) if vals else 0.0
            w_raw[cat] = m
            w_rel[cat] = m - go_mean
        season_weights_raw[int(season)] = w_raw
        season_weights[int(season)] = w_rel

    log("  three seasons' weight vectors (runs, relative to generic out = 0):")
    header = f"{'cat':6}" + "".join(f"{s:>12}" for s in sorted(season_weights))
    log("  " + header)
    for cat in CATS:
        row = "".join(f"{season_weights[s][cat]:>12.4f}" for s in sorted(season_weights))
        log(f"  {cat:6}{row}")
    log(f"  generic-out mean run value by season: "
        f"{ {s: round(v,4) for s,v in season_generic_out.items()} }")

    return season_weights, season_weights_raw, season_generic_out


# ============================================================================
# main
# ============================================================================
def main():
    log("loading PA table + split...")
    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}
    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)
    log(f"rows={len(rows)} train_games={len(train_g)} test_games={len(test_g)} "
        f"batters={n_bat} pitchers={n_pit}")

    hp, ref_dev = load_shape_d_hp()

    # ---------------- STEP 0 ----------------
    step0_result, calibration = step0_calibration(
        rows, train_g, test_g, hp, ref_dev, BI, PI, n_bat, n_pit, season_idx)
    if step0_result.get("decision") == "ABORT":
        log("ABORTING: step 0 sanity gate failed. Writing partial result and exiting.")
        OUT_JSON.write_text(json.dumps({"step0_calibration": step0_result}, indent=1))
        return

    # ---------------- STEP 1 ----------------
    season_weights, season_weights_raw, season_generic_out = step1_season_weights()
    seasons_sorted = sorted(season_weights)  # [2024, 2025, 2026]
    weight_vec = {s: np.array([season_weights[s][c] for c in CATS]) for s in seasons_sorted}

    # ---------------- STEPS 2-3: fit shape D on ALL data ----------------
    log("STEPS 2-3: fitting shape D on ALL data (train+test) -- descriptive "
        "leaderboard fit, NOT a performance claim (honest number is the "
        f"train-only {step0_result['frozen_test']:.5f} from step 0).")
    node_params_all, ok_all = fit_all_nodes(rows, hp, BI, PI, n_bat, n_pit, season_idx)
    log(f"  all-data fit converged={ok_all}")

    bat_pa = Counter(r["batter"] for r in rows)
    pit_pa = Counter(r["pitcher"] for r in rows)
    w_bat_raw = np.array([bat_pa.get(b, 0) for b in bats], dtype=float)
    w_pit_raw = np.array([pit_pa.get(p, 0) for p in pits], dtype=float)
    w_bat = w_bat_raw / w_bat_raw.sum()
    w_pit = w_pit_raw / w_pit_raw.sum()

    X_fixed = structural(rows, season_idx).mean(axis=0)
    log(f"  X_fixed (PA-weighted mean structural covariate over ALL data) = {X_fixed}")

    C = np.array([npm["alpha"] + float(X_fixed @ npm["beta"]) for npm in node_params_all])
    PSI = np.array([npm["psi"] for npm in node_params_all])
    B = [npm["b"] for npm in node_params_all]
    Q = [npm["q"] for npm in node_params_all]

    def compute_avg_over_bat(B_, Q_, C_, PSI_, w_pit_, n_bat_, n_pit_):
        out = np.zeros((n_bat_, N_CAT))
        for bi in range(n_bat_):
            node_p_vec = [ao_p(C_[ni] + B_[ni][bi] + Q_[ni], PSI_[ni]) for ni in range(N_NODES)]
            probs_vs_all = cat_matrix_from_node_p(node_p_vec, n_pit_)
            out[bi] = w_pit_ @ probs_vs_all
        return out

    def compute_avg_over_pit(B_, Q_, C_, PSI_, w_bat_, n_bat_, n_pit_):
        out = np.zeros((n_pit_, N_CAT))
        for pi in range(n_pit_):
            node_p_vec = [ao_p(C_[ni] + B_[ni] + Q_[ni][pi], PSI_[ni]) for ni in range(N_NODES)]
            probs_vs_all = cat_matrix_from_node_p(node_p_vec, n_bat_)
            out[pi] = w_bat_ @ probs_vs_all
        return out

    log("  computing average-over-population probability vectors "
        "(batter vs standardized pitcher population, pitcher vs standardized batter population)...")
    avg_over_bat = compute_avg_over_bat(B, Q, C, PSI, w_pit, n_bat, n_pit)
    avg_over_pit = compute_avg_over_pit(B, Q, C, PSI, w_bat, n_bat, n_pit)
    assert np.allclose(avg_over_bat.sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(avg_over_pit.sum(axis=1), 1.0, atol=1e-8)

    if calibration is not None:
        log("  applying STEP-0 CALIBRATED transform to population probability vectors")
        a_hat, b_hat = calibration
        avg_over_bat = apply_calibration(avg_over_bat, a_hat, b_hat)
        avg_over_pit = apply_calibration(avg_over_pit, a_hat, b_hat)
    else:
        log("  using RAW model probabilities (step 0 decision)")

    # ---------------- name lookups ----------------
    name_bat, name_pit = {}, {}
    with open(common.PA_TABLE) as fh:
        for r in csv.DictReader(fh):
            bkey = r["batter_career"] or r["batter_pid"]
            pkey = r["pitcher_career"] or r["pitcher_pid"]
            name_bat.setdefault(bkey, r["batter_name"])
            name_pit.setdefault(pkey, r["pitcher_name"])

    # ---------------- batter-season / pitcher-season tables ----------------
    batter_season_pa = Counter((r["batter"], r["season"]) for r in rows)
    pitcher_season_bf = Counter((r["pitcher"], r["season"]) for r in rows)
    pitcher_season_starter_votes = defaultdict(Counter)
    for r in rows:
        pitcher_season_starter_votes[(r["pitcher"], r["season"])][r["is_starter"]] += 1

    QUAL_BAT_PA = 150
    QUAL_STARTER_BF = 150
    QUAL_RELIEVER_BF = 50

    qualified_batters = [(b, s) for (b, s), pa in batter_season_pa.items() if pa >= QUAL_BAT_PA]
    log(f"  batter-seasons qualifying (>= {QUAL_BAT_PA} PA): {len(qualified_batters)} "
        f"of {len(batter_season_pa)} total batter-seasons")

    pitcher_role = {}  # (p,s) -> "starter"/"reliever"
    for k, votes in pitcher_season_starter_votes.items():
        pitcher_role[k] = "starter" if votes[True] >= votes[False] else "reliever"
    qualified_starters = [(p, s) for (p, s), bf in pitcher_season_bf.items()
                           if pitcher_role[(p, s)] == "starter" and bf >= QUAL_STARTER_BF]
    qualified_relievers = [(p, s) for (p, s), bf in pitcher_season_bf.items()
                            if pitcher_role[(p, s)] == "reliever" and bf >= QUAL_RELIEVER_BF]
    log(f"  pitcher-seasons qualifying: starters(>= {QUAL_STARTER_BF} BF)={len(qualified_starters)}  "
        f"relievers(>= {QUAL_RELIEVER_BF} BF)={len(qualified_relievers)}")

    # ---------------- STEP 4: baselines ----------------
    log("STEP 4: baselines (measured / empirical, not model-predicted)")
    qualified_batter_set = set(qualified_batters)
    baseline_a, baseline_b = {}, {}
    for s in seasons_sorted:
        season_rows = [r for r in rows if r["season"] == s]
        cnt_all = Counter(r["cat"] for r in season_rows)
        n_all = sum(cnt_all.values())
        baseline_a[s] = sum((cnt_all.get(c, 0) / n_all) * season_weights[s][c] for c in CATS)

        qual_rows = [r for r in season_rows if (r["batter"], s) in qualified_batter_set]
        cnt_q = Counter(r["cat"] for r in qual_rows)
        n_q = sum(cnt_q.values())
        baseline_b[s] = sum((cnt_q.get(c, 0) / n_q) * season_weights[s][c] for c in CATS) if n_q else None
        log(f"  season {s}: baseline_a(all PA)={baseline_a[s]:+.5f}  "
            f"baseline_b(qualified only, n={n_q})={baseline_b[s]:+.5f}  "
            f"offset(b-a)*350={ (baseline_b[s]-baseline_a[s])*350:+.3f} runs/350PA")

    n_all_pa = sum(1 for r in rows if r["season"] in baseline_a)
    overall_offset_350 = sum(
        (baseline_b[s] - baseline_a[s]) * 350 * (sum(1 for r in rows if r["season"] == s) / len(rows))
        for s in seasons_sorted
    )
    log(f"  PA-weighted overall offset (qualified-only minus all-PA baseline): "
        f"{overall_offset_350:+.3f} runs/350PA")

    # ---------------- STEP 2: batter RAA / 350 PA ----------------
    def batter_raa(b, s):
        p = avg_over_bat[BI[b]]
        exp_rpa = float(p @ weight_vec[s])
        return (exp_rpa - baseline_a[s]) * 350.0

    batter_rows = []
    for (b, s) in qualified_batters:
        raa = batter_raa(b, s)
        pa = batter_season_pa[(b, s)]
        nm = name_bat.get(b, b)
        batter_rows.append(dict(batter_id=b, name=nm, season=s, pa=pa, raa_350=raa))
    batter_rows.sort(key=lambda d: -d["raa_350"])

    log("STEP 2: top 15 qualified batter-seasons (runs above average / 350 PA):")
    for row in batter_rows[:15]:
        log(f"  {row['name']:25} {row['season']}  PA={row['pa']:4d}  {row['raa_350']:+7.2f}")
    log("STEP 2: bottom 5 qualified batter-seasons:")
    for row in batter_rows[-5:]:
        log(f"  {row['name']:25} {row['season']}  PA={row['pa']:4d}  {row['raa_350']:+7.2f}")

    # ---------------- STEP 3: pitcher RAA / 100 BF ----------------
    def pitcher_raa(p, s):
        pv = avg_over_pit[PI[p]]
        exp_rpa = float(pv @ weight_vec[s])
        return (exp_rpa - baseline_a[s]) * 100.0

    pitcher_rows = []
    for (p, s) in qualified_starters + qualified_relievers:
        raa = pitcher_raa(p, s)
        bf = pitcher_season_bf[(p, s)]
        nm = name_pit.get(p, p)
        pitcher_rows.append(dict(pitcher_id=p, name=nm, season=s, bf=bf,
                                  role=pitcher_role[(p, s)], raa_100=raa))
    # sort ascending: negative = fewer runs allowed than average = better pitcher
    pitcher_rows.sort(key=lambda d: d["raa_100"])

    log("STEP 3: top 15 pitcher-seasons (runs above average ALLOWED / 100 BF; "
        "negative = better / fewer runs allowed than average):")
    for row in pitcher_rows[:15]:
        log(f"  {row['name']:25} {row['season']}  {row['role']:9} BF={row['bf']:4d}  {row['raa_100']:+7.2f}")

    # ---------------- STEP 5: bootstrap rank stability ----------------
    log("STEP 5: bootstrap over games (20 replicates), refitting shape D at "
        "FIXED per-node hyperparameters each time (no re-search).")
    all_games = sorted({r["game_id"] for r in rows})
    rows_by_game = defaultdict(list)
    for r in rows:
        rows_by_game[r["game_id"]].append(r)

    original_raa = {(b, s): batter_raa(b, s) for (b, s) in qualified_batters}
    ranked_orig = sorted(qualified_batters, key=lambda k: -original_raa[k])
    rank_orig = {k: i + 1 for i, k in enumerate(ranked_orig)}
    top20_orig = ranked_orig[:20]
    top5_orig = set(ranked_orig[:5])
    top1_key = ranked_orig[0]
    log(f"  original #1 batter: {name_bat.get(top1_key[0], top1_key[0])} "
        f"{top1_key[1]}  RAA/350={original_raa[top1_key]:+.2f}")

    n_reps = 20
    rng = np.random.RandomState(20260830)
    rep_rank_changes = []
    top5_preserved = 0
    top1_values = []
    n_dropped = 0
    n_ok = 0
    for rep in range(n_reps):
        sampled_games = rng.choice(all_games, size=len(all_games), replace=True)
        boot_rows = []
        for g in sampled_games:
            boot_rows.extend(rows_by_game[g])
        node_params_b, ok_b = fit_all_nodes(boot_rows, hp, BI, PI, n_bat, n_pit, season_idx)
        if not ok_b:
            n_dropped += 1
            log(f"  rep {rep}: FAILED to converge -- dropped")
            continue
        n_ok += 1

        bat_pa_b = Counter(r["batter"] for r in boot_rows)
        pit_pa_b = Counter(r["pitcher"] for r in boot_rows)
        w_bat_b = np.array([bat_pa_b.get(b, 0) for b in bats], dtype=float)
        w_pit_b = np.array([pit_pa_b.get(p, 0) for p in pits], dtype=float)
        w_bat_b = w_bat_b / w_bat_b.sum()
        w_pit_b = w_pit_b / w_pit_b.sum()
        X_fixed_b = structural(boot_rows, season_idx).mean(axis=0)
        C_b = np.array([npm["alpha"] + float(X_fixed_b @ npm["beta"]) for npm in node_params_b])
        PSI_b = np.array([npm["psi"] for npm in node_params_b])
        B_b = [npm["b"] for npm in node_params_b]
        Q_b = [npm["q"] for npm in node_params_b]

        avg_over_bat_b = compute_avg_over_bat(B_b, Q_b, C_b, PSI_b, w_pit_b, n_bat, n_pit)
        if calibration is not None:
            avg_over_bat_b = apply_calibration(avg_over_bat_b, *calibration)

        raa_b = {}
        for (b, s) in qualified_batters:
            p = avg_over_bat_b[BI[b]]
            exp_rpa = float(p @ weight_vec[s])
            raa_b[(b, s)] = (exp_rpa - baseline_a[s]) * 350.0

        ranked_b = sorted(qualified_batters, key=lambda k: -raa_b[k])
        rank_b = {k: i + 1 for i, k in enumerate(ranked_b)}
        for k in top20_orig:
            rep_rank_changes.append(abs(rank_b[k] - rank_orig[k]))
        if set(ranked_b[:5]) == top5_orig:
            top5_preserved += 1
        top1_values.append(raa_b[top1_key])
        log(f"  rep {rep}: converged, top1 RAA/350={raa_b[top1_key]:+.2f}, "
            f"top5 preserved={set(ranked_b[:5]) == top5_orig}")

    mean_abs_rank_change = float(np.mean(rep_rank_changes)) if rep_rank_changes else None
    top5_frac = top5_preserved / n_ok if n_ok else None
    ci90 = np.percentile(top1_values, [5, 95]).tolist() if top1_values else None
    log(f"  bootstrap done: {n_ok} converged / {n_reps} attempted, {n_dropped} dropped")
    log(f"  mean |rank change| for original top-20 batters: {mean_abs_rank_change}")
    log(f"  top-5 set preserved in {top5_preserved}/{n_ok} replicates ({top5_frac})")
    log(f"  90% interval on original #1 batter's RAA/350: {ci90}")

    # ---------------- write result ----------------
    out = dict(
        step0_calibration=step0_result,
        step1_season_weights=season_weights,
        step1_season_weights_raw=season_weights_raw,
        step1_season_generic_out=season_generic_out,
        all_data_fit_converged=bool(ok_all),
        n_batter_seasons_total=len(batter_season_pa),
        n_batter_seasons_qualified=len(qualified_batters),
        n_pitcher_seasons_starters_qualified=len(qualified_starters),
        n_pitcher_seasons_relievers_qualified=len(qualified_relievers),
        qualifiers=dict(batter_pa=QUAL_BAT_PA, starter_bf=QUAL_STARTER_BF, reliever_bf=QUAL_RELIEVER_BF),
        baselines=dict(
            all_pa={str(s): baseline_a[s] for s in seasons_sorted},
            qualified_only={str(s): baseline_b[s] for s in seasons_sorted},
            offset_runs_per_350pa={str(s): (baseline_b[s] - baseline_a[s]) * 350.0 for s in seasons_sorted},
            overall_pa_weighted_offset_runs_per_350pa=overall_offset_350,
        ),
        batter_leaderboard_top15=batter_rows[:15],
        batter_leaderboard_bottom5=batter_rows[-5:],
        pitcher_leaderboard_top15=pitcher_rows[:15],
        bootstrap=dict(
            n_reps_attempted=n_reps, n_reps_converged=n_ok, n_reps_dropped=n_dropped,
            mean_abs_rank_change_top20=mean_abs_rank_change,
            top5_preserved_fraction=top5_frac,
            top1_batter=dict(name=name_bat.get(top1_key[0], top1_key[0]), season=top1_key[1],
                              original_raa_350=original_raa[top1_key],
                              ci90_raa_350=ci90),
        ),
    )
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
