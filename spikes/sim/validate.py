"""Validation driver: runs both advancement models (naive, empirical) on the
frozen TEST games, N sims/game, and reports run-total and win-probability
accuracy against two baselines. Writes spikes/sim/result.json.

Actual outcomes (runs, winner, ties) are read from games/<season>/<id>.json's
`linescore.totals` -- the authoritative box score -- rather than re-summed
from pa_table, since ties are only visible there (the 10th linescore column
reads 0-0 for the 82 games the league settles with an unparsed HR derby) and
pa_table's per-PA data is exactly what the disposed-game exclusion is
protecting against.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPIKES = HERE.parent
sys.path.insert(0, str(SPIKES))
sys.path.insert(0, str(SPIKES / "value"))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import run_expectancy as RE_MOD  # noqa: E402  (load_disposed_ids only)
from data import load_pa_full  # noqa: E402
from simulator import SimModel, build_game_context, simulate_game  # noqa: E402

N_SIMS = 2000
SEED = 20260901
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def load_actual(season, game_id):
    path = SPIKES.parent / "games" / str(season) / f"{game_id}.json"
    d = json.loads(path.read_text())
    tot = d["linescore"]["totals"]
    return int(tot["away"]["R"]), int(tot["home"]["R"])


# ============================================================================
# baselines
# ============================================================================
def train_baselines(rows, train_g, disposed):
    """Per (team, season) mean runs scored/allowed from TRAIN (non-disposed)
    games, plus the overall non-tied home-win rate -- used by the two win
    -probability baselines."""
    by_game = {}
    for r in rows:
        by_game.setdefault(r["game_id"], []).append(r)

    rs = {}   # (team, season) -> list of runs scored
    ra = {}   # (team, season) -> list of runs allowed
    home_wins = away_wins = ties = 0
    n_used = 0
    for gid, grs in by_game.items():
        if gid not in train_g or gid in disposed:
            continue
        season = grs[0]["season"]
        home_team = grs[0]["home_team"]
        away_rows = [r for r in grs if not r["batting_is_home"]]
        if not away_rows:
            continue
        away_team = away_rows[0]["batting_team"]
        try:
            away_r, home_r = load_actual(season, gid)
        except FileNotFoundError:
            continue
        n_used += 1
        rs.setdefault((home_team, season), []).append(home_r)
        ra.setdefault((home_team, season), []).append(away_r)
        rs.setdefault((away_team, season), []).append(away_r)
        ra.setdefault((away_team, season), []).append(home_r)
        if home_r > away_r:
            home_wins += 1
        elif away_r > home_r:
            away_wins += 1
        else:
            ties += 1

    const_home_win_rate = home_wins / (home_wins + away_wins)
    league_rs = {s: [] for s in set(k[1] for k in rs)}
    for (t, s), vals in rs.items():
        league_rs[s].extend(vals)
    league_avg = {s: float(np.mean(v)) for s, v in league_rs.items()}

    log(f"TRAIN baselines built on {n_used} games: home_wins={home_wins} away_wins={away_wins} "
        f"ties={ties}  const_home_win_rate={const_home_win_rate:.4f}")
    for s, v in sorted(league_avg.items()):
        log(f"  season {s}: league avg runs/team-game = {v:.2f}")

    team_mean = {}
    for k, v in rs.items():
        team_mean[k] = (float(np.mean(v)), float(np.mean(ra[k])), len(v))
    return dict(const_home_win_rate=const_home_win_rate, team_mean=team_mean,
                league_avg=league_avg, n_games=n_used)


def pyth_win_prob(baselines, home_team, away_team, season, k=2.0):
    league_avg = baselines["league_avg"].get(season, np.mean(list(baselines["league_avg"].values())))

    def rate(team):
        v = baselines["team_mean"].get((team, season))
        if v is None:
            return league_avg, league_avg
        rs, ra, n = v
        return rs, ra

    hrs, hra = rate(home_team)
    ars, ara = rate(away_team)
    p_home = hrs ** k / (hrs ** k + hra ** k)
    p_away = ars ** k / (ars ** k + ara ** k)
    if p_home + p_away - 2 * p_home * p_away <= 0:
        return 0.5
    return (p_home - p_home * p_away) / (p_home + p_away - 2 * p_home * p_away)


# ============================================================================
# metrics
# ============================================================================
def brier(p, y):
    p, y = np.asarray(p), np.asarray(y)
    return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p, y = np.asarray(p), np.asarray(y)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_curve(p, y, n_bins=10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            out.append(dict(lo=float(lo), hi=float(hi), n=0, mean_pred=None, actual_rate=None))
        else:
            out.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()),
                             mean_pred=float(p[m].mean()), actual_rate=float(y[m].mean())))
    return out


def run_total_metrics(pred_means, pred_lo50, pred_hi50, pred_lo90, pred_hi90, actual):
    pred_means, actual = np.asarray(pred_means), np.asarray(actual)
    bias = float(np.mean(pred_means - actual))
    mae = float(np.mean(np.abs(pred_means - actual)))
    cov50 = float(np.mean((actual >= np.asarray(pred_lo50)) & (actual <= np.asarray(pred_hi50))))
    cov90 = float(np.mean((actual >= np.asarray(pred_lo90)) & (actual <= np.asarray(pred_hi90))))
    return dict(bias=bias, mae=mae, coverage_50=cov50, coverage_90=cov90, n=len(actual))


# ============================================================================
# main
# ============================================================================
def main():
    log("loading PA rows + split...")
    rows = load_pa_full()
    split = json.loads((SPIKES / "split.json").read_text())
    train_g, test_g = set(split["train_games"]), set(split["test_games"])
    disposed = RE_MOD.load_disposed_ids()
    log(f"train_games={len(train_g)} test_games={len(test_g)} disposed={len(disposed)}")

    by_game = {}
    for r in rows:
        by_game.setdefault(r["game_id"], []).append(r)

    baselines = train_baselines(rows, train_g, disposed)

    log("loading SimModel (shaped_train.npz + empirical_transitions.npz)...")
    model = SimModel()

    test_ids = sorted(g for g in test_g if g not in disposed)
    n_disposed_excluded = len(test_g) - len(test_ids)
    log(f"test games after excluding {n_disposed_excluded} disposed: {len(test_ids)}")

    ctxs = {}
    n_skip_lineup = 0
    n_short = []
    for gid in test_ids:
        ctx = build_game_context(model, by_game[gid])
        if ctx is None:
            n_skip_lineup += 1
            continue
        ctxs[gid] = ctx
        if ctx.n_innings != 9:
            n_short.append((gid, ctx.n_innings))
    log(f"usable test games: {len(ctxs)}  (skipped {n_skip_lineup} for incomplete/missing lineup data)")
    log(f"non-9-inning test games simulated at their real length: {len(n_short)} "
        f"(innings histogram: {sorted(set(i for _, i in n_short))})")

    actuals = {}
    n_actual_missing = 0
    for gid in list(ctxs):
        season = ctxs[gid].season
        try:
            away_r, home_r = load_actual(season, gid)
        except FileNotFoundError:
            n_actual_missing += 1
            del ctxs[gid]
            continue
        actuals[gid] = (away_r, home_r)
    if n_actual_missing:
        log(f"dropped {n_actual_missing} games with no linescore file")
    log(f"final validation set: {len(ctxs)} games")

    n_ties = sum(1 for a, h in actuals.values() if a == h)
    log(f"of these, {n_ties} ({100*n_ties/len(actuals):.1f}%) are tied after regulation "
        f"(HR-derby winner unrecoverable) -- excluded from win-probability scoring below, "
        f"included in run-total scoring.")

    results = {}
    rng = np.random.default_rng(SEED)
    for adv in ("naive", "empirical"):
        log(f"=== simulating advancement model: {adv} ===")
        t_adv0 = time.time()
        per_game = []
        for gid, ctx in ctxs.items():
            t0 = time.time()
            out = simulate_game(model, ctx, adv, N_SIMS, rng)
            elapsed = time.time() - t0
            away_r, home_r = actuals[gid]
            ar, hr = out["away_runs"], out["home_runs"]
            p_home_win = float(out["home_win"].mean())
            p_tie = float(out["tie"].mean())
            per_game.append(dict(
                game_id=gid, season=ctx.season, n_innings=ctx.n_innings,
                actual_away=away_r, actual_home=home_r, actual_tie=(away_r == home_r),
                pred_away_mean=float(ar.mean()), pred_home_mean=float(hr.mean()),
                pred_away_p25=float(np.percentile(ar, 25)), pred_away_p75=float(np.percentile(ar, 75)),
                pred_away_p05=float(np.percentile(ar, 5)), pred_away_p95=float(np.percentile(ar, 95)),
                pred_home_p25=float(np.percentile(hr, 25)), pred_home_p75=float(np.percentile(hr, 75)),
                pred_home_p05=float(np.percentile(hr, 5)), pred_home_p95=float(np.percentile(hr, 95)),
                p_home_win=p_home_win, p_tie=p_tie, sim_seconds=elapsed,
            ))
        t_adv_total = time.time() - t_adv0
        log(f"  {adv}: simulated {len(per_game)} games in {t_adv_total:.1f}s "
            f"({t_adv_total/len(per_game)*1000:.1f} ms/game for {N_SIMS} sims)")

        pred_means = [g["pred_away_mean"] for g in per_game] + [g["pred_home_mean"] for g in per_game]
        actual_all = [g["actual_away"] for g in per_game] + [g["actual_home"] for g in per_game]
        lo50 = [g["pred_away_p25"] for g in per_game] + [g["pred_home_p25"] for g in per_game]
        hi50 = [g["pred_away_p75"] for g in per_game] + [g["pred_home_p75"] for g in per_game]
        lo90 = [g["pred_away_p05"] for g in per_game] + [g["pred_home_p05"] for g in per_game]
        hi90 = [g["pred_away_p95"] for g in per_game] + [g["pred_home_p95"] for g in per_game]
        rt = run_total_metrics(pred_means, lo50, hi50, lo90, hi90, actual_all)
        log(f"  {adv} RUN TOTALS: bias={rt['bias']:+.3f}  MAE={rt['mae']:.3f}  "
            f"coverage50={rt['coverage_50']:.3f} (nominal .50)  "
            f"coverage90={rt['coverage_90']:.3f} (nominal .90)  n={rt['n']} team-games")

        nontied = [g for g in per_game if not g["actual_tie"]]
        y = np.array([1.0 if g["actual_home"] > g["actual_away"] else 0.0 for g in nontied])
        p_sim = np.array([g["p_home_win"] for g in nontied])
        p_const = np.full(len(nontied), baselines["const_home_win_rate"])
        p_pyth = np.array([pyth_win_prob(baselines, ctxs[g["game_id"]].home_team,
                                          ctxs[g["game_id"]].away_team, g["season"])
                            for g in nontied])

        wp = dict(
            n_nontied=len(nontied),
            sim=dict(brier=brier(p_sim, y), logloss=logloss(p_sim, y),
                     calibration=calibration_curve(p_sim, y)),
            baseline_constant=dict(value=baselines["const_home_win_rate"],
                                    brier=brier(p_const, y), logloss=logloss(p_const, y)),
            baseline_pythagorean=dict(brier=brier(p_pyth, y), logloss=logloss(p_pyth, y),
                                       calibration=calibration_curve(p_pyth, y)),
        )
        log(f"  {adv} WIN PROB (n={wp['n_nontied']} non-tied games):")
        log(f"    sim:          brier={wp['sim']['brier']:.4f}  logloss={wp['sim']['logloss']:.4f}")
        log(f"    const base:   brier={wp['baseline_constant']['brier']:.4f}  "
            f"logloss={wp['baseline_constant']['logloss']:.4f}  (p={baselines['const_home_win_rate']:.4f})")
        log(f"    pythag base:  brier={wp['baseline_pythagorean']['brier']:.4f}  "
            f"logloss={wp['baseline_pythagorean']['logloss']:.4f}")

        results[adv] = dict(run_totals=rt, win_prob=wp, wall_clock=dict(
            total_seconds=t_adv_total, n_games=len(per_game),
            seconds_per_game=t_adv_total / len(per_game),
            n_sims_per_game=N_SIMS,
            ms_per_1000_sims=t_adv_total / len(per_game) / N_SIMS * 1000 * 1000 / 1000), per_game=per_game)

    # naive vs empirical gap
    gap = dict(
        run_total_bias_gap=results["empirical"]["run_totals"]["bias"] - results["naive"]["run_totals"]["bias"],
        run_total_mae_gap=results["empirical"]["run_totals"]["mae"] - results["naive"]["run_totals"]["mae"],
        win_brier_gap=results["empirical"]["win_prob"]["sim"]["brier"] - results["naive"]["win_prob"]["sim"]["brier"],
        win_logloss_gap=results["empirical"]["win_prob"]["sim"]["logloss"] - results["naive"]["win_prob"]["sim"]["logloss"],
    )
    log("=== NAIVE vs EMPIRICAL gap (empirical minus naive; negative = empirical better) ===")
    for k, v in gap.items():
        log(f"  {k}: {v:+.4f}")

    out = dict(
        n_sims=N_SIMS, seed=SEED,
        n_test_games_total=len(test_g), n_disposed_excluded=n_disposed_excluded,
        n_skip_lineup=n_skip_lineup, n_actual_missing=n_actual_missing,
        n_validation_games=len(ctxs), n_tied_games=n_ties,
        baselines=dict(const_home_win_rate=baselines["const_home_win_rate"],
                       n_train_games_for_baselines=baselines["n_games"]),
        naive=results["naive"], empirical=results["empirical"],
        naive_vs_empirical_gap=gap,
    )
    (HERE / "result.json").write_text(json.dumps(out, indent=1, default=float))
    log("wrote result.json")


if __name__ == "__main__":
    main()
