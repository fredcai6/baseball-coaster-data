"""Runs all 5 measurements from the spike brief and writes
spikes/advance/result.json. Console output (this script's prints) is meant
to be captured to spikes/advance/run.log by the caller.

PYTHONPATH=pipeline .venv/bin/python spikes/advance/measure.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

ADVANCE_DIR = os.path.dirname(os.path.abspath(__file__))
SPIKES_DIR = os.path.dirname(ADVANCE_DIR)
REPO_ROOT = os.path.dirname(SPIKES_DIR)
sys.path.insert(0, ADVANCE_DIR)
sys.path.insert(0, os.path.join(SPIKES_DIR, "value"))

import naive  # noqa: E402
import model  # noqa: E402
import records as records_mod  # noqa: E402

RE_RESULT_PATH = os.path.join(SPIKES_DIR, "value", "re_result.json")
RESULT_PATH = os.path.join(ADVANCE_DIR, "result.json")

MIN_CELL_N = 5  # empirical-table fallback threshold used at prediction time
THIN_CELL_N = 30  # task-5 "thin cell" threshold from the brief


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def mae(xs):
    return sum(abs(x) for x in xs) / len(xs) if xs else None


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def dist_summary(diffs):
    return {
        "n": len(diffs),
        "bias_mean": mean(diffs),
        "mae": mae(diffs),
        "sd": statistics.pstdev(diffs) if len(diffs) > 1 else 0.0,
        "p10": pct(diffs, 0.10), "p50": pct(diffs, 0.50), "p90": pct(diffs, 0.90),
        "min": min(diffs) if diffs else None, "max": max(diffs) if diffs else None,
    }


def main():
    print("=" * 70)
    print("Loading PA records (spikes/advance/records_cache.json) ...")
    data = records_mod.build(force=False)
    all_records = data["pa_records"]
    half_summaries = data["half_summaries"]
    print(json.dumps(data["meta"], indent=2))

    train_records = [r for r in all_records if r["split"] == "train"]
    test_records = [r for r in all_records if r["split"] == "test"]
    train_halves = [h for h in half_summaries if h["split"] == "train"]
    test_halves = [h for h in half_summaries if h["split"] == "test"]
    print(f"train PA={len(train_records)} test PA={len(test_records)}")
    print(f"train halves={len(train_halves)} test halves={len(test_halves)}")

    result = {"meta": data["meta"], "n_train_pa": len(train_records), "n_test_pa": len(test_records),
              "n_train_halves": len(train_halves), "n_test_halves": len(test_halves)}

    # -----------------------------------------------------------------
    # fit naive_v1 params + build empirical tables (TRAIN only)
    # -----------------------------------------------------------------
    params = naive.fit_params(train_records)
    print("\nnaive_v1 fitted params (TRAIN):")
    print(json.dumps(params, indent=2))
    result["naive_v1_params"] = params

    emp_full = model.build_empirical_table(train_records, mode="full")
    emp_full_marginal = model.build_marginal_table(train_records, mode="full")
    emp_within = model.build_empirical_table(train_records, mode="within")

    def dist_v0(b, o, c, s):
        return model.v0_dist(b, o, c, s)

    def dist_v1(b, o, c, s):
        return model.v1_dist(b, o, c, s, params)

    def dist_emp(b, o, c, s):
        return model.empirical_dist(emp_full, b, o, c, min_n=MIN_CELL_N, fallback_table=emp_full_marginal)

    dist_fns = {"naive_v0": dist_v0, "naive_v1": dist_v1, "empirical": dist_emp}

    # -----------------------------------------------------------------
    # TASK 1: half-inning replay on TEST, actual outcome sequence held at
    # truth, exact expectation-propagation over the (<=25)-state space.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 1: half-inning replay (TEST)")
    half_diffs = {m: [] for m in dist_fns}
    per_game = {m: defaultdict(lambda: [0.0, 0]) for m in dist_fns}  # game_id -> [pred_sum, ]
    per_game_actual = defaultdict(float)

    for h in test_halves:
        actual = h["actual_runs"]
        per_game_actual[h["game_id"]] += actual
        for mname, fn in dist_fns.items():
            state_dist = {((False, False, False), 0): 1.0}
            expected_runs = 0.0
            for pa in h["cat_sequence"]:
                bases_in = tuple(pa["bases_before"])
                outs_in = pa["outs_before"]
                cat = pa["cat"]
                is_sac = pa["is_sac"]
                new_dist = defaultdict(float)
                for (bases, outs), p in state_dist.items():
                    if p <= 0:
                        continue
                    for nb, no, runs, bp in fn(bases, outs, cat, is_sac):
                        w = p * bp
                        expected_runs += w * runs
                        if no < 3:
                            new_dist[(nb, no)] += w
                state_dist = new_dist
            half_diffs[mname].append(expected_runs - actual)
            per_game[mname][h["game_id"]][0] += expected_runs

    for mname in dist_fns:
        result.setdefault("task1_replay", {})[mname] = {
            "half_inning": dist_summary(half_diffs[mname]),
        }
    print(f"half-innings used: {len(test_halves)}")
    for mname in dist_fns:
        s = result["task1_replay"][mname]["half_inning"]
        print(f"  {mname:10s} half-inning (pred-actual): bias={s['bias_mean']:+.4f} "
              f"MAE={s['mae']:.4f} sd={s['sd']:.4f} n={s['n']}")

    # per-game roll-up
    game_diffs = {m: [] for m in dist_fns}
    for gid, actual in per_game_actual.items():
        for mname in dist_fns:
            pred = per_game[mname][gid][0]
            game_diffs[mname].append(pred - actual)
    for mname in dist_fns:
        result["task1_replay"][mname]["per_game"] = dist_summary(game_diffs[mname])
    print(f"\ngames used: {len(per_game_actual)}")
    for mname in dist_fns:
        s = result["task1_replay"][mname]["per_game"]
        print(f"  {mname:10s} per-game    (pred-actual): bias={s['bias_mean']:+.4f} "
              f"MAE={s['mae']:.4f} sd={s['sd']:.4f} n={s['n']}")

    # -----------------------------------------------------------------
    # TASK 2: decompose empirical-vs-naive_v0 gap into within-PA vs
    # between-PA, on TEST records (actual games, not simulated).
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 2: within-PA vs between-PA decomposition (TEST, vs naive_v0)")
    overall_within, overall_between = 0.0, 0.0
    overall_within_abs, overall_between_abs = 0.0, 0.0
    by_cat = defaultdict(lambda: {"within": 0.0, "between": 0.0, "within_abs": 0.0, "between_abs": 0.0, "n": 0})
    # secondary: residual after naive_v1 (expected value, not a draw)
    overall_within_v1, overall_within_v1_abs = 0.0, 0.0

    for r in test_records:
        bases_before = tuple(r["bases_before"])
        _, _, naive_runs = naive.apply_v0(bases_before, r["outs_before"], r["cat"], r["is_sac"])
        v1_branches = model.v1_dist(bases_before, r["outs_before"], r["cat"], r["is_sac"], params)
        naive_v1_runs = sum(bp * runs for _, _, runs, bp in v1_branches)

        g_within = r["runs_pa"] - naive_runs
        g_between = r["between_runs"]
        g_within_v1 = r["runs_pa"] - naive_v1_runs

        overall_within += g_within
        overall_between += g_between
        overall_within_abs += abs(g_within)
        overall_between_abs += abs(g_between)
        overall_within_v1 += g_within_v1
        overall_within_v1_abs += abs(g_within_v1)

        c = by_cat[r["cat"]]
        c["within"] += g_within
        c["between"] += g_between
        c["within_abs"] += abs(g_within)
        c["between_abs"] += abs(g_between)
        c["n"] += 1

    denom = overall_within + overall_between
    denom_abs = overall_within_abs + overall_between_abs
    task2 = {
        "baseline": "naive_v0",
        "n_test_pa": len(test_records),
        "overall": {
            "sum_within_pa_gap": overall_within, "sum_between_pa_gap": overall_between,
            "share_within_net": overall_within / denom if denom else None,
            "share_between_net": overall_between / denom if denom else None,
            "sum_abs_within_pa_gap": overall_within_abs, "sum_abs_between_pa_gap": overall_between_abs,
            "share_within_abs": overall_within_abs / denom_abs if denom_abs else None,
            "share_between_abs": overall_between_abs / denom_abs if denom_abs else None,
        },
        "residual_after_naive_v1": {
            "sum_within_pa_gap_v1": overall_within_v1,
            "sum_abs_within_pa_gap_v1": overall_within_v1_abs,
            "note": "same within-PA gap but measured against naive_v1's expected runs "
                    "instead of naive_v0's point prediction -- shows what the 3 fitted "
                    "refinements already capture vs what's left.",
        },
        "by_category": {},
    }
    for cat in model.CATS:
        c = by_cat.get(cat)
        if not c:
            continue
        d = c["within"] + c["between"]
        d_abs = c["within_abs"] + c["between_abs"]
        task2["by_category"][cat] = {
            "n": c["n"], "sum_within_pa_gap": c["within"], "sum_between_pa_gap": c["between"],
            "share_within_net": c["within"] / d if d else None,
            "share_within_abs": c["within_abs"] / d_abs if d_abs else None,
        }
    result["task2_decomposition"] = task2

    print(f"n TEST PA: {len(test_records)}")
    print(f"sum within-PA gap (runs) : {overall_within:+.1f}")
    print(f"sum between-PA gap (runs): {overall_between:+.1f}")
    print(f"share within (net)  : {task2['overall']['share_within_net']:.3f}")
    print(f"share between (net) : {task2['overall']['share_between_net']:.3f}")
    print(f"share within (abs)  : {task2['overall']['share_within_abs']:.3f}")
    print(f"share between (abs) : {task2['overall']['share_between_abs']:.3f}")
    print("\nby category (net share within):")
    for cat in model.CATS:
        row = task2["by_category"].get(cat)
        if row and row["share_within_net"] is not None:
            print(f"  {cat:6s} n={row['n']:6d}  within={row['sum_within_pa_gap']:+8.1f}  "
                  f"between={row['sum_between_pa_gap']:+8.1f}  share_within_net={row['share_within_net']:+.3f}")

    # -----------------------------------------------------------------
    # TASK 3: RE24 recomputation under each model, TRAIN category
    # frequencies, compared to spikes/value/re_result.json's pooled table.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 3: RE24 recomputation vs re_result.json pooled table")
    cat_counts = Counter(r["cat"] for r in train_records)
    n_train = sum(cat_counts.values())
    cat_probs = {c: cat_counts[c] / n_train for c in model.CATS}
    other_records = [r for r in train_records if r["cat"] == "OTHER"]
    p_sac_other = sum(1 for r in other_records if r["is_sac"]) / len(other_records) if other_records else 0.0
    print("TRAIN category frequencies:", {c: round(v, 4) for c, v in cat_probs.items()})
    print(f"P(is_sac | OTHER) on TRAIN = {p_sac_other:.4f} (n={len(other_records)})")

    def dist_v0_sim(b, o, c):
        return model.v0_dist(b, o, c, False)

    def dist_v1_sim(b, o, c):
        return model.v1_dist_marginal_sac(b, o, c, p_sac_other, params)

    def dist_emp_sim(b, o, c):
        return model.empirical_dist(emp_full, b, o, c, min_n=MIN_CELL_N, fallback_table=emp_full_marginal)

    sim_dist_fns = {"naive_v0": dist_v0_sim, "naive_v1": dist_v1_sim, "empirical": dist_emp_sim}

    with open(RE_RESULT_PATH) as f:
        re_truth = json.load(f)["re_table_pooled"]

    task3 = {"cat_probs_train": cat_probs, "p_sac_other_train": p_sac_other, "models": {}}
    for mname, fn in sim_dist_fns.items():
        RE, n_iter = model.solve_re24(fn, cat_probs)
        diffs = {}
        for (bases, outs) in model.ALL_STATES:
            key = model.state_key(bases, outs)
            truth = re_truth[key]["re"]
            diffs[key] = RE[(bases, outs)] - truth if truth is not None else None
        valid = {k: v for k, v in diffs.items() if v is not None}
        abs_vals = [abs(v) for v in valid.values()]
        worst = sorted(valid.items(), key=lambda kv: -abs(kv[1]))[:6]
        task3["models"][mname] = {
            "n_iter": n_iter,
            "re_table": {model.state_key(b, o): RE[(b, o)] for (b, o) in model.ALL_STATES},
            "diff_vs_truth": diffs,
            "max_abs_diff": max(abs_vals) if abs_vals else None,
            "mean_abs_diff": mean(abs_vals),
            "worst_states": worst,
        }
        print(f"\n  {mname}: converged in {n_iter} iters. "
              f"max|diff|={task3['models'][mname]['max_abs_diff']:.4f}  "
              f"mean|diff|={task3['models'][mname]['mean_abs_diff']:.4f}")
        print(f"    worst states: {worst}")
    result["task3_re24"] = task3

    # -----------------------------------------------------------------
    # TASK 4: stability of the empirical table across seasons and teams.
    # Uses ALL non-disposed PA records (train+test) -- this is a diagnostic
    # about the corpus, not a held-out prediction exercise.
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 4: cross-season / cross-team stability")
    by_season_records = defaultdict(list)
    for r in all_records:
        by_season_records[r["season"]].append(r)
    seasons = sorted(by_season_records)
    season_tables = {s: model.build_empirical_table(by_season_records[s], mode="full") for s in seasons}

    cell_spreads = []
    for bases in model.ALL_BASES:
        for outs in (0, 1, 2):
            for cat in model.CATS:
                key = (bases, outs, cat)
                means, ns = [], []
                ok = True
                for s in seasons:
                    cell = season_tables[s].get(key)
                    if cell is None or cell["n"] < THIN_CELL_N:
                        ok = False
                        break
                    means.append(cell["mean_runs"])
                    ns.append(cell["n"])
                if not ok:
                    continue
                spread = max(means) - min(means)
                cell_spreads.append({
                    "state": model.state_key(bases, outs), "cat": cat,
                    "season_means": dict(zip(seasons, means)), "season_ns": dict(zip(seasons, ns)),
                    "spread": spread,
                })
    cell_spreads.sort(key=lambda c: -c["spread"])
    task4_season = {
        "n_cells_with_min_n_all_seasons": len(cell_spreads),
        "thin_cell_threshold": THIN_CELL_N,
        "mean_spread": mean([c["spread"] for c in cell_spreads]),
        "median_spread": pct([c["spread"] for c in cell_spreads], 0.5),
        "top10_by_spread": cell_spreads[:10],
    }
    print(f"cells with >= {THIN_CELL_N} obs in all 3 seasons: {len(cell_spreads)} / 240")
    print(f"mean season-to-season spread (runs): {task4_season['mean_spread']:.4f}")
    print(f"median season-to-season spread (runs): {task4_season['median_spread']:.4f}")
    print("top 10 cells by season spread:")
    for c in cell_spreads[:10]:
        print(f"  {c['state']} {c['cat']:6s} spread={c['spread']:.3f}  means={c['season_means']}")

    # per-team spread on 3 target cells
    def rec_matches_target(r, target):
        b = tuple(r["bases_before"])
        if target == "on1_1B":
            return b == (True, False, False) and r["cat"] == "1B"
        if target == "on2_1B":
            return b == (False, True, False) and r["cat"] == "1B"
        if target == "on3_F_lt2outs":
            return b == (False, False, True) and r["cat"] == "F" and r["outs_before"] < 2
        raise ValueError(target)

    targets = ["on1_1B", "on2_1B", "on3_F_lt2outs"]
    MIN_TEAM_N = 10
    task4_team = {}
    for target in targets:
        by_season_team = defaultdict(lambda: defaultdict(list))
        for r in all_records:
            if rec_matches_target(r, target):
                runs = r["runs_pa"] + r["between_runs"]
                by_season_team[r["season"]][r["batting_team"]].append(runs)
        season_rows = {}
        all_spreads = []
        for s in seasons:
            team_means = {}
            for team, vals in by_season_team[s].items():
                if len(vals) >= MIN_TEAM_N:
                    team_means[team] = {"n": len(vals), "mean_runs": mean(vals)}
            if len(team_means) >= 2:
                ms = [v["mean_runs"] for v in team_means.values()]
                spread = max(ms) - min(ms)
                sd = statistics.pstdev(ms)
                all_spreads.append(spread)
            else:
                spread, sd = None, None
            season_rows[s] = {
                "n_teams": len(team_means), "team_means": team_means,
                "spread_max_minus_min": spread, "sd_across_teams": sd,
            }
        task4_team[target] = {
            "season_rows": season_rows,
            "mean_spread_across_seasons": mean([v for v in all_spreads if v is not None]),
        }
        print(f"\n  target={target}:")
        for s in seasons:
            row = season_rows[s]
            sp = row["spread_max_minus_min"]
            sp_s = f"{sp:.3f}" if sp is not None else "n/a"
            print(f"    season {s}: n_teams={row['n_teams']:3d}  spread(max-min)={sp_s}  sd={row['sd_across_teams']}")

    result["task4_stability"] = {"by_season": task4_season, "by_team": task4_team}

    # -----------------------------------------------------------------
    # TASK 5: cell sizes -- thin-cell count on the TRAIN empirical table
    # (the table actually used to drive naive_v0/v1's expected-runs
    # residual comparison and the empirical replay/RE24 models above).
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TASK 5: cell sizes (TRAIN empirical table, 24 states x 10 outcomes = 240 cells)")
    thin_cells = []
    zero_cells = []
    all_ns = {}
    for bases in model.ALL_BASES:
        for outs in (0, 1, 2):
            for cat in model.CATS:
                key = (bases, outs, cat)
                cell = emp_full.get(key)
                n = cell["n"] if cell else 0
                all_ns[f"{model.state_key(bases, outs)}|{cat}"] = n
                if n < THIN_CELL_N:
                    thin_cells.append({"state": model.state_key(bases, outs), "cat": cat, "n": n})
                if n == 0:
                    zero_cells.append({"state": model.state_key(bases, outs), "cat": cat})
    thin_cells.sort(key=lambda c: c["n"])
    task5 = {
        "total_cells": 240, "thin_threshold": THIN_CELL_N,
        "n_thin_cells": len(thin_cells), "n_zero_cells": len(zero_cells),
        "thin_cells": thin_cells, "cell_n": all_ns,
    }
    result["task5_cell_sizes"] = task5
    print(f"cells with < {THIN_CELL_N} TRAIN obs: {len(thin_cells)} / 240 "
          f"({len(zero_cells)} have zero obs)")
    for c in thin_cells:
        print(f"  {c['state']} {c['cat']:6s} n={c['n']}")

    # -----------------------------------------------------------------
    with open(RESULT_PATH, "w") as f:
        json.dump(result, f, indent=2, sort_keys=False, default=str)
    print("\n" + "=" * 70)
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
