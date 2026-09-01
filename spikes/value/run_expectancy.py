"""Run-expectancy matrix and linear weights for the Pioneer League simulated
corpus (3 seasons, games/{2024,2025,2026}/*.json).

Method
------
1. Exclude the 17 games listed in corrections/dispositions.json (known
   incomplete play-by-play -- missing plate appearances would corrupt any
   half-inning that contains one).
2. Within each remaining game, group events by (inning, half). Only
   `plate_appearance` and `runner_event` kinds carry base/out state and
   `runs_on_play`; `substitution` and `inning_summary` are bookkeeping and are
   skipped. Events are ordered by `seq`.
3. A half-inning is "clean" if: it starts at 000|0, and for every consecutive
   pair of state-bearing events, bases_after[i] == bases_before[i+1] and
   outs_after[i] == outs_before[i+1] (no gap a missing event could have left).
   Dirty half-innings are discarded and counted.
4. For each clean half-inning, walk state-bearing events in order and, for
   every `plate_appearance`, record the state before it (bases_before,
   outs_before) and the runs scored from that point to the end of the half
   (inclusive of the PA's own runs_on_play and any later runner events'
   runs_on_play, e.g. steals of home / wild pitches / passed balls).
5. RE(state) = mean of those per-state observations. Cells with fewer than
   100 observations are flagged as thin.
6. Linear weights: for each PA, run_value = RE(after) - RE(before) +
   runs_on_play, with RE(any 3-out state) defined as 0. RE(after)/RE(before)
   are looked up in the POOLED (all-season) table. Averaged by outcome.type.

Run: PYTHONPATH=pipeline .venv/bin/python spikes/value/run_expectancy.py
(no third-party deps beyond the stdlib are required; run under the repo venv
per the task's environment note anyway.)
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GAMES_GLOB = os.path.join(REPO_ROOT, "games", "{season}", "*.json")
DISPOSITIONS_PATH = os.path.join(REPO_ROOT, "corrections", "dispositions.json")
OUT_JSON = os.path.join(REPO_ROOT, "spikes", "value", "re_result.json")

SEASONS = ["2024", "2025", "2026"]
STATE_KINDS = {"plate_appearance", "runner_event"}

MLB_REF = {
    "single": 0.47,
    "double": 0.77,
    "triple": 1.04,
    "home_run": 1.40,
    "walk": 0.29,
    "hit_by_pitch": 0.32,
    "generic_out": -0.27,
}

# Outcome types that end the PA in an out (used for the "generic out" rollup
# used as the zero point of the linear-weights scale). Fielders_choice and
# reached_on_error are NOT included -- the batter is safe in the corpus's
# encoding of those, so they are not "generic outs" even though a defender
# recorded one on someone.
OUT_TYPES = {
    "strikeout_swinging", "strikeout_looking", "strikeout",
    "groundout", "grounded_into_double_play", "flyout", "popout",
    "infield_fly", "foul_out", "lineout", "batter_interference",
}


def state_key(bases, outs):
    b = "".join("1" if x else "0" for x in bases)
    return f"{b}|{outs}"


def load_disposed_ids():
    with open(DISPOSITIONS_PATH) as f:
        d = json.load(f)
    return {x["game_id"] for x in d["dispositions"]}


def iter_games(season):
    for path in sorted(glob.glob(GAMES_GLOB.format(season=season))):
        with open(path) as f:
            yield path, json.load(f)


def half_innings(events):
    """Group state-bearing events by (inning, half), sorted by seq."""
    groups = defaultdict(list)
    for e in events:
        if e.get("kind") in STATE_KINDS:
            groups[(e["inning"], e["half"])].append(e)
    for key in groups:
        groups[key].sort(key=lambda e: e["seq"])
    return groups


def check_clean(evs):
    """Return True if the half-inning's state chain has no gaps."""
    if not evs:
        return False
    d0 = evs[0]["_derived"]
    if tuple(d0["bases_before"]) != (False, False, False) or d0["outs_before"] != 0:
        return False
    for i in range(len(evs) - 1):
        da = evs[i]["_derived"]
        db = evs[i + 1]["_derived"]
        if tuple(da["bases_after"]) != tuple(db["bases_before"]):
            return False
        if da["outs_after"] != db["outs_before"]:
            return False
    return True


def main():
    disposed = load_disposed_ids()

    # Observations for the RE table: season -> state -> list[runs_remaining]
    re_obs = {s: defaultdict(list) for s in SEASONS}
    re_obs["pooled"] = defaultdict(list)

    # PA-level records for linear weights (pooled across seasons).
    pa_records = []  # dicts: outcome_type, before_key, after_key, runs_on_play, season

    n_games_total = 0
    n_games_excluded = 0
    n_games_used = 0
    n_pa_rows_lost_to_exclusion = 0

    n_halves_total = 0
    n_halves_clean = 0
    n_halves_dirty = 0

    total_runs_clean_halves = 0
    n_clean_half_list_for_avg = 0  # == n_halves_clean

    sample_check_rows = []  # for correctness check 1

    for season in SEASONS:
        for path, game in iter_games(season):
            n_games_total += 1
            game_id = game.get("game_id") or os.path.splitext(os.path.basename(path))[0]
            if game_id in disposed:
                n_games_excluded += 1
                n_pa_rows_lost_to_exclusion += sum(
                    1 for e in game["events"] if e.get("kind") == "plate_appearance"
                )
                continue
            n_games_used += 1

            groups = half_innings(game["events"])
            for (inning, half), evs in groups.items():
                n_halves_total += 1
                if not check_clean(evs):
                    n_halves_dirty += 1
                    continue
                n_halves_clean += 1

                total_runs = sum(e["_derived"]["runs_on_play"] for e in evs)
                total_runs_clean_halves += total_runs

                # suffix sum of runs from event i (inclusive) to end of half
                n = len(evs)
                suffix = [0] * (n + 1)
                for i in range(n - 1, -1, -1):
                    suffix[i] = suffix[i + 1] + evs[i]["_derived"]["runs_on_play"]

                half_pa_values = []  # for the correctness-check sample

                for i, e in enumerate(evs):
                    d = e["_derived"]
                    before_key = state_key(d["bases_before"], d["outs_before"])
                    runs_remaining = suffix[i]

                    re_obs["pooled"][before_key].append(runs_remaining)
                    re_obs[season][before_key].append(runs_remaining)

                    if e["kind"] == "plate_appearance":
                        after_key = state_key(d["bases_after"], d["outs_after"])
                        pa_records.append(
                            {
                                "season": season,
                                "game_id": game_id,
                                "inning": inning,
                                "half": half,
                                "outcome_type": e["outcome"]["type"],
                                "before_key": before_key,
                                "after_key": after_key,
                                "outs_after": d["outs_after"],
                                "runs_on_play": d["runs_on_play"],
                            }
                        )
                        half_pa_values.append(
                            (before_key, after_key, d["outs_after"], d["runs_on_play"])
                        )

                sample_check_rows.append(
                    {
                        "game_id": game_id,
                        "inning": inning,
                        "half": half,
                        "total_runs": total_runs,
                        "pa_values": half_pa_values,
                        # check_clean() guarantees the half starts at 000|0
                        "start_key": "000|0",
                        "has_runner_event": any(e["kind"] == "runner_event" for e in evs),
                        "truncated": evs[-1]["_derived"]["outs_after"] < 3,
                    }
                )

    # --- build RE tables ----------------------------------------------
    def build_re_table(obs_by_state):
        table = {}
        for bases in ("000", "100", "010", "001", "110", "101", "011", "111"):
            for outs in (0, 1, 2):
                key = f"{bases}|{outs}"
                vals = obs_by_state.get(key, [])
                n = len(vals)
                mean = sum(vals) / n if n else None
                sd = statistics.pstdev(vals) if n > 1 else (0.0 if n == 1 else None)
                table[key] = {
                    "n": n,
                    "re": mean,
                    "sd": sd,
                    "thin": n < 100,
                }
        return table

    re_tables = {"pooled": build_re_table(re_obs["pooled"])}
    for season in SEASONS:
        re_tables[season] = build_re_table(re_obs[season])

    def re_lookup(table, key):
        outs = int(key.split("|")[1])
        if outs >= 3:
            return 0.0
        cell = table[key]
        return cell["re"] if cell["re"] is not None else 0.0

    pooled_table = re_tables["pooled"]

    # --- linear weights --------------------------------------------------
    by_type_values = defaultdict(list)
    for rec in pa_records:
        re_before = re_lookup(pooled_table, rec["before_key"])
        re_after = 0.0 if rec["outs_after"] >= 3 else re_lookup(pooled_table, rec["after_key"])
        run_value = re_after - re_before + rec["runs_on_play"]
        rec["run_value"] = run_value
        by_type_values[rec["outcome_type"]].append(run_value)

    weights_table = {}
    for outcome_type, vals in by_type_values.items():
        n = len(vals)
        mean = sum(vals) / n
        sd = statistics.pstdev(vals) if n > 1 else 0.0
        se = sd / math.sqrt(n) if n > 0 else None
        weights_table[outcome_type] = {"n": n, "mean_run_value": mean, "se": se}

    # weighted mean value of a "generic out"
    out_vals = []
    for t in OUT_TYPES:
        out_vals.extend(by_type_values.get(t, []))
    generic_out_mean = sum(out_vals) / len(out_vals) if out_vals else None
    generic_out_n = len(out_vals)
    generic_out_sd = statistics.pstdev(out_vals) if len(out_vals) > 1 else 0.0
    generic_out_se = generic_out_sd / math.sqrt(generic_out_n) if generic_out_n else None

    weights_rel_to_out = {}
    for outcome_type, row in weights_table.items():
        weights_rel_to_out[outcome_type] = row["mean_run_value"] - generic_out_mean
    weights_rel_to_out["generic_out (rollup)"] = 0.0

    # --- correctness check 1: PA run-value sum vs runs scored in half ----
    #
    # Telescoping algebra: summing run_value_i = RE(after_i) - RE(before_i) +
    # runs_i over a half-inning collapses the middle terms only where
    # before_{i+1} == after_i for EVERY consecutive pair. That holds between
    # two plate appearances with nothing between them, but a half-inning
    # starts at 000|0 (RE > 0, not 0) and, if it contains a runner_event
    # (steal / wild pitch / passed ball), that event also changes the state
    # without any PA being charged for it. So the raw identity is:
    #     sum(PA run_value) == total_runs - RE(start_state)              [if no runner_event, ends at 3 outs]
    # NOT sum(PA run_value) == total_runs as a bare literal match. We report
    # both the naive (raw) discrepancy and the algebra-adjusted one, and
    # separate out halves with a runner_event or a truncated (walk-off)
    # ending, since those are exactly where the adjusted identity can still
    # miss.
    max_raw_discrepancy = 0.0
    max_adj_discrepancy = 0.0
    worst_half = None
    n_checked = 0
    total_abs_raw = 0.0
    total_abs_adj = 0.0
    n_clean_complete = 0  # no runner_event, ends at 3 outs
    clean_complete_max_adj = 0.0
    n_halves_with_nonpa_runs = 0
    for h in sample_check_rows:
        pa_sum = 0.0
        pa_runs_only = 0
        for before_key, after_key, outs_after, runs_on_play in h["pa_values"]:
            re_before = re_lookup(pooled_table, before_key)
            re_after = 0.0 if outs_after >= 3 else re_lookup(pooled_table, after_key)
            pa_sum += re_after - re_before + runs_on_play
            pa_runs_only += runs_on_play
        raw_disc = abs(pa_sum - h["total_runs"])
        re_start = re_lookup(pooled_table, h["start_key"])
        adj_disc = abs((pa_sum + re_start) - h["total_runs"])

        total_abs_raw += raw_disc
        total_abs_adj += adj_disc
        n_checked += 1
        if raw_disc > max_raw_discrepancy:
            max_raw_discrepancy = raw_disc
        if adj_disc > max_adj_discrepancy:
            max_adj_discrepancy = adj_disc
            worst_half = h
        if pa_runs_only != h["total_runs"]:
            n_halves_with_nonpa_runs += 1
        if (not h["has_runner_event"]) and (not h["truncated"]):
            n_clean_complete += 1
            if adj_disc > clean_complete_max_adj:
                clean_complete_max_adj = adj_disc
    mean_raw_discrepancy = total_abs_raw / n_checked if n_checked else None
    mean_adj_discrepancy = total_abs_adj / n_checked if n_checked else None

    # --- correctness check 2: RE(000|0) vs average runs per half-inning --
    re_000_0 = pooled_table["000|0"]["re"]
    avg_runs_per_half = (
        total_runs_clean_halves / n_halves_clean if n_halves_clean else None
    )

    # --- run environment drift across seasons -----------------------------
    season_re_000_0 = {s: re_tables[s]["000|0"]["re"] for s in SEASONS}
    season_avg_runs = {}
    for season in SEASONS:
        obs = re_obs[season]
        # avg runs per half-inning per season = RE(000|0) for that season's
        # own table is the cleanest read since 000|0 only occurs once, at the
        # start of each clean half-inning, one row per half.
        season_avg_runs[season] = re_tables[season]["000|0"]["re"]

    # --- MLB comparison -----------------------------------------------
    mlb_comparison = {}
    label_map = {
        "single": "single",
        "double": "double",
        "triple": "triple",
        "home_run": "home_run",
        "walk": "walk",
        "hit_by_pitch": "hit_by_pitch",
    }
    for our_type, mlb_key in label_map.items():
        ours = weights_table.get(our_type, {}).get("mean_run_value")
        mlb_comparison[mlb_key] = {
            "ours": ours,
            "mlb": MLB_REF[mlb_key],
            "diff": (ours - MLB_REF[mlb_key]) if ours is not None else None,
        }
    mlb_comparison["generic_out"] = {
        "ours": generic_out_mean,
        "mlb": MLB_REF["generic_out"],
        "diff": (generic_out_mean - MLB_REF["generic_out"]) if generic_out_mean is not None else None,
    }

    result = {
        "games": {
            "total_files": n_games_total,
            "excluded_disposed": n_games_excluded,
            "used": n_games_used,
            "pa_rows_lost_to_exclusion": n_pa_rows_lost_to_exclusion,
        },
        "half_innings": {
            "total": n_halves_total,
            "clean": n_halves_clean,
            "dirty_discarded": n_halves_dirty,
        },
        "re_table_pooled": pooled_table,
        "re_table_by_season": {s: re_tables[s] for s in SEASONS},
        "linear_weights": weights_table,
        "generic_out_rollup": {
            "n": generic_out_n,
            "mean_run_value": generic_out_mean,
            "se": generic_out_se,
            "member_types": sorted(OUT_TYPES),
        },
        "weights_relative_to_out": weights_rel_to_out,
        "mlb_comparison": mlb_comparison,
        "correctness_checks": {
            "pa_run_value_sum_vs_runs_scored": {
                "n_half_innings_checked": n_checked,
                "note": (
                    "raw = |sum(PA run_value) - total_runs|; adjusted = "
                    "|sum(PA run_value) + RE(start_state) - total_runs|, which "
                    "is the algebraically correct form of this identity "
                    "(see comment above). Adjusted is exact (float noise "
                    "only) for half-innings with no runner_event and a clean "
                    "3-out ending."
                ),
                "max_abs_discrepancy_raw": max_raw_discrepancy,
                "mean_abs_discrepancy_raw": mean_raw_discrepancy,
                "max_abs_discrepancy_adjusted": max_adj_discrepancy,
                "mean_abs_discrepancy_adjusted": mean_adj_discrepancy,
                "n_clean_complete_half_innings": n_clean_complete,
                "max_abs_discrepancy_adjusted_clean_complete_only": clean_complete_max_adj,
                "n_halves_with_nonpa_scoring_events": n_halves_with_nonpa_runs,
                "worst_half": None
                if worst_half is None
                else {
                    "game_id": worst_half["game_id"],
                    "inning": worst_half["inning"],
                    "half": worst_half["half"],
                    "total_runs": worst_half["total_runs"],
                },
            },
            "re_000_0_vs_avg_runs_per_half": {
                "re_000_0_pooled": re_000_0,
                "avg_runs_per_half_pooled": avg_runs_per_half,
                "diff": (re_000_0 - avg_runs_per_half) if (re_000_0 is not None and avg_runs_per_half is not None) else None,
            },
            "malformed_half_innings_discarded": n_halves_dirty,
        },
        "season_run_environment": {
            "re_000_0_by_season": season_re_000_0,
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, sort_keys=False)

    # ------------------------------------------------------------------
    # console report
    # ------------------------------------------------------------------
    print("=" * 70)
    print("GAMES")
    print(f"  total files:        {n_games_total}")
    print(f"  excluded (disposed): {n_games_excluded}  (PA rows lost: {n_pa_rows_lost_to_exclusion})")
    print(f"  used:               {n_games_used}")
    print()
    print("HALF-INNINGS")
    print(f"  total:   {n_halves_total}")
    print(f"  clean:   {n_halves_clean}")
    print(f"  dirty (discarded, chain-break/gap): {n_halves_dirty}")
    print()
    print("POOLED 24-STATE RUN EXPECTANCY MATRIX (bases|outs -> RE, n, thin?)")
    header = f"{'bases':>6} | " + " | ".join(f"outs={o}" for o in (0, 1, 2))
    print(header)
    for bases in ("000", "100", "010", "001", "110", "101", "011", "111"):
        cells = []
        for outs in (0, 1, 2):
            c = pooled_table[f"{bases}|{outs}"]
            flag = "*" if c["thin"] else " "
            re_s = f"{c['re']:.3f}" if c["re"] is not None else "  n/a"
            cells.append(f"{re_s}{flag}(n={c['n']})")
        print(f"{bases:>6} | " + " | ".join(cells))
    print("  (* = fewer than 100 observations, flagged as thin)")
    print()
    print("PER-SEASON RE(000|0) [bases empty, 0 out] -- run-environment drift check")
    for s in SEASONS:
        c = re_tables[s]["000|0"]
        print(f"  {s}: RE={c['re']:.4f}  n={c['n']}")
    print(f"  pooled: RE={pooled_table['000|0']['re']:.4f}")
    print()
    print("LINEAR WEIGHTS (mean run value by outcome.type, pooled)")
    print(f"{'type':<28}{'n':>8}{'mean_rv':>12}{'se':>10}{'rel_to_out':>12}")
    for t, row in sorted(weights_table.items(), key=lambda kv: -kv[1]["mean_run_value"]):
        se_s = f"{row['se']:.4f}" if row["se"] is not None else "n/a"
        print(f"{t:<28}{row['n']:>8}{row['mean_run_value']:>12.4f}{se_s:>10}{weights_rel_to_out[t]:>12.4f}")
    print(f"{'generic_out (rollup)':<28}{generic_out_n:>8}{generic_out_mean:>12.4f}{generic_out_se:>10.4f}{0.0:>12.4f}")
    print()
    print("MLB COMPARISON")
    print(f"{'type':<14}{'ours':>10}{'mlb':>10}{'diff':>10}")
    for k, row in mlb_comparison.items():
        ours_s = f"{row['ours']:.3f}" if row["ours"] is not None else "n/a"
        diff_s = f"{row['diff']:+.3f}" if row["diff"] is not None else "n/a"
        print(f"{k:<14}{ours_s:>10}{row['mlb']:>10.3f}{diff_s:>10}")
    print()
    print("CORRECTNESS CHECKS")
    cc1 = result["correctness_checks"]["pa_run_value_sum_vs_runs_scored"]
    print(f"  [1] PA run-value sum vs runs scored, over {cc1['n_half_innings_checked']} half-innings:")
    print(f"      RAW      max |discrepancy| = {cc1['max_abs_discrepancy_raw']:.6f}   mean = {cc1['mean_abs_discrepancy_raw']:.6f}")
    print(f"               (raw sum telescopes to total_runs - RE(start_state), NOT total_runs -- expected, see code comment)")
    print(f"      ADJUSTED max |discrepancy| = {cc1['max_abs_discrepancy_adjusted']:.6f}   mean = {cc1['mean_abs_discrepancy_adjusted']:.6f}")
    print(f"               (adding back RE(start_state) before comparing)")
    print(f"      on the {cc1['n_clean_complete_half_innings']} half-innings with no runner_event and a clean 3-out ending,")
    print(f"      adjusted max |discrepancy| = {cc1['max_abs_discrepancy_adjusted_clean_complete_only']:.2e}  (exact, to floating point)")
    print(f"      half-innings where PA-only runs != total half runs (steals/WP/PB contributed runs): {cc1['n_halves_with_nonpa_scoring_events']}")
    cc2 = result["correctness_checks"]["re_000_0_vs_avg_runs_per_half"]
    print(f"  [2] RE(000|0) pooled = {cc2['re_000_0_pooled']:.4f} vs avg runs/half-inning = {cc2['avg_runs_per_half_pooled']:.4f} (diff {cc2['diff']:+.4f})")
    print(f"  [3] malformed half-innings discarded: {n_halves_dirty} (of {n_halves_total} total)")
    print("=" * 70)


if __name__ == "__main__":
    main()
