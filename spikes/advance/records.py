"""Walk every non-disposed game once and build a flat list of PA-level
records with enough ground truth to drive every measurement in this spike:
naive/empirical replay, the within-PA vs between-PA decomposition, RE24
recomputation, and season/team stability.

Reuses spikes/value/run_expectancy.py's walker (iter_games, half_innings,
check_clean, load_disposed_ids) rather than re-deriving the clean-half-inning
logic. spikes/common.py's OUTCOME_MAP supplies the 10-category mapping.

A record is one plate_appearance event. Fields:
  game_id, season, half_id (game_id, inning, half) - str key for grouping
  batting_team, outs_before, bases_before (tuple of 3 bool: on1,on2,on3)
  cat (10-category), raw_type (outcome.type), is_sac
  outs_recorded, runs_pa (this PA's own runs_on_play)
  bases_after, outs_after (this PA's own _derived -- "within-PA" truth)
  between_runs (sum of runs_on_play of runner_events between this PA and the
    next state-bearing event in the half)
  state_before_next (bases, outs) actually observed right before the next PA
    begins (i.e. this PA's bases_after/outs_after rolled forward through any
    intervening runner_events) -- None if this is the last PA of the half
  is_last_pa (bool), half_actual_runs (total runs_on_play in the whole half,
    across PA + runner_event, repeated on every record of that half),
  seq_in_half (0-based index of this PA among PAs in the half)

Also builds a per-half-inning summary list (one row per clean half-inning):
  half_id, game_id, season, in_split, n_pa, actual_runs, cat_sequence
  (ordered list of (bases_before, outs_before, cat, is_sac) for replay)

Cached to spikes/advance/records_cache.json so the ~1300 game walk (matching
run_expectancy.py's own walk) happens once across all measurement scripts.
"""
from __future__ import annotations

import json
import os
import sys

ADVANCE_DIR = os.path.dirname(os.path.abspath(__file__))
SPIKES_DIR = os.path.dirname(ADVANCE_DIR)
REPO_ROOT = os.path.dirname(SPIKES_DIR)
sys.path.insert(0, SPIKES_DIR)
sys.path.insert(0, os.path.join(SPIKES_DIR, "value"))

import common  # noqa: E402
import run_expectancy as re_mod  # noqa: E402

CACHE_PATH = os.path.join(ADVANCE_DIR, "records_cache.json")


def _bases_tuple(b):
    return (bool(b[0]), bool(b[1]), bool(b[2]))


def build(force=False):
    if os.path.exists(CACHE_PATH) and not force:
        with open(CACHE_PATH) as f:
            return json.load(f)

    rows = common.load_pa()
    train_games, test_games = common.get_split(rows)

    disposed = re_mod.load_disposed_ids()

    pa_records = []
    half_summaries = []

    n_games_used = 0
    n_halves_dirty = 0
    n_halves_clean = 0

    for season in re_mod.SEASONS:
        for path, game in re_mod.iter_games(season):
            game_id = game.get("game_id") or os.path.splitext(os.path.basename(path))[0]
            if game_id in disposed:
                continue
            n_games_used += 1
            in_split = "train" if game_id in train_games else (
                "test" if game_id in test_games else None
            )
            if in_split is None:
                # game not in either frozen split bucket (shouldn't happen,
                # but don't silently mis-split if it does)
                continue

            groups = re_mod.half_innings(game["events"])
            for (inning, half), evs in groups.items():
                if not re_mod.check_clean(evs):
                    n_halves_dirty += 1
                    continue
                n_halves_clean += 1

                half_id = f"{game_id}|{inning}|{half}"
                actual_runs = sum(e["_derived"]["runs_on_play"] for e in evs)

                pa_idx_in_half = [i for i, e in enumerate(evs) if e["kind"] == "plate_appearance"]
                cat_sequence = []
                seq_in_half = 0

                for j, i in enumerate(pa_idx_in_half):
                    e = evs[i]
                    d = e["_derived"]
                    raw_type = e["outcome"]["type"]
                    cat = common.OUTCOME_MAP.get(raw_type)
                    if cat is None:
                        raise ValueError(f"unmapped outcome_type {raw_type!r}")
                    # pa_table's own is_sac column (verified by cross-check)
                    # matches "sac" appearing case-insensitively anywhere in
                    # outcome.modifiers -- raw scoring uses both "SAC" and
                    # "sacrifice fly" / "sacrifice bunt" style strings.
                    is_sac = any("sac" in m.lower() for m in (e["outcome"].get("modifiers") or []))

                    bases_before = _bases_tuple(d["bases_before"])
                    outs_before = d["outs_before"]
                    bases_after = _bases_tuple(d["bases_after"])
                    outs_after = d["outs_after"]
                    runs_pa = d["runs_on_play"]

                    # walk forward through any runner_events before the next
                    # PA (or end of half) to get between-PA movement and the
                    # state actually observed right before the next PA.
                    between_runs = 0
                    state_bases, state_outs = bases_after, outs_after
                    next_pa_i = pa_idx_in_half[j + 1] if j + 1 < len(pa_idx_in_half) else None
                    k = i + 1
                    while k < (next_pa_i if next_pa_i is not None else len(evs)):
                        ek = evs[k]
                        if ek["kind"] == "runner_event":
                            dk = ek["_derived"]
                            between_runs += dk["runs_on_play"]
                            state_bases = _bases_tuple(dk["bases_after"])
                            state_outs = dk["outs_after"]
                        k += 1

                    is_last_pa = next_pa_i is None

                    rec = {
                        "game_id": game_id, "season": int(season),
                        "half_id": half_id, "batting_team": e["batting_team"],
                        "split": in_split, "seq_in_half": seq_in_half,
                        "outs_before": outs_before, "bases_before": list(bases_before),
                        "cat": cat, "raw_type": raw_type, "is_sac": is_sac,
                        "outs_recorded": e["outcome"]["outs_recorded"],
                        "runs_pa": runs_pa,
                        "bases_after": list(bases_after), "outs_after": outs_after,
                        "between_runs": between_runs,
                        "state_before_next_bases": list(state_bases),
                        "state_before_next_outs": state_outs,
                        "is_last_pa": is_last_pa,
                        "half_actual_runs": actual_runs,
                    }
                    pa_records.append(rec)
                    cat_sequence.append({
                        "bases_before": list(bases_before), "outs_before": outs_before,
                        "cat": cat, "is_sac": is_sac,
                    })
                    seq_in_half += 1

                half_summaries.append({
                    "half_id": half_id, "game_id": game_id, "season": int(season),
                    "split": in_split, "n_pa": len(pa_idx_in_half),
                    "actual_runs": actual_runs, "cat_sequence": cat_sequence,
                })

    out = {
        "meta": {
            "n_games_used": n_games_used,
            "n_halves_clean": n_halves_clean,
            "n_halves_dirty": n_halves_dirty,
            "n_pa_records": len(pa_records),
            "n_half_summaries": len(half_summaries),
        },
        "pa_records": pa_records,
        "half_summaries": half_summaries,
    }
    with open(CACHE_PATH, "w") as f:
        json.dump(out, f)
    return out


if __name__ == "__main__":
    d = build(force=True)
    print(json.dumps(d["meta"], indent=2))
    print("train halves:", sum(1 for h in d["half_summaries"] if h["split"] == "train"))
    print("test halves :", sum(1 for h in d["half_summaries"] if h["split"] == "test"))
