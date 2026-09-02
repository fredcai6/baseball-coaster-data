"""Full-fidelity PA loader for spikes/sim/.

common.load_pa() (DO NOT EDIT) drops the columns this simulator needs for
lineups, bullpens and base/out state -- batting_team, fielding_team,
order_slot, outs_before, bases_before, outs_recorded, runs_on_play, half,
seq. This module reads pa_table.csv directly (same file, same row order) and
re-does common.load_pa's category mapping and bats/throws profile join so
every spikes/sim/ script sees ONE consistent, fuller row shape instead of
each re-deriving it. Nothing here modifies common.py or its output; this is
purely additive columns from the same CSV.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

OUTCOME_MAP = common.OUTCOME_MAP
CI = common.CAT_INDEX


def _b(v):
    return v == "1"


def bases_tuple(s):
    """'101' -> (True, False, True), base1/base2/base3 order (matches
    pa_table's bases_before / base_out_state string convention)."""
    return (s[0] == "1", s[1] == "1", s[2] == "1")


def bases_code(bt):
    return (1 if bt[0] else 0) | (2 if bt[1] else 0) | (4 if bt[2] else 0)


def load_pa_full(with_handedness=True):
    bats, throws = {}, {}
    if with_handedness and common.PROFILES.exists():
        with open(common.PROFILES) as fh:
            for r in csv.DictReader(fh):
                if not r["person_id"]:
                    continue
                key = (int(r["season"]), r["person_id"])
                if r["bats"]:
                    bats[key] = r["bats"]
                if r["throws"]:
                    throws[key] = r["throws"]

    rows = []
    with open(common.PA_TABLE) as fh:
        for r in csv.DictReader(fh):
            cat = OUTCOME_MAP.get(r["outcome_type"])
            if cat is None:
                raise ValueError(f"unmapped outcome_type {r['outcome_type']!r}")
            season = int(r["season"])
            rows.append({
                "game_id": r["game_id"], "season": season, "seq": int(r["seq"]),
                "batter": r["batter_career"] or r["batter_pid"],
                "pitcher": r["pitcher_career"] or r["pitcher_pid"],
                "batter_person": r["batter_person"], "pitcher_person": r["pitcher_person"],
                "y": CI[cat], "cat": cat,
                "outcome_type": r["outcome_type"],
                "tto": int(r["tto"]), "pitcher_bf": int(r["pitcher_bf"]),
                "is_starter": _b(r["pitcher_is_starter"]),
                "batting_is_home": _b(r["batting_is_home"]),
                "batting_at_home_park": _b(r["batting_at_home_park"]),
                "batting_team": r["batting_team"], "fielding_team": r["fielding_team"],
                "home_team": r["home_team"],
                "inning": int(r["inning"]), "half": r["half"],
                "order_slot": int(r["order_slot"]) if r["order_slot"] else None,
                "outs_before": int(r["outs_before"]),
                "bases_before": bases_tuple(r["bases_before"]),
                "outs_recorded": int(r["outs_recorded"]),
                "runs_on_play": int(r["runs_on_play"]),
                "bats": bats.get((season, r["batter_person"])),
                "throws": throws.get((season, r["pitcher_person"])),
            })
    return rows
