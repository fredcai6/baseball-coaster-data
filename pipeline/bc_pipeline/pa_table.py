"""Flatten the play-by-play corpus into one row per plate appearance.

The corpus is 1,485 game files whose `events[]` spine is the authoritative
play-by-play. Every analysis consumer otherwise re-walks those files and
re-derives the same columns, which is where a shared bug would hide. This
module is that walk, done once.

Two rules the caller does not have to remember, because the table encodes them:

  * **`player_id` does not join across games.** 10% of player records carry a
    synthetic `syn:<side>:<n>` id assigned by boxscore row order, so the same
    value denotes a different person in every file (`syn:away:8` binds to 99
    distinct display names in 2026 alone). The join keys this table exposes are
    `batter_career` / `pitcher_career` (cross-season) and `batter_person` /
    `pitcher_person` (season-stable). The raw `*_pid` columns are carried for
    traceability back to the source file and are explicitly NOT join keys.

  * **`outcome` is an object, not a string.** Its `type` is the closed taxonomy;
    reading `outcome` as a scalar silently yields nothing rather than failing.

The table asserts nothing the corpus does not already say. Batted-ball class,
plate-appearance bookkeeping (AB/H/BB/SO) and matchup context (times through the
order, batters faced) are folds over the spine, not new claims — which is what
lets `tests/test_pa_table.py` check them against the boxscore, a table parsed
from a different region of the source page.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

TABLE_VERSION = "0.1.0"

# --- outcome taxonomy folds ------------------------------------------------
# The closed set of outcome types the schema permits, partitioned by what each
# one means for the classical counting stats. A type absent from every set here
# is a schema addition the builder has not been taught; `classify` raises on it
# rather than silently scoring it as an out.

STRIKEOUTS = {"strikeout_swinging", "strikeout_looking", "strikeout"}
WALKS = {"walk", "intentional_walk"}
HITS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

# Not an at-bat: the batter was not charged one. Walks, HBP and award-of-first
# on interference reach base without consuming an AB.
NOT_AT_BAT = WALKS | {"hit_by_pitch", "sacrifice", "reached_on_interference"}

# A sacrifice is NOT a member of the outcome-type taxonomy -- the `sacrifice`
# type covers only 36 of the corpus's 1,895 sacrifices. The rest are ordinary
# `flyout` / `groundout` / `lineout` / `foul_out` / `popout` / `fielders_choice`
# / `reached_on_error` rows carrying a `SAC` or `sacrifice fly` MODIFIER, which
# spans eight distinct types. Reading the type alone over-counts AB on every one
# of them; the boxscore caught this on 1,749 player-games the first time this
# table was folded. AB eligibility is therefore a function of type AND modifiers.
SAC_MODIFIERS = {"SAC", "sacrifice fly"}

# In play, i.e. the defense had a chance at it.
BATTED_BALL = {
    "groundout": "gb",
    "grounded_into_double_play": "gb",
    "flyout": "fb",
    "popout": "pu",
    "infield_fly": "pu",
    "foul_out": "fb",
    "lineout": "ld",
}

# Reached without a hit and without a walk.
OTHER_IN_PLAY = {"fielders_choice", "reached_on_error"}

# Outs that are neither struck out nor cleanly a batted-ball class.
OTHER_OUTS = {"batter_interference"}

KNOWN_TYPES = (
    STRIKEOUTS
    | WALKS
    | set(HITS)
    | {"hit_by_pitch", "sacrifice", "reached_on_interference"}
    | set(BATTED_BALL)
    | OTHER_IN_PLAY
    | OTHER_OUTS
)

# --- spray ----------------------------------------------------------------
# The source names either a fielder or a prose location, never a coordinate.
# Both are folded to one coarse field axis. Handedness is NOT recorded anywhere
# in this corpus, so pull/oppo cannot be derived -- only absolute direction.

_FIELD_BY_POS = {
    "lf": "left", "ss": "left", "3b": "left",
    "cf": "center", "p": "center", "c": "center", "2b": "right",
    "rf": "right", "1b": "right",
}

_FIELD_BY_LOCATION = {
    "left field": "left", "down the lf line": "left", "left side": "left",
    "shortstop": "left", "third base": "left",
    "left center": "left_center",
    "center field": "center", "up the middle": "center",
    "right center": "right_center",
    "right field": "right", "down the rf line": "right", "right side": "right",
    "first base": "right", "second base": "right",
}


def spray_field(outcome):
    """Coarse absolute field direction, or None when the source named neither."""
    loc = outcome.get("location")
    if loc and loc in _FIELD_BY_LOCATION:
        return _FIELD_BY_LOCATION[loc]
    fielders = outcome.get("fielders") or []
    if fielders:
        return _FIELD_BY_POS.get(fielders[0])
    return None


def classify(outcome):
    """Fold one outcome object into the counting-stat primitives.

    Takes the whole outcome, not just its type, because sacrifice status lives
    in `modifiers` (see SAC_MODIFIERS) and AB eligibility depends on it.

    Returns a dict of the flags the boxscore also reports, so the two can be
    reconciled. Raises on an unknown type: a schema addition must be taught
    here deliberately, not absorbed as an out.
    """
    otype = outcome.get("type")
    if otype not in KNOWN_TYPES:
        raise ValueError(f"unknown outcome type {otype!r}; teach pa_table.py before building")
    modifiers = set(outcome.get("modifiers") or ())
    is_sac = bool(modifiers & SAC_MODIFIERS)
    return {
        "is_k": otype in STRIKEOUTS,
        "is_bb": otype in WALKS,
        "is_ibb": otype == "intentional_walk",
        "is_hbp": otype == "hit_by_pitch",
        "is_hit": otype in HITS,
        "bases": HITS.get(otype, 0),
        "is_hr": otype == "home_run",
        "is_sac": is_sac,
        "is_ab": otype not in NOT_AT_BAT and not is_sac,
        "is_bip": otype in BATTED_BALL or otype in OTHER_IN_PLAY or otype in HITS,
        "bb_type": BATTED_BALL.get(otype),
    }


# --- pitch sequence -------------------------------------------------------
#
# The source publishes a per-PA pitch string ("BSBFB") and the displayed count.
# Letters: B ball, K called strike, S swinging strike, F foul, H hit by pitch.
#
# The string holds every pitch EXCEPT a ball put in play -- a batted ball is
# not a pitch-result letter, so it is absent and must be added back. Verified
# on the 2026 season against the source's own displayed count: 43,073 of
# 43,308 PAs reconcile exactly (99.46%), the residual being source scoring
# errors the check names rather than absorbs (e.g. "struck out swinging (0-0)"
# with an empty string, which no legal PA can be).
#
# ONLY 2026 CARRIES THIS. 2024 and 2025 publish neither field -- not sparsely,
# not partially: zero of 82,518 PAs. Anything built on n_pitches is a
# one-season model and must say so.

PITCH_LETTERS = frozenset("BKSFH")

#: Outcomes whose deciding pitch IS in the string (it resolved without contact).
#: Everything else ended on a batted ball, which the string does not record.
TERMINAL_IN_SEQ = STRIKEOUTS | WALKS | {"hit_by_pitch"}


def replay_count(seq):
    """Fold a pitch string into the (balls, strikes) it implies.

    A foul at two strikes does not advance the count, which is why the string
    can be longer than balls+strikes. Returns None on an unknown letter rather
    than guessing -- a new letter is a schema change to be taught here.
    """
    balls = strikes = 0
    for ch in seq:
        if ch not in PITCH_LETTERS:
            return None
        if ch in "BH":          # the source's displayed count treats a HBP as a ball
            balls += 1
        elif ch in "KS":
            strikes += 1
        elif ch == "F" and strikes < 2:
            strikes += 1
    return balls, strikes


def pitch_detail(event, otype):
    """Pitch-count fields for one PA, or all-None when the season lacks them.

    `count_ok` is the reconciliation flag, not a filter: rows that fail it are
    still emitted with their derived n_pitches so a consumer can decide. It is
    False only when the source's own two fields disagree with each other.
    """
    count = event.get("count")
    if not count or count.get("balls") is None:
        return {"pitch_seq": None, "n_pitches": None, "count_balls": None,
                "count_strikes": None, "count_ok": None}

    seq = event.get("pitches") or ""
    n = len(seq) + (0 if otype in TERMINAL_IN_SEQ else 1)

    replayed = replay_count(seq)
    if replayed is None:
        ok = False
    else:
        balls, strikes = replayed
        # The deciding pitch pushes past the displayed cap (ball four, strike
        # three), so back it out before comparing with what the source printed.
        if otype in STRIKEOUTS:
            expected = (balls, strikes - 1)
        elif otype in WALKS:
            expected = (balls - 1, strikes)
        else:
            expected = (balls, strikes)
        ok = expected == (count["balls"], count["strikes"])

    return {"pitch_seq": seq, "n_pitches": n, "count_balls": count["balls"],
            "count_strikes": count["strikes"], "count_ok": ok}


COLUMNS = [
    # provenance
    "game_id", "season", "date", "seq",
    # identity -- career/person join, pid for traceability only
    "batter_career", "batter_person", "batter_pid", "batter_name",
    "pitcher_career", "pitcher_person", "pitcher_pid", "pitcher_name",
    # sides
    "batting_team", "fielding_team", "home_team", "batting_is_home",
    # matchup context
    "inning", "half", "tto", "pitcher_bf", "pitcher_is_starter",
    "order_slot", "pa_number_of_batter",
    # state before the pitch
    "outs_before", "base_out_state", "bases_before", "score_diff_batting",
    # outcome
    "outcome_type", "outs_recorded", "runs_on_play", "spray", "location", "fielder1",
    "is_k", "is_bb", "is_ibb", "is_hbp", "is_hit", "bases", "is_hr", "is_sac",
    "is_ab", "is_bip", "bb_type",
    # pitch sequence -- 2026 only, null for 2024/2025 (see pitch_detail)
    "pitch_seq", "n_pitches", "count_balls", "count_strikes", "count_ok",
]


def rows_for_game(game):
    """Yield one dict per plate appearance in a single game file."""
    gid = game["game_id"]
    season = game["season"]
    date = game["date"]
    players = game["players"]
    teams = game.get("teams") or {}
    home_team = (teams.get("home") or {}).get("team_id")

    # Starting batting order, by team. Substitutions carry a null slot often
    # enough that only the starting nine are trusted here; a sub gets None.
    slot_of = {}
    for team_id, lu in (game.get("lineups") or {}).items():
        for entry in lu.get("batting_order") or []:
            slot_of[(team_id, entry["player_id"])] = entry.get("slot")

    first_pitcher = {}      # fielding_team -> pid of the game's first pitcher
    bf_count = {}           # pitcher pid -> batters faced so far this game
    tto_count = {}          # (pitcher pid, batter pid) -> times faced this game

    for e in game.get("events") or []:
        if e.get("kind") != "plate_appearance":
            continue
        batter = e.get("batter") or {}
        pitcher = e.get("pitcher") or {}
        bpid, ppid = batter.get("player_id"), pitcher.get("player_id")
        if not (bpid and ppid):
            continue

        outcome = e.get("outcome") or {}
        otype = outcome.get("type")
        flags = classify(outcome)

        d = e.get("_derived") or {}
        ft = e["fielding_team"]
        bt = e["batting_team"]
        first_pitcher.setdefault(ft, ppid)
        bf_count[ppid] = bf_count.get(ppid, 0) + 1
        tto_count[(ppid, bpid)] = tto_count.get((ppid, bpid), 0) + 1

        away_before = d.get("away_score_before")
        home_before = d.get("home_score_before")
        batting_is_home = e["half"] == "bottom"
        if away_before is None or home_before is None:
            score_diff = None
        elif batting_is_home:
            score_diff = home_before - away_before
        else:
            score_diff = away_before - home_before

        bases_before = d.get("bases_before")
        fielders = outcome.get("fielders") or []
        brec = players.get(bpid) or {}
        prec = players.get(ppid) or {}

        row = {
            "game_id": gid, "season": season, "date": date, "seq": e["seq"],
            "batter_career": brec.get("career_id"),
            "batter_person": brec.get("person_id"),
            "batter_pid": bpid,
            "batter_name": brec.get("name") or batter.get("name_raw"),
            "pitcher_career": prec.get("career_id"),
            "pitcher_person": prec.get("person_id"),
            "pitcher_pid": ppid,
            "pitcher_name": prec.get("name") or pitcher.get("name_raw"),
            "batting_team": bt, "fielding_team": ft, "home_team": home_team,
            "batting_is_home": batting_is_home,
            "inning": e["inning"], "half": e["half"],
            "tto": tto_count[(ppid, bpid)],
            "pitcher_bf": bf_count[ppid],
            "pitcher_is_starter": first_pitcher[ft] == ppid,
            "order_slot": slot_of.get((bt, bpid)),
            "pa_number_of_batter": d.get("pa_number_of_batter"),
            "outs_before": d.get("outs_before"),
            "base_out_state": d.get("base_out_state"),
            "bases_before": (
                "".join("1" if b else "0" for b in bases_before)
                if bases_before is not None else None
            ),
            "score_diff_batting": score_diff,
            "outcome_type": otype,
            "outs_recorded": outcome.get("outs_recorded", 0),
            "runs_on_play": d.get("runs_on_play"),
            "spray": spray_field(outcome),
            "location": outcome.get("location"),
            "fielder1": fielders[0] if fielders else None,
        }
        row.update(flags)
        row.update(pitch_detail(e, otype))
        yield row


def iter_game_files(games_dir):
    for path in sorted(Path(games_dir).glob("*/*.json")):
        yield path


def build(games_dir):
    """Walk the corpus and return every plate-appearance row, in file order."""
    rows = []
    for path in iter_game_files(games_dir):
        with open(path) as fh:
            game = json.load(fh)
        rows.extend(rows_for_game(game))
    return rows


def _fmt(v):
    if v is None:
        return ""
    if v is True:
        return "1"
    if v is False:
        return "0"
    return v


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r[k]) for k in COLUMNS})
    return out_path
