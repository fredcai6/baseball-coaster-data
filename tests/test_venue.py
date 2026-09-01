"""Venue, and the home-field covariate it corrects.

`venue` (schema 1.13.0) was sitting in the archived HTML the whole time -- the
source page's "Other Information" table -- and the parser had never looked at
it. Recovering it cost zero fetches and settled a question nothing in the
corpus could previously answer: WAS THE DESIGNATED HOME TEAM ACTUALLY AT HOME?

Usually yes. Not always. The 2025 Colorado Springs Sky Sox played 26 of their
37 designated-home games in the opponent's ballpark, whole series at a time,
and they shared Blocktickets Park with the Rocky Mountain Vibes for the other
11. So `half == "bottom"` is a fact about BATTING ORDER and is not a proxy for
home-field advantage; `batting_at_home_park` is. Every model on the board had
been using the former for the latter.

The numbers below are pinned so a re-parse or a canonicalization edit that
moves them has to say so out loud.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from bc_pipeline import pa_table, venue

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_DIR = REPO_ROOT / "games"


@pytest.fixture(scope="module")
def games():
    return [json.loads(p.read_text()) for p in pa_table.iter_game_files(GAMES_DIR)]


# --------------------------------------------------------------- coverage --

def test_every_game_states_a_venue_block_even_when_it_has_no_venue(games):
    """Required and explicit, on the `record_shape` argument: an absent marker
    must never be ambiguous between "the page stated no venue" and "this file
    predates the field"."""
    missing = [g["game_id"] for g in games if "venue" not in g]
    assert not missing, f"{len(missing)} game(s) carry no venue block: {missing[:5]}"
    bad = [g["game_id"] for g in games
           if set(g["venue"]) != {"raw", "field"}]
    assert not bad, f"venue block with unexpected keys: {bad[:5]}"


def test_venue_coverage_is_pinned(games):
    with_venue = [g for g in games if g["venue"]["raw"]]
    assert len(games) == 1485
    assert len(with_venue) == 1483, (
        f"{len(with_venue)} games carry a venue string, expected 1483. The two "
        f"known gaps are Long Beach Coast games whose Other Information table "
        f"holds only Attendance and Duration."
    )
    silent = sorted(g["game_id"] for g in games if not g["venue"]["raw"])
    assert silent == ["20260605_vawp", "20260610_3see"]


def test_the_field_the_string_came_from_is_recorded(games):
    """`Stadium` and `Location` are not equally trustworthy -- Location is
    sometimes a street address and sometimes only a city -- so which row the
    value came from is recorded rather than left for the caller to guess."""
    by_field = collections.Counter(g["venue"]["field"] for g in games)
    assert by_field["Stadium"] == 1030
    assert by_field["Location"] == 453
    assert by_field[None] == 2
    for g in games:
        assert (g["venue"]["field"] is None) == (g["venue"]["raw"] is None)


def test_canonicalization_folds_the_corpus_onto_sixteen_parks(games):
    raws = {g["venue"]["raw"] for g in games if g["venue"]["raw"]}
    parks = {venue.canonicalize(r) for r in raws}
    assert len(raws) == 81, f"{len(raws)} distinct raw venue strings, expected 81"
    assert len(parks) == 16, f"{len(parks)} canonical parks, expected 16: {sorted(parks)}"


def test_no_raw_string_is_season_ambiguous(games):
    """The canonicalization map is SEASON-BLIND, and two near-identical strings
    resolve to different parks: 'Colorado Springs' -> UCHealth Park (the Vibes'
    2024 home) and 'Colorado Springs, CO' -> Blocktickets Park (their 2025
    home, after the move). That is only safe while each string stays inside one
    season. If a later fetch puts either string in the other's season, the map
    silently returns the wrong park -- so pin it here rather than discover it
    through a wrong park factor."""
    seasons = collections.defaultdict(set)
    for g in games:
        if g["venue"]["raw"]:
            seasons[g["venue"]["raw"]].add(g["season"])
    for s in ("Colorado Springs", "Colorado Springs, CO"):
        assert len(seasons[s]) == 1, (
            f"{s!r} now appears in seasons {sorted(seasons[s])}; the season-blind "
            f"CANONICAL_VENUE map cannot distinguish them any more."
        )


# ------------------------------------------------- the home-field covariate --

def test_home_park_index_takes_the_mode_not_the_assumption(games):
    """Assuming a team's home park from its designated-home games would give
    the Sky Sox whichever park they visited most. The mode survives because 11
    beats any single opponent's 6 -- so the 26 exceptions stay exceptions."""
    idx = pa_table.home_park_index(games)
    sky = [k for k in idx if k[1] == 2025 and idx[k] == "Blocktickets Park"]
    assert len(sky) == 2, (
        "Blocktickets Park should be the 2025 home park of exactly two clubs -- "
        "the Vibes, and the Sky Sox who shared it."
    )


def test_designated_home_matches_actual_park_on_1457_of_1485(games):
    idx = pa_table.home_park_index(games)
    agree = disagree = unknown = 0
    for g in games:
        park = pa_table.venue_canonical_of(g)
        home_id = g["teams"]["home"]["team_id"]
        own = idx.get((home_id, g["season"]))
        if park is None or own is None:
            unknown += 1
        elif park == own:
            agree += 1
        else:
            disagree += 1
    assert (agree, disagree, unknown) == (1457, 26, 2), (
        f"agree={agree} disagree={disagree} unknown={unknown}; expected "
        f"1457/26/2. The 26 are the 2025 Sky Sox playing designated-home games "
        f"in the opponent's park."
    )


def test_at_home_park_falls_back_to_batting_order_only_when_venue_is_unknown():
    assert pa_table._at_home_park("Dehler Park", "Dehler Park", False) is True
    assert pa_table._at_home_park("Dehler Park", "Suplizio Field", True) is False
    # unknown venue, or a team whose own park we cannot name: keep the old
    # designated-home reading, but as a stated fallback rather than silently.
    assert pa_table._at_home_park(None, "Dehler Park", True) is True
    assert pa_table._at_home_park(None, "Dehler Park", False) is False
    assert pa_table._at_home_park("Dehler Park", None, True) is True


def test_both_sides_flip_when_the_designated_home_team_is_on_the_road(games):
    """The correction is not "remove the Sky Sox's home advantage". In a game
    the Sky Sox hosted at Dehler Park, BILLINGS really was at home -- so the
    nominal away team gains the advantage the nominal home team loses."""
    idx = pa_table.home_park_index(games)
    g = next(x for x in games if x["game_id"] == "20250603_ihyw")
    rows = list(pa_table.rows_for_game(g, idx))
    home_rows = [r for r in rows if r["batting_is_home"]]
    away_rows = [r for r in rows if not r["batting_is_home"]]
    assert home_rows and away_rows
    assert all(r["batting_at_home_park"] is False for r in home_rows)
    assert all(r["batting_at_home_park"] is True for r in away_rows)
