"""Tests for schema 1.2.0 (issue #30, g2b): count and substitution.player_out made nullable.

Protected intent: additive-only MINOR bump. Every 1.1.0-valid game file stays valid;
only new optional/nullable surface is added. null means "the source genuinely omitted
this," never a guess.
"""
from __future__ import annotations

import copy

import pytest

from _support import load_fixture, validate_game

FIXTURE_NAME = "game_20260709_h94w_top1.json"


@pytest.fixture
def fx() -> dict:
    return load_fixture(FIXTURE_NAME)


def _first_plate_appearance(d: dict) -> dict:
    for e in d["events"]:
        if e["kind"] == "plate_appearance":
            return e
    raise AssertionError("fixture has no plate_appearance event")


def test_schema_version_is_1_2_0():
    from _support import load_schema

    schema = load_schema()
    assert "1.2.0" in schema["$comment"]


def test_plate_appearance_count_null_validates(fx):
    """A plate_appearance event with count: null (no count-tail in the source line)
    must validate under the bumped 1.2.0 schema."""
    other = copy.deepcopy(fx)
    pa = _first_plate_appearance(other)
    pa["count"] = None
    validate_game(other)  # raises on invalid; no exception == valid


def test_substitution_player_out_null_validates(fx):
    """A substitution event with player_out: null (a bare DH-slot-entry naming only
    the incoming player) must validate under the bumped 1.2.0 schema."""
    other = copy.deepcopy(fx)
    sub_event = {
        "seq": 999,
        "inning": 1,
        "half": "top",
        "kind": "substitution",
        "batting_team": other["teams"]["away"]["team_id"],
        "fielding_team": other["teams"]["home"]["team_id"],
        "narrative": "Cole Robinson to dh.",
        "scoring_play": False,
        "substitution": {
            "slot": None,
            "player_out": None,
            "player_in": next(iter(other["players"])),
            "kind": "offensive",
            "after_event_seq": 0,
        },
    }
    other["events"].append(sub_event)
    validate_game(other)  # raises on invalid; no exception == valid


def test_existing_1_1_0_fixture_still_validates(fx):
    """Additive-only guarantee: the pre-1.2.0 fixture (non-null count/player_out
    everywhere) is unmodified and still validates under the bumped schema."""
    validate_game(fx)
