"""Tests for schema 1.3.0 (issue #31, g2b): outcome.type enum gains
`foul_out` and `strikeout`.

Protected intent: additive-only MINOR bump. Every 1.2.0-valid game file stays
valid; only the two new closed-taxonomy enum values are added. Human-ratified
via Admiral (issue #31) -- see the schema $comment VERSION HISTORY entry.
"""
from __future__ import annotations

import copy

import pytest

from _support import load_fixture, load_schema, validate_game

FIXTURE_NAME = "game_20260709_h94w_top1.json"


@pytest.fixture
def fx() -> dict:
    return load_fixture(FIXTURE_NAME)


def _first_plate_appearance(d: dict) -> dict:
    for e in d["events"]:
        if e["kind"] == "plate_appearance":
            return e
    raise AssertionError("fixture has no plate_appearance event")


def test_schema_version_is_1_3_0():
    schema = load_schema()
    assert "1.3.0" in schema["$comment"]


def test_outcome_type_enum_gains_foul_out_and_strikeout():
    schema = load_schema()
    enum = set(schema["$defs"]["outcome"]["properties"]["type"]["enum"])
    assert "foul_out" in enum
    assert "strikeout" in enum
    assert len(enum) == 19


def test_foul_out_event_validates(fx):
    """A plate_appearance event with outcome.type foul_out, fielders
    populated with the catching position, must validate under 1.3.0."""
    other = copy.deepcopy(fx)
    pa = _first_plate_appearance(other)
    pa["outcome"] = {
        "type": "foul_out",
        "modifiers": [],
        "fielders": ["1b"],
        "outs_recorded": 1,
        "location": None,
    }
    validate_game(other)  # raises on invalid; no exception == valid


def test_strikeout_event_validates(fx):
    """A plate_appearance event with outcome.type strikeout, fielders []
    (no fielder named), must validate under 1.3.0."""
    other = copy.deepcopy(fx)
    pa = _first_plate_appearance(other)
    pa["outcome"] = {
        "type": "strikeout",
        "modifiers": [],
        "fielders": [],
        "outs_recorded": 1,
        "location": None,
    }
    validate_game(other)  # raises on invalid; no exception == valid


def test_existing_1_2_0_fixture_still_validates(fx):
    """Additive-only guarantee: the pre-1.3.0 fixture (no foul_out/strikeout
    events) is unmodified and still validates under the bumped schema."""
    validate_game(fx)
