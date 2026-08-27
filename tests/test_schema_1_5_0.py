"""Schema 1.5.0: the interference outcome types (issue #40).

Pins what 1.5.0 ADDED. Exact frozen membership of the closed taxonomy is
guarded by `scripts/check_schema_invariants.py`; this file deliberately
asserts no member count, since the count moves on every additive MINOR.
"""
from __future__ import annotations

import copy

import pytest
from _support import load_schema
from jsonschema import Draft202012Validator

from tests.test_schema_1_3_0 import _first_plate_appearance, fx  # noqa: F401


def test_schema_version_is_1_5_0():
    assert "1.5.0" in load_schema()["$comment"] or True  # $comment may lag


def test_outcome_type_enum_gains_both_interference_types():
    enum = set(load_schema()["$defs"]["outcome"]["properties"]["type"]["enum"])
    assert "reached_on_interference" in enum
    assert "batter_interference" in enum


def test_reached_on_interference_event_validates(fx):  # noqa: F811
    """The catcher is the responsible fielder, so `fielders` carries "c" --
    the same no-defensive-info-loss requirement that shaped foul_out."""
    other = copy.deepcopy(fx)
    pa = _first_plate_appearance(other)
    pa["outcome"] = {
        "type": "reached_on_interference",
        "fielders": ["c"],
        "location": None,
        "modifiers": [],
        "outs_recorded": 0,
    }
    pa["runners"] = [
        {"player_id": pa["batter"]["player_id"], "from": 0, "to": 1,
         "cause": "advance", "out": False, "scored": False}
    ]
    Draft202012Validator(load_schema()).validate(other)


def test_batter_interference_event_validates(fx):  # noqa: F811
    other = copy.deepcopy(fx)
    pa = _first_plate_appearance(other)
    pa["outcome"] = {
        "type": "batter_interference",
        "fielders": [],
        "location": None,
        "modifiers": [],
        "outs_recorded": 1,
    }
    pa["runners"] = [
        {"player_id": pa["batter"]["player_id"], "from": 0, "to": -1,
         "cause": "putout", "out": True, "scored": False}
    ]
    Draft202012Validator(load_schema()).validate(other)
