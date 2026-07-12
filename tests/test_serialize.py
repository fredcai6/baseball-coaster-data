"""Tests for bc_pipeline.serialize: the canonical serializer + semantic-equality contract.

Protected intent (schema $comment, DECISION.md sec4): two game files are "the same game"
iff deep-equal after deleting root `meta` and EVERY `_derived` block, at any depth.
"""
from __future__ import annotations

import copy
import json

import pytest

from _support import load_fixture, validate_game

from bc_pipeline import serialize

FIXTURE_NAME = "game_20260709_h94w_top1.json"


@pytest.fixture
def fx() -> dict:
    return load_fixture(FIXTURE_NAME)


def test_hand_fixture_loads_and_validates_against_schema(fx):
    """The existing hand fixture loads and passes jsonschema validation against the frozen schema."""
    validate_game(fx)  # raises on invalid; no exception == valid


def test_semantic_equal_identical_fixture_is_true(fx):
    """A game file compared against an unmodified deep copy of itself is semantically equal."""
    other = copy.deepcopy(fx)
    assert serialize.semantic_equal(fx, other) is True
    # inputs must not be mutated by the comparison
    assert fx == load_fixture(FIXTURE_NAME)


def test_semantic_equal_true_when_only_meta_changes(fx):
    """Changing root `meta` alone does not affect semantic equality (meta is provenance-only)."""
    original = copy.deepcopy(fx)
    other = copy.deepcopy(fx)
    other["meta"]["parser_version"] = "some-other-parser-version"
    other["meta"]["source_sha256"] = "0" * 64
    other["meta"]["parse"]["warnings"] = ["a different re-parse run"]
    assert serialize.semantic_equal(fx, other) is True
    # inputs must not be mutated by the comparison
    assert fx == original


def test_semantic_equal_true_when_only_event_derived_changes(fx):
    """Changing an event's `_derived` block alone does not affect semantic equality
    (it's a regenerable cache, excluded from semantic equality at any depth)."""
    original = copy.deepcopy(fx)
    other = copy.deepcopy(fx)
    other["events"][0]["_derived"]["outs_before"] = 99
    other["events"][0]["_derived"]["base_out_state"] = "999|9"
    other["events"][0]["_derived"]["new_analytic_key_not_in_original"] = "whatever"
    assert serialize.semantic_equal(fx, other) is True
    # inputs must not be mutated by the comparison
    assert fx == original


def test_semantic_equal_false_when_event_outcome_type_changes(fx):
    """Changing real event content (outcome.type) breaks semantic equality."""
    original = copy.deepcopy(fx)
    other = copy.deepcopy(fx)
    assert other["events"][0]["outcome"]["type"] == "single"
    other["events"][0]["outcome"]["type"] = "double"
    assert serialize.semantic_equal(fx, other) is False
    # inputs must not be mutated by the comparison
    assert fx == original


def test_canonical_dumps_is_idempotent(fx):
    """canonical_dumps(x) == canonical_dumps(json.loads(canonical_dumps(x)))."""
    once = serialize.canonical_dumps(fx)
    assert isinstance(once, str)
    assert once.endswith("\n")
    twice = serialize.canonical_dumps(json.loads(once))
    assert once == twice
