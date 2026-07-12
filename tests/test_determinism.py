"""Determinism test: parsing the same raw HTML bytes twice with the same
parser version produces the same game (modulo `meta` and every `_derived`
block, per `serialize.semantic_equal`'s contract).

Protected intent: parse is a pure function of (html bytes, parser version)
-- no wall-clock/random/dict-iteration-order leakage into the asserted
game content.
"""
from __future__ import annotations

from _support import SAMPLES_DIR

from bc_pipeline import parse, serialize


def _load(name: str) -> str:
    path = SAMPLES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


FINAL_HTML = _load("boxscore_20260709_final.html")
SOURCE_URL = "https://longbeachcoast.com/sports/bsb/2026/boxscores/20260709_h94w.xml"


def test_parsing_twice_is_semantically_equal():
    run1 = parse.parse_game(FINAL_HTML, source_url=SOURCE_URL, fetched_at="2026-07-11T00:00:00Z")
    run2 = parse.parse_game(FINAL_HTML, source_url=SOURCE_URL, fetched_at="2026-07-11T00:01:00Z")
    assert serialize.semantic_equal(run1, run2)


def test_parsing_twice_produces_byte_identical_canonical_dumps_minus_meta():
    run1 = parse.parse_game(FINAL_HTML, source_url=SOURCE_URL, fetched_at="2026-07-11T00:00:00Z")
    run2 = parse.parse_game(FINAL_HTML, source_url=SOURCE_URL, fetched_at="2026-07-11T00:01:00Z")
    dump1 = serialize.canonical_dumps(serialize.strip_volatile(run1))
    dump2 = serialize.canonical_dumps(serialize.strip_volatile(run2))
    assert dump1 == dump2


def test_idempotency_key_is_stable_across_reparses():
    key1 = parse.idempotency_key(FINAL_HTML)
    key2 = parse.idempotency_key(FINAL_HTML)
    assert key1 == key2
