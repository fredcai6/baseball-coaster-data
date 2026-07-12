"""Fixture-promotion protocol — first REAL exercise.

One of the 5 DH-pitching-substitution lines from the live sample was an
``unparsed[]`` residue under schema 1.0.0 (``substitution.slot`` required 1-9,
a DH pitching change has no batting-order slot). After the additive-MINOR
evolution to 1.1.0 (nullable slot), it is a real substitution event. This test
re-derives the promoted line's parse from the live sample and asserts it matches
the committed promotion fixture, so the fixture stays golden rather than
hand-copied. See tests/fixtures/PROMOTION_PROTOCOL.md and docs/design/DECISION.md §7.
"""

import bc_pipeline.grammar as grammar
import bc_pipeline.parse as parse
from tests._support import SAMPLES_DIR, load_fixture

SOURCE_URL = "https://longbeachcoast.com/sports/bsb/2026/boxscores/20260709_h94w.xml"
FETCHED_AT = "2026-07-11T00:00:00Z"


def _parse_final() -> dict:
    html = (SAMPLES_DIR / "boxscore_20260709_final.html").read_text(encoding="utf-8")
    return parse.parse_game(html, source_url=SOURCE_URL, fetched_at=FETCHED_AT)


def test_grammar_recognizes_the_promoted_line():
    fx = load_fixture("promoted/dh_pitching_sub_promotion.json")
    cg = grammar.parse_clause_group(fx["raw_line"])
    assert cg.kind == "substitution"
    assert cg.substitution.player_in == fx["grammar_clausegroup"]["player_in"]
    assert cg.substitution.player_out == fx["grammar_clausegroup"]["player_out"]


def test_live_sample_reproduces_the_promoted_substitution_event():
    fx = load_fixture("promoted/dh_pitching_sub_promotion.json")
    expected = fx["promoted_substitution_event"]
    game = _parse_final()
    match = [
        e
        for e in game["events"]
        if e["kind"] == "substitution" and e["narrative"] == fx["raw_line"]
    ]
    assert len(match) == 1, "the promoted line must appear exactly once as a substitution event"
    assert match[0] == expected
    # And it is no longer an unparsed[] residue.
    assert all(fx["raw_line"] not in u["raw"] for u in game["unparsed"])
