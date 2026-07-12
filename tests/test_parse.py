"""Tests for bc_pipeline.parse: raw boxscore HTML -> full schema-valid
``final`` game dict.

Protected intent: correctness over coverage. Every PBP line becomes an
event OR a verbatim ``unparsed[]`` entry -- never dropped, never guessed.
This gate's close criteria: (1) the full real sample parses schema-valid
with every one of the 122 PBP cells accounted for by events+unparsed, (2)
the parser's own Top-1 (first 9 events, the away half of the 1st) matches
the hand fixture (the strongest oracle) under `serialize.semantic_equal`.
"""
from __future__ import annotations

from _support import SAMPLES_DIR, load_fixture, validate_game

from bc_pipeline import parse, serialize


def _load(name: str) -> str:
    path = SAMPLES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


FINAL_HTML = _load("boxscore_20260709_final.html")
SOURCE_URL = "https://longbeachcoast.com/sports/bsb/2026/boxscores/20260709_h94w.xml"
FETCHED_AT = "2026-07-11T00:00:00Z"


def _parse_final() -> dict:
    return parse.parse_game(FINAL_HTML, source_url=SOURCE_URL, fetched_at=FETCHED_AT)


# --- schema validity + full-sample coverage --------------------------------


def test_full_sample_is_schema_valid():
    game = _parse_final()
    validate_game(game)  # raises on failure


def test_full_sample_events_plus_unparsed_account_for_every_pbp_cell():
    # g4's own real-sample sweep (tests/test_grammar.py) counts exactly 122
    # `<td class="text">` cells across all 9 innings with 0 GrammarMiss.
    # events+unparsed here must reproduce that total: nothing dropped.
    game = _parse_final()
    assert len(game["events"]) + len(game["unparsed"]) == 122


def test_full_sample_unparsed_is_only_the_dh_pitching_subs():
    # This sample's only unparsed residue is the 5 pure pitching
    # substitutions under the two-way-DH rule (neither player ever holds a
    # batting-order slot -- see parse.py's build_events substitution
    # handling and the IMPLEMENTER_RESULT for the full grounding).
    game = _parse_final()
    assert len(game["unparsed"]) == 5
    for u in game["unparsed"]:
        assert "pitching substitution" in u["reason"]
        assert u["raw"].strip().endswith(".")


def test_full_sample_kind_counts_match_grammar_coverage_minus_subs():
    # g4's coverage evidence: plate_appearance 87, runner_event 13,
    # inning_summary 17, substitution 5 (total 122). Substitutions all
    # route to unparsed here (see above), so events[] kind counts are the
    # first three, unchanged.
    game = _parse_final()
    kinds: dict = {}
    for e in game["events"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    assert kinds == {
        "plate_appearance": 87,
        "runner_event": 13,
        "inning_summary": 17,
    }


def test_full_sample_linescore_matches_fixture_oracle():
    game = _parse_final()
    fixture = load_fixture("game_20260709_h94w_top1.json")
    assert game["linescore"] == fixture["linescore"]


def test_full_sample_box_batting_first_six_away_rows_match_fixture():
    game = _parse_final()
    fixture = load_fixture("game_20260709_h94w_top1.json")
    away_team_id = game["teams"]["away"]["team_id"]
    assert (
        game["box"]["batting"][away_team_id][:6]
        == fixture["box"]["batting"][away_team_id]
    )


def test_full_sample_header_fields():
    game = _parse_final()
    assert game["game_id"] == "20260709_h94w"
    assert game["season"] == 2026
    assert game["status"] == "final"
    assert game["date"] == "2026-07-09"
    assert game["source"]["provider"] == "prestosports"
    assert game["source"]["site"] == "longbeachcoast.com"
    assert game["teams"]["home"]["team_id"] == "maotayco79j2g2lx"
    assert game["teams"]["home"]["name"] == "Long Beach Coast"
    assert game["teams"]["away"]["name"] == "Yuba-Sutter Freebirds"


def test_full_sample_meta_parse_integrity_signals():
    game = _parse_final()
    meta = game["meta"]
    assert meta["parser_version"] == parse.PARSER_VERSION
    assert meta["source_url"] == SOURCE_URL
    assert meta["source_sha256"] == parse.sha256_hex(FINAL_HTML)
    assert meta["fetched_at"] == FETCHED_AT
    assert meta["derived_replayer_version"] == "unreplayed"
    assert meta["parse"]["events_count"] == len(game["events"])
    assert meta["parse"]["unparsed_count"] == len(game["unparsed"])
    assert meta["parse"]["replayable"] is False


# --- Top-1 agreement (the strongest oracle) ---------------------------------


def test_top1_events_semantic_equal_hand_fixture():
    game = _parse_final()
    fixture = load_fixture("game_20260709_h94w_top1.json")
    top9 = game["events"][:9]
    assert serialize.semantic_equal({"events": top9}, {"events": fixture["events"]})


def test_top1_events_all_belong_to_inning_1_top():
    game = _parse_final()
    for e in game["events"][:9]:
        assert e["inning"] == 1
        assert e["half"] == "top"
    assert game["events"][9]["half"] == "bottom" or game["events"][9]["inning"] == 2


# --- idempotency key ---------------------------------------------------------


def test_idempotency_key_is_hash_plus_parser_version():
    key = parse.idempotency_key(FINAL_HTML)
    assert parse.sha256_hex(FINAL_HTML) in key
    assert parse.PARSER_VERSION in key


def test_idempotency_key_changes_with_different_html():
    assert parse.idempotency_key(FINAL_HTML) != parse.idempotency_key(FINAL_HTML + " ")
