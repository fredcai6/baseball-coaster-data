"""Regression tests for issue #30 gate g2 (Family 2: abbreviated/truncated/
suffix-comma player-name resolution).

Protected intent: PlayerTable.resolve() NEVER guesses -- an ambiguous match,
even after this gate's new prefix-match fallback, must stay honestly
unresolved ((None, False)) and route the line to unparsed[]. This file
re-derives real corpus lines (not hand-typed output) through the actual
parse.build_events()/identity.PlayerTable.resolve() code, so a future
regression is caught immediately. See tests/fixtures/PROMOTION_PROTOCOL.md
and tests/fixtures/synthetic_taxonomy_tail/family2_comma_suffix_promotion.json
for the promoted fixture (the comma-suffix shape).
"""
from __future__ import annotations

from _support import load_fixture

from bc_pipeline import identity, parse as parse_mod


# --- promoted fixture: comma-suffix shape ("Rojas, Jr") ---------------------


def _run_promoted_comma_suffix_line():
    fx = load_fixture("synthetic_taxonomy_tail/family2_comma_suffix_promotion.json")
    pbp = fx["step_c_and_d_promoted_fixture"]["synthetic_input"]["pbp_line"]

    home = identity.TeamIdentity(
        team_id="syn:team:home",
        name="Synthetic Home",
        players={
            "fgyxjb0r734dgsdl": identity.PlayerEntry(
                player_id="fgyxjb0r734dgsdl",
                name="Freddy Rojas Jr.",
                last_name="Rojas",
                team_id="syn:team:home",
                positions=["dh"],
            ),
        },
    )
    away = identity.TeamIdentity(
        team_id="syn:team:away",
        name="Synthetic Away",
        players={
            "syn:away:1": identity.PlayerEntry(
                player_id="syn:away:1",
                name="Alex Preece",
                last_name="Preece",
                team_id="syn:team:away",
                positions=["p"],
            ),
        },
    )
    player_table = identity.PlayerTable(home=home, away=away)
    line = parse_mod.PbpLine(
        inning=pbp["inning"],
        half=pbp["half"],
        line_index=pbp["line_index"],
        text=pbp["text"],
        is_strong=pbp["is_strong"],
    )
    events, unparsed, _subs = parse_mod.build_events([line], player_table)
    return fx, events, unparsed


def test_comma_suffix_line_reproduces_the_promoted_fixture():
    fx, events, unparsed = _run_promoted_comma_suffix_line()
    expected = fx["step_c_and_d_promoted_fixture"]["build_events_output"]
    assert events == expected["events"]
    assert unparsed == expected["unparsed"] == []


def test_comma_suffix_line_no_longer_lands_in_unparsed():
    _fx, _events, unparsed = _run_promoted_comma_suffix_line()
    assert unparsed == []


# --- sibling shapes: truncated surnames, re-derived from real corpus data --
#
# These re-create just the roster entries needed (real player_id/name/
# last_name/team_id, copied verbatim from the named games/**  file's
# committed players[] table) rather than the whole game, per
# PROMOTION_PROTOCOL.md's identity.PlayerTable construction convention.


def _two_sided_table(home_players, away_players, home_id="h", away_id="a"):
    home = identity.TeamIdentity(team_id=home_id, name="Home", players=home_players)
    away = identity.TeamIdentity(team_id=away_id, name="Away", players=away_players)
    return identity.PlayerTable(home=home, away=away)


def test_truncated_substitution_name_resolves_real_corpus_shape():
    # Real line: games/2025/20250520_u80r.json unparsed[] (pre-fix):
    # "C. Richardso to p for L. Short." -- real roster has "Conner
    # Richardson" (last_name "Richardson", home side) and "Luke Short"
    # (last_name "Short", home side).
    home_players = {
        "a8k7z5bbeuii76ei": identity.PlayerEntry(
            player_id="a8k7z5bbeuii76ei",
            name="Conner Richardson",
            last_name="Richardson",
            team_id="h",
            positions=["p"],
        ),
        "y93tig5ow0mslgob": identity.PlayerEntry(
            player_id="y93tig5ow0mslgob",
            name="Luke Short",
            last_name="Short",
            team_id="h",
            positions=["p"],
        ),
    }
    table = _two_sided_table(home_players, {})
    line = parse_mod.PbpLine(
        inning=1,
        half="top",
        line_index=0,
        text="C. Richardso to p for L. Short.",
        is_strong=False,
    )
    events, unparsed, _subs = parse_mod.build_events([line], table)
    assert unparsed == []
    assert events[0]["substitution"]["player_in"] == "a8k7z5bbeuii76ei"
    assert events[0]["substitution"]["player_out"] == "y93tig5ow0mslgob"


def test_truncated_runner_event_name_resolves_real_corpus_shape():
    # Real line: games/2025/20250520_4bkm.json unparsed[] (pre-fix):
    # "J. McLaughli stole second." -- real roster has "JD McLaughlin"
    # (last_name "McLaughlin", home side).
    home_players = {
        "lxy5m1w6pu28csl0": identity.PlayerEntry(
            player_id="lxy5m1w6pu28csl0",
            name="JD McLaughlin",
            last_name="McLaughlin",
            team_id="h",
            positions=["ss"],
        ),
    }
    table = _two_sided_table(home_players, {})
    line = parse_mod.PbpLine(
        inning=1, half="bottom", line_index=0, text="J. McLaughli stole second.", is_strong=False
    )
    events, unparsed, _subs = parse_mod.build_events([line], table)
    assert unparsed == []
    assert events[0]["runners"][0]["player_id"] == "lxy5m1w6pu28csl0"


def test_truncated_hyphenated_surname_resolves_real_corpus_shape():
    # Real line: games/2024/20240521_gq1b.json unparsed[] (pre-fix):
    # "T. Clark-Chi advanced to second on a wild pitch." -- real roster has
    # "Tyler Clark-Chiapparelli" (last_name "Clark-Chiapparelli", home side).
    home_players = {
        "dizcmqk3f9odli2s": identity.PlayerEntry(
            player_id="dizcmqk3f9odli2s",
            name="Tyler Clark-Chiapparelli",
            last_name="Clark-Chiapparelli",
            team_id="h",
            positions=["2b"],
        ),
    }
    table = _two_sided_table(home_players, {})
    line = parse_mod.PbpLine(
        inning=1,
        half="bottom",
        line_index=0,
        text="T. Clark-Chi advanced to second on a wild pitch.",
        is_strong=False,
    )
    events, unparsed, _subs = parse_mod.build_events([line], table)
    assert unparsed == []
    assert events[0]["runners"][0]["player_id"] == "dizcmqk3f9odli2s"
