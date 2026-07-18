"""Fixture-promotion protocol regression test — issue #31 gate g2b's
`foul_out` PRIMARY_RULES row.

Re-derives the promoted synthetic fixture's ``build_events`` output from a
fresh run of real code (never hand-typed) and asserts it matches the
committed fixture for BOTH an infield and an outfield foul_out, so a future
grammar refactor that regresses the new row (or drops the fielders
passthrough -- the human's hard requirement) is caught immediately. See
tests/fixtures/PROMOTION_PROTOCOL.md step 5.
"""

from bc_pipeline import identity, parse as parse_mod
from tests._support import load_fixture


def _player_table():
    away = identity.TeamIdentity(
        team_id="syn:team:away",
        name="Synthetic Away",
        players={
            "syn:away:1": identity.PlayerEntry(
                player_id="syn:away:1",
                name="Pat Smith",
                last_name="Smith",
                team_id="syn:team:away",
                positions=["cf"],
            ),
        },
    )
    home = identity.TeamIdentity(
        team_id="syn:team:home",
        name="Synthetic Home",
        players={
            "syn:home:1": identity.PlayerEntry(
                player_id="syn:home:1",
                name="Jordan Lee",
                last_name="Lee",
                team_id="syn:team:home",
                positions=["p"],
            ),
        },
    )
    return identity.PlayerTable(home=home, away=away)


def _run_case(case: dict):
    pbp = case["synthetic_input"]["pbp_line"]
    player_table = _player_table()
    line = parse_mod.PbpLine(
        inning=pbp["inning"],
        half=pbp["half"],
        line_index=pbp["line_index"],
        text=pbp["text"],
        is_strong=pbp["is_strong"],
    )
    events, unparsed, _subs = parse_mod.build_events([line], player_table)
    return events, unparsed


def test_foul_out_infield_reproduces_the_promoted_fixture():
    fx = load_fixture("synthetic_taxonomy_tail/foul_out_promotion.json")
    case = fx["step_c_and_d_promoted_fixture"]["infield"]
    events, unparsed = _run_case(case)
    expected = case["build_events_output"]
    assert events == expected["events"]
    assert unparsed == expected["unparsed"] == []
    assert events[0]["outcome"]["fielders"] == ["1b"]


def test_foul_out_outfield_reproduces_the_promoted_fixture():
    fx = load_fixture("synthetic_taxonomy_tail/foul_out_promotion.json")
    case = fx["step_c_and_d_promoted_fixture"]["outfield"]
    events, unparsed = _run_case(case)
    expected = case["build_events_output"]
    assert events == expected["events"]
    assert unparsed == expected["unparsed"] == []
    assert events[0]["outcome"]["fielders"] == ["rf"]


def test_foul_out_no_longer_lands_in_unparsed():
    fx = load_fixture("synthetic_taxonomy_tail/foul_out_promotion.json")
    for case in (
        fx["step_c_and_d_promoted_fixture"]["infield"],
        fx["step_c_and_d_promoted_fixture"]["outfield"],
    ):
        _events, unparsed = _run_case(case)
        assert unparsed == []
