"""Fixture-promotion protocol regression test — issue #30 gate g1's `singled`
location-phrase extension ("up the middle").

Re-derives the promoted synthetic fixture's ``build_events`` output from a
fresh run of real code (never hand-typed) and asserts it matches the
committed fixture, so a future grammar refactor that regresses the new
`singled` alternation (or the shared batter/runner-assembly logic) is caught
immediately. See tests/fixtures/PROMOTION_PROTOCOL.md step 5.
"""

from bc_pipeline import identity, parse as parse_mod
from tests._support import load_fixture


def _run_promoted_line():
    fx = load_fixture(
        "synthetic_taxonomy_tail/singled_up_the_middle_promotion.json"
    )
    synth = fx["step_c_and_d_promoted_fixture"]["synthetic_input"]
    pbp = synth["pbp_line"]

    away = identity.TeamIdentity(
        team_id="syn:team:away",
        name="Synthetic Away",
        players={
            "syn:away:1": identity.PlayerEntry(
                player_id="syn:away:1",
                name="Kyle Schmack",
                last_name="Schmack",
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


def test_singled_up_the_middle_reproduces_the_promoted_fixture():
    fx, events, unparsed = _run_promoted_line()
    expected = fx["step_c_and_d_promoted_fixture"]["build_events_output"]
    assert events == expected["events"]
    assert unparsed == expected["unparsed"] == []


def test_singled_up_the_middle_no_longer_lands_in_unparsed():
    _fx, _events, unparsed = _run_promoted_line()
    assert unparsed == []
