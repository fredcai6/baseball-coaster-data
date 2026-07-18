"""Fixture-promotion protocol regression test -- g3 REWORK (commander-31's
parse.py-assembly design call): the try-both-sides substitution
side-resolution fallback in parse.py's substitution assembly branch.

Re-derives the promoted synthetic fixture's ``build_events`` output from a
fresh run of real code (never hand-typed) and asserts it matches the
committed fixture, so a future assembly refactor that regresses the
fallback (or reintroduces the pre-rework batting-side-only assumption) is
caught immediately. See tests/fixtures/PROMOTION_PROTOCOL.md step 5.
"""

from bc_pipeline import identity, parse as parse_mod
from tests._support import load_fixture


def _run_promoted_line():
    fx = load_fixture(
        "synthetic_taxonomy_tail/two_name_dh_sub_fielding_side_fallback_promotion.json"
    )
    synth = fx["step_c_and_d_promoted_fixture"]["synthetic_input"]
    pbp = synth["pbp_line"]

    away = identity.TeamIdentity(
        team_id="syn:team:away",
        name="Synthetic Away",
        players={
            "syn:away:1": identity.PlayerEntry(
                player_id="syn:away:1",
                name="J. McLaughli",
                last_name="McLaughlin",
                team_id="syn:team:away",
                positions=["dh", "cf"],
            ),
            "syn:away:2": identity.PlayerEntry(
                player_id="syn:away:2",
                name="A. Sczepkows",
                last_name="Sczepkowski",
                team_id="syn:team:away",
                positions=["dh", "cf"],
            ),
        },
    )
    home = identity.TeamIdentity(
        team_id="syn:team:home",
        name="Synthetic Home",
        players={
            "syn:home:1": identity.PlayerEntry(
                player_id="syn:home:1",
                name="Home Pitcher",
                last_name="Pitcher",
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


def test_dh_sub_fielding_side_fallback_reproduces_the_promoted_fixture():
    fx, events, unparsed = _run_promoted_line()
    expected = fx["step_c_and_d_promoted_fixture"]["build_events_output"]
    assert events == expected["events"]
    assert unparsed == expected["unparsed"] == []


def test_dh_sub_fielding_side_fallback_no_longer_lands_in_unparsed():
    _fx, _events, unparsed = _run_promoted_line()
    assert unparsed == []


def test_dh_sub_fielding_side_fallback_kind_stays_offensive():
    # kind is grammar-layer position semantics, decoupled from which side
    # assembly resolved against -- must stay "offensive" even though this
    # sub resolved on the FIELDING side.
    _fx, events, _unparsed = _run_promoted_line()
    assert events[0]["substitution"]["kind"] == "offensive"
