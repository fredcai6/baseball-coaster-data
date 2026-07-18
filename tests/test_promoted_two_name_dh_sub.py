"""Fixture-promotion protocol regression test -- issue #31 gate g3's
generalized two-name substitution row, issue #32 (the '<in> to dh for <out>.'
DH-sub shape specifically).

Re-derives the promoted synthetic fixture's ``build_events`` output from a
fresh run of real code (never hand-typed) and asserts it matches the
committed fixture, so a future grammar/assembly refactor that regresses the
generalized `_SUBSTITUTION_RE` or its position-branched `kind` assignment is
caught immediately. See tests/fixtures/PROMOTION_PROTOCOL.md step 5.
"""

from bc_pipeline import identity, parse as parse_mod
from tests._support import load_fixture


def _run_promoted_line():
    fx = load_fixture("synthetic_taxonomy_tail/two_name_dh_sub_promotion.json")
    synth = fx["step_c_and_d_promoted_fixture"]["synthetic_input"]
    pbp = synth["pbp_line"]

    away = identity.TeamIdentity(
        team_id="syn:team:away",
        name="Synthetic Away",
        players={
            "syn:away:1": identity.PlayerEntry(
                player_id="syn:away:1",
                name="P. DePasqual",
                last_name="DePasqual",
                team_id="syn:team:away",
                positions=["dh"],
            ),
            "syn:away:2": identity.PlayerEntry(
                player_id="syn:away:2",
                name="J. Impedugli",
                last_name="Impedugli",
                team_id="syn:team:away",
                positions=["dh"],
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


def test_two_name_dh_sub_reproduces_the_promoted_fixture():
    fx, events, unparsed = _run_promoted_line()
    expected = fx["step_c_and_d_promoted_fixture"]["build_events_output"]
    assert events == expected["events"]
    assert unparsed == expected["unparsed"] == []


def test_two_name_dh_sub_no_longer_lands_in_unparsed():
    _fx, _events, unparsed = _run_promoted_line()
    assert unparsed == []


def test_two_name_dh_sub_kind_is_offensive_not_hardcoded_pitching():
    # The regression this whole gate targets: `_build_substitution` used to
    # hardcode kind="pitching" unconditionally. Confirms the promoted DH-sub
    # event is NOT mislabeled "pitching".
    _fx, events, _unparsed = _run_promoted_line()
    assert events[0]["substitution"]["kind"] == "offensive"
