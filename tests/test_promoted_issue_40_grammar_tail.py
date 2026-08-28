"""Fixture-promotion regression test -- issue #40's grammar tail.

Every case is re-derived from a fresh ``build_events`` run and compared
against the committed synthetic fixture, so a later grammar refactor that
regresses one of these shapes is caught immediately (see
tests/fixtures/PROMOTION_PROTOCOL.md step 5).

Beyond the newly-covered narrative shapes, this file pins the three things
issue #40 changed about the parser's POSTURE, each of which is a behaviour a
refactor could silently undo:

* a name capture that swallowed an unmatched lead clause fails LOUD;
* the two StatCrew source defects are recovered but DISCLOSED, never silent;
* each inference REFUSES when the evidence does not force an answer.
"""

import pytest

from bc_pipeline import grammar, identity, parse as parse_mod
from tests._support import load_fixture

FIXTURE = "synthetic_taxonomy_tail/issue_40_grammar_tail_promotion.json"

_AWAY = [
    ("syn:away:1", "Pat Smith", "cf"),
    ("syn:away:2", "Robin Vega", "1b"),
    ("syn:away:3", "Casey Doyle", "2b"),
]
_HOME = [
    ("syn:home:1", "Jordan Lee", "c"),
    ("syn:home:2", "Alex Nunez", "ss"),
    ("syn:home:3", "Sam Ortiz", "3b"),
    ("syn:home:4", "Drew Park", "rf"),
    ("syn:home:5", "Tay Quinn", "lf"),
    ("syn:home:6", "Val Reyes", "1b"),
    ("syn:home:7", "Nico Stone", "2b"),
    ("syn:home:8", "Emery Tran", "cf"),
    ("syn:home:9", "Wren Uhl", "dh"),
    # Rows past the ninth ARE the boxscore pitching order, in appearance
    # order -- which is what the blank-incoming-pitcher rule reads.
    ("syn:home:10", "Ira Vance", "p"),
    ("syn:home:11", "Kai Wolfe", "p"),
]


#: The synthetic boxscore Pitchers table, in appearance order -- the evidence
#: the blank-incoming-pitcher rule reads. Deliberately NOT the same thing as
#: "roster rows past the ninth batter": a reliever who never batted has no
#: batting row, and on these lines he is very often the man being inferred.
BOX_PITCHING_ORDER = {
    "syn:team:home": ["syn:home:10", "syn:home:11"],
    "syn:team:away": [],
}


def _side(team_id, name, rows):
    return identity.TeamIdentity(
        team_id=team_id,
        name=name,
        players={
            pid: identity.PlayerEntry(
                player_id=pid,
                name=nm,
                last_name=nm.split()[-1],
                team_id=team_id,
                positions=[pos],
            )
            for pid, nm, pos in rows
        },
    )


def _player_table():
    return identity.PlayerTable(
        home=_side("syn:team:home", "Synthetic Home", _HOME),
        away=_side("syn:team:away", "Synthetic Away", _AWAY),
    )


def _run(texts):
    lines = [
        parse_mod.PbpLine(
            inning=1, half="top", line_index=i, text=t, is_strong=False
        )
        for i, t in enumerate(texts)
    ]
    events, unparsed, _subs, inferred = parse_mod.build_events(
        lines, _player_table(), box_pitching_order=BOX_PITCHING_ORDER
    )
    return events, unparsed, inferred


@pytest.mark.parametrize("case_name", sorted(load_fixture(FIXTURE)["cases"]))
def test_promoted_case_reproduces_the_fixture(case_name):
    case = load_fixture(FIXTURE)["cases"][case_name]
    events, unparsed, inferred = _run(case["synthetic_input"]["pbp_lines"])
    expected = case["build_events_output"]
    assert unparsed == expected["unparsed"]
    assert events == expected["events"]
    assert inferred == expected["inferred"]


@pytest.mark.parametrize("case_name", sorted(load_fixture(FIXTURE)["cases"]))
def test_promoted_case_no_longer_lands_in_unparsed(case_name):
    case = load_fixture(FIXTURE)["cases"][case_name]
    _events, unparsed, _inferred = _run(case["synthetic_input"]["pbp_lines"])
    assert unparsed == []


# --------------------------------------------------------------------------
# Posture 1: a swallowed name capture fails LOUD.
# --------------------------------------------------------------------------


def test_name_capture_containing_a_verb_is_rejected():
    assert grammar._name_capture_is_swallowed("D. Covino advanced to second")
    assert grammar._name_capture_is_swallowed("C. Bess singled,")
    assert not grammar._name_capture_is_swallowed("D. Covino")


def test_name_capture_containing_a_comma_is_rejected_but_a_suffix_is_not():
    assert grammar._name_capture_is_swallowed("Smith, out at second")
    # A genuine comma-bearing surname suffix is NOT evidence of a swallow.
    assert not grammar._name_capture_is_swallowed("Herron, Jr")


def test_unmatched_lead_clause_is_a_grammar_miss_not_a_bogus_name():
    """The failure mode issue #40 exists to remove: rather than a plausible
    parse whose only symptom is an unresolvable name, an unhandled compound
    is an honest miss."""
    result = grammar.parse_clause_group(
        "D. Covino advanced to second on a spaceship, advanced to third."
    )
    assert isinstance(result, grammar.GrammarMiss)
    assert "spaceship" in result.raw


def test_a_covered_compound_chains_instead_of_swallowing_the_lead():
    result = grammar.parse_clause_group(
        "D. Covino advanced to second on an error by p, advanced to third."
    )
    assert not isinstance(result, grammar.GrammarMiss)
    assert [(r.name_token, r.destination) for r in result.runners] == [
        ("D. Covino", "second"),
        ("D. Covino", "third"),
    ]


# --------------------------------------------------------------------------
# Posture 2: a fielder's-choice out is recorded, never swept into modifiers.
# --------------------------------------------------------------------------


def test_fielders_choice_out_clause_is_structural_not_a_modifier():
    result = grammar.parse_clause_group(
        "E. Yake reached on a fielder's choice, out at second ss to 2b."
    )
    assert result.primary.forced_out_at == "second"
    assert result.primary.modifiers == []


def test_fielders_choice_out_is_refused_when_the_force_base_is_empty():
    """Nobody on first, so an out at second is not a force and the line names
    no runner -- assert nothing, and say so."""
    _events, unparsed, inferred = _run(
        ["Pat Smith reached on a fielder's choice, out at second ss to 2b."]
    )
    assert inferred == []
    assert len(unparsed) == 1
    assert "names no runner" in unparsed[0]["reason"]


# --------------------------------------------------------------------------
# Posture 3: every inference is disclosed, and refused when not forced.
# --------------------------------------------------------------------------


def test_every_inference_is_disclosed_with_its_rule_and_evidence():
    _events, _unparsed, inferred = _run(
        [
            "Robin Vega tripled to right field.",
            "Pat Smith singled to center field, RBI; Robin Vega Robin Vega.",
        ]
    )
    assert [i["rule"] for i in inferred] == ["doubled_name_scored"]
    entry = inferred[0]
    assert entry["raw"].endswith("Robin Vega Robin Vega.")
    assert entry["location"] == {"inning": 1, "half": "top", "line_index": 1}
    # The disclosure carries the evidence, not just the conclusion.
    assert "54/54" in entry["asserted"]


def test_blank_incoming_pitcher_is_refused_when_no_successor_exists():
    """The outgoing pitcher is LAST in the boxscore order, so the order does
    not force a reliever -- 7 of the corpus's 58 such lines are exactly this
    and are left unparsed rather than guessed."""
    _events, unparsed, inferred = _run(["/  for Kai Wolfe."])
    assert inferred == []
    assert len(unparsed) == 1
    assert "does not force one" in unparsed[0]["reason"]


def test_a_clean_line_infers_nothing():
    _events, unparsed, inferred = _run(["Pat Smith singled to left field."])
    assert unparsed == []
    assert inferred == []
