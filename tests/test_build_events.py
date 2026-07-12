"""Tests for bc_pipeline.parse.build_events: the PURE ClauseGroups+panes+
player_table -> events[] fold.

PURE means: no HTML/Node objects appear anywhere in this module. Player
identity is built directly as `identity.PlayerTable`/`TeamIdentity`/
`PlayerEntry` dataclasses (no `html_struct.parse_html` call), and PBP lines
are plain `parse.PbpLine` records. build_events itself calls
`grammar.parse_clause_group` on each line's text, so these narrative
strings are real StatCrew-shaped sentences (not mocked ClauseGroups) --
this exercises the grammar+build_events seam together while staying free
of any DOM traversal.

Protected intent under test: runner `from`/`to` are ASSERTED PRIMITIVES
tracked forward as an in-order fold over base occupancy -- including the
resistant case where two runners on the same play swap/shift bases (one
clause's target base is another runner's source base), which a naive
"look at the live map after each clause" fold gets wrong (see the
IMPLEMENTER_RESULT's bug note).
"""
from __future__ import annotations

from bc_pipeline import identity
from bc_pipeline.parse import PbpLine, build_events

HOME_ID = "hometeam1234abcd"
AWAY_ID = "syn:team:away"


def _entry(pid, name, team_id, positions=()):
    return identity.PlayerEntry(
        player_id=pid,
        name=name,
        last_name=name.split()[-1],
        team_id=team_id,
        positions=list(positions),
    )


def _make_table():
    home = identity.TeamIdentity(
        team_id=HOME_ID,
        name="Home",
        players={"p1": _entry("p1", "Home Pitcher", HOME_ID, ["p"])},
    )
    away = identity.TeamIdentity(
        team_id=AWAY_ID,
        name="Away",
        players={
            "a1": _entry("a1", "Alpha One", AWAY_ID, ["ss"]),
            "a2": _entry("a2", "Beta Two", AWAY_ID, ["2b"]),
            "a3": _entry("a3", "Gamma Three", AWAY_ID, ["3b"]),
        },
    )
    return identity.PlayerTable(home=home, away=away)


def _line(inning, half, idx, text, strong=False):
    return PbpLine(inning=inning, half=half, line_index=idx, text=text, is_strong=strong)


# --- batter-as-runner synthesis ---------------------------------------------


def test_batter_own_movement_is_synthesized_as_a_from_zero_runner():
    table = _make_table()
    lines = [_line(1, "top", 0, "Alpha One singled to left field (1-0 B).")]
    events, unparsed, _subs = build_events(lines, table)
    assert unparsed == []
    assert len(events) == 1
    runners = events[0]["runners"]
    assert len(runners) == 1
    assert runners[0] == {
        "player_id": "a1",
        "from": 0,
        "to": 1,
        "cause": "batted_ball",
        "out": False,
        "scored": False,
    }


def test_strikeout_batter_runner_is_from_zero_to_negative_one_out():
    table = _make_table()
    lines = [_line(1, "top", 0, "Beta Two struck out swinging (0-2 KKK).")]
    events, unparsed, _subs = build_events(lines, table)
    assert unparsed == []
    r = events[0]["runners"][0]
    assert r == {
        "player_id": "a2",
        "from": 0,
        "to": -1,
        "cause": "putout",
        "out": True,
        "scored": False,
    }
    assert events[0]["outcome"]["outs_recorded"] == 1


# --- base occupancy folds forward, updating from event to event ------------


def test_runner_from_tracks_the_base_they_reached_in_a_prior_event():
    table = _make_table()
    lines = [
        _line(1, "top", 0, "Alpha One singled to left field (1-0 B)."),
        _line(
            1,
            "top",
            1,
            "Beta Two singled to right field (1-0 B); Alpha One advanced to second.",
        ),
    ]
    events, unparsed, _subs = build_events(lines, table)
    assert unparsed == []
    second_event_runners = {r["player_id"]: r for r in events[1]["runners"]}
    alpha = second_event_runners["a1"]
    assert alpha["from"] == 1  # Alpha reached 1st in the PRIOR event
    assert alpha["to"] == 2


def test_two_runners_shifting_bases_on_the_same_line_both_read_the_pre_line_state():
    # Regression: Alpha is on 2nd, Beta is on 1st. "Alpha stole third; Beta
    # stole second." -- Beta's destination (2nd) is exactly Alpha's ORIGIN.
    # A fold that mutates the live occupancy map clause-by-clause and reads
    # it back for the NEXT clause would clobber Alpha's base-2 entry with
    # Beta's own write before Beta's own "from" is read, corrupting Beta's
    # asserted `from` to something other than 1. Both must read the
    # occupancy as it stood BEFORE this line's own movements.
    table = _make_table()
    lines = [
        _line(1, "top", 0, "Alpha One singled to left field (1-0 B)."),
        _line(
            1,
            "top",
            1,
            "Beta Two singled to right field (1-0 B); Alpha One advanced to second.",
        ),
        _line(1, "top", 2, "Alpha One stole third; Beta Two stole second."),
    ]
    events, unparsed, _subs = build_events(lines, table)
    assert unparsed == []
    runners = {r["player_id"]: r for r in events[2]["runners"]}
    assert runners["a1"] == {
        "player_id": "a1",
        "from": 2,
        "to": 3,
        "cause": "stolen_base",
        "out": False,
        "scored": False,
    }
    assert runners["a2"] == {
        "player_id": "a2",
        "from": 1,
        "to": 2,
        "cause": "stolen_base",
        "out": False,
        "scored": False,
    }


def _make_table_with_home_batter():
    home = identity.TeamIdentity(
        team_id=HOME_ID,
        name="Home",
        players={"h1": _entry("h1", "Delta Four", HOME_ID, ["c"])},
    )
    away = identity.TeamIdentity(
        team_id=AWAY_ID,
        name="Away",
        players={"a1": _entry("a1", "Alpha One", AWAY_ID, ["ss"])},
    )
    return identity.PlayerTable(home=home, away=away)


def test_base_occupancy_resets_at_the_start_of_each_half():
    # Alpha (away) reaches 1st in the top half and is left on base (the
    # inning ends via the summary line, no third out modeled here -- this
    # test only cares about occupancy carry-over, not out-counting). Delta
    # (home) leads off the bottom half with a standalone pickoff clause: if
    # base occupancy leaked across the half boundary, Delta could
    # spuriously resolve against Alpha's still-tracked base-1 entry. Since
    # base_occ is reset, Delta has no tracked base at all, so `from` falls
    # back to 0 (pickoff, no named destination -> to == from == 0) rather
    # than inheriting Alpha's stale base-1 occupancy.
    table = _make_table_with_home_batter()
    lines = [
        _line(1, "top", 0, "Alpha One singled to left field (1-0 B)."),
        _line(1, "top", 1, "Inning Summary: 0 Runs, 1 Hits, 0 Errors, 1 LOB"),
        _line(1, "bottom", 0, "Delta Four Failed pickoff attempt."),
    ]
    events, unparsed, _subs = build_events(lines, table)
    assert unparsed == []
    pickoff = events[2]
    assert pickoff["kind"] == "runner_event"
    r = pickoff["runners"][0]
    assert r["player_id"] == "h1"
    assert r["from"] == 0
    assert r["to"] == 0


# --- RBI/earned per-run assertions -------------------------------------------


def test_rbi_and_earned_are_asserted_on_the_scoring_runner_only():
    table = _make_table()
    lines = [
        _line(1, "top", 0, "Alpha One singled to left field (1-0 B)."),
        _line(
            1,
            "top",
            1,
            "Beta Two singled to right field (1-0 B); Alpha One advanced to third.",
        ),
        _line(
            1,
            "top",
            2,
            "Gamma Three singled to center field, RBI (1-0 B); Beta Two advanced to second; Alpha One scored.",
        ),
    ]
    events, unparsed, _subs = build_events(lines, table)
    assert unparsed == []
    runners = {r["player_id"]: r for r in events[2]["runners"]}
    scorer = runners["a1"]
    assert scorer["scored"] is True
    assert scorer["earned"] is True
    assert scorer["rbi"] is True
    non_scorer = runners["a2"]
    assert non_scorer["scored"] is False
    assert "earned" not in non_scorer
    assert "rbi" not in non_scorer


# --- same-runner multi-clause chaining WITHIN one event ---------------------


def test_same_runner_two_clauses_one_event_chains_from_previous_to():
    # Regression for the seq50/seq51 Mata bug (found via g6 replay): a single
    # event names ONE runner twice ("advanced to second on a passed ball,
    # advanced to third"). g4 grammar correctly emits TWO RunnerMovements
    # (to second cause passed_ball, then to third cause advance). build_events
    # must CHAIN them: the second clause's `from` is the first clause's `to`
    # (2), NOT the runner's event-start base (1) -- and the runner's tracked
    # final base must be 3 so the NEXT event sees him on third (where he then
    # scores from).
    table = _make_table()
    lines = [
        # Alpha leads off and reaches 1st.
        _line(1, "top", 0, "Alpha One singled to left field (1-0 B)."),
        # seq50 analogue: batter walks; Alpha (on 1st) takes two bases on one
        # play across two chained clauses.
        _line(
            1,
            "top",
            1,
            "Beta Two walked (3-2 BBBFFB); Alpha One advanced to second on a passed ball, advanced to third.",
        ),
        # seq51 analogue: next batter doubles; Alpha scores FROM THIRD.
        _line(
            1,
            "top",
            2,
            "Gamma Three doubled to center field, RBI (1-0 B); Alpha One scored.",
        ),
    ]
    events, unparsed, _subs = build_events(lines, table)
    assert unparsed == []

    # seq50 analogue: Alpha's two same-event clauses collapse to ONE net-path
    # record 1 -> 3 (the collapsed form the commander blessed, and the ONLY
    # form g6's replayer accepts: it validates every emitted `from` against the
    # base occupancy frozen at event START, so a second entry with from=2 -- a
    # base not occupied before the event -- reads as an illegal transition).
    alpha_clauses = [r for r in events[1]["runners"] if r["player_id"] == "a1"]
    assert len(alpha_clauses) == 1
    assert alpha_clauses[0] == {
        "player_id": "a1",
        "from": 1,  # the runner's true event-start base
        "to": 3,  # his final base this event, NOT the intermediate 2
        "cause": "passed_ball",  # the initiating mechanism (first clause's cause)
        "out": False,
        "scored": False,
    }

    # seq51 analogue: Alpha scores from THIRD (his correct tracked final base) --
    # the whole point of the fix: before it, he'd have scored from the stale
    # penultimate base (2).
    alpha_score = [r for r in events[2]["runners"] if r["player_id"] == "a1"]
    assert len(alpha_score) == 1
    assert alpha_score[0]["from"] == 3
    assert alpha_score[0]["to"] == 4
    assert alpha_score[0]["scored"] is True


def test_runner_from_is_always_a_base_the_runner_currently_occupies():
    # Strict transition invariant over the whole chained sequence: replay the
    # asserted runner primitives forward and confirm every runner.from equals
    # the base that runner occupied immediately before the clause fired (0 for
    # a batter out of the box; a real base 1-3 for someone already on; -1 is
    # never a valid `from` here). This is exactly the illegal-transition check
    # g6 performs independently.
    table = _make_table()
    lines = [
        _line(1, "top", 0, "Alpha One singled to left field (1-0 B)."),
        _line(
            1,
            "top",
            1,
            "Beta Two walked (3-2 BBBFFB); Alpha One advanced to second on a passed ball, advanced to third.",
        ),
        _line(
            1,
            "top",
            2,
            "Gamma Three doubled to center field, RBI (1-0 B); Alpha One scored.",
        ),
    ]
    events, unparsed, _subs = build_events(lines, table)
    assert unparsed == []

    occ: dict = {}  # base -> pid, folded from the asserted primitives alone
    half = None
    for e in events:
        if (e["inning"], e["half"]) != half:
            occ = {}
            half = (e["inning"], e["half"])
        runners = e.get("runners", [])
        # Two-phase per event, mirroring g6's own replayer: FIRST validate
        # every `from` against the occupancy frozen at the START of the event
        # (a batter forcing the runner ahead means one runner's destination is
        # briefly another's origin mid-apply -- checking against the frozen
        # snapshot, not a partially-mutated map, is the correct model). THEN
        # apply vacate+occupy.
        snapshot = dict(occ)
        for r in runners:
            frm, pid = r["from"], r["player_id"]
            if frm == 0:
                assert pid not in snapshot.values(), (
                    f"{pid} claims from=0 but was already on base {snapshot}"
                )
            else:
                assert snapshot.get(frm) == pid, (
                    f"{pid} claims from={frm} but base {frm} held "
                    f"{snapshot.get(frm)!r} at event start (occupancy {snapshot})"
                )
        for r in runners:
            frm, pid, to = r["from"], r["player_id"], r["to"]
            if frm in (1, 2, 3) and occ.get(frm) == pid:
                del occ[frm]
            if not r["out"] and to not in (-1, 4):
                occ[to] = pid


# --- unrecognized clauses route to unparsed[], never dropped/guessed -------


def test_grammar_miss_routes_to_unparsed_with_location():
    table = _make_table()
    lines = [_line(2, "bottom", 3, "This is not a recognized PBP sentence at all")]
    events, unparsed, _subs = build_events(lines, table)
    assert events == []
    assert len(unparsed) == 1
    assert unparsed[0]["raw"] == "This is not a recognized PBP sentence at all"
    assert unparsed[0]["location"] == {"inning": 2, "half": "bottom", "line_index": 3}
    assert unparsed[0]["reason"]


def test_ambiguous_batter_name_routes_to_unparsed_never_guessed():
    # Two "One"-surnamed players on the away side -> resolve() returns
    # unresolved -- never a guess.
    home = identity.TeamIdentity(team_id=HOME_ID, name="Home", players={})
    away = identity.TeamIdentity(
        team_id=AWAY_ID,
        name="Away",
        players={
            "a1": _entry("a1", "Alpha One", AWAY_ID),
            "a2": _entry("a2", "Zeta One", AWAY_ID),
        },
    )
    table = identity.PlayerTable(home=home, away=away)
    lines = [_line(1, "top", 0, "Alpha One singled to left field (1-0 B).")]
    events, unparsed, _subs = build_events(lines, table)
    assert events == []
    assert len(unparsed) == 1
    assert "did not resolve" in unparsed[0]["reason"]
