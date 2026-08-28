"""Grammar coverage for the dropped third strike, the shared error-spelling
fragment, and the three source defects handled alongside them.

Protected intent, in one sentence per group:

* A dropped third strike records the outs the PLAY made -- one when the
  batter is retired, none when he reaches -- never the outs the primary verb
  implies. Getting this wrong is invisible in the parse and shows up as an
  `outs_per_half` failure, so it is asserted on the event, not on the clause.
* An error is one fragment with several spellings. Every row that accepts an
  error accepts all of them; this file fails if a spelling is ever added to
  one row and not its siblings -- the divergence that produced most of the
  shapes below.
* Tokens that annotate a play without changing base-out state assert nothing,
  and in particular ", caught stealing" trailing a movement clause does not
  retire the runner: StatCrew writes that out on its own following line, and
  counting both put four outs in an inning.
"""
from __future__ import annotations

from bc_pipeline import grammar, identity, parse as parse_mod
from bc_pipeline.grammar import GrammarMiss, parse_clause_group


def _group(line: str):
    cg = parse_clause_group(line)
    assert not isinstance(cg, GrammarMiss), f"{line!r} -> {getattr(cg, 'reason', '')}"
    return cg


def _table():
    away = identity.TeamIdentity(team_id="syn:team:away", name="Away")
    home = identity.TeamIdentity(team_id="syn:team:home", name="Home")
    for pid, nm, last in (
        ("a1", "Alex Rios", "Rios"),
        ("a2", "Blake Wilhoite", "Wilhoite"),
        ("a3", "Tom Smith", "Smith"),
        ("a4", "Chris Hanson", "Hanson"),
    ):
        away.players[pid] = identity.PlayerEntry(
            player_id=pid, name=nm, last_name=last, team_id="syn:team:away"
        )
    home.players["h1"] = identity.PlayerEntry(
        player_id="h1", name="Pat Kane", last_name="Kane", team_id="syn:team:home"
    )
    return identity.PlayerTable(home=home, away=away)


def _event(text: str):
    line = parse_mod.PbpLine(
        inning=1, half="top", line_index=0, text=text, is_strong=False
    )
    events, unparsed, _subs, _inf = parse_mod.build_events([line], _table())
    assert unparsed == [], unparsed
    assert len(events) == 1
    return events[0]


# --- dropped third strike: the two dispositions --------------------------


def test_dropped_third_strike_retired_records_exactly_one_out():
    # Strike three gets away and the catcher throws him out. The pitcher is
    # credited a strikeout AND the batter is retired, but that is ONE out,
    # not two: the clause merges into the batter's own record rather than
    # appending a second retired runner.
    ev = _event("A. Rios struck out swinging, grounded out to c unassisted.")
    assert ev["outcome"]["type"] == "strikeout_swinging"
    assert ev["outcome"]["outs_recorded"] == 1
    assert [(r["player_id"], r["out"], r["to"]) for r in ev["runners"]] == [
        ("a1", True, -1)
    ]


def test_dropped_third_strike_safe_on_error_records_no_out():
    # The mirror case: the pitcher still gets his strikeout, but the batter is
    # standing on first and the play retired nobody. `outs_recorded` must
    # follow the PLAY, not the verb.
    ev = _event("A. Rios struck out swinging, reached first on an error by c.")
    assert ev["outcome"]["type"] == "strikeout_swinging"
    assert ev["outcome"]["outs_recorded"] == 0
    assert [(r["player_id"], r["out"], r["to"]) for r in ev["runners"]] == [
        ("a1", False, 1)
    ]


def test_dropped_third_strike_safe_on_error_carries_chained_advance():
    ev = _event(
        "Blake Wilhoite struck out swinging, reached first on a throwing "
        "error by c, advanced to third (3-2 BBBKFS)."
    )
    assert ev["outcome"]["outs_recorded"] == 0
    assert [(r["from"], r["to"], r["out"]) for r in ev["runners"]] == [(0, 3, False)]


# --- one error fragment, every spelling ----------------------------------


def test_every_error_spelling_is_accepted_on_the_same_row():
    # "a muffed throw by F" is the spelling that was missing. If a future
    # edit adds a spelling to one row only, one of these will fail.
    for phrase in (
        "an error by c",
        "a throwing error by c",
        "a fielding error by c",
        "a muffed throw by c",
    ):
        cg = _group(f"A. Rios reached first on {phrase}.")
        assert cg.primary.outcome_type == "reached_on_error"
        assert cg.primary.fielders == ["c"]


def test_assist_inside_an_error_phrase_is_accepted_mid_clause():
    # ", assist by 3b" is a fielding credit nothing in the schema records; it
    # must not stop the clause that carries it from parsing.
    cg = _group(
        "T. Smith reached first on a muffed throw by 1b, assist by 3b, "
        "advanced to second."
    )
    assert cg.primary.outcome_type == "reached_on_error"


# --- annotations that assert nothing --------------------------------------


def test_caught_stealing_suffix_on_a_movement_clause_records_no_out():
    # The out belongs to the FOLLOWING line ("C. Hanson out at third c to
    # ss"). Retiring him here as well produced four outs in an inning.
    ev = _event(
        "C. Hanson advanced to second on an error by 2b, assist by p, "
        "caught stealing."
    )
    # A runner-only line: no batter, so no `outcome` block -- the assertion
    # is on the runner record the replayer folds.
    assert ev["kind"] == "runner_event"
    assert [(r["to"], r["out"]) for r in ev["runners"]] == [(2, False)]


def test_interference_between_scored_and_its_unearned_keeps_the_run_unearned():
    # An end-only strip left "interference" separating "scored" from its own
    # "unearned", the bare `scored` row won, and the run was recorded EARNED
    # on a line that says unearned.
    cg = _group("A. Rios advanced to third, scored, interference, unearned.")
    scored = [r for r in cg.runners if r.scored]
    assert len(scored) == 1
    assert scored[0].unearned is True


def test_standalone_dropped_foul_ball_line_still_asserts_nothing():
    # The spliced-fragment strip must not touch the standalone spelling.
    cg = _group("A. Rios Dropped foul ball, E3.")
    assert list(cg.runners) == []


def test_spliced_dropped_foul_ball_lets_the_real_outcome_through():
    # Both no-separator spellings StatCrew emits.
    assert _group("A. Rios walkedDropped foul ball, E3 (3-2 BKBBFFB).").primary.outcome_type == "walk"
    assert _group(
        "A. Rios Dropped foul ball, E3struck out swinging. (2 out)"
    ).primary.outcome_type == "strikeout_swinging"
    assert _group(
        "A. Rios Dropped foul ball, E2, homered to right field (2-2 BKFFB)."
    ).primary.outcome_type == "home_run"


def test_repeated_subject_with_no_run_in_the_clause_is_read_as_scored():
    # "T. Smith advanced to third on an error by 3b, T. Smith, unearned" --
    # the template re-emits the subject where the verb belongs. An earlier
    # pass dropped it as noise; that was wrong, and wrong in the falsifiable
    # direction. All four affected games came up short by exactly one run in
    # exactly that inning against the linescore oracle. Same defect, same
    # meaning as the standalone "M. Moralez M. Moralez." shape 1.10.0
    # measured at 54/54.
    cg = _group(
        "A. Rios singled to left field; T. Smith advanced to third on an "
        "error by 3b, T. Smith, unearned."
    )
    moved = [r for r in cg.runners if r.name_token == "T. Smith"]
    assert [(r.destination, r.scored) for r in moved] == [
        ("third", False),
        ("home", True),
    ]
    # It is an inference, so it must be DISCLOSED, never pass as something
    # the line said in words.
    assert moved[-1].inferred
    assert moved[-1].unearned is True


def test_repeated_subject_beside_an_explicit_scored_is_only_redundant():
    # The other half of the conditional. Here the clause already records the
    # run, so the re-emitted name adds nothing -- dropping it is correct, and
    # nothing is inferred because nothing was asserted beyond the line.
    cg = _group(
        "A. Rios singled to left field; T. Smith advanced to third, scored "
        "on an error by 3b, T. Smith, unearned."
    )
    moved = [r for r in cg.runners if r.name_token == "T. Smith"]
    assert sum(1 for r in moved if r.scored) == 1
    assert not any(r.inferred for r in moved)


def test_unearned_stays_attached_to_its_run_on_the_primary_chain():
    # `scored, unearned` is a PAIR. The primary chain lifts modifier tokens
    # out of the movement chain, and lifting "unearned" away from "scored"
    # left the bare `scored` row to win -- recording as EARNED a run the line
    # says is unearned. `earned` is a real emitted field, so that is a wrong
    # fact, not a lost one.
    cg = _group(
        "A. Rios reached first on an error by 1b, advanced to second on an "
        "error by lf, A. Rios, unearned, RBI."
    )
    scored = [r for r in cg.runners if r.scored]
    assert len(scored) == 1
    assert scored[0].unearned is True
    assert "RBI" in cg.primary.modifiers


def test_reached_on_error_records_movement_as_runners_not_modifier_strings():
    # The row's tail was the only unrestricted `.*` in the table and it ate
    # real content: 248 events across 228 games stored movement clauses as
    # MODIFIER STRINGS, so those runners never moved in runners[].
    cg = _group("A. Rios reached first on an error by cf to right center, advanced to second.")
    assert cg.primary.location == "right center"
    assert [(r.destination, r.scored) for r in cg.runners] == [("second", False)]
    assert not any("advanced" in m for m in cg.primary.modifiers)
