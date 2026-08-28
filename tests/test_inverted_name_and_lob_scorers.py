"""The inverted-name source defect, and LOB's post-plate-appearance scorers.

Protected intent:

* "H. Casey" is Casey Harford. StatCrew sometimes writes a name as
  "<surname initial>. <FIRST name>", so the token's second word is not a
  surname at all and every surname tier legitimately finds nothing. The
  inverted reading is a LAST resort and must stay one: it fires only after
  all four surname tiers come up empty, and only on a unique match. Measured
  null: zero tokens in the corpus that resolve normally would also resolve
  this way, so it can never overrule a correct answer.

* A runner who SCORES after the half's last plate appearance is not left on
  base. `check_lob` measures occupancy at that plate appearance, where he is
  still standing on third -- so he has to be subtracted. This does NOT touch
  the runner merely RETIRED after the last plate appearance, whom the box
  does count and whom the plate-appearance rule exists to keep.
"""
from __future__ import annotations

from bc_pipeline import identity, replay


def _team():
    t = identity.TeamIdentity(team_id="syn:team:away", name="Away")
    for pid, nm, last in (
        ("p1", "Casey Harford", "Harford"),
        ("p2", "Drew Kane", "Kane"),
    ):
        t.players[pid] = identity.PlayerEntry(
            player_id=pid, name=nm, last_name=last, team_id="syn:team:away"
        )
    return identity.PlayerTable(home=identity.TeamIdentity("syn:team:home", "Home"), away=t)


def test_inverted_name_token_resolves_when_no_surname_tier_matches():
    # "H. Casey": H -> Harford, Casey -> his FIRST name. No player has the
    # surname "Casey", so every surname tier fails first.
    assert _team().resolve("Casey", "away", "H. Casey") == ("p1", True)


def test_inverted_reading_never_overrules_a_normal_surname_match():
    # A normal "<initial>. <SURNAME>" token still resolves as a surname, and
    # the inverted tier is never reached.
    assert _team().resolve("Kane", "away", "D. Kane") == ("p2", True)


def test_inverted_reading_refuses_when_the_initial_disagrees():
    # "Q. Casey" -- no surname "Casey", and no first-name-Casey player whose
    # surname starts with Q. Never guess.
    assert _team().resolve("Casey", "away", "Q. Casey") == (None, False)


def test_inverted_reading_needs_a_two_token_initialled_form():
    # A bare surname carries no initial to invert; the tier must not fire.
    assert _team().resolve("Casey", "away", "Casey") == (None, False)


# --- LOB: post-plate-appearance scorers -----------------------------------


def _half(events):
    """One half-inning plus its summary, shaped as check_lob reads it."""
    return {
        "events": events,
        "meta": {"parse": {}},
    }


def _pa(seq, runners):
    return {
        "seq": seq, "inning": 1, "half": "top", "kind": "plate_appearance",
        "batter": {"player_id": "b%d" % seq, "name_raw": None, "resolved": True},
        "runners": runners, "narrative": "", "scoring_play": False,
    }


def _runner_event(seq, runners):
    return {
        "seq": seq, "inning": 1, "half": "top", "kind": "runner_event",
        "runners": runners, "narrative": "", "scoring_play": False,
    }


def _summary(seq, lob):
    return {
        "seq": seq, "inning": 1, "half": "top", "kind": "inning_summary",
        "summary": {"R": 0, "H": 0, "E": 0, "LOB": lob}, "narrative": "",
    }


def test_runner_who_scores_after_the_last_pa_is_not_left_on_base():
    # Batter reaches first, runner is on third at the end of the batting;
    # the runner then scores. LOB is 1 (the batter), not 2.
    game = _half([
        _pa(1, [{"player_id": "r1", "from": 0, "to": 3, "cause": "batted_ball",
                 "out": False, "scored": False}]),
        _pa(2, [{"player_id": "b2", "from": 0, "to": 1, "cause": "batted_ball",
                 "out": False, "scored": False}]),
        _runner_event(3, [{"player_id": "r1", "from": 3, "to": 4, "cause": "advance",
                           "out": False, "scored": True, "earned": True, "rbi": False}]),
        _summary(4, 1),
    ])
    assert replay.check_lob(game, {}).ok


def test_runner_merely_retired_after_the_last_pa_is_still_left_on_base():
    # The case the plate-appearance rule exists for -- unchanged. The box
    # counts a runner caught stealing between plate appearances.
    game = _half([
        _pa(1, [{"player_id": "r1", "from": 0, "to": 1, "cause": "batted_ball",
                 "out": False, "scored": False}]),
        _runner_event(2, [{"player_id": "r1", "from": 1, "to": -1,
                           "cause": "caught_stealing", "out": True, "scored": False}]),
        _summary(3, 1),
    ])
    assert replay.check_lob(game, {}).ok
