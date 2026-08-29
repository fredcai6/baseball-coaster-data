"""The `boxscore_only` record shape (schema 1.12.0).

Twice in three seasons the league published a completed final whose
play-by-play it never wrote: `20250520_iiqj` and `20250521_jyjy`. Both carry
a full linescore and a full batting box -- 68 and 60 at-bats -- and both were
refused for the life of the corpus, because a page with no inning panes
arrived through the same door as an ordinary pre-game page.

Admitting them is a real risk and these tests are drawn against it. The
danger is not that the shape exists; it is that the shape becomes a hole a
page with nothing in it can fall through, which is exactly how
`20260809_3555` scored as fully validated for three seasons. So the shape
has a floor at every layer that can enforce one: the parser refuses a
paneless page with no at-bats, the schema refuses a `boxscore_only` record
carrying events, and `replay.check_has_content` refuses a record whose box
totals zero at-bats before `check_box_linescore` is allowed to say anything
about it.
"""
from __future__ import annotations

import copy
import json

import pytest
from _support import SAMPLES_DIR, load_schema
from jsonschema import Draft202012Validator

from bc_pipeline import parse, replay

SAMPLE = "boxscore_20250520_iiqj_boxscore_only.html"
SOURCE_URL = "https://www.pioneerleague.com/sports/bsb/2025/boxscores/20250520_iiqj.xml"


def _html(name: str) -> str:
    return (SAMPLES_DIR / name).read_text(encoding="utf-8")


def _parsed() -> dict:
    return parse.parse_game(
        _html(SAMPLE),
        source_url=SOURCE_URL,
        fetched_at="1970-01-01T00:00:00Z",
        parsed_at="1970-01-01T00:00:00Z",
    )


# --- the parser -------------------------------------------------------------


def test_a_paneless_page_with_a_real_boxscore_parses_as_boxscore_only():
    game = _parsed()
    assert game["record_shape"] == "boxscore_only"
    assert game["status"] == "final"
    assert game["events"] == []
    assert game["unparsed"] == []
    at_bats = sum(
        row["AB"] for rows in game["box"]["batting"].values() for row in rows
    )
    assert at_bats == 68, "the whole reason this game is worth committing"
    assert game["linescore"]["totals"]["away"]["R"] == 9


def test_a_boxscore_only_record_carries_no_reconstructed_lineup():
    """`_build_lineups` reads the substitution sequence to order the slots
    and there is none. Position-duplicate lineup reconstruction was scored on
    this corpus at 54.8%, so the box row order is not a substitute: a
    half-right batting order would be worse than an empty one, because
    nothing downstream could tell which it had."""
    game = _parsed()
    assert len(game["lineups"]) == 2
    for side in game["lineups"].values():
        assert side == {"batting_order": [], "substitutions": []}


def test_a_pre_game_page_is_still_refused():
    """The shape must not become a door for the pages it sits next to. A
    pre-game page has no `linescore` element at all, which is what the
    parser's refusal turns on."""
    with pytest.raises(parse.NonFinalPageError, match="no PBP panes"):
        parse.parse_game(
            _html("boxscore_20260710_today.html"),
            source_url="https://x/sports/bsb/2026/boxscores/20260710_todo.xml",
            fetched_at="1970-01-01T00:00:00Z",
        )


def test_a_paneless_page_whose_box_records_no_at_bat_is_refused(monkeypatch):
    """The floor. `20260809_3555` is the paned version of this page -- twenty
    box rows totalling zero at-bats -- and it passed every oracle for three
    seasons because there was nothing to disagree with. The new branch gets
    the same guard rather than inheriting the same hole."""
    real = parse._parse_box_batting

    def empty_box(root, player_table):
        return {
            team: [dict(row, AB=0) for row in rows]
            for team, rows in real(root, player_table).items()
        }

    monkeypatch.setattr(parse, "_parse_box_batting", empty_box)
    with pytest.raises(parse.NonFinalPageError, match="records no at-bat"):
        _parsed()


# --- the schema -------------------------------------------------------------


def test_the_parsed_record_validates_against_the_game_schema():
    game = replay.replay_game(_parsed(), _html(SAMPLE))
    errors = list(Draft202012Validator(load_schema()).iter_errors(game))
    assert errors == [], [(list(e.path), e.message) for e in errors]


def test_the_schema_refuses_a_boxscore_only_record_that_carries_events():
    """A record shape only the parser believes in is one a hand-edit can
    break silently."""
    game = replay.replay_game(_parsed(), _html(SAMPLE))
    smuggled = copy.deepcopy(game)
    smuggled["events"] = [{"seq": 1, "kind": "inning_summary"}]
    assert list(Draft202012Validator(load_schema()).iter_errors(smuggled))


def test_the_schema_refuses_a_play_by_play_record_with_nothing_in_it():
    """The converse floor, and the one that matters more: this is
    `20260809_3555`'s shape."""
    game = replay.replay_game(_parsed(), _html(SAMPLE))
    empty = copy.deepcopy(game)
    empty["record_shape"] = "play_by_play"
    assert list(Draft202012Validator(load_schema()).iter_errors(empty))


# --- the replayer -----------------------------------------------------------


def test_a_boxscore_only_record_is_validated_by_the_checks_that_can_read_it():
    """Not excused from validation -- validated by a smaller set. The five
    event oracles have no play-by-play to read, so running them would be
    five vacuous passes; `content` and `box_linescore` read no events and
    run on every game, including this one."""
    game = replay.replay_game(_parsed(), _html(SAMPLE))
    assert game["meta"]["parse"]["replayable"] is True
    assert game["meta"]["parse"]["warnings"] == []


def test_the_box_and_the_linescore_are_what_validate_it():
    """And they are not the same table read twice: the batting box sums to
    9 runs and 14 hits, and the linescore -- re-derived from the HTML by the
    replayer's own extractor, not taken from the parse -- says 9 and 14 in
    its totals and 9 across its inning cells."""
    game = _parsed()
    oracle = replay.extract_oracle(_html(SAMPLE), game)
    assert replay.check_box_linescore(game, oracle).ok
    away = game["teams"]["away"]["team_id"]
    assert sum(r["R"] for r in game["box"]["batting"][away]) == 9
    assert oracle["linescore"]["totals"]["away"]["R"] == 9
    assert sum(c for c in oracle["linescore"]["innings"]["away"] if c is not None) == 9


def test_a_corrupted_box_is_caught_on_a_boxscore_only_record():
    """The check has to be able to FAIL here, or admitting the shape means
    admitting two records nothing can refuse."""
    game = _parsed()
    oracle = replay.extract_oracle(_html(SAMPLE), game)
    corrupted = copy.deepcopy(game)
    away = corrupted["teams"]["away"]["team_id"]
    corrupted["box"]["batting"][away][0]["R"] += 1
    result = replay.check_box_linescore(corrupted, oracle)
    assert not result.ok
    assert "away box R 10 != linescore total R 9" in result.warnings[0]


def test_an_empty_box_is_refused_before_box_linescore_can_pass_it():
    """0 == 0 is a pass, so `content` has to get there first. This is the
    3555 lesson applied to the new shape: the check that cannot pass
    vacuously sits under the one that can."""
    game = _parsed()
    oracle = replay.extract_oracle(_html(SAMPLE), game)
    emptied = copy.deepcopy(game)
    emptied["box"]["batting"] = {
        team: [dict(row, AB=0, R=0, H=0) for row in rows]
        for team, rows in emptied["box"]["batting"].items()
    }
    zero_linescore = copy.deepcopy(oracle)
    for side in ("away", "home"):
        zero_linescore["linescore"]["totals"][side] = {"R": 0, "H": 0, "E": 0}
        zero_linescore["linescore"]["innings"][side] = [0]
    assert replay.check_box_linescore(emptied, zero_linescore).ok, (
        "box_linescore passes vacuously on an empty game -- which is why it "
        "is not allowed to be the floor"
    )
    assert not replay.check_has_content(emptied, zero_linescore).ok


def test_the_ordinary_shape_still_runs_every_check():
    """A `play_by_play` record must not accidentally inherit the smaller
    set: that would silently stop validating 1,485 games."""
    html = _html("boxscore_20260709_final.html")
    game = parse.parse_game(
        html,
        source_url="https://longbeachcoast.com/sports/bsb/2026/boxscores/20260709_h94w.xml",
        fetched_at="1970-01-01T00:00:00Z",
    )
    assert game["record_shape"] == "play_by_play"
    broken = replay.replay_game(game, html)
    assert broken["meta"]["parse"]["replayable"] is True
    # Corrupt something only an EVENT oracle reads, and confirm it is caught.
    corrupted = copy.deepcopy(game)
    for event in corrupted["events"]:
        if event["kind"] == "plate_appearance":
            event["kind"] = "runner_event"
            break
    assert replay.replay_game(corrupted, html)["meta"]["parse"]["replayable"] is False


def test_json_round_trip_is_stable():
    game = replay.replay_game(_parsed(), _html(SAMPLE))
    assert json.loads(json.dumps(game)) == game
