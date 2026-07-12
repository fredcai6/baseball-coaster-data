"""Tests for bc_pipeline.replay: the INDEPENDENT oracle/fold/5-check
replayer (issue #19 gate g6, spec D2 independence).

Protected intent: replay.py re-derives the linescore/box oracle from raw
HTML with its OWN code (no shared table-reader with parse.py -- enforced by
test_no_circular_import.py) and folds the asserted `runners[]` primitives
forward into `_derived`, entirely independently of the parser's own
numbers. Each of the 5 checks gets a passing case AND a dedicated synthetic
bad-sequence fixture proving it fails ONLY that check (isolation).
"""
from __future__ import annotations

import json
from pathlib import Path

from _support import FIXTURES_DIR, SAMPLES_DIR, load_fixture

from bc_pipeline import parse, replay

SYNTH_DIR = FIXTURES_DIR / "synthetic_bad_sequences"

SOURCE_URL = "https://longbeachcoast.com/sports/bsb/2026/boxscores/20260709_h94w.xml"
FETCHED_AT = "2026-07-11T00:00:00Z"


def _load_html(name: str) -> str:
    path = SAMPLES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _load_synth(name: str) -> dict:
    with (SYNTH_DIR / name).open("r", encoding="utf-8") as f:
        return json.load(f)


FINAL_HTML = _load_html("boxscore_20260709_final.html")


def _parse_real_game() -> dict:
    # Test-only use of parse.py to obtain the real input `game` dict fed
    # into replay_game -- replay.py ITSELF never imports parse (see
    # test_no_circular_import.py); this is just how the test constructs a
    # realistic parsed-game fixture to replay against.
    return parse.parse_game(FINAL_HTML, source_url=SOURCE_URL, fetched_at=FETCHED_AT)


# ---------------------------------------------------------------------------
# fold_base_out vs. the hand fixture's hand-computed _derived
# ---------------------------------------------------------------------------


def test_fold_base_out_reproduces_hand_fixture_derived():
    fixture = load_fixture("game_20260709_h94w_top1.json")
    foldable = [e for e in fixture["events"] if e["kind"] in ("plate_appearance", "runner_event")]
    derived = replay.fold_base_out(fixture["events"])
    # 9 total events in the Top-1 fixture, but the 9th (seq 8) is the
    # inning_summary -- only the 8 plate_appearance/runner_event entries are
    # foldable and carry a hand-computed `_derived`.
    assert len(derived) == len(foldable) == 8

    for i, (ev, d) in enumerate(zip(foldable, derived)):
        expected = ev["_derived"]
        for key in expected:
            assert d[key] == expected[key], (
                f"event seq={ev['seq']} key={key!r}: got {d[key]!r} expected {expected[key]!r}"
            )
        # And the fold must not invent extra keys the hand fixture doesn't
        # expect for this event (e.g. pa_number_of_batter on a runner_event).
        assert set(d.keys()) == set(expected.keys()), (
            f"event seq={ev['seq']}: fold produced {sorted(d.keys())}, "
            f"fixture has {sorted(expected.keys())}"
        )


# ---------------------------------------------------------------------------
# extract_oracle: independent re-derivation from raw HTML
# ---------------------------------------------------------------------------


def test_extract_oracle_linescore_matches_parser_linescore_on_real_sample():
    game = _parse_real_game()
    oracle = replay.extract_oracle(FINAL_HTML, game)
    assert oracle["linescore"] == game["linescore"]


def test_extract_oracle_box_batting_matches_parser_box_on_real_sample():
    game = _parse_real_game()
    oracle = replay.extract_oracle(FINAL_HTML, game)
    for team_id, lines in game["box"]["batting"].items():
        assert oracle["box"]["batting"][team_id] == lines


def test_extract_oracle_linescore_matches_hand_fixture_top1_oracle():
    game = _parse_real_game()
    oracle = replay.extract_oracle(FINAL_HTML, game)
    fixture = load_fixture("game_20260709_h94w_top1.json")
    assert oracle["linescore"] == fixture["linescore"]


# ---------------------------------------------------------------------------
# replay_game on the real full sample: all 5 checks pass, replayable=true
# ---------------------------------------------------------------------------


def test_replay_game_real_sample_is_replayable_true():
    game = _parse_real_game()
    replayed = replay.replay_game(game, FINAL_HTML)
    assert replayed["meta"]["parse"]["replayable"] is True, replayed["meta"]["parse"]["warnings"]
    assert replayed["meta"]["parse"]["warnings"] == game["meta"]["parse"]["warnings"]
    assert replayed["meta"]["derived_replayer_version"] == replay.REPLAYER_VERSION


def test_replay_game_stamps_derived_on_every_foldable_event_and_only_those():
    game = _parse_real_game()
    replayed = replay.replay_game(game, FINAL_HTML)
    for ev in replayed["events"]:
        if ev["kind"] in ("plate_appearance", "runner_event"):
            assert "_derived" in ev
        else:
            assert "_derived" not in ev


def test_replay_game_does_not_mutate_input():
    game = _parse_real_game()
    before = json.dumps(game, sort_keys=True)
    replay.replay_game(game, FINAL_HTML)
    after = json.dumps(game, sort_keys=True)
    assert before == after


def test_replay_game_each_individual_check_passes_on_real_sample():
    game = _parse_real_game()
    oracle = replay.extract_oracle(FINAL_HTML, game)
    # Fold first so _derived is populated the way the checks expect.
    derived_list = replay.fold_base_out(game["events"])
    di = 0
    for ev in game["events"]:
        if ev["kind"] in ("plate_appearance", "runner_event"):
            ev["_derived"] = derived_list[di]
            di += 1
    for name, fn in replay._CHECKS:
        result = fn(game, oracle)
        assert result.ok, f"{name} failed on real sample: {result.warnings}"


def test_replay_game_corrupted_copy_fails_the_right_check_and_flags_replayable_false():
    game = _parse_real_game()
    corrupted = json.loads(json.dumps(game))
    # Flip one event's outcome: turn Isaac Nunez's leadoff single (seq 0) into
    # a strikeout with no runner reaching base, without touching anything
    # else -- this desyncs the folded runs from the (untouched) real oracle
    # linescore/box, and it also desyncs the batter's own PA-implied outcome.
    target = corrupted["events"][0]
    assert target["outcome"]["type"] == "single"
    target["outcome"]["type"] = "strikeout_swinging"
    target["runners"] = [
        {
            "player_id": target["batter"]["player_id"],
            "from": 0,
            "to": -1,
            "cause": "putout",
            "out": True,
            "scored": False,
        }
    ]

    replayed = replay.replay_game(corrupted, FINAL_HTML)
    assert replayed["meta"]["parse"]["replayable"] is False
    assert replayed["meta"]["parse"]["warnings"], "a failed check must set a warning"
    # It never raises past the caller -- reaching this line proves that.


def test_replay_game_never_raises_on_broken_html():
    game = _parse_real_game()
    replayed = replay.replay_game(game, "<html>not a real boxscore page</html>")
    assert replayed["meta"]["parse"]["replayable"] is False
    assert replayed["meta"]["parse"]["warnings"]


# ---------------------------------------------------------------------------
# Per-check isolation: each synthetic bad-sequence fixture fails ONLY its
# targeted check while the other four still pass.
# ---------------------------------------------------------------------------

_ALL_CHECK_NAMES = [name for name, _ in replay._CHECKS]


def _run_all_checks(game: dict, oracle: dict) -> dict:
    return {name: fn(game, oracle) for name, fn in replay._CHECKS}


def test_good_baseline_passes_all_five_checks():
    data = _load_synth("good_baseline.json")
    results = _run_all_checks(data["game"], data["oracle"])
    for name, result in results.items():
        assert result.ok, f"{name} unexpectedly failed on the clean baseline: {result.warnings}"


def _assert_isolated_failure(fixture_name: str, failing_check: str):
    data = _load_synth(fixture_name)
    results = _run_all_checks(data["game"], data["oracle"])
    for name, result in results.items():
        if name == failing_check:
            assert not result.ok, f"{failing_check} was expected to fail on {fixture_name}"
        else:
            assert result.ok, (
                f"{name} unexpectedly failed on {fixture_name} (should only break "
                f"{failing_check}): {result.warnings}"
            )


def test_bad_linescore_fails_only_check_linescore():
    _assert_isolated_failure("bad_linescore.json", "linescore")


def test_bad_outs_per_half_fails_only_check_outs_per_half():
    _assert_isolated_failure("bad_outs_per_half.json", "outs_per_half")


def test_bad_lob_fails_only_check_lob():
    _assert_isolated_failure("bad_lob.json", "lob")


def test_bad_pa_counts_fails_only_check_pa_counts():
    _assert_isolated_failure("bad_pa_counts.json", "pa_counts")


def test_bad_illegal_transitions_fails_only_check_illegal_transitions():
    _assert_isolated_failure("bad_illegal_transitions.json", "illegal_transitions")
