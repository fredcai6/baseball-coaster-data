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


# --- issue #40 / schema 1.5.0: interference in the pa_counts formula ------
#
# Catcher's interference awards the batter first base and is a plate
# appearance that is NOT an at-bat, so it sits in neither AB nor BB. Without
# an explicit term every such PA fails by exactly one. This is an oracle
# DEFINITION change, which #33 requires be deliberate and separately tested.


def _pa_counts_case(outcome_type, ab, bb):
    """One batter, one plate appearance of `outcome_type`, against a box row
    carrying `ab`/`bb`. Returns the check's warnings."""
    pid = "syn:home:1"
    game = {
        "events": [
            {
                "kind": "plate_appearance",
                "seq": 1,
                "inning": 1,
                "half": "bottom",
                "batter": {"player_id": pid, "name_raw": "X", "resolved": True},
                "outcome": {"type": outcome_type, "modifiers": [], "fielders": [],
                            "location": None, "outs_recorded": 0},
                "runners": [],
            }
        ]
    }
    oracle = {"box": {"batting": {"t1": [{"player_id": pid, "AB": ab, "BB": bb}]}}}
    return replay.check_pa_counts(game, oracle).warnings


def test_catchers_interference_is_a_pa_but_not_an_at_bat():
    """AB=0, BB=0 and one interference PA must reconcile."""
    assert _pa_counts_case("reached_on_interference", ab=0, bb=0) == []


def test_catchers_interference_would_fail_without_its_term():
    """Guard the guard: a plain walk with AB=0/BB=0 still fails, so the case
    above is passing because of the new term and not because the check is
    vacuous."""
    assert _pa_counts_case("walk", ab=0, bb=0) != []


def test_batter_interference_is_charged_as_an_at_bat():
    """The batter is retired, so it is already inside box.AB and must NOT
    get its own term -- AB=1 reconciles, AB=0 does not."""
    assert _pa_counts_case("batter_interference", ab=1, bb=0) == []
    assert _pa_counts_case("batter_interference", ab=0, bb=0) != []


# --- issue #40: a sacrifice FLY is a sacrifice too -------------------------
#
# StatCrew spells the bunt ", SAC" but the fly ", sacrifice fly" in full.
# check_pa_counts tested `"SAC" in modifiers` by exact membership, so every
# sacrifice fly was missed. A sacrifice is not charged as an at-bat, so
# events_PA came out one HIGHER than box.AB + box.BB + ... and the game
# failed pa_counts. That one token accounted for the +1 direction in 284 of
# 299 pa_counts mismatches across clean-parse games.


def _pa_counts_with_modifiers(outcome_type, modifiers, ab, bb):
    pid = "syn:home:1"
    game = {
        "events": [
            {
                "kind": "plate_appearance", "seq": 1, "inning": 1, "half": "bottom",
                "batter": {"player_id": pid, "name_raw": "X", "resolved": True},
                "outcome": {"type": outcome_type, "modifiers": modifiers, "fielders": [],
                            "location": None, "outs_recorded": 1},
                "runners": [],
            }
        ]
    }
    oracle = {"box": {"batting": {"t1": [{"player_id": pid, "AB": ab, "BB": bb}]}}}
    return replay.check_pa_counts(game, oracle).warnings


def test_sacrifice_fly_is_a_plate_appearance_but_not_an_at_bat():
    """AB=0 and one sac-fly PA must reconcile."""
    assert _pa_counts_with_modifiers("flyout", ["sacrifice fly", "RBI"], ab=0, bb=0) == []


def test_sacrifice_bunt_spelling_still_counts():
    assert _pa_counts_with_modifiers("groundout", ["SAC", "bunt"], ab=0, bb=0) == []


def test_an_ordinary_flyout_is_still_an_at_bat():
    """Guard the guard: without the sacrifice modifier, AB=0 must FAIL, so
    the two cases above pass because of the modifier and not because the
    check went vacuous."""
    assert _pa_counts_with_modifiers("flyout", ["RBI"], ab=0, bb=0) != []
    assert _pa_counts_with_modifiers("flyout", ["RBI"], ab=1, bb=0) == []


def test_sac_modifier_set_covers_both_spellings():
    from bc_pipeline.replay import _SAC_MODIFIERS

    assert "SAC" in _SAC_MODIFIERS
    assert "sacrifice fly" in _SAC_MODIFIERS


# --- issue #40: LOB is measured at plate-appearance boundaries -------------
#
# check_lob REFOLDS from runners[] and never trusts an input _derived, so
# these fixtures carry real runner primitives rather than hand-written state.


def _runner(pid, frm, to, out=False, scored=False, cause="batted_ball"):
    return {"player_id": pid, "from": frm, "to": to, "out": out,
            "scored": scored, "cause": cause}


def _lob_game(events, box_lob):
    return {"events": events + [
        {"kind": "inning_summary", "seq": 99, "inning": 1, "half": "bottom",
         "summary": {"R": 0, "H": 0, "E": 0, "LOB": box_lob}}
    ]}


def _lob_warnings(events, box_lob):
    return replay.check_lob(_lob_game(events, box_lob), {}).warnings


def _pa_ev(seq, runners):
    return {"kind": "plate_appearance", "seq": seq, "inning": 1, "half": "bottom",
            "batter": {"player_id": "b", "name_raw": "B", "resolved": True},
            "outcome": {"type": "single", "modifiers": [], "fielders": [],
                        "location": None, "outs_recorded": 0},
            "runners": runners}


def _runner_ev(seq, runners):
    return {"kind": "runner_event", "seq": seq, "inning": 1, "half": "bottom",
            "runners": runners}


def test_runner_retired_between_plate_appearances_is_still_left_on_base():
    """A batter singles with two out, then is picked off for the third. The
    box counts him; the old rule (occupancy after the last EVENT) did not.
    This was the entire -1 cohort: 222 half-innings."""
    events = [
        _pa_ev(1, [_runner("r1", 0, 1)]),
        _runner_ev(2, [_runner("r1", 1, -1, out=True, cause="pickoff")]),
    ]
    assert _lob_warnings(events, box_lob=1) == []
    # And the old reading would have said 0.
    assert _lob_warnings(events, box_lob=0) != []


def test_batter_making_the_third_out_is_not_left_on_base():
    """Guard the other direction: when the final PA is itself the out,
    occupancy after it is what counts."""
    events = [_pa_ev(1, [_runner("b", 0, -1, out=True, cause="putout")])]
    assert _lob_warnings(events, box_lob=0) == []
    assert _lob_warnings(events, box_lob=1) != []


def test_runners_still_on_base_after_the_final_plate_appearance_count():
    events = [
        _pa_ev(1, [_runner("r1", 0, 1)]),
        _pa_ev(2, [_runner("r2", 0, 1), _runner("r1", 1, 2)]),
    ]
    assert _lob_warnings(events, box_lob=2) == []


def test_reached_and_did_not_score_is_NOT_the_rule():
    """The rejected alternative. A runner who reached, advanced and was then
    retired between PAs counts once -- not once per base he touched."""
    events = [
        _pa_ev(1, [_runner("r1", 0, 1)]),
        _runner_ev(2, [_runner("r1", 1, 2, cause="stolen_base")]),
        _runner_ev(3, [_runner("r1", 2, -1, out=True, cause="pickoff")]),
    ]
    assert _lob_warnings(events, box_lob=1) == []
# --- issue #40: half-inning chronological ordering -------------------------


def test_half_order_puts_top_before_bottom():
    """`max()` on the raw (inning, half) tuple compares the half
    LEXICOGRAPHICALLY, and "top" > "bottom" -- so the last half of a game
    came out as the TOP of the final inning. Every walk-off was then
    reported as a short half-inning, because `is_last_half` was False for
    every bottom half and the exception could never fire."""
    from bc_pipeline.replay import _half_order

    keys = [(9, "top"), (9, "bottom"), (8, "bottom"), (1, "top")]
    assert max(keys, key=_half_order) == (9, "bottom")
    assert sorted(keys, key=_half_order) == [
        (1, "top"), (8, "bottom"), (9, "top"), (9, "bottom"),
    ]
    # The raw comparison this replaced gets it wrong:
    assert max(keys) == (9, "top")


def test_walkoff_half_with_two_outs_is_accepted():
    """Bottom of the last inning, home takes the lead, inning ends before a
    third out."""
    game = {"events": [
        {"kind": "plate_appearance", "seq": 1, "inning": 9, "half": "top",
         "batter": {"player_id": "a", "name_raw": "A", "resolved": True},
         "outcome": {"type": "groundout", "modifiers": [], "fielders": [],
                     "location": None, "outs_recorded": 3},
         "runners": [{"player_id": f"a{i}", "from": 0, "to": -1, "out": True,
                      "scored": False, "cause": "putout"} for i in range(3)]},
        {"kind": "plate_appearance", "seq": 2, "inning": 9, "half": "bottom",
         "batter": {"player_id": "h", "name_raw": "H", "resolved": True},
         "outcome": {"type": "groundout", "modifiers": [], "fielders": [],
                     "location": None, "outs_recorded": 2},
         "runners": [{"player_id": f"h{i}", "from": 0, "to": -1, "out": True,
                      "scored": False, "cause": "putout"} for i in range(2)]},
        {"kind": "plate_appearance", "seq": 3, "inning": 9, "half": "bottom",
         "batter": {"player_id": "w", "name_raw": "W", "resolved": True},
         "outcome": {"type": "home_run", "modifiers": [], "fielders": [],
                     "location": None, "outs_recorded": 0},
         "runners": [{"player_id": "w", "from": 0, "to": 4, "out": False,
                      "scored": True, "cause": "batted_ball"}]},
    ]}
    oracle = {"linescore": {"innings": {"away": [None] * 9, "home": [None] * 9},
                            "totals": {"away": {"R": 0}, "home": {"R": 1}}}}
    assert replay.check_outs_per_half(game, oracle).warnings == []
# --- issue #40: trailing linescore columns are table padding ---------------


def _linescore_game(events):
    return {"events": events}


def _ls_oracle(away, home, away_total=0, home_total=0):
    return {"linescore": {"innings": {"away": away, "home": home},
                          "totals": {"away": {"R": away_total},
                                     "home": {"R": home_total}}},
            "box": {"batting": {}}}


def _one_inning(inning, half, runs=0):
    return {"kind": "plate_appearance", "seq": inning * 10 + (0 if half == "top" else 1),
            "inning": inning, "half": half,
            "batter": {"player_id": "b", "name_raw": "B", "resolved": True},
            "outcome": {"type": "home_run" if runs else "groundout", "modifiers": [],
                        "fielders": [], "location": None, "outs_recorded": 0 if runs else 1},
            "runners": ([{"player_id": "b", "from": 0, "to": 4, "out": False,
                          "scored": True, "cause": "batted_ball"}] if runs else
                        [{"player_id": "b", "from": 0, "to": -1, "out": True,
                          "scored": False, "cause": "putout"}])}


def test_trailing_zero_inning_with_no_events_is_not_a_mismatch():
    """The boxscore renders linescore columns past the end of the game and
    fills them with 0 -- extra-inning columns on a 9-inning game, a 9th and
    10th column on a 7-inning doubleheader. 0 runs and 0 events agree."""
    events = [_one_inning(1, "top"), _one_inning(1, "bottom")]
    oracle = _ls_oracle(away=[0, 0, 0], home=[0, 0, 0])
    assert replay.check_linescore(_linescore_game(events), oracle).warnings == []


def test_trailing_inning_with_a_nonzero_expectation_is_still_reported():
    """A non-zero run count with no events is genuinely missing play."""
    events = [_one_inning(1, "top"), _one_inning(1, "bottom")]
    oracle = _ls_oracle(away=[0, 3], home=[0, 0], away_total=3)
    ws = replay.check_linescore(_linescore_game(events), oracle).warnings
    assert any("has no folded events" in w for w in ws)


def test_interior_inning_with_no_events_is_still_reported():
    """Deliberately narrow: only TRAILING innings are excused. An interior
    gap would be genuinely missing play-by-play."""
    events = [_one_inning(1, "top"), _one_inning(1, "bottom"),
              _one_inning(3, "top"), _one_inning(3, "bottom")]
    oracle = _ls_oracle(away=[0, 0, 0], home=[0, 0, 0])
    ws = replay.check_linescore(_linescore_game(events), oracle).warnings
    assert any("inning 2" in w for w in ws)
