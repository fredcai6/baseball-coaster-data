"""Tests for bc_pipeline.reparse_summary: the machine-checkable re-parse
delta (issue #19 gate g7).

Protected intent: `summarize`/`diff` must be stable, serializable, and
correct -- zero deltas on two identical runs, correct (non-zero, isolated)
deltas when runs genuinely differ -- because a re-parse's golden-regeneration
gate depends on this delta being trustworthy.
"""
from __future__ import annotations

import copy
import json

from _support import SAMPLES_DIR

from bc_pipeline import parse, reparse_summary, replay
from bc_pipeline.reparse_summary import diff, summarize

SOURCE_URL = "https://longbeachcoast.com/sports/bsb/2026/boxscores/20260709_h94w.xml"
FETCHED_AT = "2026-07-11T00:00:00Z"


def _load_html(name: str) -> str:
    with (SAMPLES_DIR / name).open("r", encoding="utf-8") as f:
        return f.read()


def _real_game() -> dict:
    html = _load_html("boxscore_20260709_final.html")
    game = parse.parse_game(html, source_url=SOURCE_URL, fetched_at=FETCHED_AT)
    return replay.replay_game(game, html)


REAL_GAME = _real_game()


# ---------------------------------------------------------------------------
# summarize()
# ---------------------------------------------------------------------------


def test_summarize_shape_and_known_sample_counts():
    summary = summarize(REAL_GAME)
    assert set(summary.keys()) == {
        "replayable",
        "replay_pass_rate",
        "unparsed_count",
        "unparsed_rate",
        "event_type_counts",
    }
    # Known live-sample shape (see g7 handoff): 117 events, 5 unparsed (the 5
    # DH pitching-sub lines -- the floated schema gap, live pending-promotion
    # example; see tests/fixtures/PROMOTION_PROTOCOL.md).
    assert sum(summary["event_type_counts"].values()) == 117
    assert summary["unparsed_count"] == 5
    total_lines = 117 + 5
    assert summary["unparsed_rate"] == 5 / total_lines
    assert isinstance(summary["replayable"], bool)
    assert summary["replay_pass_rate"] in (0.0, 1.0)


def test_summarize_is_json_serializable_and_side_effect_free():
    game_copy = copy.deepcopy(REAL_GAME)
    summary = summarize(REAL_GAME)
    json.dumps(summary)  # must not raise
    assert REAL_GAME == game_copy  # summarize() never mutates its input


def test_summarize_on_a_summary_is_idempotent():
    summary = summarize(REAL_GAME)
    assert summarize(summary) == summary


def test_summarize_handles_a_game_with_no_lines_at_all():
    empty = {"events": [], "unparsed": [], "meta": {"parse": {"replayable": True}}}
    summary = summarize(empty)
    assert summary["unparsed_rate"] == 0.0
    assert summary["unparsed_count"] == 0
    assert summary["event_type_counts"] == {}
    assert summary["replayable"] is True
    assert summary["replay_pass_rate"] == 1.0


# ---------------------------------------------------------------------------
# diff() -- zero-delta on identical runs
# ---------------------------------------------------------------------------


def test_diff_is_all_zero_on_two_identical_game_dicts():
    d = diff(REAL_GAME, copy.deepcopy(REAL_GAME))
    assert d == {
        "replay_delta": 0.0,
        "unparsed_rate_delta": 0.0,
        "event_type_count_deltas": {},
    }


def test_diff_is_all_zero_on_two_identical_summaries():
    s = summarize(REAL_GAME)
    d = diff(s, copy.deepcopy(s))
    assert d == {
        "replay_delta": 0.0,
        "unparsed_rate_delta": 0.0,
        "event_type_count_deltas": {},
    }


def test_diff_accepts_a_mix_of_game_and_summary_inputs():
    s = summarize(REAL_GAME)
    assert diff(REAL_GAME, s) == diff(s, REAL_GAME) == {
        "replay_delta": 0.0,
        "unparsed_rate_delta": 0.0,
        "event_type_count_deltas": {},
    }


# ---------------------------------------------------------------------------
# diff() -- correct deltas on a genuinely differing pair
# ---------------------------------------------------------------------------


def test_diff_reports_correct_deltas_when_an_event_is_dropped_and_unparsed_grows():
    """Simulate a regression: one plate_appearance silently drops out of
    events[] (as if a grammar rule broke) and reappears in unparsed[]
    instead. The delta must isolate exactly that: one event_type_count
    decrement, unparsed_rate increases, replay untouched."""
    mutated = copy.deepcopy(REAL_GAME)
    dropped = None
    for i, ev in enumerate(mutated["events"]):
        if ev["kind"] == "plate_appearance":
            dropped = mutated["events"].pop(i)
            break
    assert dropped is not None
    mutated["unparsed"].append(
        {
            "location": {"inning": dropped["inning"], "half": dropped["half"], "line_index": 0},
            "raw": dropped["narrative"],
            "reason": "SYNTHETIC test regression: simulated dropped event",
        }
    )

    d = diff(REAL_GAME, mutated)

    assert d["event_type_count_deltas"] == {"plate_appearance": -1}
    base_summary = summarize(REAL_GAME)
    mutated_summary = summarize(mutated)
    assert mutated_summary["unparsed_count"] == base_summary["unparsed_count"] + 1
    assert d["unparsed_rate_delta"] == (
        mutated_summary["unparsed_rate"] - base_summary["unparsed_rate"]
    )
    assert d["unparsed_rate_delta"] > 0
    # Neither run's events[]/unparsed[] total line count changed (one moved
    # from events to unparsed), and replayable status is untouched by this
    # summary-level mutation (meta.parse.replayable wasn't edited).
    assert d["replay_delta"] == 0.0


def test_diff_reports_replay_delta_when_replayable_flips():
    a = {"events": [], "unparsed": [], "meta": {"parse": {"replayable": True}}}
    b = {"events": [], "unparsed": [], "meta": {"parse": {"replayable": False}}}
    d = diff(a, b)
    assert d["replay_delta"] == -1.0
    d2 = diff(b, a)
    assert d2["replay_delta"] == 1.0


def test_diff_omits_unchanged_event_kinds_and_reports_only_changed_ones():
    a = {"events": [{"kind": "x"}, {"kind": "y"}], "unparsed": [], "meta": {}}
    b = {"events": [{"kind": "x"}, {"kind": "y"}, {"kind": "y"}, {"kind": "z"}], "unparsed": [], "meta": {}}
    d = diff(a, b)
    assert d["event_type_count_deltas"] == {"y": 1, "z": 1}
    assert "x" not in d["event_type_count_deltas"]


def test_diff_result_is_json_serializable():
    d = diff(REAL_GAME, REAL_GAME)
    json.dumps(d)  # must not raise
