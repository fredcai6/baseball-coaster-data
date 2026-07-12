"""Golden fixture test for the live sample (issue #19 gate g7).

Protected intent: `tests/fixtures/golden/game_20260709_h94w.json` is the full
parse+replay output of the sample game (events + unparsed + `_derived`,
meta timestamps normalized so the file never depends on wall-clock time).
This test re-derives that output FRESH from the sample HTML on every run and
asserts it is byte-for-byte the same shape as the committed golden --
`_derived` INCLUDED, only the two volatile meta timestamp fields excluded
(via the same `normalize_meta_timestamps` the regeneration helper uses, so
test and regeneration never drift from each other's idea of "volatile").

If this test ever fails, that is NOT license to silently overwrite the
golden: run `PYTHONPATH=pipeline py -m bc_pipeline.reparse_summary` first
(read-only) to see the reparse-summary delta, and only pass `--write` once
that delta is understood and accepted (see
`tests/fixtures/PROMOTION_PROTOCOL.md`).
"""
from __future__ import annotations

import json

from _support import FIXTURES_DIR, SAMPLES_DIR

from bc_pipeline import parse, reparse_summary, replay

GOLDEN_PATH = FIXTURES_DIR / "golden" / "game_20260709_h94w.json"


def _load_html(name: str) -> str:
    with (SAMPLES_DIR / name).open("r", encoding="utf-8") as f:
        return f.read()


def _fresh_normalized_sample_run() -> dict:
    html = _load_html("boxscore_20260709_final.html")
    game = parse.parse_game(
        html,
        source_url=reparse_summary.SAMPLE_SOURCE_URL,
        fetched_at=reparse_summary.SAMPLE_FETCHED_AT,
        parsed_at=reparse_summary.SAMPLE_PARSED_AT,
    )
    replayed = replay.replay_game(game, html)
    return reparse_summary.normalize_meta_timestamps(replayed)


def test_golden_fixture_exists_and_is_reasonably_sized():
    assert GOLDEN_PATH.exists(), f"golden fixture missing: {GOLDEN_PATH}"
    size = GOLDEN_PATH.stat().st_size
    assert 0 < size < 1_000_000, f"golden fixture size {size} bytes out of expected range"


def test_fresh_parse_replay_matches_committed_golden():
    fresh = _fresh_normalized_sample_run()
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        golden = json.load(f)

    # Full-value equality: includes every `_derived` block (never stripped
    # here, unlike serialize.semantic_equal -- this golden is asserting the
    # REPLAYER'S output too, not just parser identity) and every meta field
    # except the two normalized timestamps.
    assert fresh == golden, (
        "fresh parse+replay of the live sample no longer matches the "
        "committed golden. Run `PYTHONPATH=pipeline py -m "
        "bc_pipeline.reparse_summary` (read-only) to see the reparse-summary "
        "delta before deciding whether to regenerate."
    )

    # Sanity: the golden genuinely carries `_derived` on every foldable
    # event (a golden with `_derived` silently stripped would still pass a
    # naive `==` against an equally-stripped fresh run, so assert its
    # presence directly).
    foldable_kinds = ("plate_appearance", "runner_event")
    foldable = [e for e in golden["events"] if e["kind"] in foldable_kinds]
    assert foldable, "golden has no foldable events to carry _derived"
    assert all("_derived" in e for e in foldable), (
        "golden is missing `_derived` on at least one foldable event"
    )


def test_golden_meta_timestamps_are_normalized():
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        golden = json.load(f)
    assert golden["meta"]["fetched_at"] == reparse_summary.NORMALIZED_TIMESTAMP
    assert golden["meta"]["parsed_at"] == reparse_summary.NORMALIZED_TIMESTAMP


def test_regenerate_golden_readonly_reports_zero_delta_when_unchanged():
    """The regeneration helper, called read-only (no --write), reports a
    ZERO reparse-summary delta against the currently-committed golden --
    this IS the gate: regeneration is never silent, and when nothing has
    changed the delta proves it."""
    delta = reparse_summary.regenerate_golden(write=False)
    assert delta == {
        "replay_delta": 0.0,
        "unparsed_rate_delta": 0.0,
        "event_type_count_deltas": {},
    }
