"""reparse_summary -- machine-checkable re-parse delta (issue #19 gate g7).

Protected intent: a re-parse of the sample (or any game) must never silently
overwrite a committed golden fixture. This module gives the caller a
MACHINE-CHECKABLE summary of one parse+replay run (``summarize``) and a
MACHINE-CHECKABLE delta between two runs (``diff``) -- both stable,
serializable dicts, safe to compare, log, or assert on in a test -- so golden
regeneration is always GATED by an explicit, visible delta rather than an
ambient overwrite.

Three pieces:

* ``summarize(game)`` -- one run's shape: replay pass/fail, unparsed rate,
  event-type counts.
* ``diff(run_a, run_b)`` -- the delta between two runs. Each of ``run_a``/
  ``run_b`` may be either a full parsed(+replayed) game dict OR an
  already-computed summary dict (auto-detected); ``diff`` calls
  ``summarize`` on any raw game it is handed.
* ``normalize_meta_timestamps`` + ``regenerate_golden`` -- the golden-fixture
  half of the gate: a deterministic meta-timestamp normalization so a golden
  file never depends on wall-clock time, and a regeneration helper that
  ALWAYS prints the reparse-summary delta vs. the currently-committed golden
  before writing anything (an unexpected delta is visible, never silent).

This module reads (never mutates) the g2-g6 modules' output; it does not
import ``parse``/``replay`` at module scope beyond what's needed for the
``__main__`` regeneration entry, so importing ``reparse_summary`` alone (as
the test suite does to exercise ``summarize``/``diff``) never requires HTML
fixtures to be present.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Union

from . import serialize

GameOrSummary = Dict[str, object]

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_HTML_PATH = REPO_ROOT / "tests" / "samples" / "boxscore_20260709_final.html"
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden" / "game_20260709_h94w.json"

# Fixed sample identity (issue #19's live sample; see g7 handoff / tests).
SAMPLE_SOURCE_URL = (
    "https://longbeachcoast.com/sports/bsb/2026/boxscores/20260709_h94w.xml"
)
SAMPLE_FETCHED_AT = "2026-07-11T00:00:00Z"
SAMPLE_PARSED_AT = "2026-07-11T00:00:00Z"

# Sentinel the golden fixture's meta timestamps are pinned to, and that
# `normalize_meta_timestamps` rewrites any fresh run's timestamps to before
# comparison -- so golden comparison NEVER depends on wall-clock time, no
# matter what `fetched_at`/`parsed_at` a given parse call used.
NORMALIZED_TIMESTAMP = "1970-01-01T00:00:00Z"
_META_TIMESTAMP_KEYS = ("fetched_at", "parsed_at")


# ---------------------------------------------------------------------------
# summarize / diff -- the machine-checkable delta contract.
# ---------------------------------------------------------------------------


def _is_summary(x: GameOrSummary) -> bool:
    """True iff `x` looks like an already-computed summarize() result rather
    than a raw game dict (a game always has `events`/`unparsed`; a summary
    never does)."""
    return "event_type_counts" in x and "events" not in x and "unparsed" not in x


def summarize(game: GameOrSummary) -> dict:
    """Summarize one parse(+replay) run of `game` into a small, stable,
    JSON-serializable dict:

    - `replayable` (bool) / `replay_pass_rate` (1.0 or 0.0 for a single run) --
      from `game["meta"]["parse"]["replayable"]` (False/absent if the game was
      never replayed).
    - `unparsed_count` / `unparsed_rate` -- `len(unparsed)` over
      `len(events) + len(unparsed)` total narrative lines (0.0 if there were
      no lines at all).
    - `event_type_counts` -- a `{event.kind: count}` dict, sorted by key for a
      stable diff.

    Idempotent and side-effect free; does not mutate `game`.
    """
    if _is_summary(game):
        # Already a summary -- return a defensive copy so callers can't
        # accidentally mutate a cached summary through the return value.
        return copy.deepcopy(game)

    events = game.get("events", []) or []
    unparsed = game.get("unparsed", []) or []
    total_lines = len(events) + len(unparsed)
    unparsed_count = len(unparsed)
    unparsed_rate = (unparsed_count / total_lines) if total_lines else 0.0

    parse_meta = (game.get("meta") or {}).get("parse") or {}
    replayable = bool(parse_meta.get("replayable", False))

    counts = Counter(ev.get("kind", "unknown") for ev in events)
    event_type_counts = {k: counts[k] for k in sorted(counts)}

    return {
        "replayable": replayable,
        "replay_pass_rate": 1.0 if replayable else 0.0,
        "unparsed_count": unparsed_count,
        "unparsed_rate": unparsed_rate,
        "event_type_counts": event_type_counts,
    }


def diff(run_a: GameOrSummary, run_b: GameOrSummary) -> dict:
    """Machine-checkable delta between two runs (each may be a raw game dict
    or an already-computed `summarize()` dict -- auto-detected).

    Returns a stable, JSON-serializable dict:

    - `replay_delta` -- `b.replay_pass_rate - a.replay_pass_rate` (`0.0` when
      both pass or both fail).
    - `unparsed_rate_delta` -- `b.unparsed_rate - a.unparsed_rate`.
    - `event_type_count_deltas` -- `{kind: b_count - a_count}`, one entry per
      event kind that CHANGED (a kind unchanged between the two runs is
      omitted, not emitted as an explicit `0`, so a no-op re-parse always
      diffs to `event_type_count_deltas == {}`).

    Two identical runs (same game object, or two independent parses of
    identical input) always diff to all-zero/`{}`. This is the gate: a
    regeneration helper computes this delta and shows it BEFORE writing a new
    golden -- see `regenerate_golden`.
    """
    a = summarize(run_a)
    b = summarize(run_b)

    keys = set(a["event_type_counts"]) | set(b["event_type_counts"])
    event_type_count_deltas = {}
    for k in sorted(keys):
        delta = b["event_type_counts"].get(k, 0) - a["event_type_counts"].get(k, 0)
        if delta != 0:
            event_type_count_deltas[k] = delta

    return {
        "replay_delta": b["replay_pass_rate"] - a["replay_pass_rate"],
        "unparsed_rate_delta": b["unparsed_rate"] - a["unparsed_rate"],
        "event_type_count_deltas": event_type_count_deltas,
    }


# ---------------------------------------------------------------------------
# Golden meta-timestamp normalization + gated regeneration.
# ---------------------------------------------------------------------------


def normalize_meta_timestamps(game: dict) -> dict:
    """Return a deep copy of `game` with `meta.fetched_at`/`meta.parsed_at`
    rewritten to a fixed sentinel (`NORMALIZED_TIMESTAMP`).

    Every other `meta` field (`parser_version`, `source_url`,
    `source_sha256`, `derived_replayer_version`, `parse.*`) is left exactly
    as computed -- only the two wall-clock-derived timestamp fields are
    volatile from one run to the next, so only those are normalized. This is
    what lets the committed golden fixture, and a fresh parse+replay of the
    same input, compare equal regardless of when either run happened.
    """
    out = copy.deepcopy(game)
    meta = out.get("meta")
    if isinstance(meta, dict):
        for key in _META_TIMESTAMP_KEYS:
            if key in meta:
                meta[key] = NORMALIZED_TIMESTAMP
    return out


def _parse_and_replay_sample():
    # Imported lazily so `summarize`/`diff` (and importing this module in
    # general) never require the g5/g6 modules or sample HTML to be present.
    from . import parse as parse_mod
    from . import replay as replay_mod

    with SAMPLE_HTML_PATH.open("r", encoding="utf-8") as f:
        html = f.read()
    game = parse_mod.parse_game(
        html,
        source_url=SAMPLE_SOURCE_URL,
        fetched_at=SAMPLE_FETCHED_AT,
        parsed_at=SAMPLE_PARSED_AT,
    )
    return replay_mod.replay_game(game, html)


def regenerate_golden(write: bool = False) -> dict:
    """Re-parse+replay the live sample, compare its reparse-summary against
    the CURRENTLY COMMITTED golden (if any), print/return that delta, and
    only overwrite `GOLDEN_PATH` when `write=True`.

    This is the gate: calling this with `write=False` (the default) is
    always safe and shows exactly what would change; `write=True` is the
    explicit, deliberate act of accepting that delta. Never called from the
    test suite with `write=True` -- the golden test (`tests/test_golden.py`)
    only ever calls with the default, read-only form.
    """
    fresh = normalize_meta_timestamps(_parse_and_replay_sample())

    if GOLDEN_PATH.exists():
        with GOLDEN_PATH.open("r", encoding="utf-8") as f:
            committed = json.load(f)
        delta = diff(committed, fresh)
    else:
        delta = {
            "replay_delta": None,
            "unparsed_rate_delta": None,
            "event_type_count_deltas": None,
            "note": "no committed golden yet at GOLDEN_PATH",
        }

    if write:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GOLDEN_PATH.open("w", encoding="utf-8") as f:
            f.write(serialize.canonical_dumps(fresh))

    return delta


def _main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m bc_pipeline.reparse_summary",
        description=(
            "Re-parse+replay the live sample and print the reparse-summary "
            "delta vs. the committed golden fixture. Pass --write to accept "
            "the delta and overwrite the golden; without it, this is a "
            "read-only preview (the gate)."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="overwrite tests/fixtures/golden/game_20260709_h94w.json with the fresh run",
    )
    args = parser.parse_args()
    delta = regenerate_golden(write=args.write)
    print(json.dumps(delta, indent=2, sort_keys=True))
    if args.write:
        print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    _main()
