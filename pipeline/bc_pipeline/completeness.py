"""Completeness report generator for backfill runs (g2, #20).

Consumes one or more :class:`bc_pipeline.backfill.BackfillResult` objects
(the CLI passes the whole multi-season backfill's results) and produces a
JSON-serializable completeness report -- honest per-game and per-season
fetch/parse/replay/UNPARSED accounting, plus a CLI that exits nonzero when
the observed LINE-level UNPARSED rate crosses a threshold.

This module never re-parses or re-touches ``games/**`` -- it reads
``BackfillResult``/``GameOutcome``/``SeasonSummary`` exactly as
``bc_pipeline.backfill`` defines them (read that module's dataclasses,
around lines 113-198, before changing anything here) and never invents a
new field on the game files themselves.

**Two distinct rates, both reported honestly (rework, #20 g2 attempt 2):**
an earlier draft of this module conflated a GAME-level failure count with
the term "UNPARSED rate". The Admiral's ruling (spec D4/T4) fixed the real
UNPARSED rate as a LINE-level metric -- the share of PBP narrative lines
landing in a game's ``unparsed[]`` list -- so this module now reports BOTH,
under distinct names, neither one dropped:

* ``failure_rate`` -- GAME-level: what fraction of discovered games failed
  to parse or turned out not replayable. This is the metric attempt-1
  mistakenly called ``unparsed_rate``; it is renamed here, not removed --
  it remains a valuable, honestly-reported number in its own right.
* ``unparsed_rate`` -- LINE-level: what fraction of PBP narrative lines, in
  games this run actually parsed, ended up in ``unparsed[]``. THIS is the
  real UNPARSED rate, and it is what the CLI threshold/exit-code is keyed
  on (see "Threshold mechanism" below).

**Failure-rate definition** (game-level): a game counts against the rate if
its outcome is ``"parse_failed"``, OR its outcome is ``"parsed"`` but
``replayable`` is ``False``. ``"non_final"`` is EXCLUDED from the numerator
-- a not-yet-final schedule entry is an expected, non-alarming outcome, not
a parse failure -- but it still counts in the denominator
(``games_discovered``), since it was a real discovered game this run looked
at. ``"skipped_already_committed"`` is also excluded from the numerator (it
succeeded in a *previous* run; this run did not re-examine its content) but
counts in the denominator too, since it is still a discovered game. In
short::

    failure_rate = (games_parse_failed + (games_parsed - games_replayable))
                    / games_discovered

**UNPARSED-rate definition** (line-level -- the real UNPARSED metric, fixed
by the Admiral's ruling): ``parse.py`` stamps ``meta.parse.events_count``
and ``meta.parse.unparsed_count`` on every successfully parsed game (see
``parse.py`` lines ~882-892; ``bc_pipeline.backfill.GameOutcome`` threads
both numbers through as ``events_count``/``unparsed_count``, ``None`` for
any outcome that never went through a parse this run). Per game, when both
counts are available::

    line_unparsed_rate = unparsed_count / (events_count + unparsed_count)

A game with no ``events_count``/``unparsed_count`` (``non_final``,
``parse_failed``, ``skipped_already_committed`` -- this run never parsed
its content) is excluded ENTIRELY from both the numerator and the
denominator of the aggregate below -- never treated as a 0% game, never
fabricated. ``league.unparsed_rate`` and each
``by_season["<year>"].unparsed_rate`` are TOTALS-based, not an average of
per-game rates::

    unparsed_rate = sum(unparsed_count over parsed games)
                     / sum(events_count + unparsed_count over parsed games)

The totals form is used deliberately instead of averaging per-game rates:
it weights every narrative line equally regardless of which game produced
it, so a 10,000-line game with 100 unparsed lines moves the aggregate more
than a 50-line game with 5 unparsed lines -- exactly matching the question
"what fraction of all PBP lines did we fail to parse", which is the honest
question this metric answers. A simple average of per-game rates would
instead weight every GAME equally regardless of size, which is a different
(and less honest, for this purpose) question.

**Threshold mechanism** (evidence-grounded, see ``DEFAULT_THRESHOLD`` below):
g3's zero-fetch slice validation (issue #20) ran this exact pipeline against
all 10 already-archived real games and observed a league-wide LINE-level
``unparsed_rate`` of **0.1457** (182 unparsed lines / 1249 total narrative
lines) -- a known, diagnosed "grammar tail" (illegal base-runner
transitions, PA-count/linescore/LOB mismatches on a subset of plays; filed
as issue #30, to be closed by a labeled re-parse once the full corpus's
unparsed inventory is available). ``DEFAULT_THRESHOLD`` is set to
**0.20 (20%)** -- the observed 0.1457 plus a ~0.054 safety margin, rounded
up -- so the overnight full-corpus run (which is *expected* to reproduce
this same grammar tail across ~1,257 games) exits 0 against the known
baseline and only exits loud (nonzero) if the corpus-wide rate is
materially WORSE than what g3 already characterized -- i.e. a genuinely new
regression, not the already-diagnosed and already-filed residue. The
threshold and CLI nonzero-exit are keyed on the LINE-level
``league.unparsed_rate`` (not ``failure_rate`` -- the game-level number is
still reported, but it does not gate the run: g3's slice, for reference,
had ``failure_rate`` 1.0 (10/10 games not fully replayable) alongside the
much smaller *line*-level 0.1457, which is exactly why the two are tracked
as separate, differently-scaled numbers rather than one conflated metric).
``--threshold`` still lets any run override this constant without a code
change (see ``main``/``build_arg_parser`` below) if the observed
distribution shifts again.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bc_pipeline.backfill import BackfillResult

__all__ = [
    "DEFAULT_THRESHOLD",
    "build_completeness_report",
    "build_arg_parser",
    "main",
]

#: Evidence-grounded threshold for the LINE-level ``unparsed_rate`` -- see
#: module docstring's "Threshold mechanism": g3's real zero-fetch slice
#: (issue #20) observed 0.1457 league-wide; this is that value + a ~0.054
#: safety margin, rounded up.
DEFAULT_THRESHOLD: float = 0.20

#: Outcomes that count as "not alarming" and are tracked separately from
#: ``enumerated_failures`` -- see ``non_final_games`` below.
_NON_FINAL_OUTCOME = "non_final"

_LEAGUE_FIELDS = (
    "games_discovered",
    "games_fetched",
    "games_parsed",
    "games_replayable",
    "games_non_final",
    "games_parse_failed",
    "games_skipped_already_committed",
)


def _empty_totals() -> dict:
    return {field: 0 for field in _LEAGUE_FIELDS}


def _empty_line_totals() -> dict:
    """Accumulator for the LINE-level rate: raw sums, not yet divided."""
    return {"unparsed_lines": 0, "total_lines": 0}


def _failure_rate(totals: dict) -> float:
    """Fraction of discovered games that failed to parse or turned out not
    replayable -- GAME-level, see this module's "Failure-rate definition"
    docstring section. Renamed from an earlier draft's ``unparsed_rate``;
    kept, not dropped -- see module docstring.

    Returns ``0.0`` when ``games_discovered`` is zero (no games seen at all
    is not itself a failure -- avoids a division by zero on an empty input
    rather than reporting a misleading rate).
    """
    discovered = totals["games_discovered"]
    if discovered == 0:
        return 0.0
    failed_count = totals["games_parse_failed"] + (
        totals["games_parsed"] - totals["games_replayable"]
    )
    return failed_count / discovered


def _unparsed_rate(line_totals: dict) -> float:
    """Fraction of PBP narrative lines that landed in ``unparsed[]``, across
    every game actually parsed this run/input -- LINE-level, the real
    UNPARSED rate, see this module's "UNPARSED-rate definition" docstring
    section.

    Returns ``0.0`` when no game with a known ``events_count``/
    ``unparsed_count`` was seen (never divide by zero, never fabricate a
    rate for games that were never parsed).
    """
    total_lines = line_totals["total_lines"]
    if total_lines == 0:
        return 0.0
    return line_totals["unparsed_lines"] / total_lines


def build_completeness_report(
    results: Iterable[BackfillResult],
    *,
    threshold: float | None = None,
) -> dict:
    """Aggregate one or more :class:`BackfillResult`\\ s into a JSON-
    serializable completeness report dict.

    Every :class:`~bc_pipeline.backfill.GameOutcome` across every
    :class:`BackfillResult` passed in is enumerated somewhere in the
    output -- never silently dropped or averaged away (the "honest-null"
    constraint this report exists to satisfy).
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD

    league_totals = _empty_totals()
    league_line_totals = _empty_line_totals()
    by_season: dict[int, dict] = {}
    by_season_line_totals: dict[int, dict] = {}
    enumerated_failures: list[dict] = []
    non_final_games: list[dict] = []

    for result in results:
        for season, summary in result.seasons.items():
            season_totals = by_season.setdefault(season, _empty_totals())
            # games_discovered is derived from the actual per-game outcome
            # count for this season (below), not from SeasonSummary fields,
            # so it is correct even if a future SeasonSummary field is added
            # or renamed.
            season_totals["games_fetched"] += summary.fetched
            season_totals["games_parsed"] += summary.parsed
            season_totals["games_replayable"] += summary.replayable
            season_totals["games_non_final"] += summary.non_final
            season_totals["games_parse_failed"] += summary.parse_failed
            season_totals["games_skipped_already_committed"] += (
                summary.skipped_already_committed
            )

            league_totals["games_fetched"] += summary.fetched
            league_totals["games_parsed"] += summary.parsed
            league_totals["games_replayable"] += summary.replayable
            league_totals["games_non_final"] += summary.non_final
            league_totals["games_parse_failed"] += summary.parse_failed
            league_totals["games_skipped_already_committed"] += (
                summary.skipped_already_committed
            )

        for game in result.games:
            season_totals = by_season.setdefault(game.season, _empty_totals())
            season_totals["games_discovered"] += 1
            league_totals["games_discovered"] += 1

            is_parse_failed = game.outcome == "parse_failed"
            is_unreplayable_parsed = game.outcome == "parsed" and game.replayable is False

            if is_parse_failed or is_unreplayable_parsed:
                reason = game.reason
                if reason is None and is_unreplayable_parsed:
                    reason = (
                        "warnings: " + "; ".join(game.warnings)
                        if game.warnings
                        else "parsed but replay marked this game not replayable"
                    )
                enumerated_failures.append(
                    {
                        "game_id": game.game_id,
                        "season": game.season,
                        "url": game.url,
                        "outcome": game.outcome,
                        "reason": reason,
                    }
                )
            elif game.outcome == _NON_FINAL_OUTCOME:
                non_final_games.append(
                    {
                        "game_id": game.game_id,
                        "season": game.season,
                        "url": game.url,
                        "reason": game.reason,
                    }
                )

            # LINE-level accumulation: only games this run actually parsed
            # carry non-None events_count/unparsed_count (see GameOutcome) --
            # anything else (non_final/parse_failed/skipped_already_committed)
            # is excluded entirely, never fabricated as a zero.
            if game.events_count is not None and game.unparsed_count is not None:
                season_line_totals = by_season_line_totals.setdefault(
                    game.season, _empty_line_totals()
                )
                line_total = game.events_count + game.unparsed_count
                season_line_totals["unparsed_lines"] += game.unparsed_count
                season_line_totals["total_lines"] += line_total
                league_line_totals["unparsed_lines"] += game.unparsed_count
                league_line_totals["total_lines"] += line_total

    league_totals["failure_rate"] = _failure_rate(league_totals)
    league_totals["unparsed_rate"] = _unparsed_rate(league_line_totals)

    by_season_out: dict[str, dict] = {}
    for season in sorted(by_season):
        season_totals = by_season[season]
        season_totals["failure_rate"] = _failure_rate(season_totals)
        season_line_totals = by_season_line_totals.get(season, _empty_line_totals())
        season_totals["unparsed_rate"] = _unparsed_rate(season_line_totals)
        by_season_out[str(season)] = season_totals

    # Threshold + nonzero exit are keyed on the LINE-level unparsed_rate --
    # the real UNPARSED metric per the Admiral's ruling -- not failure_rate.
    exceeded = league_totals["unparsed_rate"] > threshold

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "league": league_totals,
        "by_season": by_season_out,
        "enumerated_failures": enumerated_failures,
        "non_final_games": non_final_games,
        "threshold": {
            "value": threshold,
            "exceeded": exceeded,
        },
    }


def _load_backfill_result(path: str) -> BackfillResult:
    """Load one ``BackfillResult.to_dict()``-shaped JSON file back into the
    minimal shape ``build_completeness_report`` needs (a duck-typed stand-in
    for :class:`BackfillResult`/``GameOutcome``/``SeasonSummary`` -- the CLI
    reads serialized JSON, never a live Python object)."""
    from bc_pipeline.backfill import GameOutcome, SeasonSummary

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seasons = {
        int(season_key): SeasonSummary(
            season=int(season_key),
            fetched=summary.get("fetched", 0),
            skipped_already_done=summary.get("skipped_already_done", 0),
            parsed=summary.get("parsed", 0),
            replayable=summary.get("replayable", 0),
            non_final=summary.get("non_final", 0),
            parse_failed=summary.get("parse_failed", 0),
            skipped_already_committed=summary.get("skipped_already_committed", 0),
        )
        for season_key, summary in data.get("seasons", {}).items()
    }
    games = [
        GameOutcome(
            url=g["url"],
            season=g["season"],
            game_id=g["game_id"],
            outcome=g["outcome"],
            reason=g.get("reason"),
            replayable=g.get("replayable"),
            warnings=list(g.get("warnings", [])),
            unparsed_count=g.get("unparsed_count"),
            events_count=g.get("events_count"),
        )
        for g in data.get("games", [])
    ]
    return BackfillResult(seasons=seasons, games=games)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bc_pipeline.completeness",
        description=(
            "Aggregate one or more backfill-result JSON files into a completeness "
            "report (artifacts/latest/completeness.json) and exit nonzero if the "
            "observed LINE-level UNPARSED rate crosses a threshold."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more BackfillResult.to_dict()-shaped JSON files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/latest/completeness.json",
        metavar="PATH",
        help="Where to write the completeness report (default: artifacts/latest/completeness.json).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="RATE",
        help=(
            "LINE-level UNPARSED-rate threshold (0..1); exit nonzero if the "
            f"league-wide rate exceeds this value (default: {DEFAULT_THRESHOLD}, "
            "a provisional placeholder -- see module docstring)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code (0 clean, 1 if the
    league-wide LINE-level UNPARSED rate exceeds ``--threshold``)."""
    args = build_arg_parser().parse_args(argv)

    results = [_load_backfill_result(path) for path in args.input]
    report = build_completeness_report(results, threshold=args.threshold)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    league = report["league"]
    if report["threshold"]["exceeded"]:
        print(
            f"[COMPLETENESS] FAILED: league-wide unparsed_rate "
            f"{league['unparsed_rate']:.4f} exceeds threshold {args.threshold:.4f} "
            f"(failure_rate {league['failure_rate']:.4f}).",
            file=sys.stderr,
        )
        bad_seasons = [
            season
            for season, totals in report["by_season"].items()
            if totals["unparsed_rate"] > args.threshold
        ]
        if bad_seasons:
            print(
                f"[COMPLETENESS] Season(s) over threshold: {', '.join(bad_seasons)}.",
                file=sys.stderr,
            )
        print(
            f"[COMPLETENESS] {len(report['enumerated_failures'])} enumerated failure(s); "
            f"see {output_path} for full detail.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[COMPLETENESS] OK: league-wide unparsed_rate {league['unparsed_rate']:.4f} "
        f"(threshold {args.threshold:.4f}; failure_rate {league['failure_rate']:.4f}); "
        f"{league['games_discovered']} game(s) discovered, "
        f"{len(report['enumerated_failures'])} enumerated failure(s). Wrote {output_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
