"""Completeness report generator for backfill runs (g2, #20).

Consumes one or more :class:`bc_pipeline.backfill.BackfillResult` objects
(the CLI passes the whole multi-season backfill's results) and produces a
JSON-serializable completeness report -- honest per-game and per-season
fetch/parse/replay/UNPARSED accounting, plus a CLI that exits nonzero when
the observed UNPARSED rate crosses a threshold.

This module never re-parses or re-touches ``games/**`` -- it reads
``BackfillResult``/``GameOutcome``/``SeasonSummary`` exactly as
``bc_pipeline.backfill`` defines them (read that module's dataclasses,
around lines 113-198, before changing anything here) and never invents a
new field on the game files themselves.

**UNPARSED-rate definition** (this term is not otherwise defined anywhere
in the codebase -- this gate chooses it): a game counts against the
UNPARSED rate if its outcome is ``"parse_failed"``, OR its outcome is
``"parsed"`` but ``replayable`` is ``False``. ``"non_final"`` is EXCLUDED
from the numerator -- a not-yet-final schedule entry is an expected,
non-alarming outcome, not a parse failure -- but it still counts in the
denominator (``games_discovered``), since it was a real discovered game
this run looked at. ``"skipped_already_committed"`` is also excluded from
the numerator (it succeeded in a *previous* run; this run did not
re-examine its content) but counts in the denominator too, since it is
still a discovered game. In short::

    unparsed_rate = (games_parse_failed + (games_parsed - games_replayable))
                    / games_discovered

**Threshold mechanism** (provisional, see ``DEFAULT_THRESHOLD`` below):
the full multi-season corpus this report will eventually score does not
exist yet at development time (this gate runs before g3's slice
validation), so a hand-picked placeholder is used: 5% (0.05). The intended
mechanism, once real data exists, is "observed league-wide unparsed_rate
across the full backfill slice + a fixed safety margin (e.g. +2
percentage points)" -- i.e. re-derive this constant from g3's actual
numbers rather than guessing forever. Until then, ``--threshold`` lets a
real run override this constant without a code change (see ``main``/
``build_arg_parser`` below), and 0.05 is deliberately generous (rather
than tight) so a provisional value does not spuriously fail an otherwise
healthy early run.
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

#: Provisional threshold -- see module docstring's "Threshold mechanism" for
#: the justification and the plan to replace this with an evidence-grounded
#: value once the full backfill corpus exists (g3).
DEFAULT_THRESHOLD: float = 0.05

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


def _unparsed_rate(totals: dict) -> float:
    """Fraction of discovered games that did NOT end up replayable, per the
    UNPARSED-rate definition documented in this module's docstring.

    Returns ``0.0`` when ``games_discovered`` is zero (no games seen at all
    is not itself an unparsed game -- avoids a division by zero on an empty
    input rather than reporting a misleading rate).
    """
    discovered = totals["games_discovered"]
    if discovered == 0:
        return 0.0
    unparsed_count = totals["games_parse_failed"] + (
        totals["games_parsed"] - totals["games_replayable"]
    )
    return unparsed_count / discovered


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
    by_season: dict[int, dict] = {}
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
                    }
                )

    league_totals["unparsed_rate"] = _unparsed_rate(league_totals)

    by_season_out: dict[str, dict] = {}
    for season in sorted(by_season):
        season_totals = by_season[season]
        season_totals["unparsed_rate"] = _unparsed_rate(season_totals)
        by_season_out[str(season)] = season_totals

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
            "observed UNPARSED rate crosses a threshold."
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
            "UNPARSED-rate threshold (0..1); exit nonzero if the league-wide rate "
            f"exceeds this value (default: {DEFAULT_THRESHOLD}, a provisional "
            "placeholder -- see module docstring)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code (0 clean, 1 if the
    league-wide UNPARSED rate exceeds ``--threshold``)."""
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
            f"{league['unparsed_rate']:.4f} exceeds threshold {args.threshold:.4f}.",
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
        f"(threshold {args.threshold:.4f}); {league['games_discovered']} game(s) discovered, "
        f"{len(report['enumerated_failures'])} enumerated failure(s). Wrote {output_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
