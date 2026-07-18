"""Season-level event-frequency aggregation (g1, issue #21).

Aggregates ``events[].outcome.type`` -- the CLOSED 19-type taxonomy at
``schemas/game.schema.json`` ``$defs.outcome.properties.type.enum`` -- from
``games/**`` into season+league **team** and **player** event-frequency
tables. This module reads (never mutates) ``games/**``; it never re-parses,
re-derives, or fabricates an outcome.

**Two sub-tables per key, at BOTH the team and player level** (same event
pass, two keyings):

* ``batting`` -- keyed by ``batting_team`` / ``batter.player_id``: what a
  team/player did AT THE PLATE (offense).
* ``pitching`` -- keyed by ``fielding_team`` / ``pitcher.player_id``: what a
  team/player ALLOWED (defense/pitching allowed).

**Rate definition** (explicit, no ambiguity)::

    rate = outcome_type_count / total_plate_appearances_for_that_key

For a ``batting`` entry, the denominator is the total plate appearances that
team/player BATTED in (this season, or league-wide for the ``league``
bucket). For a ``pitching`` entry, the denominator is the total plate
appearances that team/player FACED. Both are counted by construction (every
``plate_appearance`` event increments exactly one outcome-type count and the
same key's ``total_plate_appearances``), so ``sum(counts.values()) ==
total_plate_appearances`` always holds -- never a fabricated or
independently-estimated denominator.

**One combined artifact**, ``league``/``by_season`` nesting mirroring
``completeness.json``'s existing shape (``bc_pipeline.completeness``):
top-level ``meta``, ``league.{batting,pitching}.{teams,players}`` (totals
across every game this run aggregated), and ``by_season.<season>.
{batting,pitching}.{teams,players}`` (per-season breakdown). Every
outcome-type count/rate table always carries all 19 taxonomy keys (0 when a
type never occurred for that key) -- never sparse, never silently omitted.

**Honest-Null coverage**: ``meta.coverage`` reports the LINE-level
unparsed-rate across the aggregated corpus (from each game's
``meta.parse.events_count``/``unparsed_count``, stamped by ``parse.py`` --
never recomputed here) and an explicit note that outcome-type counts are
drawn only from ``events[]``: a source line the parser could not classify
(landing in ``unparsed[]``) is NOT represented in any count below, and may
under-count rare event types. Never imputed, never fabricated -- the
Honest-Null Clause.

**Determinism**: every count/rate table is emitted with keys sorted
alphabetically; ``meta.generated_at`` is a plain wall-clock timestamp for
humans, but the no-commit CLI guard (below) always compares two runs with
that one volatile field normalized to a FIXED SENTINEL first --
``NORMALIZED_TIMESTAMP``/``normalize_generated_at``, mirroring
``reparse_summary.py``'s ``NORMALIZED_TIMESTAMP``/``normalize_meta_
timestamps`` idiom, written LOCALLY here (this module does not import
``reparse_summary`` or ``completeness`` -- it mirrors their SHAPE only, per
the issue #21 fence).

**CLI**: ``python -m bc_pipeline.frequencies --input games/ --output
artifacts/latest/frequencies.json`` regenerates and writes the artifact.
``--check-no-commit`` instead regenerates in memory, compares against the
currently-committed ``--output`` file with ``generated_at`` normalized on
BOTH sides, and reports a distinct signal without writing: exit 0 + a
"NO-OP" message when nothing but the timestamp would change, exit 2 + a
"CHANGED" message otherwise. The actual git commit/no-commit decision is a
later gate's (``refresh.py``, issue #21 g2) job -- this CLI only reports the
comparison result.

**Scope**: CONTEXT-FREE outcome-type counts/rates ONLY. No state-dependent
stat (run expectancy, base-out, LOB rate, win probability) belongs here --
that is out of scope by the launch order's pre-ruling.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

__all__ = [
    "OUTCOME_TYPES",
    "NORMALIZED_TIMESTAMP",
    "build_frequencies",
    "normalize_generated_at",
    "load_games",
    "build_arg_parser",
    "main",
]

#: The closed 19-type outcome taxonomy (schemas/game.schema.json
#: $defs.outcome.properties.type.enum), listed here in the schema's own
#: order for readability. Every artifact count/rate table is emitted with
#: keys sorted ALPHABETICALLY (see `_finalize_rates`) regardless of this
#: tuple's order -- this ordering is documentation only, never a
#: determinism dependency.
OUTCOME_TYPES: tuple[str, ...] = (
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "intentional_walk",
    "hit_by_pitch",
    "strikeout_swinging",
    "strikeout_looking",
    "groundout",
    "flyout",
    "lineout",
    "popout",
    "fielders_choice",
    "reached_on_error",
    "grounded_into_double_play",
    "sacrifice",
    "foul_out",
    "strikeout",
)

#: Fixed sentinel `meta.generated_at` is normalized to for the no-commit
#: comparison -- mirrors reparse_summary.py's NORMALIZED_TIMESTAMP idiom,
#: written LOCALLY here per the issue #21 fence (never import
#: reparse_summary.py).
NORMALIZED_TIMESTAMP: str = "1970-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Accumulation (internal, mutable) -- finalized into the immutable artifact
# shape by `_finalize_*` below.
# ---------------------------------------------------------------------------


def _empty_key_entry() -> dict:
    return {
        "total_plate_appearances": 0,
        "counts": {t: 0 for t in OUTCOME_TYPES},
    }


def _empty_side_tables() -> dict:
    return {"teams": {}, "players": {}}


def _empty_batting_pitching() -> dict:
    return {"batting": _empty_side_tables(), "pitching": _empty_side_tables()}


def _add_pa(table: dict, key: str, outcome_type: str) -> None:
    entry = table.setdefault(key, _empty_key_entry())
    entry["total_plate_appearances"] += 1
    entry["counts"][outcome_type] += 1


def _finalize_rates(entry: dict) -> dict:
    """Turn one accumulator entry (total + counts) into the final,
    alphabetically-sorted-keys {total_plate_appearances, counts, rates}
    shape. `rates[t] = counts[t] / total` -- see module docstring's Rate
    definition; `0.0` only in the unreachable-in-practice case of a zero
    total (an entry is only ever created alongside its first PA)."""
    total = entry["total_plate_appearances"]
    counts = entry["counts"]
    rates = {t: (counts[t] / total if total else 0.0) for t in OUTCOME_TYPES}
    return {
        "total_plate_appearances": total,
        "counts": dict(sorted(counts.items())),
        "rates": dict(sorted(rates.items())),
    }


def _finalize_side_tables(side: dict) -> dict:
    return {
        "teams": {k: _finalize_rates(v) for k, v in sorted(side["teams"].items())},
        "players": {
            k: _finalize_rates(v) for k, v in sorted(side["players"].items())
        },
    }


def _accumulate_game(game: dict, *buckets: dict) -> tuple[int | None, int | None]:
    """Fold one parsed game dict's `plate_appearance` events into every
    accumulator dict in `buckets` (each shaped like `_empty_batting_
    pitching()`) -- e.g. the league-wide accumulator AND that game's season
    accumulator, in one pass over `events[]`.

    Returns `(events_count, unparsed_count)` as stamped on this game by
    `parse.py` (`meta.parse`) -- `None, None` if that provenance block is
    absent -- for the caller's Honest-Null coverage accounting. Never
    recomputed from `events`/`unparsed` here; always read verbatim from the
    game's own provenance.
    """
    for event in game.get("events", []):
        if event.get("kind") != "plate_appearance":
            continue
        outcome = event.get("outcome") or {}
        outcome_type = outcome.get("type")
        if outcome_type not in OUTCOME_TYPES:
            # Never happens for a schema-valid game (closed taxonomy) --
            # guard rather than silently mis-bucket an unrecognized value.
            continue

        batting_team = event["batting_team"]
        fielding_team = event["fielding_team"]
        batter_id = (event.get("batter") or {}).get("player_id")
        pitcher_id = (event.get("pitcher") or {}).get("player_id")

        for bucket in buckets:
            _add_pa(bucket["batting"]["teams"], batting_team, outcome_type)
            _add_pa(bucket["pitching"]["teams"], fielding_team, outcome_type)
            if batter_id:
                _add_pa(bucket["batting"]["players"], batter_id, outcome_type)
            if pitcher_id:
                _add_pa(bucket["pitching"]["players"], pitcher_id, outcome_type)

    parse_meta = (game.get("meta") or {}).get("parse") or {}
    events_count = parse_meta.get("events_count")
    unparsed_count = parse_meta.get("unparsed_count")
    return events_count, unparsed_count


def build_frequencies(games: Iterable[dict], *, generated_at: str | None = None) -> dict:
    """Aggregate `games` (an iterable of parsed ``games/**``-shaped dicts)
    into the season+league frequency artifact dict.

    Pure function: never reads or writes any file, never mutates `games` or
    any element of it. `generated_at` defaults to the current UTC wall-clock
    time (ISO-8601, `Z`-suffixed) if not given -- pass an explicit value for
    a reproducible test.
    """
    league_acc = _empty_batting_pitching()
    by_season_acc: dict[int, dict] = {}
    games_by_season: dict[int, int] = {}
    parser_versions: set[str] = set()
    total_narrative_lines = 0
    total_unparsed_lines = 0
    total_games = 0

    for game in games:
        total_games += 1
        season = game["season"]
        season_acc = by_season_acc.setdefault(season, _empty_batting_pitching())
        games_by_season[season] = games_by_season.get(season, 0) + 1

        parser_version = (game.get("meta") or {}).get("parser_version")
        if parser_version:
            parser_versions.add(parser_version)

        events_count, unparsed_count = _accumulate_game(game, league_acc, season_acc)
        if events_count is not None and unparsed_count is not None:
            total_narrative_lines += events_count + unparsed_count
            total_unparsed_lines += unparsed_count

    by_season_out = {
        str(season): {
            "batting": _finalize_side_tables(acc["batting"]),
            "pitching": _finalize_side_tables(acc["pitching"]),
        }
        for season, acc in sorted(by_season_acc.items())
    }

    unparsed_rate = (
        total_unparsed_lines / total_narrative_lines if total_narrative_lines else 0.0
    )

    return {
        "meta": {
            "generated_at": generated_at
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "parser_versions": sorted(parser_versions),
            "games_included": {
                "total": total_games,
                "by_season": {
                    str(season): count
                    for season, count in sorted(games_by_season.items())
                },
            },
            "coverage": {
                "total_narrative_lines": total_narrative_lines,
                "total_unparsed_lines": total_unparsed_lines,
                "unparsed_rate": unparsed_rate,
                "note": (
                    "Outcome-type counts/rates are drawn only from "
                    "events[]; verbatim source lines the parser could not "
                    "classify (unparsed[]) are not represented in any "
                    "count/rate below and may under-count rare event "
                    "types -- roughly on the scale of unparsed_rate. "
                    "Never imputed or fabricated (Honest-Null Clause)."
                ),
            },
        },
        "league": {
            "batting": _finalize_side_tables(league_acc["batting"]),
            "pitching": _finalize_side_tables(league_acc["pitching"]),
        },
        "by_season": by_season_out,
    }


# ---------------------------------------------------------------------------
# CLI / file I/O.
# ---------------------------------------------------------------------------


def load_games(input_dir: str | Path) -> list[dict]:
    """Load every `*.json` file under `input_dir` (recursively), sorted by
    path for deterministic aggregation order, returning the parsed dicts.
    Pure I/O helper; never mutates or writes anything under `input_dir`
    (``games/**`` is write-once/read-only from this module's perspective)."""
    root = Path(input_dir)
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*.json"))
    ]


def normalize_generated_at(artifact: dict) -> dict:
    """Return a deep copy of `artifact` with `meta.generated_at` rewritten to
    `NORMALIZED_TIMESTAMP` -- mirrors reparse_summary.py's
    `normalize_meta_timestamps`, written LOCALLY here (never imported) per
    the issue #21 fence. Used by the `--check-no-commit` CLI guard and by
    tests proving determinism modulo wall-clock time."""
    out = copy.deepcopy(artifact)
    meta = out.get("meta")
    if isinstance(meta, dict) and "generated_at" in meta:
        meta["generated_at"] = NORMALIZED_TIMESTAMP
    return out


def _write_artifact(artifact: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bc_pipeline.frequencies",
        description=(
            "Aggregate games/**'s events[].outcome.type into a season+league "
            "batting/pitching event-frequency artifact "
            "(artifacts/latest/frequencies.json)."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="games",
        metavar="DIR",
        help="games/** root to aggregate (default: games).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/latest/frequencies.json",
        metavar="PATH",
        help="Where to write the artifact (default: artifacts/latest/frequencies.json).",
    )
    parser.add_argument(
        "--check-no-commit",
        action="store_true",
        help=(
            "Regenerate in memory and compare (generated_at normalized on "
            "both sides) against the currently-committed --output file "
            "instead of writing; exit 0 + 'NO-OP' when only the timestamp "
            "would differ, exit 2 + 'CHANGED' otherwise. Does not write "
            "--output. The actual commit/no-commit decision is a later "
            "gate's job -- this only reports the comparison."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code: 0 on a clean write (or a
    no-commit-check that found no real change), 2 when `--check-no-commit`
    finds a real change."""
    args = build_arg_parser().parse_args(argv)

    games = load_games(args.input)
    fresh = build_frequencies(games)
    output_path = Path(args.output)

    if args.check_no_commit:
        if output_path.exists():
            committed = json.loads(output_path.read_text(encoding="utf-8"))
            changed = normalize_generated_at(committed) != normalize_generated_at(fresh)
        else:
            changed = True

        if changed:
            print(
                "[FREQUENCIES] CHANGED: regenerated artifact differs from "
                f"the committed {output_path} (or none is committed yet); "
                "commit needed.",
                file=sys.stderr,
            )
            return 2

        print(
            "[FREQUENCIES] NO-OP: regenerated artifact matches the "
            f"committed {output_path} (generated_at normalized on both "
            "sides); nothing to commit."
        )
        return 0

    _write_artifact(fresh, output_path)
    print(
        f"[FREQUENCIES] wrote {output_path} "
        f"({fresh['meta']['games_included']['total']} game(s))."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
