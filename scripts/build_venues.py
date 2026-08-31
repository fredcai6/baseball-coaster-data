#!/usr/bin/env python
"""Validate `reference/venues.json`, the franchise -> ballpark reference table.

This is NOT a `bc_pipeline`-style builder and is deliberately not wired into
`scripts/check_artifacts_current.py`. Every other file under `artifacts/latest/`
is a pure function of `games/**` (README caller-contract clause 2) -- this one
is not, and cannot be: `meta.source_url` on every game file 403s from any
container host (a CloudFront IP block, confirmed non-transient), and the raw
scraped HTML the games were parsed from is deliberately never committed
(clause 3), so there is no `games/**`-reachable venue data to derive FROM.
`reference/venues.json` is hand-researched from external sources instead
(mostly Wikipedia and Pioneer League/local-news coverage -- see each row's
`source`), which is exactly why it lives at `reference/venues.json` and not
`artifacts/latest/venues.json`: putting a non-derived, non-regenerable file
under the mutable-and-regenerated-from-games/** tier would misrepresent what
it is, even though today's fixed `check_artifacts_current.py` builder list
would not literally choke on the extra file.

What this script CAN check mechanically, and does:

  1. `games/**` gives us the true, closed set of franchise_ids and the
     season(s) each one appears in -- read straight from the corpus, the one
     part of this problem that IS a pure function of games/**. Every
     franchise_id found there must have an entry in `reference/venues.json`,
     and vice versa (no orphan rows for a franchise that has left the
     corpus, no missing row for one that is in it). `--check` fails loudly,
     naming the mismatch, if this ever drifts (e.g. a new season adds a
     16th franchise before this table is updated).
  2. Every franchise's `seasons_in_corpus` must equal what games/** actually
     shows -- this is the one field in the file that IS a derived fact, and
     it is checked against a fresh scan every run rather than trusted as
     typed.
  3. Structural/vocabulary invariants on the hand-researched fields
     themselves. Venues are keyed `season -> venue`, one explicit row per
     season the franchise appears in -- deliberately NOT year ranges, because
     independent-league clubs change stadia more often than a range implies
     and continuity should never be inferred silently. So: exactly one row
     per corpus season (no gaps, no rows for seasons the club was not in),
     `confidence` is one of the three documented values, `park_season_link`
     is `evidenced` or `assumed` (every season must state whether its park
     was researched for THAT year or merely carried forward), a travelling
     club carries no park in any season, and any populated `park_name`
     carries a `source` URL (the no-invention rule: a populated row must be
     attributable).

This script never touches the hand-researched content (park names, coords,
elevations, ...) -- there is nothing here to regenerate. `--check` (the
default) exits non-zero on any violation; there is no write mode.

Run:  python scripts/build_venues.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_DIR = REPO_ROOT / "games"
VENUES_PATH = REPO_ROOT / "reference" / "venues.json"

VALID_CONFIDENCE = {"verified", "probable", "unknown"}
VALID_PARK_SEASON_LINK = {"evidenced", "assumed"}


def _franchise_seasons_from_games(games_dir: Path) -> dict[str, set[int]]:
    """Return {franchise_id: {season, ...}} straight from games/**."""
    seasons: dict[str, set[int]] = {}
    for path in sorted(games_dir.rglob("*.json")):
        game = json.loads(path.read_text(encoding="utf-8"))
        season = game["season"]
        for side in ("home", "away"):
            fid = game["teams"][side]["franchise_id"]
            seasons.setdefault(fid, set()).add(season)
    return seasons


def _check_row(fid: str, row: dict, corpus_seasons: set[int]) -> list[str]:
    """Structural checks on one franchise entry (now keyed season -> venue)."""
    failures: list[str] = []
    prefix = f"{fid}:"

    seasons_in_corpus = row.get("seasons_in_corpus")
    if seasons_in_corpus is None:
        failures.append(f"{prefix} missing `seasons_in_corpus`")
    elif set(seasons_in_corpus) != corpus_seasons:
        failures.append(
            f"{prefix} seasons_in_corpus={sorted(seasons_in_corpus)} does not "
            f"match games/** ({sorted(corpus_seasons)})"
        )

    travelling = row.get("travelling", False)
    seasons = row.get("seasons")
    if not isinstance(seasons, dict):
        failures.append(f"{prefix} missing `seasons` map (season -> venue)")
        return failures

    # Exactly one row per season the franchise actually appears in: no year
    # ranges, no gaps, no rows for seasons the club was not in the corpus.
    have = {int(k) for k in seasons}
    for missing in sorted(corpus_seasons - have):
        failures.append(f"{prefix} no venue row for corpus season {missing}")
    for extra in sorted(have - corpus_seasons):
        failures.append(
            f"{prefix} venue row for season {extra}, which is not one of this "
            f"franchise's corpus seasons {sorted(corpus_seasons)}"
        )

    for season in sorted(have & corpus_seasons):
        venue = seasons[str(season)]
        vprefix = f"{prefix} seasons[{season}]"

        confidence = venue.get("confidence")
        if confidence not in VALID_CONFIDENCE:
            failures.append(
                f"{vprefix} confidence={confidence!r} is not one of "
                f"{sorted(VALID_CONFIDENCE)}"
            )

        link = venue.get("park_season_link")
        if link not in VALID_PARK_SEASON_LINK:
            failures.append(
                f"{vprefix} park_season_link={link!r} is not one of "
                f"{sorted(VALID_PARK_SEASON_LINK)} -- every season must say "
                "whether its park was researched for that year or assumed"
            )

        park_name = venue.get("park_name")

        if travelling or venue.get("travelling"):
            # A club with no home park must not carry one, in any season.
            if park_name is not None:
                failures.append(
                    f"{vprefix} travelling club carries park_name={park_name!r} "
                    "-- a travelling club must not be given a home park"
                )
            continue

        if park_name is not None and not venue.get("source"):
            failures.append(
                f"{vprefix} park_name={park_name!r} is populated but `source` "
                "is missing -- every populated row must carry a source URL "
                "(no-invention rule)"
            )
        if park_name is None and confidence != "unknown":
            failures.append(
                f"{vprefix} park_name is null but confidence={confidence!r} "
                "(expected 'unknown' for a null park)"
            )

    return failures


def check(repo_root: Path) -> list[str]:
    games_dir = repo_root / "games"
    venues_path = repo_root / "reference" / "venues.json"

    if not venues_path.exists():
        return [f"{venues_path} does not exist"]

    doc = json.loads(venues_path.read_text(encoding="utf-8"))
    franchises = doc.get("franchises", {})

    corpus = _franchise_seasons_from_games(games_dir)

    failures: list[str] = []

    missing = set(corpus) - set(franchises)
    for fid in sorted(missing):
        failures.append(
            f"{fid} appears in games/** (seasons {sorted(corpus[fid])}) but has "
            f"no entry in {venues_path}"
        )

    orphaned = set(franchises) - set(corpus)
    for fid in sorted(orphaned):
        failures.append(
            f"{fid} has an entry in {venues_path} but does not appear in "
            "games/** -- stale row"
        )

    for fid in sorted(set(franchises) & set(corpus)):
        failures.extend(_check_row(fid, franchises[fid], corpus[fid]))

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_venues.py",
        description=(
            "Validate reference/venues.json against games/** (franchise_id "
            "coverage and seasons only -- the hand-researched park facts are "
            "never touched)."
        ),
    )
    parser.add_argument("--repo-root", default=None, metavar="PATH")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else REPO_ROOT

    failures = check(repo_root)
    if failures:
        print(
            f"INVALID: {len(failures)} problem(s) in {VENUES_PATH.relative_to(REPO_ROOT)}:",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  [X] {failure}", file=sys.stderr)
        return 1

    n_franchises = len(json.loads(VENUES_PATH.read_text(encoding="utf-8"))["franchises"])
    print(
        f"OK: {VENUES_PATH.relative_to(REPO_ROOT)} covers all "
        f"{n_franchises} franchise(s) in games/**, with seasons and row "
        "structure consistent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
