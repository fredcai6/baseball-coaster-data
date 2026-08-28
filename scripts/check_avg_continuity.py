#!/usr/bin/env python3
"""Cross-game completeness oracle: the source's own running batting average.

Every boxscore batting row carries `AVG`, and it is CUMULATIVE-TO-DATE for
that player's season -- not the single game's rate. That makes it ground
truth the source publishes about games we may not have: walk a player's
season in date order, accumulate AB and H from our own boxscores, and the
running quotient must reproduce the published AVG on every row.

It is the only check in this repository that reads ACROSS games. Every
other one validates a game against itself, which means a corpus that is
internally consistent but INCOMPLETE looks perfect to all of them. This
session's repeated lesson is that whatever has no oracle turns out to be
wrong, and completeness had no oracle.

WHAT A DIVERGENCE MEANS. The published AVG includes at-bats we do not have;
ours is computed only from what we hold. So a mismatch says our AB/H total
is short, and the shortfall is CONSTANT from the moment it appears -- the
error never heals, it just dilutes as the denominator grows. The tell is
therefore the FIRST divergence, not the size of the gap.

When many players on one team first diverge in the SAME game, the games
missing are the ones immediately before it. That is how the two known gaps
below were found; neither was visible to any within-game check.

This is a DIAGNOSTIC, not a gate. It is expected to report the known gaps,
and it exits 0 unless --strict is passed. Use it after a fetch to find out
what the fetch missed.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: A published AVG is rounded to three places, so compare at that precision
#: with half a unit of slack rather than demanding exact float equality.
TOLERANCE = 0.0006

#: Below this many at-bats a single hit moves the average by more than the
#: rounding tolerance can absorb, so early-season rows are noisy by
#: construction rather than by defect.
MIN_AB = 1


def _gap_before(previous: "str | None", date: str) -> list:
    """Dates between a team's previous game and this one, exclusive.

    A divergence that begins in a game a team played the day after its last
    one is NOT explained by a missing game, and saying so is the point --
    the oracle should not manufacture a gap where the schedule has none.
    """
    if previous is None:
        return []
    import datetime

    start = datetime.date.fromisoformat(previous)
    end = datetime.date.fromisoformat(date)
    return [
        (start + datetime.timedelta(days=i)).isoformat()
        for i in range(1, (end - start).days)
    ]


def _schedules(games_dir: Path):
    """team_id -> sorted dates, and game_id -> (date, [(team_id, name)])."""
    import collections as _c

    schedule = _c.defaultdict(list)
    meta = {}
    for path in sorted(games_dir.glob("*/*.json")):
        game = json.loads(path.read_text(encoding="utf-8"))
        sides = [
            (game["teams"][side]["team_id"], game["teams"][side]["name"])
            for side in ("away", "home")
        ]
        meta[game["game_id"]] = (game["date"], sides)
        for team_id, _ in sides:
            schedule[team_id].append(game["date"])
    for dates in schedule.values():
        dates.sort()
    return schedule, meta


def _load(games_dir: Path):
    """(season, person_id) -> [(date, game_id, AB, H, AVG), ...]."""
    seasons = collections.defaultdict(list)
    skipped = 0
    for path in sorted(games_dir.glob("*/*.json")):
        game = json.loads(path.read_text(encoding="utf-8"))
        for rows in game["box"]["batting"].values():
            for row in rows:
                avg = row.get("AVG")
                if avg in (None, ""):
                    continue
                player = game["players"].get(row["player_id"]) or {}
                person = player.get("person_id")
                if not person:
                    # A synthetic id is per-GAME; `syn:away:3` in two games
                    # is two different people, so accumulating across games
                    # on it would be meaningless. Skipped, and counted.
                    skipped += 1
                    continue
                seasons[(game["season"], person)].append(
                    (game["date"], game["game_id"], row["AB"], row["H"], avg)
                )
    return seasons, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default=str(REPO_ROOT / "games"))
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any person-season fails to reconcile",
    )
    ap.add_argument(
        "--min-cluster",
        type=int,
        default=3,
        help="report a game as a suspected corpus gap when at least this "
        "many person-seasons first diverge there (default 3)",
    )
    args = ap.parse_args()

    games_dir = Path(args.games)
    seasons, skipped = _load(games_dir)
    schedule, meta = _schedules(games_dir)
    reconciled = 0
    first_divergence: collections.Counter = collections.Counter()
    failures = 0

    for rows in seasons.values():
        rows.sort()
        at_bats = hits = 0
        first = None
        for date, game_id, ab, h, avg in rows:
            at_bats += ab
            hits += h
            try:
                published = float(avg)
            except (TypeError, ValueError):
                continue
            if at_bats < MIN_AB:
                continue
            computed = round(hits / at_bats, 3)
            if abs(computed - published) > TOLERANCE and first is None:
                first = game_id
        if first is None:
            reconciled += 1
        else:
            failures += 1
            first_divergence[first] += 1

    total = reconciled + failures
    print(f"person-seasons          {total}")
    print(f"  reconcile exactly     {reconciled} ({reconciled / total:.1%})")
    print(f"  diverge               {failures}")
    print(f"box rows skipped        {skipped} (synthetic id, no person_id)")

    clusters = [
        (game_id, n)
        for game_id, n in first_divergence.most_common()
        if n >= args.min_cluster
    ]
    if clusters:
        print()
        print(
            f"SUSPECTED CORPUS GAPS -- {len(clusters)} game(s) where "
            f"{args.min_cluster}+ person-seasons first diverge."
        )
        print("The missing games are the ones immediately BEFORE each of these.")
        for game_id, n in clusters:
            date, teams = meta.get(game_id, (None, []))
            print(f"  {n:4d} person-seasons first diverge at {game_id}  {date}")
            for team_id, name in teams:
                prior = [d for d in schedule.get(team_id, []) if d < date]
                gap = _gap_before(prior[-1] if prior else None, date)
                if gap:
                    print(f"         {name:28s} plays {prior[-1]}, then {date}"
                          f"  -- MISSING {', '.join(gap)}")
                elif not prior:
                    print(f"         {name:28s} has no earlier game -- season "
                          "opener is later than the rest of the league")

    singletons = failures - sum(n for _, n in clusters)
    print()
    print(
        f"{singletons} divergence(s) are not clustered -- individual "
        "anomalies rather than a missing game."
    )

    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
