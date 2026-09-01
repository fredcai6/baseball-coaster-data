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

WHAT CHANGED, 2026-08-28 -- READ THIS BEFORE ACTING ON THE OUTPUT. A live
schedule walk on that date discovered exactly 1,486 boxscore URLs across the
three seasons (2024: 498, 2025: 490, 2026: 498), and the raw archive holds
all 1,486 of them: 1,484 committed, plus 20250520_iiqj and 20250521_jyjy,
which the site still publishes with a boxscore and ZERO play-by-play panes
(refetched that day to confirm, still empty) and which are disclosed in
artifacts/latest/completeness.json under `non_final_games`.

So the corpus is COMPLETE with respect to what the source publishes, and a
divergence here no longer implies a missing game. Two measurements say the
same thing from inside this check:

  - 30 person-seasons fail on the player's FIRST row, where by construction
    there is nothing earlier to be missing.
  - 2025 looked 10 games short beside its neighbours. It is not. The league
    published 490 games that year.

The reading of the column is not the problem either: taken as CUMULATIVE
INCLUDING this game, 84.05% of 29,953 checkable rows reconcile; taken as
"entering this game", 9.85% do. The first is plainly the right reading.

WHAT CHANGED, 2026-08-29. 114 of the then-215 reported divergences were this
check's own rounding, not the corpus's -- see TOLERANCE. After the fix, 90 of
1,895 person-seasons diverge (95.3% reconcile) and the residual is SHARPER
rather than merely smaller: 69 of the 90 fall in eight games, all in 2025,
all between May 27 and June 21.

WHAT CHANGED, 2026-09-01 -- THE 90 ARE EXPLAINED. Issue #42, closed. The
residual is a SOURCE-SIDE artifact and needs no corpus action; nothing in
games/** is wrong, and every file matches its published page exactly.

The mechanism is the one this docstring already predicted two paragraphs up:
a CONSTANT shortfall that dilutes as the denominator grows. What was missing
was proof that the missing at-bats are UNOBTAINABLE rather than a game we
could still fetch. Both halves are now measured:

  - A paced refetch of two clustered games (20250531_txx5, 20250527_glxj)
    against their 2026-07-12 archived copies found the batting box, linescore
    and Other Information byte-identical across 51 days, once randomized
    `component-nav*` / `dropdownId` render ids are normalized. Nothing was
    retroactively rewritten, so the divergence is not a scoring correction we
    fetched on the wrong side of.

  - Walking A.J. Shaver's 2025 season, a fixed +1 H / +4 AB applied from
    20250531_txx5 onward reproduces the published AVG on 52 of 55 games
    (against 22 of 55 at zero offset); 2 of the 3 misses are the known
    doubleheader-ordering artifact. That offset is exactly one game's worth of
    batting -- a 1-for-4 the source's season-to-date carries and NO published
    boxscore contains. He has no career split (one record, 64 games), so it is
    not an identity artifact either.

So the source's stats database holds at-bats that were never published as a
boxscore. We cannot obtain them, and we should not try.

NOT ESTABLISHED, deliberately: that every one of the 90 has this shape. A
quick re-walk of all 2025 person-seasons found 146 divergences where this
check finds 90, which makes that walk a DIFFERENT diagnostic, not this one --
its counts mean nothing until the two are reconciled. Offsets did cluster at
+4/+5 AB, one game's worth, which is suggestive and no more. Recorded as a
lead. This check has been wrong about its own arithmetic before, which is
exactly why the stronger claim is not made here.

If the 90 ever get in someone's way, the fix is a CLASSIFIER IN THIS CHECK,
not a corpus change: fit the minimal (dH, dAB) per divergence and report
"N of known one-game-offset shape / M of unknown shape". The value would be
discrimination -- as it stands the 90 are camouflage, and a genuinely new
parsing defect would land in the same undifferentiated pile unnoticed. Low
priority while this stays a diagnostic that gates nothing.

Do not use a cluster here as a reason to fetch. That mistake has already
been made once on this corpus: roughly 35 games were called missing, and the
refetch found 6 of those dates had no scheduled game at all and 20 were
already committed. The 2026-09-01 refetch above is the second and last time
this needs paying for -- it returned byte-identical pages.

This is a DIAGNOSTIC, not a gate, and it exits 0 unless --strict is passed.
What it is still good for: catching a person_id that should have been two
people or two that should have been one, and re-proving completeness cheaply
if the fetch story ever changes again.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Half a unit in the published figure's last place, plus a float-comparison
#: epsilon. The published AVG is rounded to three places and the source's
#: ROUNDING MODE is not documented, so the comparison is made against the
#: UNROUNDED quotient and both conventions are accepted at an exact tie.
#:
#: Rounding our own side first is what this replaces, and it was wrong in one
#: specific way that cost 114 of 215 reported divergences. Python's `round`
#: is banker's rounding, so 20/64 = .3125 becomes .312; the league's scorer
#: rounds half up and publishes .313. Every exact-half quotient failed, and
#: 5/16, 15/48, 20/64 and 18/32 are ordinary denominators in a short season.
#: Measured on the committed corpus, the change moves 215 divergences to 90
#: and 88.7% reconciling to 95.3% -- with no game file touched, because the
#: defect was in the check.
TOLERANCE = 0.0005 + 1e-9

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
                # A row with NO published average still counts toward the
                # running AB/H -- it just cannot be checked. 20240521_gq1b's
                # Batters table omits the AVG column entirely, and skipping
                # its rows outright would leave every later game short by
                # exactly the at-bats this check exists to notice. Accumulate
                # always; check only where the source published a figure.
                avg = row.get("AVG") or None
                # "0" is the source's SENTINEL for "no figure published",
                # not a batting average of zero -- it writes ".000" for a
                # genuine zero. Measured: 11,633 rows carry "0" and 11,617
                # of them have AB=0, i.e. the player appeared but never
                # batted, while ".000" appears on 743 rows. Read as a value
                # it makes every later row for a bench appearance compare a
                # real cumulative average against 0.000; treating it as
                # missing removes 7 false divergences (231 -> 224).
                if avg == "0":
                    avg = None
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


def _first_divergence(rows):
    """The game_id where the running average first stops reproducing the
    published one, or None if the whole season reconciles.

    DOUBLEHEADERS make the order ambiguous. Two games share a date, and the
    file name does not say which was played first -- so accumulating them in
    filename order can manufacture a divergence where the data is fine. 354
    of 1,893 person-seasons contain a same-date pair, so this is not a corner
    case; sorting naively and reporting the result would have blamed a
    missing game for what is really an ordering guess.

    The published averages themselves settle it: only one order reproduces
    both rows. So try the orderings of each same-date group and keep one that
    reconciles, falling back to the first ordering when none does (a real
    divergence, which is what we are looking for).
    """
    import itertools

    by_date = collections.defaultdict(list)
    for row in rows:
        by_date[row[0]].append(row)

    at_bats = hits = 0
    for date in sorted(by_date):
        group = by_date[date]
        # More than a doubleheader would make the permutations expensive and
        # does not occur; guard rather than assume.
        orders = (
            list(itertools.permutations(group)) if len(group) <= 3 else [tuple(group)]
        )
        chosen = None
        for order in orders:
            ab, h = at_bats, hits
            if all(_row_reconciles(ab := ab + r[2], h := h + r[3], r[4]) for r in order):
                chosen = (order, ab, h)
                break
        if chosen is None:
            # No ordering of this date works: report the first game in it.
            order = orders[0]
            for row in order:
                at_bats += row[2]
                hits += row[3]
                if not _row_reconciles(at_bats, hits, row[4]):
                    return row[1]
            return order[0][1]
        order, at_bats, hits = chosen
    return None


def _row_reconciles(at_bats, hits, published):
    if at_bats < MIN_AB:
        return True
    if published is None:
        return True  # accumulated, but nothing published to check against
    try:
        target = float(published)
    except (TypeError, ValueError):
        return True
    # The quotient is deliberately NOT rounded before comparing -- see
    # TOLERANCE. A tie like .3125 sits exactly half a unit from both .312
    # and .313, so both published values reconcile and neither is guessed at.
    return abs(hits / at_bats - target) <= TOLERANCE


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
        first = _first_divergence(rows)
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
            f"CLUSTERED DIVERGENCES -- {len(clusters)} game(s) where "
            f"{args.min_cluster}+ person-seasons first diverge."
        )
        print(
            "NOT evidence of a missing game, and no longer unexplained (issue "
            "#42, closed 2026-09-01). The corpus is complete against the "
            "source's own schedule, and these are a SOURCE-SIDE artifact: the "
            "league's season-to-date carries about one game's worth of AB/H "
            "that no published boxscore holds, so the shortfall is constant "
            "and merely dilutes. A paced refetch found the pages byte-"
            "identical across 51 days. Nothing here needs fixing, and this is "
            "not a reason to fetch -- see this module's docstring."
        )
        for game_id, n in clusters:
            date, teams = meta.get(game_id, (None, []))
            print(f"  {n:4d} person-seasons first diverge at {game_id}  {date}")
            # The schedule-gap lines below describe the calendar only. A date
            # with no game is an ordinary league off-day -- every season in
            # this corpus has several -- not a hole in the corpus.
            for team_id, name in teams:
                prior = [d for d in schedule.get(team_id, []) if d < date]
                gap = _gap_before(prior[-1] if prior else None, date)
                if gap:
                    print(f"         {name:28s} plays {prior[-1]}, then {date}"
                          f"  -- no game on {', '.join(gap)}")
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
