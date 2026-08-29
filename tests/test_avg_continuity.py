"""The cross-game completeness oracle.

Every other check in this repository validates a game against ITSELF, so a
corpus that is internally consistent but INCOMPLETE looks perfect to all of
them. The source's own running batting average is cumulative-to-date, which
makes it ground truth about games we may not hold.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_avg_continuity", REPO_ROOT / "scripts" / "check_avg_continuity.py"
)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_a_gap_is_the_dates_between_two_games():
    assert mod._gap_before("2026-06-25", "2026-06-27") == ["2026-06-26"]
    assert mod._gap_before("2026-07-28", "2026-07-30") == ["2026-07-29"]


def test_consecutive_games_leave_no_gap():
    # The oracle must not manufacture a missing game where the schedule has
    # none -- a divergence beginning the day after a team's last game has
    # some other cause, and saying so is the point.
    assert mod._gap_before("2026-06-25", "2026-06-26") == []


def test_no_earlier_game_is_not_a_gap():
    # A team whose season opener is simply later than the rest of the
    # league has no previous game to measure from.
    assert mod._gap_before(None, "2025-05-22") == []


def test_the_published_average_is_cumulative_not_per_game():
    """The whole oracle rests on this. If AVG were the single game's rate,
    every one of its findings would be an artifact."""
    import glob
    import json

    seasons, _ = mod._load(REPO_ROOT / "games")
    # Take a well-populated person-season and confirm the published figure
    # tracks the RUNNING quotient rather than the game's own.
    key = max(seasons, key=lambda k: len(seasons[k]))
    rows = sorted(seasons[key])
    at_bats = hits = 0
    checked = 0
    for _date, _gid, ab, h, avg in rows[:10]:
        at_bats += ab
        hits += h
        if not at_bats:
            continue
        assert abs(hits / at_bats - float(avg)) <= mod.TOLERANCE
        checked += 1
    assert checked >= 5


# --- doubleheaders make the accumulation order ambiguous --------------------
#
# Two games share a date and nothing in the file says which was played first,
# so accumulating in filename order manufactures a divergence where the data
# is fine. 354 of 1,893 person-seasons contain a same-date pair. The
# published averages settle it: only one order reproduces both rows.


def _row(date, gid, ab, h, avg):
    return (date, gid, ab, h, avg)


def test_a_doubleheader_reconciles_in_whichever_order_works():
    # Played g2 first (2-for-4 = .500), then g1 (1-for-4, cumulative 3/8
    # = .375). Presented in the other order, which naive sorting would use.
    rows = [
        _row("2025-06-01", "g1", 4, 1, ".375"),
        _row("2025-06-01", "g2", 4, 2, ".500"),
    ]
    assert mod._first_divergence(rows) is None


def test_a_real_divergence_inside_a_doubleheader_is_still_reported():
    # No ordering of these reproduces the published figures, so the oracle
    # must still report rather than permuting its way out of a real gap.
    rows = [
        _row("2025-06-01", "g1", 4, 1, ".900"),
        _row("2025-06-01", "g2", 4, 2, ".950"),
    ]
    assert mod._first_divergence(rows) is not None


def test_an_ordinary_season_is_unaffected():
    rows = [
        _row("2025-05-20", "a", 3, 1, ".333"),
        _row("2025-05-21", "b", 4, 2, ".429"),
        _row("2025-05-22", "c", 6, 2, ".385"),
    ]
    assert mod._first_divergence(rows) is None


# --- the rounding mode is the source's, not Python's -----------------------
#
# `round(0.3125, 3)` is 0.312 -- Python rounds half to even. The league's
# scorer rounds half up and publishes .313. Rounding our own side before
# comparing therefore failed every exact-half quotient, and 5/16, 15/48,
# 20/64 and 18/32 are ordinary denominators in a short season: 114 of the
# then-215 reported divergences were this check's arithmetic rather than the
# corpus's. The comparison is against the UNROUNDED quotient now, at half a
# unit in the published figure's last place.


def test_an_exact_rounding_tie_reconciles_either_way():
    """A quotient of .3125 sits exactly half a unit from both .312 and .313.
    The source's rounding mode is not documented, so both are accepted --
    guessing one would be inventing a fact about the scoring software."""
    for published in (".312", ".313"):
        assert mod._row_reconciles(64, 20, published), published
        assert mod._row_reconciles(16, 5, published), published
    for published in (".562", ".563"):
        assert mod._row_reconciles(32, 18, published), published


def test_the_tie_tolerance_does_not_swallow_a_real_shortfall():
    """Half a unit and not a hair more. .3125 against .314 is one full unit
    out and must still be reported, or the fix would have bought its 114
    recoveries by going blind."""
    assert not mod._row_reconciles(64, 20, ".314")
    assert not mod._row_reconciles(64, 20, ".311")
    # The smallest real gap the published precision can express.
    assert mod._row_reconciles(100, 30, ".300")
    assert not mod._row_reconciles(100, 30, ".301")


def test_the_real_corpus_residual_is_the_2025_cluster_not_a_rounding_artifact():
    """Regression guard on the measurement the fix is justified by. If this
    number moves, the check changed -- read why before believing the new one."""
    seasons, _ = mod._load(REPO_ROOT / "games")
    diverging = [k for k, rows in seasons.items() if mod._first_divergence(rows)]
    assert len(diverging) == 90, (
        f"{len(diverging)} diverging person-seasons, expected 90 -- the AVG "
        f"residual moved; see issue #42 for the 2025 May-27/Jun-21 cluster"
    )
