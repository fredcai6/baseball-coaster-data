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
        assert abs(round(hits / at_bats, 3) - float(avg)) <= mod.TOLERANCE
        checked += 1
    assert checked >= 5
