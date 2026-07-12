"""Tests for the pure schedule-page walker (g1: schedule walker, pure parse, no I/O).

These tests run entirely against the local fixture in
``pipeline/tests/fixtures/schedule_fixture.txt`` (a real pioneerleague.com/
PrestoSports season-schedule page snapshot, stored with a non-``.html``
extension deliberately -- the repo's root ``.gitignore`` excludes ``*.html``
repo-wide, by design, to keep raw scraped HTML out of git per issue #16; a
test fixture that is source-controlled test data, not raw scrape output,
needs a different extension so it is not silently swallowed by that same
rule and lost on a fresh clone / in CI). No network calls are made anywhere
in this module or the module under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bc_pipeline.schedule import build_schedule_urls, final_boxscore_urls

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "schedule_fixture.txt"


@pytest.fixture(scope="module")
def schedule_html() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8", errors="replace")


def test_final_boxscore_urls_returns_known_count_from_fixture(schedule_html: str) -> None:
    # Frozen fixture assertion (not a production hard-code): as of the
    # 2026-07-10 snapshot this fixture (a team-scoped 2026 schedule page)
    # contains exactly 45 unique data-boxscore URLs, all marking final games.
    urls = final_boxscore_urls(schedule_html)
    assert len(urls) == 45
    assert len(set(urls)) == 45  # all unique


def test_final_boxscore_urls_match_expected_pattern(schedule_html: str) -> None:
    urls = final_boxscore_urls(schedule_html)
    assert all(
        u.startswith("/sports/bsb/2026/boxscores/") and u.endswith(".xml")
        for u in urls
    )


def test_final_boxscore_urls_includes_known_sample_url(schedule_html: str) -> None:
    urls = final_boxscore_urls(schedule_html)
    assert "/sports/bsb/2026/boxscores/20260519_wmxc.xml" in urls


def test_final_boxscore_urls_ignores_rows_without_data_boxscore() -> None:
    # A future/upcoming event-row has no data-boxscore attribute at all (the
    # boxscore URL isn't published as a data attribute for non-final games in
    # this markup shape) and must not appear in the result.
    html = """
    <div class="card w-100 event-row away upcoming schedule-next-event-indicator">
        <a href="/sports/bsb/2026/boxscores/20260830_zzzz.xml">Box Score</a>
    </div>
    <div class="card w-100 event-row away result has-recap no-leaders"
         data-boxscore="/sports/bsb/2026/boxscores/20260519_wmxc.xml">
    </div>
    """
    assert final_boxscore_urls(html) == [
        "/sports/bsb/2026/boxscores/20260519_wmxc.xml"
    ]


def test_final_boxscore_urls_ignores_data_boxscore_off_event_row() -> None:
    # data-boxscore only counts as a final-game signal when it's on an
    # event-row element -- a stray attribute elsewhere should not be picked up.
    html = """
    <div class="some-other-widget" data-boxscore="/sports/bsb/2026/boxscores/20260101_fake.xml"></div>
    """
    assert final_boxscore_urls(html) == []


def test_final_boxscore_urls_empty_html_returns_empty_list() -> None:
    assert final_boxscore_urls("") == []


def test_build_schedule_urls_default_covers_2026_2025_2024_in_order() -> None:
    urls = build_schedule_urls()
    assert urls == [
        "https://www.pioneerleague.com/sports/bsb/2026/schedule",
        "https://www.pioneerleague.com/sports/bsb/2025/schedule",
        "https://www.pioneerleague.com/sports/bsb/2024/schedule",
    ]


def test_build_schedule_urls_accepts_explicit_years() -> None:
    assert build_schedule_urls([2025]) == [
        "https://www.pioneerleague.com/sports/bsb/2025/schedule",
    ]
