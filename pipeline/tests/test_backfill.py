"""Tests for bc_pipeline.backfill: the until-caught-up fetch+parse+replay+
commit driver (g1, issue #20).

Everything here runs against an INJECTED fake transport and a fake
clock/sleep pair -- same technique as ``test_fetch.py`` -- so pacing logic
genuinely runs, no test sleeps for real, and no test invokes real git.

Real (large) sample HTML from the top-level ``tests/samples/`` directory is
reused here as the "final, parseable, replayable" boxscore body (it is
already proven schema-valid and replayable=True by ``tests/test_parse.py``/
``tests/test_replay.py``) -- game_id/season for the SYNTHETIC boxscore URLs
used below come from the URL/schedule-season this module assigns, not from
that sample's own embedded date, since this suite exercises the backfill
ORCHESTRATION (season loop, batching, write-once), not date-parsing
correctness (already covered elsewhere).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bc_pipeline import backfill, schedule
from bc_pipeline.config import PipelineConfig
from bc_pipeline.fetcher import ChallengeDetected, FetchResponse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOP_SAMPLES_DIR = _REPO_ROOT / "tests" / "samples"

FINAL_HTML = (_TOP_SAMPLES_DIR / "boxscore_20260709_final.html").read_text(encoding="utf-8")
NON_FINAL_HTML = (_TOP_SAMPLES_DIR / "boxscore_20260710_today.html").read_text(encoding="utf-8")

# Has a PBP pane (so NOT NonFinalPageError) but no <div class="date">, so
# parse_game blows up with a plain ValueError partway through -- the generic
# "any other parse exception" path.
BROKEN_FINAL_HTML = '<html><body><section id="pbp-inning-1"></section></body></html>'


def _season_schedule_url(season: int) -> str:
    return schedule.build_schedule_urls([season])[0]


def _box_url(season: int, slug: str) -> str:
    return f"{schedule.DEFAULT_BASE_URL}/sports/bsb/{season}/boxscores/{slug}.xml"


def _schedule_html(box_slugs: list[str], season: int) -> str:
    rows = "\n".join(
        f'<div class="card event-row result" '
        f'data-boxscore="/sports/bsb/{season}/boxscores/{slug}.xml"></div>'
        for slug in box_slugs
    )
    return f"<html><body>{rows}</body></html>"


class FakeClock:
    """Same technique as g2's ``test_fetcher.py``/g4's ``test_fetch.py``:
    clock advances only on sleep()."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._t += seconds


def make_transport(response_map: dict[str, FetchResponse], call_log: list[str]):
    def transport(url: str) -> FetchResponse:
        call_log.append(url)
        if url not in response_map:
            raise AssertionError(f"unexpected fetch for url not in response_map: {url}")
        return response_map[url]

    return transport


def make_config(tmp_path: Path, seasons: list[int]) -> PipelineConfig:
    return PipelineConfig(
        seasons=seasons,
        min_interval_seconds=10,
        jitter_seconds=0,
        archive_root=str(tmp_path / "archive"),
        checkpoint_path=str(tmp_path / "archive" / "checkpoint.json"),
    )


def run(
    config,
    response_map,
    call_log,
    clock,
    repo_root,
    commits,
    **kwargs,
):
    transport = make_transport(response_map, call_log)

    def commit_fn(paths, message):
        commits.append((tuple(str(p) for p in paths), message))

    return backfill.run_backfill(
        config,
        transport,
        repo_root=repo_root,
        sleep_fn=clock.sleep,
        clock_fn=clock.now,
        jitter_fn=lambda lo, hi: 0,
        wall_clock_fn=lambda: 1_700_000_000.0,
        print_fn=lambda _msg: None,
        commit_fn=commit_fn,
        **kwargs,
    )


# --- (a) until-caught-up: every discoverable final game gets processed -----


def test_until_caught_up_processes_every_discoverable_final_game(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    box_slugs = [f"20260401_g{i}" for i in range(1, 8)]  # 7 games, no --limit
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(box_slugs, 2026)
        ),
    }
    for slug in box_slugs:
        response_map[_box_url(2026, slug)] = FetchResponse(status_code=200, body=FINAL_HTML)

    call_log: list[str] = []
    commits: list = []
    result = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)

    assert result.challenge is None
    summary = result.seasons[2026]
    assert summary.fetched == 7
    assert summary.parsed == 7
    assert summary.replayable == 7
    assert summary.parse_failed == 0
    assert summary.non_final == 0

    for slug in box_slugs:
        assert (tmp_path / "games" / "2026" / f"{slug}.json").exists()


# --- (b) NonFinalPageError is recorded, not stubbed -------------------------


def test_non_final_page_recorded_not_stubbed(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_final1", "20260402_notfinal"], 2026)
        ),
        _box_url(2026, "20260401_final1"): FetchResponse(status_code=200, body=FINAL_HTML),
        _box_url(2026, "20260402_notfinal"): FetchResponse(status_code=200, body=NON_FINAL_HTML),
    }

    call_log: list[str] = []
    commits: list = []
    result = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)

    summary = result.seasons[2026]
    assert summary.parsed == 1
    assert summary.non_final == 1

    non_final_outcomes = [g for g in result.games if g.outcome == "non_final"]
    assert len(non_final_outcomes) == 1
    assert non_final_outcomes[0].game_id == "20260402_notfinal"
    assert non_final_outcomes[0].reason  # a reason string was recorded
    # non_final never got parsed this run -- no line counts to report.
    assert non_final_outcomes[0].events_count is None
    assert non_final_outcomes[0].unparsed_count is None

    # No stub file for the non-final game.
    assert not (tmp_path / "games" / "2026" / "20260402_notfinal.json").exists()
    assert (tmp_path / "games" / "2026" / "20260401_final1.json").exists()


def test_other_parse_exception_recorded_as_parse_failed(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_broken"], 2026)
        ),
        _box_url(2026, "20260401_broken"): FetchResponse(status_code=200, body=BROKEN_FINAL_HTML),
    }

    call_log: list[str] = []
    commits: list = []
    result = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)

    summary = result.seasons[2026]
    assert summary.parse_failed == 1
    assert summary.parsed == 0

    failed = [g for g in result.games if g.outcome == "parse_failed"]
    assert len(failed) == 1
    assert failed[0].reason  # raw is already kept by the fetch side; reason is recorded here
    # parse_failed never produced a parsed game -- no line counts to report.
    assert failed[0].events_count is None
    assert failed[0].unparsed_count is None

    assert not (tmp_path / "games" / "2026" / "20260401_broken.json").exists()


# --- (g2 rework) parsed outcomes carry line-level counts from meta.parse ---


def test_parsed_outcome_carries_unparsed_and_events_count(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=200, body=FINAL_HTML),
    }

    call_log: list[str] = []
    commits: list = []
    result = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)

    outcome = result.games[0]
    assert outcome.outcome == "parsed"
    assert outcome.events_count is not None
    assert outcome.unparsed_count is not None

    written = json.loads(
        (tmp_path / "games" / "2026" / "20260401_g1.json").read_text(encoding="utf-8")
    )
    assert outcome.events_count == written["meta"]["parse"]["events_count"]
    assert outcome.unparsed_count == written["meta"]["parse"]["unparsed_count"]

    # to_dict() round-trips both fields.
    outcome_dict = outcome.to_dict()
    assert outcome_dict["events_count"] == outcome.events_count
    assert outcome_dict["unparsed_count"] == outcome.unparsed_count


# --- (c) games/** write-once: repeat run skips, never overwrites ------------


def test_repeat_run_skips_already_committed_games(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1", "20260402_g2"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=200, body=FINAL_HTML),
        _box_url(2026, "20260402_g2"): FetchResponse(status_code=200, body=FINAL_HTML),
    }

    call_log: list[str] = []
    commits: list = []
    first = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)
    assert first.seasons[2026].parsed == 2
    assert first.seasons[2026].skipped_already_committed == 0

    committed_path = tmp_path / "games" / "2026" / "20260401_g1.json"
    before = committed_path.read_text(encoding="utf-8")

    call_log.clear()
    commits.clear()
    second = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)

    assert second.seasons[2026].skipped_already_committed == 2
    assert second.seasons[2026].parsed == 0
    # Never overwritten.
    assert committed_path.read_text(encoding="utf-8") == before
    # No boxscore was even re-fetched (fetch-side idempotency, already proven
    # in test_fetch.py, re-confirmed here at the backfill level).
    assert _box_url(2026, "20260401_g1") not in call_log
    assert _box_url(2026, "20260402_g2") not in call_log
    # And no commit was made for zero new files.
    assert commits == []


# --- (d) season order 2026 -> 2025 -> 2024 ----------------------------------


def test_season_order_2026_then_2025_then_2024(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026, 2025, 2024])
    response_map = {}
    for season in (2026, 2025, 2024):
        slug = f"{season}0401_g1"
        response_map[_season_schedule_url(season)] = FetchResponse(
            status_code=200, body=_schedule_html([slug], season)
        )
        response_map[_box_url(season, slug)] = FetchResponse(status_code=200, body=FINAL_HTML)

    call_log: list[str] = []
    commits: list = []
    result = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)

    assert list(result.seasons.keys()) == [2026, 2025, 2024]

    schedule_positions = {
        season: call_log.index(_season_schedule_url(season)) for season in (2026, 2025, 2024)
    }
    assert (
        schedule_positions[2026] < schedule_positions[2025] < schedule_positions[2024]
    ), call_log

    for season in (2026, 2025, 2024):
        assert (tmp_path / "games" / str(season) / f"{season}0401_g1.json").exists()


# --- (e) batching triggers a commit at the season boundary ------------------


def test_batching_commits_at_50_and_at_season_boundary(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    box_slugs = [f"20260401_g{i:02d}" for i in range(1, 61)]  # 60 games: one batch of 50 + 10
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(box_slugs, 2026)
        ),
    }
    for slug in box_slugs:
        response_map[_box_url(2026, slug)] = FetchResponse(status_code=200, body=FINAL_HTML)

    call_log: list[str] = []
    commits: list = []
    result = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)

    assert result.seasons[2026].parsed == 60
    assert len(commits) == 2
    (paths_1, message_1), (paths_2, message_2) = commits
    assert len(paths_1) == 50
    assert len(paths_2) == 10
    assert message_1 == "backfill(2026): games 1–50"
    assert message_2 == "backfill(2026): games 51–60"
    assert result.commits == [message_1, message_2]


def test_season_boundary_flushes_a_short_batch(tmp_path: Path) -> None:
    """A season with fewer than the batch size still gets exactly one commit
    at its own boundary, and the NEXT season starts its own numbering fresh."""
    config = make_config(tmp_path, seasons=[2026, 2025])
    response_map = {}
    for season, count in ((2026, 3), (2025, 2)):
        slugs = [f"{season}0401_g{i}" for i in range(1, count + 1)]
        response_map[_season_schedule_url(season)] = FetchResponse(
            status_code=200, body=_schedule_html(slugs, season)
        )
        for slug in slugs:
            response_map[_box_url(season, slug)] = FetchResponse(status_code=200, body=FINAL_HTML)

    call_log: list[str] = []
    commits: list = []
    result = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)

    assert len(commits) == 2
    assert commits[0][1] == "backfill(2026): games 1–3"
    assert commits[1][1] == "backfill(2025): games 1–2"


# --- challenge stops the run immediately, no further seasons walked --------


def test_challenge_stops_run_immediately_no_further_seasons(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026, 2025])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=202, body="please wait"),
    }

    call_log: list[str] = []
    commits: list = []
    result = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=None)

    assert result.challenge is not None
    assert isinstance(result.challenge, ChallengeDetected)
    assert _season_schedule_url(2025) not in call_log


# --- limit override caps total NEW fetches across the whole run ------------


def test_limit_caps_total_fetches_across_seasons(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026, 2025])
    response_map = {}
    for season in (2026, 2025):
        slugs = [f"{season}0401_g{i}" for i in range(1, 4)]
        response_map[_season_schedule_url(season)] = FetchResponse(
            status_code=200, body=_schedule_html(slugs, season)
        )
        for slug in slugs:
            response_map[_box_url(season, slug)] = FetchResponse(status_code=200, body=FINAL_HTML)

    call_log: list[str] = []
    commits: list = []
    result = run(config, response_map, call_log, FakeClock(), tmp_path, commits, limit=4)

    total_fetched = sum(s.fetched for s in result.seasons.values())
    assert total_fetched == 4


# --- CLI-level escalating challenge backoff (run_backfill_with_escalation) -


def test_escalation_backs_off_60_600_3600_then_stops_cleanly(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=202, body="please wait"),
    }

    call_log: list[str] = []
    commits: list = []
    escalation_sleeps: list[float] = []

    def commit_fn(paths, message):
        commits.append((tuple(str(p) for p in paths), message))

    transport = make_transport(response_map, call_log)
    result = backfill.run_backfill_with_escalation(
        config,
        transport,
        repo_root=tmp_path,
        sleep_fn=FakeClock().sleep,
        clock_fn=FakeClock().now,
        jitter_fn=lambda lo, hi: 0,
        wall_clock_fn=lambda: 1_700_000_000.0,
        print_fn=lambda _msg: None,
        commit_fn=commit_fn,
        escalation_sleep_fn=escalation_sleeps.append,
    )

    assert result.stopped_by_challenge
    assert escalation_sleeps == [60.0, 600.0, 3600.0]


def test_escalation_recovers_and_returns_clean_result_if_challenge_clears(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=200, body=FINAL_HTML),
    }

    call_log: list[str] = []
    escalation_sleeps: list[float] = []

    def commit_fn(paths, message):
        pass

    class _Toggle:
        def __init__(self):
            self.box_calls = 0

        def __call__(self, url: str) -> FetchResponse:
            call_log.append(url)
            if url == _box_url(2026, "20260401_g1"):
                self.box_calls += 1
                if self.box_calls <= 2:
                    return FetchResponse(status_code=202, body="please wait")
            if url not in response_map:
                raise AssertionError(f"unexpected url {url}")
            return response_map[url]

    result = backfill.run_backfill_with_escalation(
        config,
        _Toggle(),
        repo_root=tmp_path,
        sleep_fn=FakeClock().sleep,
        clock_fn=FakeClock().now,
        jitter_fn=lambda lo, hi: 0,
        wall_clock_fn=lambda: 1_700_000_000.0,
        print_fn=lambda _msg: None,
        commit_fn=commit_fn,
        escalation_sleep_fn=escalation_sleeps.append,
    )

    assert not result.stopped_by_challenge
    assert escalation_sleeps == [60.0, 600.0]
    assert result.seasons[2026].parsed == 1
