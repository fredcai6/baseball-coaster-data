"""End-to-end tests for the g4 orchestrator/CLI (schedule -> paced fetch ->
archive, composed).

Everything here runs against an INJECTED fake transport (never real
``requests``) and a fake clock/sleep pair (same technique as g2's
``test_fetcher.py``), so pacing logic genuinely runs but the whole suite
still completes in well under a second. No test in this module calls real
network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bc_pipeline import schedule
from bc_pipeline.archive import load_checkpoint
from bc_pipeline.config import PipelineConfig
from bc_pipeline.fetch import run_pipeline
from bc_pipeline.fetcher import FetchResponse

SCHEDULE_URL = schedule.build_schedule_urls([2026])[0]
BOX_A = f"{schedule.DEFAULT_BASE_URL}/sports/bsb/2026/boxscores/20260401_aaa1.xml"
BOX_B = f"{schedule.DEFAULT_BASE_URL}/sports/bsb/2026/boxscores/20260402_bbb2.xml"
BOX_C = f"{schedule.DEFAULT_BASE_URL}/sports/bsb/2026/boxscores/20260403_ccc3.xml"

SCHEDULE_HTML = """
<div class="card event-row result" data-boxscore="/sports/bsb/2026/boxscores/20260401_aaa1.xml"></div>
<div class="card event-row result" data-boxscore="/sports/bsb/2026/boxscores/20260402_bbb2.xml"></div>
<div class="card event-row result" data-boxscore="/sports/bsb/2026/boxscores/20260403_ccc3.xml"></div>
"""

_BOX_BODY = "<html>a perfectly normal, non-empty boxscore body for testing</html>"


class FakeClock:
    """Same technique as g2's ``test_fetcher.py``: clock advances only on sleep()."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._t += seconds


def make_transport(response_map: dict[str, FetchResponse], call_log: list[str]):
    """A fake transport keyed by exact URL, logging every call it receives.

    Raises if asked for a URL that isn't in ``response_map`` -- a test bug
    (unexpected fetch) should fail loudly, not silently return nonsense.
    """

    def transport(url: str) -> FetchResponse:
        call_log.append(url)
        if url not in response_map:
            raise AssertionError(f"unexpected fetch for url not in response_map: {url}")
        return response_map[url]

    return transport


def make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        seasons=[2026],
        min_interval_seconds=10,
        jitter_seconds=0,
        archive_root=str(tmp_path / "archive"),
        checkpoint_path=str(tmp_path / "archive" / "checkpoint.json"),
    )


def make_response_map(box_a_response: FetchResponse | None = None) -> dict[str, FetchResponse]:
    return {
        SCHEDULE_URL: FetchResponse(status_code=200, body=SCHEDULE_HTML),
        BOX_A: box_a_response or FetchResponse(status_code=200, body=_BOX_BODY + " a"),
        BOX_B: FetchResponse(status_code=200, body=_BOX_BODY + " b"),
        BOX_C: FetchResponse(status_code=200, body=_BOX_BODY + " c"),
    }


def run(config, response_map, call_log, clock, **kwargs):
    transport = make_transport(response_map, call_log)
    return run_pipeline(
        config,
        transport,
        sleep_fn=clock.sleep,
        clock_fn=clock.now,
        jitter_fn=lambda lo, hi: 0,
        wall_clock_fn=lambda: 1_700_000_000.0,
        print_fn=lambda _msg: None,
        **kwargs,
    )


# --- (a) normal bounded run: --limit caps fetches, archives + checkpoints ---


def test_normal_bounded_run_respects_limit_and_archives(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    call_log: list[str] = []

    result = run(config, make_response_map(), call_log, FakeClock(), limit=2)

    assert result.fetched == [BOX_A, BOX_B]
    assert result.skipped_already_done == []
    assert result.challenge is None

    checkpoint = load_checkpoint(config.checkpoint_path)
    assert set(checkpoint) == {BOX_A, BOX_B}
    assert all(entry["status"] == "done" for entry in checkpoint.values())
    # BOX_C was never even requested -- limit stops further fetches outright.
    assert BOX_C not in call_log


# --- (b) idempotency end-to-end: second run against a full checkpoint ------


def test_second_run_against_same_checkpoint_fetches_zero_new(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    call_log: list[str] = []
    response_map = make_response_map()

    first = run(config, response_map, call_log, FakeClock(), limit=None)
    assert first.fetched == [BOX_A, BOX_B, BOX_C]

    call_log.clear()
    second = run(config, response_map, call_log, FakeClock(), limit=None)

    assert second.fetched == []
    assert set(second.skipped_already_done) == {BOX_A, BOX_B, BOX_C}
    # Schedule page is refetched (cheap, always walked); no boxscore is.
    assert BOX_A not in call_log and BOX_B not in call_log and BOX_C not in call_log
    assert SCHEDULE_URL in call_log


def test_same_bounded_limit_rerun_against_exhausted_backlog_fetches_zero(
    tmp_path: Path,
) -> None:
    """Literal acceptance-criterion-2 shape: identical ``--limit N`` args on
    both invocations (not ``None`` then a different number), where N is
    large enough to exhaust the fixture's whole boxscore backlog on the
    first run. The second run, with the SAME limit, must report 0 fetched.

    This is the compensating fake-fetcher test for a real gap discovered
    live in issue #18: ``--limit`` only counts actual fetches (checkpoint-
    skipped URLs don't count against it and don't stop the loop), so a
    same-args rerun against a REAL season -- whose backlog vastly exceeds
    any small bounded run -- advances into unfetched games instead of
    reporting 0. The criterion's literal wording ("a second run with same
    args fetches nothing new") is true once the reachable backlog is
    exhausted, which this test proves with a small, fully-exhaustible fixture
    rather than a live crawl of ~1000+ real games. See README's "Raw archive
    & fetching" section, "``--limit`` is a continue-crawl bound, not a
    fetch-count assertion" paragraph, for the caller-facing statement of
    this same fact.
    """
    config = make_config(tmp_path)
    call_log: list[str] = []
    response_map = make_response_map()

    # limit=5 > the fixture's 3 boxscore URLs, so the backlog is fully
    # exhausted on the first run -- nothing is left for a same-args rerun
    # to advance into.
    first = run(config, response_map, call_log, FakeClock(), limit=5)
    assert first.fetched == [BOX_A, BOX_B, BOX_C]

    call_log.clear()
    second = run(config, response_map, call_log, FakeClock(), limit=5)

    assert second.fetched == []
    assert set(second.skipped_already_done) == {BOX_A, BOX_B, BOX_C}
    assert BOX_A not in call_log and BOX_B not in call_log and BOX_C not in call_log


def test_limit_does_not_count_already_done_urls_against_it(tmp_path: Path) -> None:
    """`--limit 1` on a fully-populated checkpoint must report 0 fetched, not
    refuse to run because URLs were "already limited out"."""
    config = make_config(tmp_path)
    call_log: list[str] = []
    response_map = make_response_map()

    run(config, response_map, call_log, FakeClock(), limit=None)
    call_log.clear()

    second = run(config, response_map, call_log, FakeClock(), limit=1)

    assert second.fetched == []
    assert set(second.skipped_already_done) == {BOX_A, BOX_B, BOX_C}


# --- (c) mid-run ChallengeDetected stops cleanly ----------------------------


def test_challenge_mid_run_stops_cleanly_and_preserves_partial_checkpoint(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    call_log: list[str] = []
    response_map = make_response_map()
    # BOX_B looks like a WAF challenge (202).
    response_map[BOX_B] = FetchResponse(status_code=202, body="please wait")

    result = run(config, response_map, call_log, FakeClock(), limit=None)

    assert result.challenge is not None
    assert result.challenge.url == BOX_B
    assert result.fetched == [BOX_A]
    # The run stopped before ever attempting BOX_C.
    assert BOX_C not in call_log

    checkpoint = load_checkpoint(config.checkpoint_path)
    assert set(checkpoint) == {BOX_A}
    assert checkpoint[BOX_A]["status"] == "done"


def test_challenge_never_retries_internally(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    call_log: list[str] = []
    response_map = make_response_map()
    response_map[BOX_A] = FetchResponse(status_code=202, body="please wait")

    run(config, response_map, call_log, FakeClock(), limit=None)

    assert call_log.count(BOX_A) == 1


# --- (d) kill-and-resume: two invocations, no re-fetch of completed URLs ---


def test_kill_and_resume_does_not_refetch_completed_urls(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    call_log: list[str] = []
    response_map = make_response_map()

    first = run(config, response_map, call_log, FakeClock(), limit=1)
    assert first.fetched == [BOX_A]

    call_log.clear()
    second = run(config, response_map, call_log, FakeClock(), limit=10)

    assert second.fetched == [BOX_B, BOX_C]
    assert BOX_A not in call_log  # never re-fetched

    checkpoint = load_checkpoint(config.checkpoint_path)
    assert set(checkpoint) == {BOX_A, BOX_B, BOX_C}


# --- --dry-run: walks schedules, never calls transport for boxscore URLs ---


def test_dry_run_reports_planned_urls_without_fetching_any_boxscore(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    call_log: list[str] = []

    result = run(config, make_response_map(), call_log, FakeClock(), dry_run=True)

    assert result.planned == [BOX_A, BOX_B, BOX_C]
    assert result.fetched == []
    # Only the schedule page was fetched -- no boxscore URL touched the transport.
    assert call_log == [SCHEDULE_URL]

    # Nothing archived, no checkpoint written.
    assert not (tmp_path / "archive" / "checkpoint.json").exists()


def test_dry_run_excludes_already_done_urls_from_planned(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    call_log: list[str] = []
    response_map = make_response_map()

    run(config, response_map, call_log, FakeClock(), limit=1)  # archives BOX_A
    call_log.clear()

    result = run(config, response_map, call_log, FakeClock(), dry_run=True)

    assert result.planned == [BOX_B, BOX_C]
    assert BOX_A not in call_log
    assert BOX_B not in call_log and BOX_C not in call_log  # dry-run touches no boxscore


# --- pacing genuinely runs (not bypassed by the orchestrator) --------------


def test_orchestrator_paces_between_fetches(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    call_log: list[str] = []
    clock = FakeClock()

    run(config, make_response_map(), call_log, clock, limit=None)

    # 4 total fetches (1 schedule + 3 boxscore) -> 3 gaps, each paced.
    assert len(clock.sleep_calls) == 3
    assert all(s >= config.min_interval_seconds for s in clock.sleep_calls)
