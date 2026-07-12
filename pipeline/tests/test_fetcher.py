"""Tests for the paced fetcher (g2: config + paced fetcher + 202/challenge backoff).

Everything here runs against an injected fake transport, fake clock, and fake
sleep function -- no real network access and no real time.sleep anywhere in
this module, so the whole suite must complete in well under a second.
"""

from __future__ import annotations

import pytest

from bc_pipeline.config import PipelineConfig
from bc_pipeline.fetcher import (
    CHALLENGE_BACKOFF_SECONDS,
    ChallengeDetected,
    FetchResponse,
    FetchResult,
    PacedFetcher,
)


class FakeClock:
    """A fully injectable clock: ``now()`` advances only when ``sleep()`` is called."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._t += seconds


def make_transport(responses):
    """Return a fake transport callable that yields ``responses`` in order."""
    it = iter(responses)

    def transport(url: str) -> FetchResponse:
        return next(it)

    return transport


def test_paced_fetcher_enforces_minimum_interval_between_call_starts() -> None:
    clock = FakeClock()
    responses = [FetchResponse(status_code=200, body="<html>a perfectly normal, non-empty response body</html>")] * 4
    transport = make_transport(responses)
    config = PipelineConfig(min_interval_seconds=10, jitter_seconds=0)
    fetcher = PacedFetcher(
        transport=transport,
        config=config,
        sleep_fn=clock.sleep,
        clock_fn=clock.now,
        jitter_fn=lambda lo, hi: 0,
    )

    for _ in range(4):
        fetcher.fetch("https://example.test/a")

    # 4 calls -> 3 gaps, each gap must be >= min_interval_seconds.
    assert len(fetcher.call_start_times) == 4
    gaps = [
        b - a
        for a, b in zip(fetcher.call_start_times, fetcher.call_start_times[1:])
    ]
    assert all(gap >= config.min_interval_seconds for gap in gaps)
    # First call has no prior call to pace against, so it triggers no sleep;
    # only the 3 subsequent calls should have paced (one sleep call each).
    assert len(clock.sleep_calls) == 3


def test_paced_fetcher_respects_jitter_upper_bound() -> None:
    clock = FakeClock()
    responses = [FetchResponse(status_code=200, body="<html>a perfectly normal, non-empty response body</html>")] * 3
    transport = make_transport(responses)
    config = PipelineConfig(min_interval_seconds=10, jitter_seconds=2)
    fetcher = PacedFetcher(
        transport=transport,
        config=config,
        sleep_fn=clock.sleep,
        clock_fn=clock.now,
        jitter_fn=lambda lo, hi: hi,  # force max jitter deterministically
    )

    for _ in range(3):
        fetcher.fetch("https://example.test/a")

    gaps = [
        b - a
        for a, b in zip(fetcher.call_start_times, fetcher.call_start_times[1:])
    ]
    for gap in gaps:
        assert config.min_interval_seconds <= gap <= config.min_interval_seconds + config.jitter_seconds + 1e-9


def test_paced_fetcher_does_not_wait_if_enough_time_already_elapsed() -> None:
    clock = FakeClock()
    responses = [FetchResponse(status_code=200, body="<html>a perfectly normal, non-empty response body</html>")] * 2
    transport = make_transport(responses)
    config = PipelineConfig(min_interval_seconds=10, jitter_seconds=0)
    fetcher = PacedFetcher(
        transport=transport,
        config=config,
        sleep_fn=clock.sleep,
        clock_fn=clock.now,
        jitter_fn=lambda lo, hi: 0,
    )

    fetcher.fetch("https://example.test/a")
    clock._t += 50  # plenty of real time already passed externally
    fetcher.fetch("https://example.test/a")

    assert clock.sleep_calls == [] or all(s == 0 for s in clock.sleep_calls)


def test_normal_response_returns_fetch_result_with_body_and_timestamp() -> None:
    clock = FakeClock()
    transport = make_transport(
        [FetchResponse(status_code=200, body="<html>a perfectly normal, non-empty response body</html>")]
    )
    config = PipelineConfig(min_interval_seconds=10, jitter_seconds=0)
    fetcher = PacedFetcher(
        transport=transport,
        config=config,
        sleep_fn=clock.sleep,
        clock_fn=clock.now,
        jitter_fn=lambda lo, hi: 0,
        wall_clock_fn=lambda: 1234.5,
    )

    result = fetcher.fetch("https://example.test/a")

    assert isinstance(result, FetchResult)
    assert result.url == "https://example.test/a"
    assert result.status_code == 200
    assert result.body == "<html>a perfectly normal, non-empty response body</html>"
    assert result.fetched_at == 1234.5


def test_status_202_raises_challenge_detected() -> None:
    clock = FakeClock()
    transport = make_transport([FetchResponse(status_code=202, body="please wait")])
    config = PipelineConfig(min_interval_seconds=10, jitter_seconds=0)
    fetcher = PacedFetcher(
        transport=transport, config=config, sleep_fn=clock.sleep, clock_fn=clock.now,
    )

    with pytest.raises(ChallengeDetected):
        fetcher.fetch("https://example.test/a")


def test_goku_props_marker_raises_challenge_detected() -> None:
    clock = FakeClock()
    body = "<html><script>window.gokuProps = {};</script></html>"
    transport = make_transport([FetchResponse(status_code=200, body=body)])
    config = PipelineConfig(min_interval_seconds=10, jitter_seconds=0)
    fetcher = PacedFetcher(
        transport=transport, config=config, sleep_fn=clock.sleep, clock_fn=clock.now,
    )

    with pytest.raises(ChallengeDetected):
        fetcher.fetch("https://example.test/a")


def test_empty_body_raises_challenge_detected() -> None:
    clock = FakeClock()
    transport = make_transport([FetchResponse(status_code=200, body="   ")])
    config = PipelineConfig(min_interval_seconds=10, jitter_seconds=0)
    fetcher = PacedFetcher(
        transport=transport, config=config, sleep_fn=clock.sleep, clock_fn=clock.now,
    )

    with pytest.raises(ChallengeDetected):
        fetcher.fetch("https://example.test/a")


def test_challenge_detection_never_retries_internally() -> None:
    clock = FakeClock()
    calls = []

    def transport(url: str) -> FetchResponse:
        calls.append(url)
        return FetchResponse(status_code=202, body="please wait")

    config = PipelineConfig(min_interval_seconds=10, jitter_seconds=0)
    fetcher = PacedFetcher(
        transport=transport, config=config, sleep_fn=clock.sleep, clock_fn=clock.now,
    )

    with pytest.raises(ChallengeDetected):
        fetcher.fetch("https://example.test/a")

    assert len(calls) == 1  # the seam calls the transport exactly once, no retry
    assert clock.sleep_calls == []  # no internal backoff sleep either


def test_challenge_backoff_constant_is_at_least_sixty_seconds() -> None:
    assert CHALLENGE_BACKOFF_SECONDS >= 60
