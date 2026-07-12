"""Paced fetcher with 202/challenge detection (g2).

This module is the fetch-side network abstraction other gates build on. It is
fully testable without any real network access: the fetcher never calls a
transport directly -- it only calls an INJECTED transport callable, so
production code can inject a real ``requests``/``httpx`` call (g4) while
tests inject a fake one (per the issue's "develop against a fake fetcher
first" mandate).

Responsibilities, precisely scoped to this gate:
    * Enforce a minimum interval (plus jitter) between the START of any two
      successive calls through the paced-fetcher seam, using an injectable
      clock/sleep pair -- never ``time.sleep``/``time.monotonic`` called
      directly without a substitution point.
    * Detect a "challenge" response (HTTP 202, the AWS WAF JS-challenge page
      marker ``window.gokuProps``, or an empty/near-empty body) and raise
      :class:`ChallengeDetected` rather than returning it as a normal fetch.
    * Never retry a detected challenge internally. Backing off >= 60s and
      resuming from checkpoint is a LATER gate's job (the caller's), not this
      seam's -- this seam only detects and signals.

Explicitly NOT this gate's job: writing raw HTML to the archive, tracking a
checkpoint file, or wiring a CLI/real transport (see g3/g4).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from bc_pipeline.config import PipelineConfig

#: A caller-owned constant documenting the required challenge backoff floor.
#: This gate does not sleep for it -- the caller (a later gate) does.
CHALLENGE_BACKOFF_SECONDS: float = 60.0

#: A body this short (after stripping whitespace) is treated as "empty/near-
#: empty" and therefore a challenge signal, even with a 200 status code.
NEAR_EMPTY_BODY_THRESHOLD_CHARS: int = 32

#: The AWS WAF JS-challenge page signature observed during recon.
WAF_CHALLENGE_MARKER: str = "window.gokuProps"


@dataclass(frozen=True)
class FetchResponse:
    """What an injected transport callable must return for one URL fetch.

    This is the seam's input-side contract: any transport (fake in tests,
    real ``requests``/``httpx`` call in production) hands back one of these.
    """

    status_code: int
    body: str


@dataclass(frozen=True)
class FetchResult:
    """What :meth:`PacedFetcher.fetch` returns for a normal (non-challenge) fetch.

    This is the "fetch result" shape g3 (raw archive/checkpoint writer)
    depends on:
        url: the URL that was fetched.
        status_code: the HTTP status code of the response.
        body: the raw response body (decoded text) -- what g3 archives.
        fetched_at: wall-clock timestamp (seconds since the epoch, i.e. what
            ``time.time()`` returns) of when this fetch completed, for g3 to
            use in archive filenames/metadata.
    """

    url: str
    status_code: int
    body: str
    fetched_at: float


class ChallengeDetected(Exception):
    """Raised when a fetch response looks like a 202/WAF-challenge/empty page.

    The caller (a later gate) is responsible for deciding to back off
    (>= :data:`CHALLENGE_BACKOFF_SECONDS`) and resume from checkpoint -- this
    exception only carries enough information to make that decision; it is
    never retried internally by the seam that raises it.
    """

    def __init__(self, url: str, status_code: int, reason: str, body_snippet: str) -> None:
        super().__init__(f"Challenge detected for {url!r} ({reason}): status={status_code}")
        self.url = url
        self.status_code = status_code
        self.reason = reason
        self.body_snippet = body_snippet


class Transport(Protocol):
    """The injectable seam: anything callable with this shape is a transport."""

    def __call__(self, url: str) -> FetchResponse: ...


def _is_challenge(response: FetchResponse) -> str | None:
    """Return a short reason string if ``response`` looks like a challenge, else None."""
    if response.status_code == 202:
        return "status_202"
    if WAF_CHALLENGE_MARKER in response.body:
        return "waf_challenge_marker"
    if len(response.body.strip()) < NEAR_EMPTY_BODY_THRESHOLD_CHARS:
        return "near_empty_body"
    return None


class PacedFetcher:
    """Fetches URLs through an injectable transport, paced and challenge-aware.

    Both the sleep function and the time-source function are injectable
    (defaulting to ``time.sleep``/``time.monotonic``) specifically so unit
    tests can substitute fakes and assert exact pacing intervals without a
    test literally taking >= 10 real seconds.
    """

    def __init__(
        self,
        transport: Transport,
        config: PipelineConfig,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
        jitter_fn: Callable[[float, float], float] = random.uniform,
        wall_clock_fn: Callable[[], float] = time.time,
    ) -> None:
        self._transport = transport
        self._config = config
        self._sleep_fn = sleep_fn
        self._clock_fn = clock_fn
        self._jitter_fn = jitter_fn
        self._wall_clock_fn = wall_clock_fn
        self._last_call_start: float | None = None
        #: Recorded start time (per ``clock_fn``) of every ``fetch`` call, in
        #: order -- test-observability seam for asserting pacing intervals.
        self.call_start_times: list[float] = []

    def fetch(self, url: str) -> FetchResult:
        """Fetch ``url`` through the injected transport, paced and challenge-checked.

        Raises :class:`ChallengeDetected` if the response looks like a
        202/WAF-challenge/empty-body page. Never retries internally.
        """
        self._wait_for_pacing()
        now = self._clock_fn()
        self._last_call_start = now
        self.call_start_times.append(now)

        response = self._transport(url)

        reason = _is_challenge(response)
        if reason is not None:
            raise ChallengeDetected(
                url=url,
                status_code=response.status_code,
                reason=reason,
                body_snippet=response.body[:200],
            )

        return FetchResult(
            url=url,
            status_code=response.status_code,
            body=response.body,
            fetched_at=self._wall_clock_fn(),
        )

    def _wait_for_pacing(self) -> None:
        if self._last_call_start is None:
            return  # first call: nothing to pace against
        jitter = self._jitter_fn(0, self._config.jitter_seconds)
        required_gap = self._config.min_interval_seconds + jitter
        elapsed = self._clock_fn() - self._last_call_start
        remaining = required_gap - elapsed
        if remaining > 0:
            self._sleep_fn(remaining)
