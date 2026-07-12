"""Fetch raw boxscore/PBP HTML from the source site to local disk.

# Implemented in issue #18/#19

g2 implemented the paced-fetcher-and-challenge-detection half in
:mod:`bc_pipeline.fetcher`; g3 implemented the raw-archive/checkpoint writer
in :mod:`bc_pipeline.archive`. This module (g4) is the composition gate: it
wires g1 (:mod:`bc_pipeline.schedule`), g2, and g3 into one runnable CLI
entrypoint, plus the one real-HTTP transport (:mod:`bc_pipeline.transport`)
allowed to exist in this whole issue.

Run it via::

    py -m bc_pipeline.fetch --limit 5
    py -m bc_pipeline.fetch --dry-run
    py -m bc_pipeline.fetch --config path/to/config.json --limit 20

(from the ``pipeline/`` directory, or anywhere with ``pipeline/`` on
``PYTHONPATH`` -- see README "Raw archive & fetching" for the exact
invocation this repo verifies against).

Orchestration/testability seam (:func:`run_pipeline`): it is the SAME
function the CLI's :func:`main` calls and that g5's live demo calls, with
only the ``transport`` argument swapped between a fake (tests, this gate)
and :func:`bc_pipeline.transport.real_transport` (production, g5) -- never a
separately written "demo path". Tests also substitute the pacing clock/sleep
functions (as g2's own tests do) so the fake-fetcher end-to-end suite runs in
well under a second despite exercising the real pacing logic.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence
from urllib.parse import urljoin

from bc_pipeline import archive, schedule
from bc_pipeline.config import PipelineConfig, load_config
from bc_pipeline.fetcher import (
    CHALLENGE_BACKOFF_SECONDS,
    ChallengeDetected,
    FetchResponse,
    FetchResult,
    PacedFetcher,
    Transport,
)

__all__ = [
    "CHALLENGE_BACKOFF_SECONDS",
    "ChallengeDetected",
    "FetchResponse",
    "FetchResult",
    "PacedFetcher",
    "Transport",
    "RunResult",
    "run_pipeline",
    "build_arg_parser",
    "main",
]


@dataclass
class RunResult:
    """Summary of one :func:`run_pipeline` invocation, for callers/tests.

    ``fetched``: boxscore URLs actually fetched-and-archived this run (counts
        against ``--limit``).
    ``skipped_already_done``: boxscore URLs the checkpoint already marked
        ``done`` -- never fetched, never counted against ``--limit``.
    ``planned``: in ``--dry-run`` mode only, the boxscore URLs that WOULD be
        fetched (checkpoint-filtered, limit-filtered); empty in a real run.
    ``schedule_urls``: season schedule-page URLs walked this run (always
        actually fetched -- see module docstring / README for why schedule
        pages are never dry-run-skipped).
    ``challenge``: the :class:`ChallengeDetected` that stopped this run, if
        any; ``None`` means the run completed (or hit its limit) cleanly.
    """

    fetched: list[str] = field(default_factory=list)
    skipped_already_done: list[str] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)
    schedule_urls: list[str] = field(default_factory=list)
    challenge: ChallengeDetected | None = None

    @property
    def stopped_by_challenge(self) -> bool:
        return self.challenge is not None


def _collect_boxscore_urls(
    fetcher: PacedFetcher,
    config: PipelineConfig,
    *,
    print_fn: Callable[[str], None],
) -> tuple[list[str], list[str], ChallengeDetected | None]:
    """Walk every season's schedule page and accumulate FINAL boxscore URLs.

    Schedule pages are fetched through the SAME ``PacedFetcher``/transport
    seam as boxscore pages (one pacing/challenge-detection policy for every
    request this run makes against the site -- the WAF does not distinguish
    schedule pages from boxscore pages, so neither should this pipeline's
    pacing). They are deliberately never archived/checkpointed: they are not
    the artifact this pipeline preserves (boxscore pages are), they are cheap
    to re-fetch (one page per season, a handful of seasons), and the season
    list they enumerate can change between runs (newly final games), so
    checkpoint-based idempotency for them would be actively wrong.

    Returns ``(schedule_urls_walked, boxscore_urls, challenge)``. If a
    challenge is detected while fetching a schedule page, walking stops
    immediately and the partial ``boxscore_urls`` collected so far (from
    already-walked seasons) is returned alongside the challenge -- season
    order is 2026-first (decision:live-demo-scope), so a challenge while
    walking preserves the most-relevant partial progress a caller could act
    on.
    """
    schedule_page_urls = schedule.build_schedule_urls(config.seasons)
    schedule_urls_walked: list[str] = []
    boxscore_urls: list[str] = []

    for schedule_url in schedule_page_urls:
        try:
            result = fetcher.fetch(schedule_url)
        except ChallengeDetected as challenge:
            print_fn(
                f"[CHALLENGE] Stopping run: {challenge.reason} while fetching "
                f"schedule page {challenge.url!r} (status={challenge.status_code}). "
                f"Back off >= {CHALLENGE_BACKOFF_SECONDS:.0f}s before re-running; "
                "the checkpoint reflects only what completed before this."
            )
            return schedule_urls_walked, boxscore_urls, challenge

        schedule_urls_walked.append(schedule_url)
        for relative_url in schedule.final_boxscore_urls(result.body):
            boxscore_urls.append(urljoin(schedule.DEFAULT_BASE_URL, relative_url))

    return schedule_urls_walked, boxscore_urls, None


def run_pipeline(
    config: PipelineConfig,
    transport: Transport,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    print_fn: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
    jitter_fn: Callable[[float, float], float] = random.uniform,
    wall_clock_fn: Callable[[], float] = time.time,
) -> RunResult:
    """Compose schedule-walk + paced-fetch + archive into one bounded run.

    This is the single orchestration function both the CLI (:func:`main`,
    real transport) and tests/g5 (fake transport) call -- see module
    docstring's "Protected Intent" note. ``sleep_fn``/``clock_fn``/
    ``jitter_fn``/``wall_clock_fn`` are exposed for the same reason g2
    exposes them on ``PacedFetcher`` directly: so tests can prove real pacing
    logic runs without a test literally waiting on it.

    Composition, in order:
        1. Build season schedule URLs (``config.seasons``, 2026 first).
        2. Fetch each schedule page (same paced/transport seam) and extract
           FINAL boxscore URLs (:func:`schedule.final_boxscore_urls`),
           accumulating across seasons. A challenge here stops the run.
        3. For each accumulated boxscore URL, in order:
           - Skip (no fetch, does not count against ``limit``) if
             ``archive.should_fetch_url`` says it's already done.
           - In ``--dry-run`` mode: record it as "planned" and move on --
             the transport is never called for boxscore URLs in dry-run.
           - Otherwise: fetch it through the paced fetcher. A challenge here
             stops the run immediately (no further URLs are attempted); a
             normal result is archived (:func:`archive.archive_result`,
             already-atomic checkpoint write) and counts against ``limit``.
        4. Stop once ``limit`` fetches (not skips) have happened.
    """
    fetcher = PacedFetcher(
        transport=transport,
        config=config,
        sleep_fn=sleep_fn,
        clock_fn=clock_fn,
        jitter_fn=jitter_fn,
        wall_clock_fn=wall_clock_fn,
    )

    schedule_urls_walked, boxscore_urls, challenge = _collect_boxscore_urls(
        fetcher, config, print_fn=print_fn
    )
    if challenge is not None:
        return RunResult(schedule_urls=schedule_urls_walked, challenge=challenge)

    fetched: list[str] = []
    skipped: list[str] = []
    planned: list[str] = []

    for url in boxscore_urls:
        if not archive.should_fetch_url(config, url):
            skipped.append(url)
            continue

        if dry_run:
            planned.append(url)
            print_fn(f"[DRY RUN] would fetch: {url}")
            continue

        if limit is not None and len(fetched) >= limit:
            break

        try:
            result = fetcher.fetch(url)
        except ChallengeDetected as detected:
            print_fn(
                f"[CHALLENGE] Stopping run: {detected.reason} while fetching "
                f"{detected.url!r} (status={detected.status_code}). Back off "
                f">= {CHALLENGE_BACKOFF_SECONDS:.0f}s before re-running; the "
                "checkpoint reflects only what completed before this -- this "
                "URL was NOT archived."
            )
            return RunResult(
                fetched=fetched,
                skipped_already_done=skipped,
                schedule_urls=schedule_urls_walked,
                challenge=detected,
            )

        archive.archive_result(result, config)
        fetched.append(url)
        print_fn(f"[FETCHED] {url}")

    return RunResult(
        fetched=fetched,
        skipped_already_done=skipped,
        planned=planned,
        schedule_urls=schedule_urls_walked,
        challenge=None,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bc_pipeline.fetch",
        description=(
            "Walk the pioneerleague.com season schedule(s), then fetch and "
            "archive FINAL-game boxscore pages, paced and challenge-aware."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a PipelineConfig JSON override file (default: in-code defaults).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of boxscore URLs actually fetched this run (bounded demo).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk schedules and print what WOULD be fetched; never fetch a boxscore URL.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code (0 clean, 1 on challenge-stop)."""
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    # Imported lazily so this module (and its tests) never needs `requests`
    # importable/mockable for anything except the CLI's real run.
    from bc_pipeline.transport import real_transport

    result = run_pipeline(
        config,
        real_transport,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    if result.stopped_by_challenge:
        return 1

    if args.dry_run:
        print(f"[DRY RUN] {len(result.planned)} URL(s) would be fetched.")
    else:
        print(
            f"Done: {len(result.fetched)} fetched, "
            f"{len(result.skipped_already_done)} already done (skipped)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
