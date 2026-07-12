"""Until-caught-up backfill driver: fetch -> parse -> replay -> commit (g1, #20).

Composes the already-shipped fetch chain (:func:`bc_pipeline.fetch.run_pipeline`,
which itself wires :mod:`bc_pipeline.schedule` / :mod:`bc_pipeline.fetcher` /
:mod:`bc_pipeline.archive`) with the already-shipped :func:`bc_pipeline.parse.
parse_game` and :func:`bc_pipeline.replay.replay_game` to produce committed
``games/<season>/<game_id>.json`` files.

This module does NOT modify any of those already-shipped modules -- it only
drives them, one season at a time (:data:`PipelineConfig.seasons`, in order),
so that boxscore URLs discovered for a season are never interleaved with a
different season's URLs, which lets this module batch/commit cleanly at each
season boundary.

Two seams, mirroring ``fetch.py``'s own split:

* :func:`run_backfill` -- the pure-ish orchestration function under test
  (fake transport/clock/sleep in tests, exactly like ``test_fetch.py``). It
  stops immediately on a :class:`~bc_pipeline.fetcher.ChallengeDetected` --
  no internal retry, no backoff -- returning a partial :class:`BackfillResult`
  the caller can inspect.
* :func:`run_backfill_with_escalation` -- the CLI-facing loop that adds the
  escalating challenge backoff (60s -> 10min -> 60min per LAUNCH_ORDER_20
  Pre-Rulings), stopping cleanly after 3 escalations in one process. This is
  where the CLI's own retry policy lives; :func:`run_backfill` itself never
  sleeps for a challenge.

``games/**`` write-once (the single most important invariant this module
proves): before writing ``games/<season>/<game_id>.json``, if a file already
exists at that path, the write is ALWAYS skipped -- this module never
overwrites a committed game file, regardless of content. The idempotency key
(``parse.idempotency_key(html)`` == ``sha256(raw html) + parser_version``,
already computed by ``parse.py`` -- see its ``idempotency_key`` function and
the ``meta.source_sha256``/``meta.parser_version`` fields it stamps onto every
parsed game) is used only to decide whether the pre-existing file's content
still matches what a re-parse of the archived HTML would produce today. A
match is the ordinary "already done, nothing to do" case. A mismatch is
recorded as an anomaly for a human to triage (a genuine re-parse, if ever
needed, is a deliberate labeled commit per the repo README's caller
contract -- never an ambient overwrite from this driver) -- the file is
still never touched either way. Computing this key does not require a full
``parse_game`` call, which lets a resumed run skip already-committed games
cheaply instead of re-running the whole parse+replay pipeline on HTML it
already successfully turned into a committed file.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from bc_pipeline import archive, parse, replay, serialize
from bc_pipeline.config import PipelineConfig, load_config
from bc_pipeline.fetch import run_pipeline
from bc_pipeline.fetcher import ChallengeDetected, Transport

__all__ = [
    "GameOutcome",
    "SeasonSummary",
    "BackfillResult",
    "run_backfill",
    "ESCALATION_BACKOFF_SECONDS",
    "MAX_ESCALATIONS",
    "run_backfill_with_escalation",
    "build_arg_parser",
    "main",
]

#: Default batch size: commit after this many NEW committed game files within
#: one season, or at the season boundary, whichever comes first.
DEFAULT_BATCH_SIZE: int = 50

#: Escalating challenge backoff, per LAUNCH_ORDER_20 Pre-Rulings: 60s, then
#: 10 minutes, then 60 minutes. Only the CLI loop (:func:`run_backfill_with_
#: escalation`) sleeps for these -- :func:`run_backfill` itself never does.
ESCALATION_BACKOFF_SECONDS: tuple[float, ...] = (60.0, 600.0, 3600.0)

#: Stop cleanly after this many escalations (i.e. this many backoff-and-retry
#: cycles) without a clean run in one process.
MAX_ESCALATIONS: int = len(ESCALATION_BACKOFF_SECONDS)

_GAME_ID_RE = re.compile(r"/boxscores/([A-Za-z0-9_]+)\.xml")


def _game_id_from_url(url: str) -> str:
    """Extract the boxscore ``game_id`` from its source URL.

    Deliberately a small LOCAL regex (not an import of ``parse.py``'s private
    ``_extract_game_id``): this module reads the game_id up front (before any
    parse call) purely to compute the write-once target path, independent of
    whatever internal helper ``parse.py`` happens to use for the same job.
    """
    m = _GAME_ID_RE.search(url)
    if not m:
        raise ValueError(f"could not extract game_id from boxscore url: {url!r}")
    return m.group(1)


def _iso_from_epoch(epoch_seconds: float) -> str:
    """Format a ``time.time()``-style epoch float as an ISO-8601 UTC string."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


@dataclass
class GameOutcome:
    """Per-URL outcome record for one boxscore this run processed.

    ``outcome`` is one of: ``"parsed"`` (parse+replay ran; may or may not be
    ``replayable``), ``"non_final"``, ``"parse_failed"``, or
    ``"skipped_already_committed"``.
    """

    url: str
    season: int
    game_id: str
    outcome: str
    reason: str | None = None
    replayable: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "season": self.season,
            "game_id": self.game_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "replayable": self.replayable,
            "warnings": list(self.warnings),
        }


@dataclass
class SeasonSummary:
    """Per-season outcome breakdown -- what g2's completeness.py consumes."""

    season: int
    fetched: int = 0
    skipped_already_done: int = 0
    parsed: int = 0
    replayable: int = 0
    non_final: int = 0
    parse_failed: int = 0
    skipped_already_committed: int = 0

    def to_dict(self) -> dict:
        return {
            "season": self.season,
            "fetched": self.fetched,
            "skipped_already_done": self.skipped_already_done,
            "parsed": self.parsed,
            "replayable": self.replayable,
            "non_final": self.non_final,
            "parse_failed": self.parse_failed,
            "skipped_already_committed": self.skipped_already_committed,
        }


@dataclass
class BackfillResult:
    """Summary of one :func:`run_backfill` invocation.

    ``seasons`` is keyed by season year, in the order those seasons were
    walked. ``games`` holds one :class:`GameOutcome` per boxscore URL this
    run processed (fetched this run OR already-archived-on-resume), across
    all seasons walked, in processing order. ``commits`` records every
    commit message this run actually made (via ``commit_fn``). ``challenge``
    is the :class:`ChallengeDetected` that stopped this run, if any -- exactly
    like ``fetch.RunResult.challenge``, this run never retries it internally.
    """

    seasons: dict[int, SeasonSummary] = field(default_factory=dict)
    games: list[GameOutcome] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    challenge: ChallengeDetected | None = None

    @property
    def stopped_by_challenge(self) -> bool:
        return self.challenge is not None

    def to_dict(self) -> dict:
        return {
            "seasons": {
                str(season): summary.to_dict() for season, summary in self.seasons.items()
            },
            "games": [g.to_dict() for g in self.games],
            "commits": list(self.commits),
            "stopped_by_challenge": self.stopped_by_challenge,
        }


def _default_commit_fn(paths: Sequence[Path], message: str, *, repo_root: Path) -> None:
    """Real ``git add`` + ``git commit`` -- never called by any unit test
    (tests always inject a fake ``commit_fn``)."""
    if not paths:
        return
    subprocess.run(
        ["git", "add", *[str(p) for p in paths]],
        cwd=str(repo_root),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(repo_root),
        check=True,
    )


class _SeasonBatcher:
    """Tracks pending newly-written game file paths for one season and fires
    ``commit_fn`` every ``batch_size`` games or when explicitly flushed
    (season boundary)."""

    def __init__(
        self,
        season: int,
        *,
        batch_size: int,
        commit_fn: Callable[[Sequence[Path], str], None],
        result_commits: list[str],
    ) -> None:
        self._season = season
        self._batch_size = batch_size
        self._commit_fn = commit_fn
        self._result_commits = result_commits
        self._pending: list[Path] = []
        self._committed_count = 0

    def add(self, path: Path) -> None:
        self._pending.append(path)
        if len(self._pending) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        start = self._committed_count + 1
        end = self._committed_count + len(self._pending)
        message = f"backfill({self._season}): games {start}–{end}"
        self._commit_fn(self._pending, message)
        self._result_commits.append(message)
        self._committed_count = end
        self._pending = []


def _process_boxscore_url(
    url: str,
    *,
    season: int,
    checkpoint: dict,
    repo_root: Path,
    summary: SeasonSummary,
) -> tuple[GameOutcome, dict | None, Path | None]:
    """Parse+replay one already-archived boxscore URL.

    Returns ``(outcome, game_or_None, out_path_or_None)``. ``game`` and
    ``out_path`` are non-``None`` only when a NEW file should be written
    (i.e. outcome is ``"parsed"`` and no committed file already exists for
    this game_id).
    """
    game_id = _game_id_from_url(url)
    out_path = repo_root / "games" / str(season) / f"{game_id}.json"

    entry = checkpoint.get(url)
    if entry is None or not entry.get("archived_path"):
        outcome = GameOutcome(
            url=url,
            season=season,
            game_id=game_id,
            outcome="parse_failed",
            reason="no archived_path found in checkpoint for this url",
        )
        summary.parse_failed += 1
        return outcome, None, None

    html = Path(entry["archived_path"]).read_text(encoding="utf-8")

    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        existing_meta = existing.get("meta", {})
        existing_key = f"{existing_meta.get('source_sha256')}:{existing_meta.get('parser_version')}"
        new_key = parse.idempotency_key(html)
        reason = None
        if existing_key != new_key:
            reason = (
                "existing committed file's idempotency key does not match a fresh "
                "re-parse of the archived HTML (content drift) -- file left untouched; "
                "float as a re-parse candidate, do not auto-overwrite"
            )
        outcome = GameOutcome(
            url=url,
            season=season,
            game_id=game_id,
            outcome="skipped_already_committed",
            reason=reason,
        )
        summary.skipped_already_committed += 1
        return outcome, None, None

    fetched_at_iso = _iso_from_epoch(entry["fetched_at"])
    try:
        game = parse.parse_game(html, source_url=url, fetched_at=fetched_at_iso)
    except parse.NonFinalPageError as exc:
        outcome = GameOutcome(
            url=url, season=season, game_id=game_id, outcome="non_final", reason=str(exc)
        )
        summary.non_final += 1
        return outcome, None, None
    except Exception as exc:  # noqa: BLE001 -- deliberate: record, never crash the run
        outcome = GameOutcome(
            url=url, season=season, game_id=game_id, outcome="parse_failed", reason=repr(exc)
        )
        summary.parse_failed += 1
        return outcome, None, None

    game = replay.replay_game(game, html)
    replayable = bool(game["meta"]["parse"]["replayable"])
    warnings = list(game["meta"]["parse"].get("warnings", []))

    outcome = GameOutcome(
        url=url,
        season=season,
        game_id=game_id,
        outcome="parsed",
        replayable=replayable,
        warnings=warnings,
    )
    summary.parsed += 1
    if replayable:
        summary.replayable += 1

    return outcome, game, out_path


def run_backfill(
    config: PipelineConfig,
    transport: Transport,
    *,
    repo_root: Path | str = ".",
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    print_fn: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
    jitter_fn: Callable[[float, float], float] = random.uniform,
    wall_clock_fn: Callable[[], float] = time.time,
    commit_fn: Callable[[Sequence[Path], str], None] | None = None,
) -> BackfillResult:
    """Fetch (until caught up) + parse + replay + commit every discoverable
    FINAL game across ``config.seasons``, in order.

    ``limit`` caps the TOTAL number of NEW boxscore fetches across the whole
    run (same continue-crawl semantics as ``fetch.run_pipeline``'s own
    ``limit`` -- an optional override for a bounded slice/test, never the
    default mode). ``None`` (the default) means: run until every configured
    season's backlog is exhausted.

    One season at a time (never interleaved), so that:
      * A challenge stops the run immediately -- no further seasons are
        walked, matching ``fetch.py``'s own challenge contract.
      * Batched commits land cleanly at season boundaries.

    For each boxscore URL this run discovers as already-archived (freshly
    fetched this run, or already ``done`` in the checkpoint from a prior
    run/resume), parses + replays it and, unless a committed file already
    exists at ``games/<season>/<game_id>.json`` (write-once -- see module
    docstring), writes it and stages it for the next batch commit.

    ``commit_fn`` defaults to real ``git add``/``git commit`` (never called
    by a unit test -- tests always inject a fake). Never pushes; pushing (if
    any) is the CLI wrapper's job.
    """
    repo_root = Path(repo_root)
    if commit_fn is None:
        commit_fn = lambda paths, message: _default_commit_fn(  # noqa: E731
            paths, message, repo_root=repo_root
        )

    result = BackfillResult()
    remaining_limit = limit

    for season in config.seasons:
        summary = SeasonSummary(season=season)
        result.seasons[season] = summary
        batcher = _SeasonBatcher(
            season,
            batch_size=batch_size,
            commit_fn=commit_fn,
            result_commits=result.commits,
        )

        season_config = PipelineConfig(
            min_interval_seconds=config.min_interval_seconds,
            jitter_seconds=config.jitter_seconds,
            seasons=[season],
            archive_root=config.archive_root,
            checkpoint_path=config.checkpoint_path,
        )

        fetch_result = run_pipeline(
            season_config,
            transport,
            limit=remaining_limit,
            dry_run=False,
            print_fn=print_fn,
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
            jitter_fn=jitter_fn,
            wall_clock_fn=wall_clock_fn,
        )

        summary.fetched = len(fetch_result.fetched)
        summary.skipped_already_done = len(fetch_result.skipped_already_done)
        if remaining_limit is not None:
            remaining_limit -= len(fetch_result.fetched)

        checkpoint = archive.load_checkpoint(config.checkpoint_path)

        for url in [*fetch_result.fetched, *fetch_result.skipped_already_done]:
            outcome, game, out_path = _process_boxscore_url(
                url,
                season=season,
                checkpoint=checkpoint,
                repo_root=repo_root,
                summary=summary,
            )
            result.games.append(outcome)
            if game is not None and out_path is not None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(serialize.canonical_dumps(game), encoding="utf-8")
                batcher.add(out_path)

        batcher.flush()

        if fetch_result.challenge is not None:
            result.challenge = fetch_result.challenge
            return result

        if remaining_limit is not None and remaining_limit <= 0:
            break

    return result


def run_backfill_with_escalation(
    config: PipelineConfig,
    transport: Transport,
    *,
    repo_root: Path | str = ".",
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    print_fn: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
    jitter_fn: Callable[[float, float], float] = random.uniform,
    wall_clock_fn: Callable[[], float] = time.time,
    commit_fn: Callable[[Sequence[Path], str], None] | None = None,
    escalation_sleep_fn: Callable[[float], None] = time.sleep,
    escalation_backoffs: Sequence[float] = ESCALATION_BACKOFF_SECONDS,
) -> BackfillResult:
    """CLI-facing wrapper: retries :func:`run_backfill` across an escalating
    challenge backoff (60s -> 10min -> 60min by default), stopping cleanly
    after ``len(escalation_backoffs)`` escalations in one process.

    ``escalation_sleep_fn`` is a SEPARATE injectable from ``sleep_fn`` (the
    latter paces individual fetches inside ``run_backfill``/``PacedFetcher``)
    specifically so a test can prove the escalating-backoff policy runs
    (right durations, right count, clean stop) without a test that sleeps
    for real -- exactly the same technique the pacing seam itself uses.
    """
    attempt = 0
    result = run_backfill(
        config,
        transport,
        repo_root=repo_root,
        limit=limit,
        batch_size=batch_size,
        print_fn=print_fn,
        sleep_fn=sleep_fn,
        clock_fn=clock_fn,
        jitter_fn=jitter_fn,
        wall_clock_fn=wall_clock_fn,
        commit_fn=commit_fn,
    )

    while result.stopped_by_challenge and attempt < len(escalation_backoffs):
        backoff = escalation_backoffs[attempt]
        print_fn(
            f"[BACKFILL] Challenge detected ({result.challenge.reason}); backing off "
            f"{backoff:.0f}s (escalation {attempt + 1}/{len(escalation_backoffs)}) "
            "before resuming from checkpoint + committed games/."
        )
        escalation_sleep_fn(backoff)
        attempt += 1
        result = run_backfill(
            config,
            transport,
            repo_root=repo_root,
            limit=limit,
            batch_size=batch_size,
            print_fn=print_fn,
            sleep_fn=sleep_fn,
            clock_fn=clock_fn,
            jitter_fn=jitter_fn,
            wall_clock_fn=wall_clock_fn,
            commit_fn=commit_fn,
        )

    if result.stopped_by_challenge:
        print_fn(
            f"[BACKFILL] Stopping after {attempt} escalation(s); no clean run achieved. "
            "Partial state preserved (checkpoint + any committed games/ files reflect "
            "everything completed so far) -- honest partial-state stop, not a crash."
        )

    return result


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m bc_pipeline.backfill",
        description=(
            "Fetch (until caught up) + parse + replay + commit every discoverable "
            "FINAL game across configured seasons, in order."
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
        help="Cap the number of boxscore URLs actually fetched this run (bounded slice/test override).",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        metavar="PATH",
        help="Repository root containing games/ (default: current directory).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to the remote after each commit this run makes (default: commit only, no push).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code (0 clean, 1 if still
    challenge-stopped after escalating backoff)."""
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    # Imported lazily so this module (and its tests) never needs `requests`
    # importable/mockable for anything except the CLI's real run.
    from bc_pipeline.transport import real_transport

    repo_root = Path(args.repo_root)

    def commit_fn(paths: Sequence[Path], message: str) -> None:
        _default_commit_fn(paths, message, repo_root=repo_root)
        if args.push:
            subprocess.run(["git", "push"], cwd=str(repo_root), check=True)

    result = run_backfill_with_escalation(
        config,
        real_transport,
        repo_root=repo_root,
        limit=args.limit,
        commit_fn=commit_fn,
    )

    if result.stopped_by_challenge:
        return 1

    total_parsed = sum(s.parsed for s in result.seasons.values())
    total_replayable = sum(s.replayable for s in result.seasons.values())
    print(
        f"Done: {total_parsed} parsed ({total_replayable} replayable), "
        f"{len(result.commits)} commit(s) made."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
