"""Refresh entrypoint: backfill -> frequency regen -> guard (g2, issue #21).

Thin orchestration ONLY. Composes two already-shipped, already-tested
modules and adds no new pick-up/idempotency/batching or aggregation logic of
its own:

* :func:`bc_pipeline.backfill.run_backfill_with_escalation` -- fetches every
  discoverable newly-FINAL game (until caught up), parses, replays, and
  commits it to ``games/<season>/<game_id>.json`` (write-once), with the
  existing escalating-challenge-backoff CLI policy. This module calls it
  UNCHANGED; every pick-up/idempotency/batching guarantee it already proves
  (see ``pipeline/tests/test_backfill.py``) is inherited verbatim, not
  re-derived here.
* :func:`bc_pipeline.frequencies.build_frequencies` (+ ``load_games`` /
  ``normalize_generated_at``) -- aggregates ``games/**`` into the season+
  league event-frequency artifact. This module calls its PUBLIC functions
  only; the aggregation algorithm itself lives entirely in
  ``bc_pipeline.frequencies`` and is never duplicated here.

**Sequencing** (:func:`run_refresh`, this module's only new logic):

1. Run the backfill escalation loop.
2. If it stopped on a challenge (``result.stopped_by_challenge``), skip
   frequency regeneration entirely -- ``games/**`` reflects only a PARTIAL
   refresh, and regenerating the frequency artifact over incomplete state
   would silently mask the stop. Return early.
3. Otherwise, regenerate the frequency artifact in memory and compare it
   (with ``meta.generated_at`` normalized on both sides, via
   ``frequencies.normalize_generated_at``) against whatever is currently
   committed at ``artifacts/latest/frequencies.json``. Equal (or "nothing
   committed yet AND nothing to aggregate") is a genuine NO-OP: nothing is
   written, nothing is committed. A real difference is written (the same
   ``json.dumps(fresh, indent=2, sort_keys=True) + "\\n"`` shape
   ``frequencies.py``'s own ``_write_artifact`` uses) and committed with the
   SAME ``commit_fn`` used for game-file commits, under its own distinct
   commit message.

``run_refresh`` mirrors ``run_backfill_with_escalation``'s own injectable-seam
shape (fake clock/sleep/wall-clock/print, real ``git`` never called by a
test) so it is fully testable against a fake transport with zero real
network and zero real git.

**CLI**: ``python -m bc_pipeline.refresh`` -- args mirror ``backfill.py``'s
own (``--config``, ``--limit``, ``--repo-root``, ``--push``) so the two
commands feel like siblings.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from bc_pipeline import backfill, frequencies
from bc_pipeline.backfill import BackfillResult
from bc_pipeline.config import PipelineConfig, load_config
from bc_pipeline.fetcher import Transport

__all__ = [
    "RefreshResult",
    "FREQUENCY_COMMIT_MESSAGE",
    "run_refresh",
    "build_arg_parser",
    "main",
]

#: Commit message used for the (distinct, second) frequency-artifact commit,
#: kept separate from any game-file batch commit made by the backfill half.
FREQUENCY_COMMIT_MESSAGE: str = "refresh: regenerate frequency artifacts"

#: Path (relative to repo_root) the frequency artifact is read from/written
#: to -- mirrors bc_pipeline.frequencies's own CLI default.
_FREQUENCIES_RELATIVE_PATH = Path("artifacts") / "latest" / "frequencies.json"


def _git_commit_fn(paths: Sequence[Path], message: str, *, repo_root: Path) -> None:
    """Real ``git add`` + ``git commit`` -- never called by any unit test
    (tests always inject a fake ``commit_fn``).

    Defined locally rather than reused from ``bc_pipeline.backfill``: that
    module's own leading-underscore equivalent is a private symbol (absent
    from ``backfill.py``'s ``__all__``), and this codebase's convention is
    that cross-module dependencies only reach for public names. This is
    plain ``git add``/``git commit`` plumbing, not the pick-up/idempotency/
    batching domain logic the "zero reimplementation" fence protects, so an
    independent copy here duplicates no business logic -- it only avoids a
    private-symbol reach into a fenced module."""
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


@dataclass
class RefreshResult:
    """Summary of one :func:`run_refresh` invocation.

    ``backfill`` is the underlying :class:`bc_pipeline.backfill.BackfillResult`
    unchanged. ``frequency_status`` is one of:

    * ``"skipped-challenge"`` -- the backfill half stopped on a challenge;
      frequency regeneration was never attempted.
    * ``"no-op"`` -- frequency regeneration ran but the regenerated artifact
      matched what was already committed (timestamp-normalized); nothing
      written, nothing committed.
    * ``"changed"`` -- the regenerated artifact differed; it was written and
      committed (``frequency_commit_message`` names that commit).
    """

    backfill: BackfillResult
    frequency_status: str
    frequency_commit_message: str | None = None

    @property
    def stopped_by_challenge(self) -> bool:
        return self.backfill.stopped_by_challenge

    def to_dict(self) -> dict:
        return {
            "backfill": self.backfill.to_dict(),
            "frequency_status": self.frequency_status,
            "frequency_commit_message": self.frequency_commit_message,
        }


def run_refresh(
    config: PipelineConfig,
    transport: Transport,
    *,
    repo_root: Path | str = ".",
    limit: int | None = None,
    batch_size: int = backfill.DEFAULT_BATCH_SIZE,
    print_fn: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
    jitter_fn: Callable[[float, float], float] = random.uniform,
    wall_clock_fn: Callable[[], float] = time.time,
    commit_fn: Callable[[Sequence[Path], str], None] | None = None,
    escalation_sleep_fn: Callable[[float], None] = time.sleep,
    escalation_backoffs: Sequence[float] = backfill.ESCALATION_BACKOFF_SECONDS,
    frequency_generated_at: str | None = None,
) -> RefreshResult:
    """Backfill any newly-final games, then regenerate the frequency artifact
    if (and only if) it actually changed.

    ``commit_fn`` (``(paths, message) -> None``) is used for BOTH the
    game-file batch commit(s) made by the backfill half AND the (distinct,
    second) frequency-artifact commit made by this function -- the same
    callable, so a test injecting a fake can observe every commit this run
    makes through one call log. Defaults to real ``git add``/``git commit``
    (never called by a unit test -- tests always inject a fake).

    Every other keyword mirrors :func:`bc_pipeline.backfill.
    run_backfill_with_escalation`'s own injectable-seam shape and is passed
    straight through to it unchanged.
    """
    repo_root = Path(repo_root).resolve()
    if commit_fn is None:

        def commit_fn(paths: Sequence[Path], message: str) -> None:
            _git_commit_fn(paths, message, repo_root=repo_root)

    backfill_result = backfill.run_backfill_with_escalation(
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
        escalation_sleep_fn=escalation_sleep_fn,
        escalation_backoffs=escalation_backoffs,
    )

    if backfill_result.stopped_by_challenge:
        print_fn(
            "[REFRESH] Backfill half stopped by a challenge after escalating "
            "backoff; games/** reflects only a PARTIAL refresh. Skipping "
            "frequency-artifact regeneration -- regenerating over an "
            "incomplete refresh is pointless and could mask the stop."
        )
        return RefreshResult(backfill=backfill_result, frequency_status="skipped-challenge")

    games_dir = repo_root / "games"
    games = frequencies.load_games(games_dir) if games_dir.exists() else []
    fresh = frequencies.build_frequencies(games, generated_at=frequency_generated_at)

    output_path = repo_root / _FREQUENCIES_RELATIVE_PATH
    if output_path.exists():
        committed = json.loads(output_path.read_text(encoding="utf-8"))
        changed = frequencies.normalize_generated_at(committed) != frequencies.normalize_generated_at(
            fresh
        )
    else:
        # No committed artifact yet: writing is warranted UNLESS there is
        # genuinely nothing to write (zero games aggregated) -- the "nothing
        # to write" edge case, treated as a NO-OP rather than committing an
        # empty artifact.
        changed = fresh["meta"]["games_included"]["total"] > 0

    if not changed:
        print_fn(
            f"[REFRESH] NO-OP: regenerated frequency artifact matches the "
            f"committed {output_path} (generated_at normalized on both "
            "sides); nothing to commit."
        )
        return RefreshResult(backfill=backfill_result, frequency_status="no-op")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commit_fn([output_path], FREQUENCY_COMMIT_MESSAGE)
    print_fn(f"[REFRESH] CHANGED: wrote + committed {output_path}.")
    return RefreshResult(
        backfill=backfill_result,
        frequency_status="changed",
        frequency_commit_message=FREQUENCY_COMMIT_MESSAGE,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bc_pipeline.refresh",
        description=(
            "One command: fetch+parse+replay+commit every discoverable newly-FINAL "
            "game (bc_pipeline.backfill), then regenerate the season+league "
            "frequency artifact (bc_pipeline.frequencies) only if it actually "
            "changed."
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
        help="Repository root containing games/ and artifacts/ (default: current directory).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to the remote after each commit this run makes (default: commit only, no push).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code (0 clean, 1 if the backfill
    half is still challenge-stopped after escalating backoff)."""
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    # Imported lazily, same rationale as backfill.py's own main(): this
    # module (and its tests) never needs `requests` importable/mockable for
    # anything except the CLI's real run.
    from bc_pipeline.transport import real_transport

    repo_root = Path(args.repo_root)

    def commit_fn(paths: Sequence[Path], message: str) -> None:
        _git_commit_fn(paths, message, repo_root=repo_root)
        if args.push:
            subprocess.run(["git", "push"], cwd=str(repo_root), check=True)

    result = run_refresh(
        config,
        real_transport,
        repo_root=repo_root,
        limit=args.limit,
        commit_fn=commit_fn,
    )

    if result.stopped_by_challenge:
        print(
            "[REFRESH] Stopping: partial state preserved (checkpoint + any "
            "committed games/ files reflect everything completed so far); "
            "frequency-artifact regeneration was skipped this run."
        )
        return 1

    total_parsed = sum(s.parsed for s in result.backfill.seasons.values())
    print(
        f"[REFRESH] Done: {total_parsed} new game(s) parsed, "
        f"{len(result.backfill.commits)} game-file commit(s) made; "
        f"frequency artifact: {result.frequency_status.upper()}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
