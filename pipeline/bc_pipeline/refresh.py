"""Refresh entrypoint: backfill -> artifact regen -> guard (g2, issue #21).

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
* :func:`bc_pipeline.person_map.build_person_map` -- rebuilds the
  within-season ``person_id`` map (issue #41). Same treatment: public
  functions only, no linking logic duplicated here.
* :func:`bc_pipeline.team_map.build_team_map` -- rebuilds the cross-season
  ``franchise_id`` registry (issue #41, team half). Note this one carries no
  drift counterpart: ``franchise_id`` is a pure function of the team name in
  each file, so ``parse`` populates it directly and it cannot fall out of
  sync the way ``person_id`` can.

**Sequencing** (:func:`run_refresh`, this module's only new logic):

1. Run the backfill escalation loop.
2. If it stopped on a challenge (``result.stopped_by_challenge``), skip
   frequency regeneration entirely -- ``games/**`` reflects only a PARTIAL
   refresh, and regenerating the frequency artifact over incomplete state
   would silently mask the stop. Return early.
3. Otherwise, regenerate EACH derived artifact in memory and compare it
   (with ``meta.generated_at`` normalized on both sides) against whatever is
   currently committed under ``artifacts/latest/``. Equal (or "nothing
   committed yet AND nothing to aggregate") is a genuine NO-OP: nothing is
   written, nothing is committed. A real difference is written (the same
   ``json.dumps(fresh, indent=2, sort_keys=True) + "\\n"`` shape
   ``frequencies.py``'s own ``_write_artifact`` uses) and committed with the
   SAME ``commit_fn`` used for game-file commits, under its own distinct
   commit message. The person map is regenerated FIRST, because it is the
   identity layer every other reading of the corpus sits on top of.

**person_id drift** (issue #41). ``person_id`` lives in two places: the
person-map artifact, which is authoritative and regenerated here, and a
materialized copy on every ``players[].person_id``, which can only be
refreshed by a labeled ``reparse(vX.Y.Z)`` commit because ``games/**`` is
write-once. A refresh that picks up new games therefore leaves the two
DIVERGED -- the new games' synthetic players are in the fresh map but carry
null in their committed files, and the map may also newly link a synthetic
that was previously unlinkable.

This module does not resolve that divergence (it has no license to rewrite
game files). It MEASURES it: ``person_id_drift`` counts the committed player
records whose stored ``person_id`` disagrees with the freshly built map.
Zero means the corpus is in sync; anything else is the honest signal that a
re-parse is due, printed as such rather than left for someone to notice.

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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from bc_pipeline import backfill, frequencies, person_map, team_map
from bc_pipeline.backfill import BackfillResult
from bc_pipeline.config import PipelineConfig, load_config
from bc_pipeline.fetcher import Transport

__all__ = [
    "RefreshResult",
    "FREQUENCY_COMMIT_MESSAGE",
    "PERSON_MAP_COMMIT_MESSAGE",
    "TEAM_MAP_COMMIT_MESSAGE",
    "run_refresh",
    "build_arg_parser",
    "main",
]

#: Commit message used for the (distinct, second) frequency-artifact commit,
#: kept separate from any game-file batch commit made by the backfill half.
FREQUENCY_COMMIT_MESSAGE: str = "refresh: regenerate frequency artifacts"

#: Commit message for the person-map artifact, kept distinct from both the
#: game-file batch commit and the frequency commit so `git log` says which
#: derived surface moved.
PERSON_MAP_COMMIT_MESSAGE: str = "refresh: regenerate person map"

#: Commit message for the franchise-map artifact.
TEAM_MAP_COMMIT_MESSAGE: str = "refresh: regenerate team map"

#: Path (relative to repo_root) the frequency artifact is read from/written
#: to -- mirrors bc_pipeline.frequencies's own CLI default.
_FREQUENCIES_RELATIVE_PATH = Path("artifacts") / "latest" / "frequencies.json"

#: Where the person-map artifact is written -- mirrors bc_pipeline.person_map's
#: own CLI default.
_PERSON_MAP_RELATIVE_PATH = Path("artifacts") / "latest" / "person_map.json"

#: Where the franchise-map artifact is written.
_TEAM_MAP_RELATIVE_PATH = Path("artifacts") / "latest" / "team_map.json"


def _regenerate_artifact(
    *,
    fresh: dict,
    output_path: Path,
    normalize: Callable[[dict], dict],
    worth_writing_when_absent: bool,
    commit_fn: Callable[[Sequence[Path], str], None],
    commit_message: str,
    label: str,
    print_fn: Callable[[str], None],
) -> str:
    """Compare one regenerated artifact against what is committed, and write +
    commit it only if it really changed. Returns ``"no-op"`` or ``"changed"``.

    Shared by the frequency and person-map halves so the compare/write/commit
    discipline exists once: both normalize ``meta.generated_at`` on BOTH sides
    before comparing (a wall-clock stamp is not a change), both write the same
    canonical text shape, and both commit through the caller's ``commit_fn``
    so a test observes every commit through one log.

    ``worth_writing_when_absent`` covers the "nothing committed yet" case: the
    caller decides whether there is genuinely anything to write (zero games
    aggregated is a NO-OP, not an empty artifact worth committing).
    """
    if output_path.exists():
        committed = json.loads(output_path.read_text(encoding="utf-8"))
        changed = normalize(committed) != normalize(fresh)
    else:
        changed = worth_writing_when_absent

    if not changed:
        print_fn(
            f"[REFRESH] NO-OP: regenerated {label} matches the committed "
            f"{output_path} (generated_at normalized on both sides); "
            "nothing to commit."
        )
        return "no-op"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commit_fn([output_path], commit_message)
    print_fn(f"[REFRESH] CHANGED: wrote + committed {output_path}.")
    return "changed"


def _person_id_drift(games: Sequence[dict], fresh_map: dict) -> int:
    """Committed player records whose stored ``person_id`` disagrees with the
    freshly built map.

    The map is authoritative; the field on ``players[]`` is a materialized
    copy that only a labeled re-parse can refresh. Counting the disagreement
    is the whole point -- it converts "the corpus might be stale" into a
    number that says whether a re-parse is worth running. A record with NO
    ``person_id`` key at all (a pre-1.7.0 file) counts as drifted, because
    that is exactly what a re-parse would fill in.
    """
    drift = 0
    for game in games:
        assigned = person_map.assignments_for_game(fresh_map, game["game_id"])
        for player_id, entry in (game.get("players") or {}).items():
            expected = (
                player_id if not person_map.is_synthetic(player_id)
                else assigned.get(player_id)
            )
            if entry.get("person_id", "\x00missing") != expected:
                drift += 1
    return drift


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

    ``person_map_status`` takes the same three values for the person-map
    artifact (issue #41). ``person_id_drift`` is the number of committed
    player records whose stored ``person_id`` disagrees with the freshly
    built map -- 0 when the corpus is in sync, otherwise the count a labeled
    re-parse would fix. It is None when regeneration never ran.
    """

    backfill: BackfillResult
    frequency_status: str
    frequency_commit_message: str | None = None
    person_map_status: str = "skipped-challenge"
    person_map_commit_message: str | None = None
    person_id_drift: int | None = None
    team_map_status: str = "skipped-challenge"
    team_map_commit_message: str | None = None

    @property
    def stopped_by_challenge(self) -> bool:
        return self.backfill.stopped_by_challenge

    def to_dict(self) -> dict:
        return {
            "backfill": self.backfill.to_dict(),
            "frequency_status": self.frequency_status,
            "frequency_commit_message": self.frequency_commit_message,
            "person_map_status": self.person_map_status,
            "person_map_commit_message": self.person_map_commit_message,
            "person_id_drift": self.person_id_drift,
            "team_map_status": self.team_map_status,
            "team_map_commit_message": self.team_map_commit_message,
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
            "artifact regeneration -- regenerating over an "
            "incomplete refresh is pointless and could mask the stop."
        )
        return RefreshResult(backfill=backfill_result, frequency_status="skipped-challenge")

    games_dir = repo_root / "games"
    games = frequencies.load_games(games_dir) if games_dir.exists() else []

    # Person map FIRST: it is the identity layer every other reading of the
    # corpus sits on top of, so if a run is interrupted between the two, the
    # surface that got regenerated is the more fundamental one.
    fresh_person_map = person_map.build_person_map(games)
    person_map_status = _regenerate_artifact(
        fresh=fresh_person_map,
        output_path=repo_root / _PERSON_MAP_RELATIVE_PATH,
        normalize=person_map.normalize_generated_at,
        worth_writing_when_absent=fresh_person_map["meta"]["games"] > 0,
        commit_fn=commit_fn,
        commit_message=PERSON_MAP_COMMIT_MESSAGE,
        label="person map",
        print_fn=print_fn,
    )

    drift = _person_id_drift(games, fresh_person_map)
    if drift:
        print_fn(
            f"[REFRESH] person_id DRIFT: {drift} committed player record(s) carry a "
            "person_id that disagrees with the regenerated map. The artifact is "
            "authoritative; games/** is write-once, so only a labeled re-parse can "
            "resync it -- run `python -m bc_pipeline.reparse --version X.Y.Z --write`."
        )

    fresh_team_map = team_map.build_team_map(games)
    team_map_status = _regenerate_artifact(
        fresh=fresh_team_map,
        output_path=repo_root / _TEAM_MAP_RELATIVE_PATH,
        normalize=team_map.normalize_generated_at,
        worth_writing_when_absent=fresh_team_map["meta"]["games"] > 0,
        commit_fn=commit_fn,
        commit_message=TEAM_MAP_COMMIT_MESSAGE,
        label="team map",
        print_fn=print_fn,
    )

    fresh = frequencies.build_frequencies(games, generated_at=frequency_generated_at)
    frequency_status = _regenerate_artifact(
        fresh=fresh,
        output_path=repo_root / _FREQUENCIES_RELATIVE_PATH,
        normalize=frequencies.normalize_generated_at,
        # No committed artifact yet: writing is warranted UNLESS there is
        # genuinely nothing to write (zero games aggregated) -- the "nothing
        # to write" edge case, treated as a NO-OP rather than committing an
        # empty artifact.
        worth_writing_when_absent=fresh["meta"]["games_included"]["total"] > 0,
        commit_fn=commit_fn,
        commit_message=FREQUENCY_COMMIT_MESSAGE,
        label="frequency artifact",
        print_fn=print_fn,
    )

    return RefreshResult(
        backfill=backfill_result,
        frequency_status=frequency_status,
        frequency_commit_message=(
            FREQUENCY_COMMIT_MESSAGE if frequency_status == "changed" else None
        ),
        person_map_status=person_map_status,
        person_map_commit_message=(
            PERSON_MAP_COMMIT_MESSAGE if person_map_status == "changed" else None
        ),
        person_id_drift=drift,
        team_map_status=team_map_status,
        team_map_commit_message=(
            TEAM_MAP_COMMIT_MESSAGE if team_map_status == "changed" else None
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bc_pipeline.refresh",
        description=(
            "One command: fetch+parse+replay+commit every discoverable newly-FINAL "
            "game (bc_pipeline.backfill), then regenerate the person map "
            "(bc_pipeline.person_map) and the season+league frequency artifact "
            "(bc_pipeline.frequencies), each only if it actually changed, and "
            "report any person_id drift a re-parse would fix."
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
        default=None,
        metavar="PATH",
        help=(
            "Repository root containing games/ and artifacts/ (default: auto-detected "
            "by walking up from the current directory for a checkout with a games/ dir)."
        ),
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

    try:
        repo_root = backfill.resolve_repo_root(args.repo_root)
    except backfill.RepoRootError as exc:
        print(f"[REFRESH] {exc}", file=sys.stderr)
        return 2

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
