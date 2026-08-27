"""Corpus-wide labeled re-parse driver.

`games/**` is write-once: a committed game file changes ONLY in an
explicitly labeled ``reparse(vX.Y.Z): ...`` commit (README caller-contract
clause 1, enforced by ``scripts/check_write_once.py``). This module is how
that commit gets made.

Both prior re-parses (v0.2.0, v0.3.0) were driven by ad-hoc scripts that
were never committed, so the most consequential operation this repo performs
-- rewriting every game file at once -- had no repeatable, tested driver and
no record of exactly what it did. This module is that driver.

Sequencing:

 1. Resolve the repo root and load the raw-archive checkpoint.
 2. COVERAGE GATE. Every committed game must have archived raw HTML. A
    PARTIAL re-parse is refused by default, because it would leave the
    corpus straddling two parser versions -- some files regenerated, some
    not -- with no marker saying which is which. ``--allow-partial`` opts
    out deliberately and is reported loudly in the summary.
 3. Re-parse + replay each game from its archived HTML, PINNING every
    player to the id the committed file already uses (see
    ``_committed_id_overrides``).
 4. Compare against the committed file under SEMANTIC equality
    (``serialize.semantic_equal``: meta and every ``_derived`` block
    stripped), so a run that changes nothing but provenance is reported as
    unchanged rather than churning the corpus.
 5. Emit a corpus-level delta -- unparsed lines, clean-parse games,
    replayable games, event-type counts, per season and overall.
 6. Write, and optionally commit with the labeled message, ONLY under
    ``--write``. The default is a dry run.

This module never fetches. It reads archived HTML that ``bc_pipeline.fetch``
or ``bc_pipeline.backfill`` already put on disk.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from bc_pipeline import archive, parse, person_map, replay, serialize
from bc_pipeline.backfill import RepoRootError, resolve_repo_root
from bc_pipeline.config import PipelineConfig, load_config

#: The labeled-commit convention `scripts/check_write_once.py` enforces.
REPARSE_SUBJECT_TEMPLATE = "reparse(v{version}): {message}"

_GAME_ID_RE = re.compile(r"/boxscores/([A-Za-z0-9_]+)\.xml")


@dataclass
class GameDelta:
    """One game's before/after, as re-parsed."""

    game_id: str
    season: int
    changed: bool
    unparsed_before: int
    unparsed_after: int
    replayable_before: bool
    replayable_after: bool


@dataclass
class ReparseResult:
    deltas: List[GameDelta] = field(default_factory=list)
    missing_archive: List[str] = field(default_factory=list)
    parse_failed: List[tuple] = field(default_factory=list)
    wrote: int = 0
    commit_subject: Optional[str] = None
    allowed_partial: bool = False
    person_map_loaded: bool = False

    def summary(self) -> dict:
        """A stable, serializable corpus-level delta."""
        by_season: Dict[str, dict] = {}
        for d in self.deltas:
            s = by_season.setdefault(
                str(d.season),
                {"games": 0, "changed": 0, "unparsed_before": 0, "unparsed_after": 0,
                 "clean_before": 0, "clean_after": 0,
                 "replayable_before": 0, "replayable_after": 0},
            )
            s["games"] += 1
            s["changed"] += int(d.changed)
            s["unparsed_before"] += d.unparsed_before
            s["unparsed_after"] += d.unparsed_after
            s["clean_before"] += int(d.unparsed_before == 0)
            s["clean_after"] += int(d.unparsed_after == 0)
            s["replayable_before"] += int(d.replayable_before)
            s["replayable_after"] += int(d.replayable_after)

        total = {k: sum(v[k] for v in by_season.values())
                 for k in next(iter(by_season.values()), {})}
        return {
            "league": total,
            "by_season": by_season,
            "missing_archive": len(self.missing_archive),
            "parse_failed": len(self.parse_failed),
            "wrote": self.wrote,
            "commit_subject": self.commit_subject,
            "allowed_partial": self.allowed_partial,
            # Reported so a run with no person-map is never mistaken for a
            # run where nothing linked (schema 1.7.0 person_id, issue #41).
            "person_map_loaded": self.person_map_loaded,
            # A game that stops replaying is the one thing a re-parse must
            # never do quietly, so it is surfaced at the top level.
            "regressions": [
                d.game_id for d in self.deltas
                if d.replayable_before and not d.replayable_after
            ],
        }


def _archived_html_by_game_id(config: PipelineConfig) -> Dict[str, Path]:
    """game_id -> archived HTML path, from the checkpoint.

    The checkpoint, not the archive directory's filenames, is the authority
    on what was fetched (README, "Raw archive & fetching").
    """
    out: Dict[str, Path] = {}
    checkpoint = archive.load_checkpoint(archive._resolve_checkpoint_path(config))
    for url, entry in checkpoint.items():
        m = _GAME_ID_RE.search(url)
        path = entry.get("archived_path")
        if not m or not path:
            continue
        p = Path(path)
        if p.is_file():
            out[m.group(1)] = p
    return out


def _committed_id_overrides(committed: dict) -> Dict[Tuple[str, str], str]:
    """(display name, team_id) -> the player_id this game already uses.

    Identity MUST survive a re-parse. The site's player-link markup changed
    from ``players?id=<16-char>`` to ``/sports/bsb/<yr>/players/<name-slug>``
    between the original fetch and any later one, and `identity.py` only
    recognizes the former. Re-deriving identity from freshly-fetched HTML
    therefore re-keys most of the roster to synthetic ``syn:<side>:<n>`` ids
    -- measured at 10.4% synthetic before, 72.9% after -- which silently
    breaks every cross-game join the corpus exists to support.

    Pinning by (name, team_id) matches 99.95% of committed players across a
    212-game sample (5,977 of 5,980). Anyone unmatched, and anyone genuinely
    new to the file, falls through to normal id assignment.

    Note this pins WITHIN-season identity, which is all the source provides:
    Presto reissues its player ids every season (218 players appearing in
    more than one season have a different id in each), so cross-season
    linkage is a separate `person_id` layer the schema already anticipates
    on `player_entry`.

    AMBIGUOUS (name, team_id) KEYS ARE DROPPED, never resolved by last-wins.
    A committed file can carry two players under one (name, team_id) -- the
    corpus has one, `20250801_s3fn`, where a box row and a PBP-declared row
    both read "R. Velazquez". Built as a plain dict comprehension, that key
    silently bound to whichever id was iterated last, which handed the BOX
    row an id the counter had not reached yet and renumbered the game's
    PBP-declared players (syn:home:3 -> :5, syn:home:4 -> :6). Pinning is
    supposed to PRESERVE identity, so an ambiguous key is refused: both
    players fall through to normal assignment, which reproduces the
    committed numbering exactly. Never guess which one the caller meant.
    """
    by_key: Dict[Tuple[str, str], str] = {}
    ambiguous: Set[Tuple[str, str]] = set()
    for pid, entry in (committed.get("players") or {}).items():
        key = (entry["name"], entry["team_id"])
        if key in by_key and by_key[key] != pid:
            ambiguous.add(key)
        by_key[key] = pid
    for key in ambiguous:
        del by_key[key]
    return by_key


def _committed_games(repo_root: Path) -> List[Path]:
    return sorted(repo_root.glob("games/*/*.json"))


#: Default location of the person-map artifact, relative to the repo root.
PERSON_MAP_PATH = "artifacts/latest/person_map.json"


def _load_person_map(repo_root: Path, path: Optional[str]) -> Optional[dict]:
    """Load the person-map artifact, or None when it is not present.

    Schema 1.7.0's ``person_id`` is populated from this file. It is
    deliberately OPTIONAL rather than a hard gate: a re-parse must still be
    runnable on a corpus that has never had one built, and a missing map
    leaves every synthetic player's ``person_id`` null (an honest unknown)
    instead of failing the whole run. The summary reports which of the two
    happened, so "no map" can never be mistaken for "nobody linked".

    Note the artifact is derived FROM ``games/**`` and then written back INTO
    it. That is safe because it is keyed only on fields a re-parse does not
    change -- season, team_id, name, player_id -- so regenerating it after
    the re-parse reproduces the same mapping (proven by an idempotence test).
    Regenerate the artifact BEFORE the re-parse whenever the corpus has grown.
    """
    artifact_path = Path(path) if path else repo_root / PERSON_MAP_PATH
    if not artifact_path.is_file():
        return None
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def run_reparse(
    *,
    repo_root: Path,
    config: PipelineConfig,
    write: bool = False,
    allow_partial: bool = False,
    limit: Optional[int] = None,
    person_map_path: Optional[str] = None,
    print_fn: Callable[[str], None] = print,
) -> ReparseResult:
    result = ReparseResult(allowed_partial=allow_partial)
    html_by_id = _archived_html_by_game_id(config)
    paths = _committed_games(repo_root)
    person_artifact = _load_person_map(repo_root, person_map_path)
    result.person_map_loaded = person_artifact is not None
    if person_artifact is None:
        print_fn(
            "[REPARSE] no person_map artifact found -- every synthetic player's "
            "person_id will be null. Run `python -m bc_pipeline.person_map` first "
            "to populate schema 1.7.0's cross-game join key."
        )

    missing = [p.stem for p in paths if p.stem not in html_by_id]
    result.missing_archive = missing
    if missing and not allow_partial:
        print_fn(
            f"[REPARSE] REFUSING: {len(missing)} of {len(paths)} committed games have no "
            f"archived raw HTML. A partial re-parse would leave the corpus straddling two "
            f"parser versions with no marker saying which file is which. Finish the archive "
            f"(python -m bc_pipeline.fetch) or pass --allow-partial deliberately."
        )
        return result

    for path in paths[: limit if limit is not None else None]:
        gid = path.stem
        html_path = html_by_id.get(gid)
        if html_path is None:
            continue
        committed = json.loads(path.read_text(encoding="utf-8"))
        html = html_path.read_text(encoding="utf-8")
        try:
            fresh = replay.replay_game(
                parse.parse_game(
                    html,
                    source_url=committed["meta"]["source_url"],
                    fetched_at=committed["meta"]["fetched_at"],
                    id_overrides=_committed_id_overrides(committed),
                    person_ids=(
                        person_map.assignments_for_game(person_artifact, gid)
                        if person_artifact is not None
                        else None
                    ),
                ),
                html,
            )
        except Exception as exc:  # noqa: BLE001 -- reported, never swallowed
            result.parse_failed.append((gid, repr(exc)))
            continue

        changed = not serialize.semantic_equal(committed, fresh)
        cm, fm = committed.get("meta", {}).get("parse", {}), fresh["meta"]["parse"]
        result.deltas.append(
            GameDelta(
                game_id=gid,
                season=int(committed["season"]),
                changed=changed,
                unparsed_before=cm.get("unparsed_count") or 0,
                unparsed_after=fm.get("unparsed_count") or 0,
                replayable_before=bool(cm.get("replayable")),
                replayable_after=bool(fm.get("replayable")),
            )
        )
        if write and changed:
            path.write_text(serialize.canonical_dumps(fresh), encoding="utf-8")
            result.wrote += 1

    return result


def _git_commit(repo_root: Path, subject: str, print_fn: Callable[[str], None]) -> None:
    subprocess.run(["git", "add", "games"], cwd=str(repo_root), check=True)
    subprocess.run(["git", "commit", "-m", subject], cwd=str(repo_root), check=True)
    print_fn(f"[REPARSE] committed: {subject}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bc_pipeline.reparse",
        description="Corpus-wide labeled re-parse of games/** from archived raw HTML.",
    )
    p.add_argument("--version", required=True,
                   help="Parser version this re-parse lands, e.g. 0.4.0 (goes in the commit subject).")
    p.add_argument("--message", default="",
                   help="Commit subject text after the reparse(vX.Y.Z): prefix.")
    p.add_argument("--write", action="store_true",
                   help="Actually rewrite changed game files (default: dry run).")
    p.add_argument("--commit", action="store_true",
                   help="With --write, also make the labeled re-parse commit.")
    p.add_argument("--allow-partial", action="store_true",
                   help="Proceed even when some committed games have no archived HTML.")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    p.add_argument("--person-map", default=None, metavar="PATH",
                   help=("person_map artifact to populate schema 1.7.0 person_id "
                         f"(default: <repo-root>/{PERSON_MAP_PATH}). Missing is "
                         "allowed and reported; synthetic person_id stays null."))
    p.add_argument("--config", default=None)
    p.add_argument("--repo-root", default=None)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        repo_root = resolve_repo_root(args.repo_root)
    except RepoRootError as exc:
        print(f"[REPARSE] {exc}", file=sys.stderr)
        return 2
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        print(f"[REPARSE] --version must be semver, got {args.version!r}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    result = run_reparse(
        repo_root=repo_root,
        config=config,
        write=args.write,
        allow_partial=args.allow_partial,
        limit=args.limit,
        person_map_path=args.person_map,
    )
    if result.missing_archive and not args.allow_partial:
        return 1

    subject = REPARSE_SUBJECT_TEMPLATE.format(
        version=args.version, message=args.message or "corpus re-parse"
    )
    result.commit_subject = subject
    print(json.dumps(result.summary(), indent=2, sort_keys=True))

    if result.parse_failed:
        print(f"[REPARSE] {len(result.parse_failed)} game(s) FAILED to parse:", file=sys.stderr)
        for gid, exc in result.parse_failed[:10]:
            print(f"  [X] {gid}: {exc}", file=sys.stderr)
        return 1
    if not args.write:
        print("[REPARSE] dry run -- nothing written. Re-run with --write to apply.")
        return 0
    if args.commit and result.wrote:
        _git_commit(repo_root, subject, print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
