#!/usr/bin/env python
"""Regenerate `artifacts/latest/completeness.json` from the committed corpus.

completeness.json is the file that states the corpus's score, and it is the
one derived artifact `check_artifacts_current.py` does NOT cover -- so it
goes stale silently, and it did. For most of 2026 it described a 2026-07-18
backfill: 1,269 games discovered, 68 replayable, a 94% failure rate, while
the corpus behind it was at 1,467 replayable. Nothing was wrong with the
file except that nothing regenerated it, and it had been hand-refreshed once
rather than rebuilt, which is the same problem deferred.

This script is the rebuild, so the answer to "is that number current?" is
`python scripts/build_completeness.py --check` rather than an argument.

WHERE EACH NUMBER COMES FROM, since the point is that none of them is typed:

  discovered  one entry per boxscore URL in the raw-archive checkpoint. That
              is the LEAGUE's own count -- the checkpoint holds exactly what
              three schedule walks yielded (1,486: 498 / 490 / 498) -- and
              deliberately not anything this repository derived.
  parsed      one per file under games/**.
  replayable  a full replay of every committed game against its archived
              HTML. Not read from `meta.parse.replayable`, which is what the
              replayer said at parse time; a stale stamp is exactly the thing
              this file exists not to repeat.
  disposed    entries in corrections/dispositions.json, each reconciled
              against the corpus by tests/test_dispositions.py.
  non_final   discovered minus committed. One game: 20260809_3555.

Run:  python scripts/build_completeness.py            # rewrite the file
      python scripts/build_completeness.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from bc_pipeline import archive, completeness, dispositions, replay, reparse  # noqa: E402
from bc_pipeline.backfill import BackfillResult, GameOutcome, SeasonSummary  # noqa: E402
from bc_pipeline.config import load_config  # noqa: E402

OUT_PATH = REPO_ROOT / "artifacts" / "latest" / "completeness.json"


def _game_id_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1].split(".")[0]


def _season_from_url(url: str) -> int:
    return int(url.split("/bsb/")[1][:4])


def build() -> dict:
    config = load_config(None)
    checkpoint = archive.load_checkpoint(archive._resolve_checkpoint_path(config))
    html_by_id = reparse._archived_html_by_game_id(config)
    ledger = dispositions.load()

    committed: dict[str, dict] = {}
    for path in sorted((REPO_ROOT / "games").rglob("*.json")):
        game = json.loads(path.read_text(encoding="utf-8"))
        committed[game["game_id"]] = game

    result = BackfillResult()
    for url in sorted(u for u in checkpoint if "/boxscores/" in u):
        game_id = _game_id_from_url(url)
        season = _season_from_url(url)
        summary = result.seasons.setdefault(season, SeasonSummary(season=season))
        game = committed.get(game_id)

        if game is None:
            entry = ledger.get(game_id)
            reason = entry["evidence"] if entry else "not committed; no disposition"
            result.games.append(GameOutcome(
                url=url, season=season, game_id=game_id,
                outcome="non_final", reason=reason,
            ))
            summary.non_final += 1
            continue

        html = Path(html_by_id[game_id]).read_text(encoding="utf-8", errors="replace")
        # Replay fresh rather than trusting meta.parse.replayable: the stamp
        # is what the replayer said when the file was written, and a report
        # built from stamps is a report that cannot notice it is out of date.
        game.setdefault("meta", {}).setdefault("parse", {})["warnings"] = []
        replayed = replay.replay_game(game, html)
        parse_meta = replayed["meta"]["parse"]
        replayable = bool(parse_meta["replayable"])
        result.games.append(GameOutcome(
            url=url, season=season, game_id=game_id, outcome="parsed",
            reason=(
                "warnings: " + "; ".join(parse_meta["warnings"])
                if parse_meta["warnings"] else None
            ),
            replayable=replayable,
            warnings=list(parse_meta["warnings"]),
            events_count=len(replayed["events"]),
            unparsed_count=len(replayed["unparsed"]),
        ))
        summary.parsed += 1
        summary.replayable += int(replayable)

    return completeness.build_completeness_report([result])


def _comparable(report: dict) -> dict:
    """The report minus its generation timestamp, which changes every run."""
    stripped = json.loads(json.dumps(report))
    stripped.pop("meta", None)
    return stripped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit nonzero if the committed file is stale; write nothing.")
    args = parser.parse_args(argv)

    fresh = build()
    accounting = fresh["accounting"]

    if args.check:
        if not OUT_PATH.exists():
            print(f"FAIL: {OUT_PATH} does not exist", file=sys.stderr)
            return 1
        committed = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if _comparable(committed) != _comparable(fresh):
            print(
                f"FAIL: {OUT_PATH.relative_to(REPO_ROOT)} is stale -- regenerate "
                f"with `python scripts/build_completeness.py`",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {OUT_PATH.relative_to(REPO_ROOT)} is current with games/**.")
        return 0

    OUT_PATH.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT_PATH.relative_to(REPO_ROOT)}\n"
        f"  discovered {accounting['games_discovered']}"
        f"  replay-validating {accounting['games_replay_validating']}"
        f"  disposed {accounting['games_disposed']}\n"
        f"  ACCOUNTED {accounting['games_accounted']}"
        f"  ({accounting['accounted_rate'] * 100:.2f}%)"
        f"  unaccounted {accounting['games_unaccounted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
