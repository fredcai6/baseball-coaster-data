"""person_map — corpus-level person identity across games (issue #41, Gap 2).

`players[].player_id` is a FILE-LOCAL id. That is fine for a real 16-char
Presto id, which is stable for a whole season, but a synthetic
``syn:<side>:<n>`` is assigned by ROW ORDER within one boxscore, so the same
id means a different person in every game. Measured on the 1,484-game
corpus: ``syn:away:8`` is bound to **99 distinct display names** in 2026
alone. Any consumer that joins on ``player_id`` across games therefore
silently mixes people together.

This module builds the join layer that fixes it: a ``person_id`` that is
stable across every game of a season, written to
``artifacts/latest/person_map.json`` (the mutable tier -- README caller
contract clause 2; ``games/**`` stays write-once and is only READ here).

**The key is ``(season, team_id, name)``.** ``team_id`` is the anchor that
makes this tractable: it is ALWAYS a real Presto id, never synthetic, even
on the team-site pages whose player rows carry no ids at all (measured: 0
synthetic team ids in the corpus). Adding it to the key collapses the
apparent ambiguity -- the 120 ``(season, name)`` pairs holding more than one
real id drop to **4** once team is included, because the rest are the same
name on different teams (a separate cross-team question, reported below as
a measured negative, never guessed at here).

**Assignment rules**, in the codebase's standing doctrine -- link only on
strong evidence, enumerate the rest with a reason, never guess:

* A **real** ``player_id`` is its own ``person_id``. Presto reissues ids
  between seasons but not within one, so a real id is already the stable
  within-season key this layer is trying to reach.
* A **synthetic** ``player_id`` resolves through its group:

  ``REAL_ANCHOR``
      The group holds exactly one real id. That id is canonical and every
      synthetic in the group consolidates onto it. This is the big win: 217
      groups, 7,726 player records.
  ``MINTED``
      The group holds no real id at all (the player is synthetic in every
      game we have). There is nothing to anchor to, so a stable id is minted
      from the group key -- see ``mint_person_id``. 94 multi-id groups plus
      82 single-id ones, all of which still need minting because a lone
      ``syn:away:3`` is not unique across games either.
  ``MULTI_REAL``
      The group holds two or more real ids. Which one a synthetic belongs to
      is not determined by the evidence available here, so it stays
      UNLINKED with that reason. (The real ids remain their own persons.)
  ``SAME_GAME_CONFLICT``
      Two of the group's ids occur in the SAME game. One person cannot hold
      two ids in one game, so either they are two people or the file has a
      defect; either way the merge is not supported. UNLINKED.
  ``NON_PERSON_NAME``
      The "name" is not a person -- the corpus contains 8 records literally
      named ``/``, from the ``/ for X`` source defect where StatCrew omitted
      the incoming player's name. UNLINKED.

The ordering matters: ``NON_PERSON_NAME`` and ``SAME_GAME_CONFLICT`` are
REFUSALS and are checked before either linking rule, so no defect can be
merged away by an otherwise-valid anchor.

**What this deliberately does NOT do.** It does not link a person across
TEAMS within a season (134 ``(season, name)`` pairs sit on more than one
team), and it does not link across seasons at all (Gap 1 -- Presto reissues
every player id, and every team id, each season). Both are reported in
``meta.not_attempted`` as measured negatives rather than left implicit.

**Determinism**: every map and list is emitted in sorted order;
``meta.generated_at`` is a wall-clock stamp for humans, normalized to
``NORMALIZED_TIMESTAMP`` on both sides of the ``--check-no-commit``
comparison -- the same idiom as ``frequencies.py`` and
``reparse_summary.py``, written LOCALLY here rather than imported.

**CLI**: ``python -m bc_pipeline.person_map --input games/ --output
artifacts/latest/person_map.json``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "NORMALIZED_TIMESTAMP",
    "PERSON_MAP_VERSION",
    "REASONS",
    "is_synthetic",
    "mint_person_id",
    "build_person_map",
    "assignments_for_game",
    "normalize_generated_at",
    "load_games",
    "build_arg_parser",
    "main",
]

#: Semver of the artifact's own shape, independent of the game schema.
PERSON_MAP_VERSION = "1.0.0"

#: Fixed sentinel `meta.generated_at` is normalized to for the no-commit
#: comparison -- mirrors frequencies.py / reparse_summary.py.
NORMALIZED_TIMESTAMP: str = "1970-01-01T00:00:00Z"

#: Every reason a group can carry, linked or not. Closed set: a group is
#: always classified, never silently dropped.
REASONS: Tuple[str, ...] = (
    "real_anchor",
    "minted",
    "multi_real_id",
    "same_game_conflict",
    "non_person_name",
)

#: Display names that are not people. The corpus contains records named
#: "/" -- StatCrew's `/ for X` substitution line with the incoming player's
#: name omitted, which the parser admits as a PBP-declared player because
#: the narrative says someone entered. It is a source defect, not a person,
#: and must never anchor or absorb a merge.
_NON_PERSON_NAMES = frozenset({"/", "-", ""})


def is_synthetic(player_id: str) -> bool:
    """True for a file-local ``syn:<side>:<n>`` id (per-GAME positional),
    False for a real 16-char Presto id (stable for a season)."""
    return player_id.startswith("syn:")


def mint_person_id(season: int, team_id: str, name: str) -> str:
    """Return the stable minted person id for a group with no real anchor.

    Derived by sha256 over the group key so it is a pure function of that
    key: regenerating the artifact from an unchanged corpus reproduces it
    exactly, on any machine, with no counter or ordering dependency.

    The ``person:`` prefix keeps the value out of BOTH other id namespaces
    -- a real Presto id is bare ``[a-z0-9]{16}`` and a synthetic is
    ``syn:<side>:<n>`` -- so a minted id can never be mistaken for, or
    collide with, a file-local ``player_id``.

    The key is hashed rather than spelled out to keep the value short enough
    to sit in every ``player_entry``; the artifact always carries the full
    ``(season, team_id, name)`` alongside it, so the mapping stays fully
    invertible for a human reading the record.
    """
    digest = hashlib.sha256(
        f"{season}\x1f{team_id}\x1f{name}".encode("utf-8")
    ).hexdigest()
    return f"person:{digest[:16]}"


def _group_key(record: dict) -> Tuple[int, str, str]:
    return (int(record["season"]), record["team_id"], record["name"])


def _collect_records(games: Iterable[dict]) -> List[dict]:
    """Flatten every game's ``players`` into one record list.

    Reads only identity fields (season, game_id, player_id, name, team_id) --
    never box stats, never events. ``games/**`` is not mutated.
    """
    records: List[dict] = []
    for game in games:
        season = int(game["season"])
        game_id = game["game_id"]
        for player_id, entry in (game.get("players") or {}).items():
            records.append(
                {
                    "season": season,
                    "game_id": game_id,
                    "player_id": player_id,
                    "name": entry["name"],
                    "team_id": entry["team_id"],
                }
            )
    records.sort(key=lambda r: (r["season"], r["game_id"], r["player_id"]))
    return records


def _classify_groups(records: List[dict]) -> Dict[Tuple[int, str, str], dict]:
    """Classify every ``(season, team_id, name)`` group and pick its canonical
    person id (or refuse, with a reason).

    Refusals are evaluated BEFORE either linking rule so that a defect can
    never be merged away by an otherwise-valid anchor.
    """
    ids_by_group: Dict[Tuple[int, str, str], set] = defaultdict(set)
    ids_by_group_game: Dict[Tuple[int, str, str, str], set] = defaultdict(set)
    for record in records:
        key = _group_key(record)
        ids_by_group[key].add(record["player_id"])
        ids_by_group_game[key + (record["game_id"],)].add(record["player_id"])

    conflicted = {
        key[:3] for key, ids in ids_by_group_game.items() if len(ids) > 1
    }

    groups: Dict[Tuple[int, str, str], dict] = {}
    for key, ids in ids_by_group.items():
        season, team_id, name = key
        real = sorted(i for i in ids if not is_synthetic(i))
        synthetic = sorted(i for i in ids if is_synthetic(i))

        if name.strip() in _NON_PERSON_NAMES:
            reason, canonical = "non_person_name", None
        elif key in conflicted:
            # One person cannot hold two ids in the same game. Either these
            # are two people or the file has a defect; neither supports a
            # merge, so refuse rather than guess which.
            reason, canonical = "same_game_conflict", None
        elif len(real) == 1:
            reason, canonical = "real_anchor", real[0]
        elif len(real) == 0:
            reason, canonical = "minted", mint_person_id(season, team_id, name)
        else:
            # Two or more real ids under one (season, team, name). Could be a
            # genuine same-name collision or source id churn; the identity
            # table alone does not say which, so any synthetic here stays
            # unlinked. The real ids remain their own persons regardless.
            reason, canonical = "multi_real_id", None

        groups[key] = {
            "season": season,
            "team_id": team_id,
            "name": name,
            "reason": reason,
            "canonical": canonical,
            "real_ids": real,
            "synthetic_ids": synthetic,
        }
    return groups


def _person_id_for(record: dict, groups: Dict[Tuple[int, str, str], dict]) -> Optional[str]:
    """The person id for one player record, or None when unlinked.

    A REAL player_id is its own person id unconditionally -- it is already
    stable for the season, and it is never absorbed into another person by
    this layer. Only synthetics resolve through the group.
    """
    player_id = record["player_id"]
    if not is_synthetic(player_id):
        return player_id
    return groups[_group_key(record)]["canonical"]


def _not_attempted(records: List[dict]) -> dict:
    """The two linkages this layer deliberately does not make, MEASURED.

    Reported so a consumer reads a number rather than inferring silence.
    An unlinked player is a measured negative, not a failure.
    """
    teams_by_season_name: Dict[Tuple[int, str], set] = defaultdict(set)
    seasons_by_name: Dict[str, set] = defaultdict(set)
    for record in records:
        teams_by_season_name[(record["season"], record["name"])].add(record["team_id"])
        seasons_by_name[record["name"]].add(record["season"])
    return {
        "cross_team_within_season": {
            "names_on_more_than_one_team": sum(
                1 for teams in teams_by_season_name.values() if len(teams) > 1
            ),
            "note": (
                "A person who changed teams mid-season gets one person_id per "
                "team. Not linked here: the key is (season, team_id, name), and "
                "no evidence in the identity table separates a mid-season move "
                "from two different people with the same name on two rosters."
            ),
        },
        "cross_season": {
            "names_in_more_than_one_season": sum(
                1 for seasons in seasons_by_name.values() if len(seasons) > 1
            ),
            "note": (
                "Gap 1, not attempted. PrestoSports reissues every player id AND "
                "every team id each season, so no id-based signal survives a "
                "season boundary and team continuity is unavailable as a "
                "corroborating signal. person_id is WITHIN-SEASON only."
            ),
        },
    }


def build_person_map(games: Iterable[dict]) -> dict:
    """Build the full person-map artifact from parsed game dicts.

    Pure function of its input: no clock read except ``meta.generated_at``,
    no filesystem access, no mutation of ``games``.
    """
    records = _collect_records(list(games))
    groups = _classify_groups(records)

    # game_id -> {player_id: person_id or None}, synthetic ids ONLY. A real
    # id is its own person id by rule, so listing all 37k of them would be
    # 90% of the file restating the identity function. An unlinked synthetic
    # is emitted EXPLICITLY as null so a consumer can tell "refused" from
    # "not in this artifact".
    assignments: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
    persons: Dict[str, dict] = {}
    linked_records = 0
    unlinked_records = 0
    reason_records: Dict[str, int] = {reason: 0 for reason in REASONS}
    reason_groups: Dict[str, int] = {reason: 0 for reason in REASONS}
    season_stats: Dict[str, dict] = {}

    for key, group in groups.items():
        reason_groups[group["reason"]] += 1

    for record in records:
        season = str(record["season"])
        stats = season_stats.setdefault(
            season,
            {
                "player_records": 0,
                "synthetic_records": 0,
                "synthetic_linked": 0,
                "synthetic_unlinked": 0,
                "persons": 0,
            },
        )
        stats["player_records"] += 1

        person_id = _person_id_for(record, groups)
        group = groups[_group_key(record)]

        if is_synthetic(record["player_id"]):
            stats["synthetic_records"] += 1
            assignments[record["game_id"]][record["player_id"]] = person_id
            reason_records[group["reason"]] += 1
            if person_id is None:
                unlinked_records += 1
                stats["synthetic_unlinked"] += 1
            else:
                linked_records += 1
                stats["synthetic_linked"] += 1

        if person_id is None:
            continue
        person = persons.setdefault(
            person_id,
            {
                "person_id": person_id,
                "season": record["season"],
                "team_id": record["team_id"],
                "name": record["name"],
                "origin": "real_anchor" if not person_id.startswith("person:") else "minted",
                "member_ids": [],
                "games": 0,
            },
        )
        person["games"] += 1
        member = [record["game_id"], record["player_id"]]
        if member not in person["member_ids"]:
            person["member_ids"].append(member)

    for person in persons.values():
        person["member_ids"].sort()
    for season, stats in season_stats.items():
        stats["persons"] = sum(1 for p in persons.values() if str(p["season"]) == season)

    unlinked_groups = sorted(
        (
            {
                "season": g["season"],
                "team_id": g["team_id"],
                "name": g["name"],
                "reason": g["reason"],
                "real_ids": g["real_ids"],
                "synthetic_ids": g["synthetic_ids"],
            }
            for g in groups.values()
            if g["canonical"] is None and g["synthetic_ids"]
        ),
        key=lambda g: (g["season"], g["team_id"], g["name"]),
    )

    synthetic_records = sum(1 for r in records if is_synthetic(r["player_id"]))
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "person_map_version": PERSON_MAP_VERSION,
            "games": len({r["game_id"] for r in records}),
            "player_records": len(records),
            "synthetic_records": synthetic_records,
            "scope": (
                "WITHIN-SEASON identity, keyed on (season, team_id, name). A real "
                "player_id is its own person_id; only synthetic syn:<side>:<n> ids "
                "are resolved here, because those are per-GAME positional and are "
                "reused by different people across games."
            ),
            "not_attempted": _not_attempted(records),
        },
        "league": {
            "groups": len(groups),
            "groups_by_reason": dict(sorted(reason_groups.items())),
            "persons": len(persons),
            "synthetic_records": synthetic_records,
            "synthetic_linked": linked_records,
            "synthetic_unlinked": unlinked_records,
            "synthetic_records_by_reason": dict(sorted(reason_records.items())),
        },
        "by_season": dict(sorted(season_stats.items())),
        "persons": dict(sorted(persons.items())),
        "assignments": {
            game_id: dict(sorted(by_pid.items()))
            for game_id, by_pid in sorted(assignments.items())
        },
        "unlinked": unlinked_groups,
    }


def assignments_for_game(artifact: dict, game_id: str) -> Dict[str, Optional[str]]:
    """Return ``{player_id: person_id or None}`` for one game.

    The re-parse driver's seam: it passes this straight to
    ``parse.parse_game(person_ids=...)``. Real ids are absent from the
    artifact's ``assignments`` by design (a real id is its own person id), so
    they are NOT filled in here either -- ``parse`` applies that rule itself,
    which keeps the identity function in exactly one place.
    """
    return dict((artifact.get("assignments") or {}).get(game_id) or {})


def load_games(input_dir: str | Path) -> list[dict]:
    """Load every `*.json` under `input_dir` recursively, sorted by path for
    deterministic aggregation order. Read-only; `games/**` is write-once."""
    root = Path(input_dir)
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*.json"))
    ]


def normalize_generated_at(artifact: dict) -> dict:
    """Deep copy of `artifact` with `meta.generated_at` set to
    `NORMALIZED_TIMESTAMP`; used by `--check-no-commit` and by determinism
    tests. Mirrors frequencies.py's helper of the same name (never imported)."""
    out = copy.deepcopy(artifact)
    meta = out.get("meta")
    if isinstance(meta, dict) and "generated_at" in meta:
        meta["generated_at"] = NORMALIZED_TIMESTAMP
    return out


def _write_artifact(artifact: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bc_pipeline.person_map",
        description=(
            "Build the within-season person_id map (artifacts/latest/"
            "person_map.json) that consolidates per-game synthetic player ids."
        ),
    )
    parser.add_argument("--input", type=str, default="games", metavar="DIR",
                        help="games/** root to read (default: games).")
    parser.add_argument("--output", type=str,
                        default="artifacts/latest/person_map.json", metavar="PATH",
                        help="Where to write the artifact.")
    parser.add_argument(
        "--check-no-commit",
        action="store_true",
        help=(
            "Regenerate in memory and compare (generated_at normalized on both "
            "sides) against the committed --output instead of writing; exit 0 + "
            "'NO-OP' when only the timestamp would differ, exit 2 + 'CHANGED' "
            "otherwise."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. 0 on a clean write (or a no-commit check that found no
    real change), 2 when `--check-no-commit` finds a real change."""
    args = build_arg_parser().parse_args(argv)
    fresh = build_person_map(load_games(args.input))
    output_path = Path(args.output)

    if args.check_no_commit:
        if output_path.exists():
            committed = json.loads(output_path.read_text(encoding="utf-8"))
            changed = normalize_generated_at(committed) != normalize_generated_at(fresh)
        else:
            changed = True
        if changed:
            print(
                "[PERSON_MAP] CHANGED: regenerated artifact differs from the "
                f"committed {output_path} (or none is committed yet); commit needed.",
                file=sys.stderr,
            )
            return 2
        print(
            "[PERSON_MAP] NO-OP: regenerated artifact matches the committed "
            f"{output_path} (generated_at normalized on both sides)."
        )
        return 0

    _write_artifact(fresh, output_path)
    league = fresh["league"]
    print(
        f"[PERSON_MAP] wrote {output_path} -- {league['persons']} person(s) over "
        f"{fresh['meta']['games']} game(s); synthetic records "
        f"{league['synthetic_linked']} linked / {league['synthetic_unlinked']} unlinked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
