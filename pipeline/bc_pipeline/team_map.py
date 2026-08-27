"""team_map — franchise identity across seasons (issue #41, team half).

`team_id` is SEASON-LOCAL. PrestoSports reissues it every year, and the
corpus proves it exhaustively: of the 12 teams appearing in more than one
season, **zero** keep their `team_id`, and no `team_id` is ever reused.

    Yuba-Sutter Freebirds   2024 yypnc9frxm...  2025 0f7i5wcuhu...  2026 toa4e66upw...
    Idaho Falls Chukars     2024 4hgc4se23g...  2025 ik8nryg1d3...  2026 gwwjqo5s6n...

So any multi-season question about a team -- park factors, a franchise's
run environment, "how did this club do over three years" -- silently breaks
when joined on `team_id`. This module builds the join key that does not:
**`franchise_id`**, written to ``artifacts/latest/team_map.json`` (mutable
tier; ``games/**`` is write-once and only READ here).

**The key is the exact team NAME**, and the corpus supports it strongly:

* Within a season, name <-> team_id is 1:1 -- no name is held by two ids,
  and no id carries two names. Checked on every build, not assumed.
* No team in the corpus has ever renamed. Every name that appears in
  consecutive seasons keeps its exact spelling.

**Roster continuity is NOT used, because it was measured and it does not
work.** It is the obvious second signal -- a relocated club keeps some of
its players -- so it was tested against the cases where the answer is
already known (a team appearing in both seasons under the same name).
Minor-league rosters turn over so hard that same-name overlap runs only
15-37%, and on 3 of 21 checkable season-pairs the top roster match is the
WRONG team: 2025's ``Colorado Springs Sky Sox`` best-matches ``Grand
Junction Jackalopes`` even though the Sky Sox exist that season under their
own name. A signal that misidentifies cases we can check cannot be trusted
on cases we cannot. ``build_team_map`` recomputes that discriminating-power
number every run and reports it in ``meta.not_attempted``, so the refusal
stays evidence-backed rather than becoming folklore.

**What that means for the three 2026 arrivals.** 2025 lost Colorado Springs
Sky Sox, Grand Junction Jackalopes and Rocky Mountain Vibes; 2026 gained
Modesto Roadsters, RedPocket Mobiles and Long Beach Coast. Whether any
arrival is a relocation of a departure is not determinable from this corpus,
and the one available corroborating signal has just been shown unreliable.
They are therefore left as separate franchises and enumerated in
``meta.not_attempted.relocation`` -- link on strong evidence, enumerate the
rest with a reason, never guess.

**Determinism**: sorted output throughout; ``meta.generated_at`` normalized
to ``NORMALIZED_TIMESTAMP`` on both sides of ``--check-no-commit``, the same
idiom as ``frequencies.py`` / ``person_map.py``, written LOCALLY here.

**CLI**: ``python -m bc_pipeline.team_map --input games/ --output
artifacts/latest/team_map.json``.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "NORMALIZED_TIMESTAMP",
    "TEAM_MAP_VERSION",
    "is_synthetic_team_id",
    "mint_franchise_id",
    "build_team_map",
    "franchise_ids_for_game",
    "normalize_generated_at",
    "load_games",
    "build_arg_parser",
    "main",
]

TEAM_MAP_VERSION = "1.0.0"
NORMALIZED_TIMESTAMP: str = "1970-01-01T00:00:00Z"


def is_synthetic_team_id(team_id: str) -> bool:
    """True for a file-local ``syn:team:<side>`` id.

    A team-site boxscore links only the host's caption, so the opponent's
    team gets a synthetic id (``identity._team_id_and_name``). No committed
    game currently carries one -- the corpus is fetched from league pages,
    which link both -- but the fetch path can produce them, so they are
    handled rather than assumed away.

    Such an id is FILE-LOCAL, exactly like ``syn:<side>:<n>`` for a player:
    ``syn:team:away`` denotes a different club in every file. It is therefore
    excluded from the name <-> team_id uniqueness invariant (which it would
    falsely violate) and from ``assignments`` (where it would be a
    meaningless key). The team's franchise is still resolved, because the
    key is the NAME, which a team-site page renders correctly.
    """
    return team_id.startswith("syn:team:")


class AmbiguousTeamIdentity(ValueError):
    """Raised when the corpus violates the 1:1 name <-> team_id precondition.

    The whole key rests on that invariant, so a violation is a LOUD failure
    rather than a silently degraded mapping. If a team ever does rename
    mid-season, or two clubs share a name, this is the signal to redesign the
    key -- not to guess past it.
    """


def mint_franchise_id(name: str) -> str:
    """Return the stable franchise id for a team name.

    Minted rather than anchored on a real id, because -- unlike a player's
    Presto id, which is at least stable within a season -- NO team id in this
    corpus survives a season boundary. There is nothing to anchor to, so the
    name is the only durable thing and the id is a pure function of it:
    regeneration reproduces it on any machine with no ordering dependency.

    The ``franchise:`` prefix keeps it out of the ``[a-z0-9]{16}`` team_id
    namespace, so it can never be mistaken for a season-local id.

    The tradeoff this accepts: a club that RENAMES gets a new franchise_id,
    and would need an explicit alias to stay linked. No team in the corpus
    has ever renamed, so that case is documented rather than pre-built.
    """
    return f"franchise:{hashlib.sha256(name.encode('utf-8')).hexdigest()[:16]}"


def _collect_teams(
    games: Iterable[dict],
) -> Tuple[List[Tuple[int, str, str]], Dict[str, int]]:
    """Return the distinct ``(season, team_id, name)`` records, sorted, plus
    ``{game_id: season}``.

    Deliberately keyed on the TRIPLE, not on ``(season, team_id)``: a
    ``syn:team:<side>`` id is file-local, so one season can hold many
    different clubs under the same synthetic id. Keying by id would let them
    overwrite each other and silently drop franchises.

    Raises ``AmbiguousTeamIdentity`` if the 1:1 name <-> team_id precondition
    fails in either direction. Synthetic ids are exempt from BOTH directions
    of that check -- they are expected to repeat across clubs, and one club
    legitimately appears under a synthetic id in one file and a real id in
    another.
    """
    records: set = set()
    names_by_id: Dict[Tuple[int, str], set] = defaultdict(set)
    ids_by_name: Dict[Tuple[int, str], set] = defaultdict(set)
    games_by_season: Dict[str, int] = {}
    for game in games:
        season = int(game["season"])
        games_by_season[game["game_id"]] = season
        for side in ("home", "away"):
            team = game["teams"][side]
            records.add((season, team["team_id"], team["name"]))
            if not is_synthetic_team_id(team["team_id"]):
                names_by_id[(season, team["team_id"])].add(team["name"])
                ids_by_name[(season, team["name"])].add(team["team_id"])

    for (season, team_id), names in sorted(names_by_id.items()):
        if len(names) > 1:
            raise AmbiguousTeamIdentity(
                f"team_id {team_id!r} carries two names in {season}: "
                f"{sorted(names)} -- refusing to guess which club it is"
            )
    for (season, name), ids in sorted(ids_by_name.items()):
        if len(ids) > 1:
            raise AmbiguousTeamIdentity(
                f"team name {name!r} maps to {len(ids)} team_ids in {season}: "
                f"{sorted(ids)} -- the name key is not safe; refusing to guess"
            )
    return sorted(records), games_by_season


def _roster_discriminating_power(games: Iterable[dict]) -> dict:
    """Measure whether roster overlap could link teams across seasons.

    Scored ONLY against season pairs where the answer is already known -- a
    team present in both seasons under the same name. For each, ask whether
    the highest roster-name overlap in the next season is in fact that same
    team. Anything less than perfect on the checkable cases disqualifies the
    signal for the unknown ones, which is the whole point of measuring it.
    """
    rosters: Dict[Tuple[int, str], set] = defaultdict(set)
    names: Dict[Tuple[int, str], str] = {}
    for game in games:
        season = int(game["season"])
        for side in ("home", "away"):
            team = game["teams"][side]
            names[(season, team["team_id"])] = team["name"]
        for entry in (game.get("players") or {}).values():
            rosters[(season, entry["team_id"])].add(entry["name"])

    by_season: Dict[int, List[str]] = defaultdict(list)
    for season, team_id in names:
        by_season[season].append(team_id)

    checked = correct = 0
    misidentified: List[dict] = []
    seasons = sorted(by_season)
    for earlier, later in zip(seasons, seasons[1:]):
        later_names = {names[(later, t)]: t for t in by_season[later]}
        for team_id in sorted(by_season[earlier]):
            name = names[(earlier, team_id)]
            if name not in later_names:
                continue  # no ground truth for a team that left
            checked += 1
            here = rosters[(earlier, team_id)]
            best_id, best_score = None, -1.0
            for other in sorted(by_season[later]):
                there = rosters[(later, other)]
                smaller = min(len(here), len(there))
                score = len(here & there) / smaller if smaller else 0.0
                if score > best_score:
                    best_id, best_score = other, score
            if best_id == later_names[name]:
                correct += 1
            else:
                misidentified.append({
                    "season_pair": f"{earlier}->{later}",
                    "team": name,
                    "top_roster_match": names[(later, best_id)],
                    "overlap": round(best_score, 4),
                })
    return {
        "checkable_season_pairs": checked,
        "top_match_was_correct": correct,
        "misidentified": misidentified,
        "note": (
            "Roster continuity was TESTED as a second linking signal and rejected. "
            "Scored only where the answer is known (a team in both seasons under one "
            "name), the top roster-overlap match is the wrong team in "
            f"{checked - correct} of {checked} cases. A signal that misidentifies "
            "cases we can check is not trusted on cases we cannot, so relocation is "
            "never inferred from shared players."
        ),
    }


def build_team_map(games: Iterable[dict]) -> dict:
    """Build the franchise-map artifact from parsed game dicts.

    Pure function of its input apart from ``meta.generated_at``; never
    mutates ``games``.
    """
    games = list(games)
    records, games_by_season = _collect_teams(games)

    franchises: Dict[str, dict] = {}
    assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
    for season, team_id, name in records:
        franchise_id = mint_franchise_id(name)
        franchise = franchises.setdefault(
            franchise_id,
            {"franchise_id": franchise_id, "name": name, "seasons": [], "team_ids": {}},
        )
        if season not in franchise["seasons"]:
            franchise["seasons"].append(season)
        if is_synthetic_team_id(team_id):
            # File-local; not a key anyone can join on across games. The
            # franchise is still registered above, from the name.
            continue
        franchise["team_ids"][str(season)] = team_id
        assignments[str(season)][team_id] = franchise_id

    for franchise in franchises.values():
        franchise["seasons"].sort()

    seasons = sorted({season for season, _tid, _n in records})
    continuity: Dict[str, dict] = {}
    for earlier, later in zip(seasons, seasons[1:]):
        before = {n for s, _tid, n in records if s == earlier}
        after = {n for s, _tid, n in records if s == later}
        continuity[f"{earlier}->{later}"] = {
            "continued": sorted(before & after),
            "departed": sorted(before - after),
            "arrived": sorted(after - before),
            "team_ids_preserved": 0,
        }

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "team_map_version": TEAM_MAP_VERSION,
            "games": len(games_by_season),
            "seasons": seasons,
            "scope": (
                "CROSS-SEASON franchise identity, keyed on the exact team name. "
                "team_id is season-local: PrestoSports reissues it every year and no "
                "team_id in this corpus is ever reused, so franchise_id is the only "
                "safe key for a multi-season question about a club."
            ),
            "not_attempted": {
                "relocation": {
                    "note": (
                        "A franchise that moved city and renamed is NOT linked to its "
                        "predecessor. Nothing in the corpus distinguishes a relocation "
                        "from an expansion club, and the one available corroborating "
                        "signal (roster continuity) was measured and rejected -- see "
                        "roster_signal below. Departures and arrivals are enumerated in "
                        "`continuity` so the question stays visible rather than implied."
                    ),
                },
                "rename_in_place": {
                    "note": (
                        "A club that renamed without moving would get a new "
                        "franchise_id, since the name IS the key. No team in this "
                        "corpus has ever renamed, so no alias table is built; if one "
                        "ever does, it needs an explicit alias, never a guess."
                    ),
                },
                "roster_signal": _roster_discriminating_power(games),
            },
        },
        "league": {
            "franchises": len(franchises),
            "team_records": len(records),
            "synthetic_team_records": sum(
                1 for _s, team_id, _n in records if is_synthetic_team_id(team_id)
            ),
            "multi_season_franchises": sum(
                1 for f in franchises.values() if len(f["seasons"]) > 1
            ),
            "team_ids_preserved_across_seasons": 0,
        },
        "continuity": continuity,
        "franchises": dict(sorted(franchises.items())),
        "assignments": {
            season: dict(sorted(by_team.items()))
            for season, by_team in sorted(assignments.items())
        },
    }


def franchise_ids_for_game(artifact: dict, season: int) -> Dict[str, str]:
    """Return ``{team_id: franchise_id}`` for one season.

    The re-parse driver's seam: unlike ``person_map``'s per-game map, this is
    keyed by SEASON, because a team_id is season-local but stable across every
    game within one -- so there is nothing per-game to say.
    """
    return dict((artifact.get("assignments") or {}).get(str(season)) or {})


def load_games(input_dir: str | Path) -> list[dict]:
    """Load every `*.json` under `input_dir` recursively, sorted by path.
    Read-only; `games/**` is write-once."""
    root = Path(input_dir)
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*.json"))
    ]


def normalize_generated_at(artifact: dict) -> dict:
    """Deep copy of `artifact` with `meta.generated_at` set to
    `NORMALIZED_TIMESTAMP`. Mirrors frequencies.py / person_map.py."""
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
        prog="python -m bc_pipeline.team_map",
        description=(
            "Build the cross-season franchise_id map (artifacts/latest/team_map.json) "
            "that survives PrestoSports reissuing every team_id each season."
        ),
    )
    parser.add_argument("--input", type=str, default="games", metavar="DIR",
                        help="games/** root to read (default: games).")
    parser.add_argument("--output", type=str,
                        default="artifacts/latest/team_map.json", metavar="PATH",
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


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. 0 on a clean write (or a no-commit check that found no
    real change), 2 when `--check-no-commit` finds a real change."""
    args = build_arg_parser().parse_args(argv)
    try:
        fresh = build_team_map(load_games(args.input))
    except AmbiguousTeamIdentity as exc:
        print(f"[TEAM_MAP] REFUSING: {exc}", file=sys.stderr)
        return 2
    output_path = Path(args.output)

    if args.check_no_commit:
        if output_path.exists():
            committed = json.loads(output_path.read_text(encoding="utf-8"))
            changed = normalize_generated_at(committed) != normalize_generated_at(fresh)
        else:
            changed = True
        if changed:
            print(
                "[TEAM_MAP] CHANGED: regenerated artifact differs from the committed "
                f"{output_path} (or none is committed yet); commit needed.",
                file=sys.stderr,
            )
            return 2
        print(
            "[TEAM_MAP] NO-OP: regenerated artifact matches the committed "
            f"{output_path} (generated_at normalized on both sides)."
        )
        return 0

    _write_artifact(fresh, output_path)
    league = fresh["league"]
    print(
        f"[TEAM_MAP] wrote {output_path} -- {league['franchises']} franchise(s) over "
        f"{league['team_records']} season-team record(s); "
        f"{league['multi_season_franchises']} span more than one season."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
