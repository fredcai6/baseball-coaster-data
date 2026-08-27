"""career_map — cross-season player identity (issue #41, Gap 1).

``person_id`` (``bc_pipeline.person_map``) is stable across every GAME of a
season, and deliberately no further: PrestoSports reissues every player id
each year, so nothing in the source survives a season boundary. This module
builds the layer above it -- **``career_id``**, one key per person across
seasons -- written to ``artifacts/latest/career_map.json``.

#41 named the blocker for this work: *"team continuity is unusable as a
linking signal until team identity is solved."* ``bc_pipeline.team_map``
solved it, so franchise continuity is now available -- and it turns out to
be the only signal here worth trusting.

**Exact display name is necessary but NOT sufficient, and the corpus proves
it.** Within a single season the corpus contains people who share a full
display name and are provably different: two ``Jack Lynch``es both played in
2024, on different franchises, appearing on the SAME DATE (30 games and 81
games respectively). One person cannot play for two clubs on one day. Three
such pairs are proven this way. A rule that linked on name alone would
merge strangers.

**Signal strength, measured on this corpus every build** (see
``meta.evidence``). For each candidate signal, how often it fires on
same-name consecutive-season pairs versus how often it fires on the NULL --
every consecutive-season person pair with a DIFFERENT name, who are
therefore definitely not the same person:

    franchise continuity   ~64% on candidates   ~7% on the null   ratio ~9
    position overlap       ~96% on candidates  ~45% on the null   ratio ~2

Position overlap fires on nearly half of unrelated people, so it is close to
a rubber stamp and is NOT used. Franchise continuity genuinely
discriminates, and independently it refuses all three proven-different pairs
above -- none of them shares a franchise.

**The rule.** Two persons in consecutive seasons are linked iff:

1. their display names are exactly equal, AND
2. that name resolves to exactly ONE person in each of the two seasons (no
   within-season ambiguity to resolve first), AND
3. they share a franchise.

Careers are the connected components of those links. Two invariants are
asserted on every build, never assumed: no career holds two persons from the
same season, and no career mixes display names.

**What is refused, and why it is refused rather than guessed.**

``franchise_changed``
    Name matches and each season is unambiguous, but the player was on a
    different franchise. This is the honest cost of the rule: a real player
    who changed clubs between seasons stays unlinked. It is also exactly the
    shape of all three proven-different pairs, and nothing else in the data
    separates the two cases. ~98 pairs.
``ambiguous_within_season``
    The name resolves to more than one person in one of the seasons -- a
    mid-season team change, or two people. ``person_map`` deliberately does
    not merge across teams within a season (a measured >2% error floor), so
    that ambiguity is inherited here rather than papered over. ~57 pairs.

Every person still receives a ``career_id``, including an unlinked one: a
singleton career is a complete, honest answer, not a missing value.

**Determinism**: sorted output throughout, and the null statistic is
computed EXACTLY over all consecutive-season pairs rather than sampled, so
the artifact never depends on a random seed. ``meta.generated_at`` is
normalized on both sides of ``--check-no-commit``.

**CLI**: ``python -m bc_pipeline.career_map --input games/ --output
artifacts/latest/career_map.json``.
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
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "NORMALIZED_TIMESTAMP",
    "CAREER_MAP_VERSION",
    "REASONS",
    "InconsistentCareer",
    "mint_career_id",
    "build_career_map",
    "career_ids_for_persons",
    "normalize_generated_at",
    "load_games",
    "build_arg_parser",
    "main",
]

CAREER_MAP_VERSION = "1.0.0"
NORMALIZED_TIMESTAMP: str = "1970-01-01T00:00:00Z"

#: Closed set of reasons a candidate consecutive-season pair was not linked.
REASONS: Tuple[str, ...] = ("franchise_changed", "ambiguous_within_season")


def mint_career_id(season: int, person_id: str) -> str:
    """Return the career id anchored on a component's EARLIEST member.

    Anchored on the earliest ``(season, person_id)`` rather than hashed over
    the whole membership, so that extending the corpus FORWARD -- adding a
    season, linking one more year onto a career -- does not re-key the
    career and invalidate every stored reference to it. The tradeoff, stated
    rather than hidden: back-filling an EARLIER season would re-key any
    career it extends backwards. The corpus grows forward, so that is the
    rare case.

    The ``career:`` prefix keeps the value out of the ``person_id``
    namespaces (a bare 16-char Presto id, or ``person:<hex>``), so the two
    layers can never be confused for one another.
    """
    digest = hashlib.sha256(f"{season}\x1f{person_id}".encode("utf-8")).hexdigest()
    return f"career:{digest[:16]}"


class _Persons:
    """Per-person facts this module joins on, collected in one pass."""

    def __init__(self) -> None:
        self.season: Dict[str, int] = {}
        #: A person can carry MORE THAN ONE display-name spelling: the corpus
        #: has real Presto ids rendered both "J. Smith" and "Jonathan Smith".
        #: Storing one name per person (last-wins) would silently pick a
        #: spelling and drop the other from name-based grouping -- the same
        #: hazard as reparse's `_committed_id_overrides`. Names are aliases of
        #: a person, so all of them are kept.
        self.names: Dict[str, Set[str]] = defaultdict(set)
        self.franchises: Dict[str, Set[str]] = defaultdict(set)
        self.positions: Dict[str, Set[str]] = defaultdict(set)
        self.dates: Dict[str, Set[str]] = defaultdict(set)
        #: (season, name) -> person_ids seen under it
        self.by_season_name: Dict[Tuple[int, str], Set[str]] = defaultdict(set)


def _collect(games: Iterable[dict]) -> _Persons:
    persons = _Persons()
    for game in games:
        season = int(game["season"])
        franchise_by_team = {
            game["teams"][side]["team_id"]: game["teams"][side].get("franchise_id")
            for side in ("home", "away")
        }
        for entry in (game.get("players") or {}).values():
            person_id = entry.get("person_id")
            if not person_id:
                # Deliberately unlinked at the person layer (a source defect or
                # a same-game conflict). Nothing to build a career on; it stays
                # a measured negative here too rather than being invented.
                continue
            persons.season[person_id] = season
            persons.names[person_id].add(entry["name"])
            franchise = franchise_by_team.get(entry["team_id"])
            if franchise:
                persons.franchises[person_id].add(franchise)
            persons.positions[person_id].update(entry.get("positions") or [])
            persons.dates[person_id].add(game["date"])
            persons.by_season_name[(season, entry["name"])].add(person_id)
    return persons


def _canonical_name(names: Iterable[str]) -> str:
    """Pick one display name to label a career, deterministically.

    Longest first, then alphabetical. Longest because the alternative
    spellings in this corpus are abbreviations of the same name ("J. Smith"
    vs "Jonathan Smith"), so the longest is the most complete rendering. Every
    spelling is kept in the career's ``aliases`` regardless -- this only
    chooses a label, never discards information.
    """
    return sorted(names, key=lambda n: (-len(n), n))[0]


def _proven_same_name_collisions(persons: _Persons) -> List[dict]:
    """Same-name pairs PROVEN to be different people.

    Two persons sharing a display name who appear on the SAME DATE for
    DIFFERENT franchises cannot be one person. This is the only hard ground
    truth available for "does this name identify one human", so it is
    recomputed every build and reported -- it is the evidence that name
    alone is not a sufficient linking rule.
    """
    proven: List[dict] = []
    for (season, name), ids in sorted(persons.by_season_name.items()):
        members = sorted(ids)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                shared_dates = persons.dates[a] & persons.dates[b]
                if shared_dates and not (persons.franchises[a] & persons.franchises[b]):
                    proven.append({
                        "season": season,
                        "name": name,
                        "person_ids": [a, b],
                        "shared_dates": sorted(shared_dates)[:3],
                    })
    return proven


def _signal_strength(persons: _Persons) -> dict:
    """How well each candidate signal separates same-name pairs from the null.

    The NULL is every consecutive-season person pair with a DIFFERENT display
    name -- definitively not the same person. Computed EXACTLY over all such
    pairs, never sampled, so the artifact stays deterministic.

    A signal that fires nearly as often on the null as on the candidates is
    not evidence, however intuitive it sounds. This is what disqualified
    position overlap.
    """
    by_season: Dict[int, List[str]] = defaultdict(list)
    for person_id, season in persons.season.items():
        by_season[season].append(person_id)
    for members in by_season.values():
        members.sort()

    seasons = sorted(by_season)
    cand = {"n": 0, "franchise": 0, "position": 0}
    null = {"n": 0, "franchise": 0, "position": 0}
    for earlier, later in zip(seasons, seasons[1:]):
        for a in by_season[earlier]:
            for b in by_season[later]:
                bucket = (
                    cand if persons.names[a] & persons.names[b] else null
                )
                bucket["n"] += 1
                if persons.franchises[a] & persons.franchises[b]:
                    bucket["franchise"] += 1
                if persons.positions[a] & persons.positions[b]:
                    bucket["position"] += 1

    def rate(bucket: dict, key: str) -> float:
        return round(bucket[key] / bucket["n"], 4) if bucket["n"] else 0.0

    def ratio(key: str) -> Optional[float]:
        null_rate = rate(null, key)
        return round(rate(cand, key) / null_rate, 2) if null_rate else None

    return {
        "same_name_pairs": cand["n"],
        "different_name_pairs_null": null["n"],
        "franchise_continuity": {
            "fires_on_same_name": rate(cand, "franchise"),
            "fires_on_null": rate(null, "franchise"),
            "likelihood_ratio": ratio("franchise"),
            "used": True,
        },
        "position_overlap": {
            "fires_on_same_name": rate(cand, "position"),
            "fires_on_null": rate(null, "position"),
            "likelihood_ratio": ratio("position"),
            "used": False,
            "note": (
                "NOT used. It fires on a large share of the null -- people who are "
                "definitively not each other -- so it barely separates the cases it "
                "is meant to decide. Kept here as a measured negative so the choice "
                "stays visible and will change if the data ever changes."
            ),
        },
    }


def _link_pairs(persons: _Persons) -> Tuple[List[Tuple[str, str]], List[dict]]:
    """Return the accepted ``(earlier, later)`` links and the refusals."""
    by_name: Dict[str, Dict[int, List[str]]] = defaultdict(dict)
    for (season, name), ids in persons.by_season_name.items():
        by_name[name][season] = sorted(ids)
    # `by_season_name` already registers a person under every spelling it was
    # rendered with, because it is keyed on the name AS WRITTEN in each file.

    links: List[Tuple[str, str]] = []
    seen_links: Set[Tuple[str, str]] = set()
    refusals: List[dict] = []
    seen_refusals: Set[Tuple[str, str, str]] = set()
    for name in sorted(by_name):
        seasons = sorted(by_name[name])
        for earlier, later in zip(seasons, seasons[1:]):
            here, there = by_name[name][earlier], by_name[name][later]
            if len(here) > 1 or len(there) > 1:
                key = (min(here + there), max(here + there), "ambiguous_within_season")
                if key not in seen_refusals:
                    seen_refusals.add(key)
                    refusals.append({
                        "name": name, "season_pair": f"{earlier}->{later}",
                        "reason": "ambiguous_within_season",
                        "person_ids": {str(earlier): here, str(later): there},
                    })
                continue
            a, b = here[0], there[0]
            if persons.franchises[a] & persons.franchises[b]:
                # A person carrying two spellings can reach the same pair from
                # either alias; dedupe so one relationship counts once.
                if (a, b) not in seen_links:
                    seen_links.add((a, b))
                    links.append((a, b))
            else:
                key = (a, b, "franchise_changed")
                if key not in seen_refusals:
                    seen_refusals.add(key)
                    refusals.append({
                        "name": name, "season_pair": f"{earlier}->{later}",
                        "reason": "franchise_changed",
                        "person_ids": {str(earlier): [a], str(later): [b]},
                    })
    return links, refusals


def _components(
    person_ids: Iterable[str], links: Iterable[Tuple[str, str]]
) -> Dict[str, List[str]]:
    """Union-find over the accepted links; every person joins a component,
    a singleton one if it linked to nobody."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for person_id in person_ids:
        find(person_id)
    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    components: Dict[str, List[str]] = defaultdict(list)
    for person_id in parent:
        components[find(person_id)].append(person_id)
    return {root: sorted(members) for root, members in components.items()}


class InconsistentCareer(ValueError):
    """A built career violates an invariant that must hold by construction.

    Raised rather than emitted, because a career holding two persons from one
    season -- or mixing display names -- means the linking rule is wrong, not
    that the data is interesting.
    """


def _assert_career_consistent(
    members: Sequence[Tuple[int, str]], alias_sets: Sequence[Set[str]]
) -> None:
    """Assert one built career's two invariants, or raise ``InconsistentCareer``.

    Both are guaranteed by the linking rule -- it refuses a name that resolves
    to more than one person in a season, and it only ever links persons that
    share a display-name spelling --
    so neither is reachable through ``build_career_map``'s public path. That is
    exactly why the check is factored out here: it is a guard against a FUTURE
    change to the rule, and keeping it callable means it can be tested
    directly instead of by a test that re-implements it.
    """
    seasons = [season for season, _ in members]
    if len(set(seasons)) != len(seasons):
        raise InconsistentCareer(
            f"career holds two persons from one season: {list(members)}"
        )
    # Members need not carry IDENTICAL name sets -- one person may be rendered
    # "J. Smith" and "Jonathan Smith" -- but every member must SHARE a spelling
    # with the rest, which is what the linking rule actually guarantees.
    shared = set(alias_sets[0]).intersection(*alias_sets) if alias_sets else set()
    if len(alias_sets) > 1 and not shared:
        raise InconsistentCareer(
            "career mixes display names with no shared spelling: "
            f"{[sorted(a) for a in alias_sets]}: {list(members)}"
        )


def build_career_map(games: Iterable[dict]) -> dict:
    """Build the cross-season career artifact from parsed game dicts.

    Pure function of its input apart from ``meta.generated_at``; never
    mutates ``games``. Raises ``InconsistentCareer`` if a built career breaks
    an invariant the rule guarantees.
    """
    games = list(games)
    persons = _collect(games)
    links, refusals = _link_pairs(persons)
    components = _components(sorted(persons.season), links)

    careers: Dict[str, dict] = {}
    assignments: Dict[str, str] = {}
    for members in components.values():
        by_season = sorted((persons.season[p], p) for p in members)
        _assert_career_consistent(by_season, [persons.names[p] for p in members])
        member_seasons = [season for season, _ in by_season]

        anchor_season, anchor_person = by_season[0]
        career_id = mint_career_id(anchor_season, anchor_person)
        all_aliases = sorted({n for p in members for n in persons.names[p]})
        careers[career_id] = {
            "career_id": career_id,
            "name": _canonical_name(all_aliases),
            "aliases": all_aliases,
            "seasons": member_seasons,
            "person_ids": [[season, p] for season, p in by_season],
            "franchises": sorted(
                {f for p in members for f in persons.franchises[p]}
            ),
        }
        for person_id in members:
            assignments[person_id] = career_id

    reason_counts = {reason: 0 for reason in REASONS}
    for refusal in refusals:
        reason_counts[refusal["reason"]] += 1

    span_counts: Dict[str, int] = defaultdict(int)
    for career in careers.values():
        span_counts[str(len(career["seasons"]))] += 1

    by_season_careers: Dict[str, dict] = {}
    for season in sorted({s for s in persons.season.values()}):
        members = [p for p, s in persons.season.items() if s == season]
        by_season_careers[str(season)] = {
            "persons": len(members),
            "in_a_multi_season_career": sum(
                1 for p in members if len(careers[assignments[p]]["seasons"]) > 1
            ),
        }

    proven = _proven_same_name_collisions(persons)
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "career_map_version": CAREER_MAP_VERSION,
            "games": len(games),
            "persons": len(persons.season),
            "scope": (
                "CROSS-SEASON player identity, one level above person_id. Two persons "
                "in consecutive seasons are linked iff their display names are exactly "
                "equal, that name resolves to exactly one person in EACH season, and "
                "they share a franchise. Careers are the connected components."
            ),
            "evidence": {
                "name_alone_is_insufficient": {
                    "proven_same_name_different_people": len(proven),
                    "cases": proven,
                    "note": (
                        "Two persons sharing a display name who appear on the SAME DATE "
                        "for DIFFERENT franchises cannot be one person. These are the "
                        "hard ground truth that a name-only rule would merge strangers; "
                        "none of them shares a franchise, which is independently why "
                        "franchise continuity is required."
                    ),
                },
                "signal_strength": _signal_strength(persons),
            },
            "not_attempted": {
                "franchise_changed": {
                    "note": (
                        "A player who changed clubs between seasons is NOT linked. This "
                        "is the honest cost of the rule: it is exactly the shape of "
                        "every proven-different pair above, and nothing else in the "
                        "data separates the two cases. Enumerated in `unlinked` so the "
                        "population stays visible and countable."
                    ),
                },
                "within_season_team_change": {
                    "note": (
                        "person_map deliberately does not merge a player across teams "
                        "WITHIN a season (a measured error floor: of the same-season "
                        "same-name cross-franchise cases, some are proven to be two "
                        "people). That ambiguity is inherited here as "
                        "`ambiguous_within_season` rather than papered over."
                    ),
                },
            },
        },
        "league": {
            "persons": len(persons.season),
            "careers": len(careers),
            "links": len(links),
            "multi_season_careers": sum(
                1 for c in careers.values() if len(c["seasons"]) > 1
            ),
            "careers_by_season_span": dict(sorted(span_counts.items())),
            "refused_pairs": len(refusals),
            "refused_by_reason": dict(sorted(reason_counts.items())),
        },
        "by_season": by_season_careers,
        "careers": dict(sorted(careers.items())),
        "assignments": dict(sorted(assignments.items())),
        "unlinked": sorted(
            refusals, key=lambda r: (r["name"], r["season_pair"])
        ),
    }


def career_ids_for_persons(artifact: dict) -> Dict[str, str]:
    """Return ``{person_id: career_id}``.

    The re-parse driver's seam. Keyed by PERSON id, not player id: ``parse``
    already resolves a player to its person (real ids resolve to themselves),
    so composing the two layers there keeps each mapping owned by exactly one
    module.
    """
    return dict(artifact.get("assignments") or {})


def load_games(input_dir: str | Path) -> list[dict]:
    """Load every `*.json` under `input_dir` recursively, sorted by path.
    Read-only; `games/**` is write-once."""
    root = Path(input_dir)
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*.json"))
    ]


def normalize_generated_at(artifact: dict) -> dict:
    """Deep copy of `artifact` with `meta.generated_at` set to
    `NORMALIZED_TIMESTAMP`. Mirrors the sibling artifact modules."""
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
        prog="python -m bc_pipeline.career_map",
        description=(
            "Build the cross-season career_id map (artifacts/latest/career_map.json), "
            "linking person_ids across seasons on name + franchise continuity."
        ),
    )
    parser.add_argument("--input", type=str, default="games", metavar="DIR",
                        help="games/** root to read (default: games).")
    parser.add_argument("--output", type=str,
                        default="artifacts/latest/career_map.json", metavar="PATH",
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
        fresh = build_career_map(load_games(args.input))
    except InconsistentCareer as exc:
        print(f"[CAREER_MAP] REFUSING: {exc}", file=sys.stderr)
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
                "[CAREER_MAP] CHANGED: regenerated artifact differs from the committed "
                f"{output_path} (or none is committed yet); commit needed.",
                file=sys.stderr,
            )
            return 2
        print(
            "[CAREER_MAP] NO-OP: regenerated artifact matches the committed "
            f"{output_path} (generated_at normalized on both sides)."
        )
        return 0

    _write_artifact(fresh, output_path)
    league = fresh["league"]
    print(
        f"[CAREER_MAP] wrote {output_path} -- {league['careers']} career(s) from "
        f"{league['persons']} person(s); {league['multi_season_careers']} span more "
        f"than one season; {league['refused_pairs']} candidate pair(s) refused."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
