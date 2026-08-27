"""Tests for bc_pipeline.career_map -- cross-season player identity
(issue #41, Gap 1).

Protected intent: this is the layer most able to do real damage. Merging two
strangers into one career silently corrupts every multi-season stat derived
from it, and unlike the within-season case there is no temporal check
available to catch it afterwards. So the rule is deliberately conservative
and most of what follows tests the REFUSALS.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bc_pipeline import career_map, parse

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = REPO_ROOT / "games"
GAME_SCHEMA_PATH = REPO_ROOT / "schemas" / "game.schema.json"

FR_A, FR_B = "franchise:aaaaaaaaaaaaaaaa", "franchise:bbbbbbbbbbbbbbbb"


def _game(game_id, season, date, franchise, players, *, away_franchise=FR_B):
    """One game: `players` is a list of (person_id, name, positions)."""
    return {
        "game_id": game_id,
        "season": season,
        "date": date,
        "teams": {
            "home": {"team_id": "home_team_id00", "name": "H", "franchise_id": franchise},
            "away": {"team_id": "away_team_id00", "name": "A", "franchise_id": away_franchise},
        },
        "players": {
            person_id: {
                "name": name, "team_id": "home_team_id00",
                "person_id": person_id, "positions": list(positions),
            }
            for person_id, name, positions in players
        },
    }


# --- linking ----------------------------------------------------------------


def test_same_name_same_franchise_links_across_seasons():
    art = career_map.build_career_map([
        _game("g1", 2025, "2025-05-01", FR_A, [("p2025", "Ann Player", ["ss"])]),
        _game("g2", 2026, "2026-05-01", FR_A, [("p2026", "Ann Player", ["ss"])]),
    ])
    assert art["assignments"]["p2025"] == art["assignments"]["p2026"]
    career = art["careers"][art["assignments"]["p2025"]]
    assert career["seasons"] == [2025, 2026]
    assert art["league"]["multi_season_careers"] == 1


def test_a_career_chains_across_three_seasons():
    art = career_map.build_career_map([
        _game("g1", 2024, "2024-05-01", FR_A, [("p2024", "Ann Player", ["ss"])]),
        _game("g2", 2025, "2025-05-01", FR_A, [("p2025", "Ann Player", ["ss"])]),
        _game("g3", 2026, "2026-05-01", FR_A, [("p2026", "Ann Player", ["ss"])]),
    ])
    assert len({art["assignments"][p] for p in ("p2024", "p2025", "p2026")}) == 1
    assert art["careers"][art["assignments"]["p2024"]]["seasons"] == [2024, 2025, 2026]


def test_career_id_is_anchored_on_the_earliest_member():
    """Anchored on the earliest member so extending the corpus FORWARD does
    not re-key an existing career and invalidate stored references."""
    two = career_map.build_career_map([
        _game("g1", 2024, "2024-05-01", FR_A, [("p2024", "Ann Player", ["ss"])]),
        _game("g2", 2025, "2025-05-01", FR_A, [("p2025", "Ann Player", ["ss"])]),
    ])
    three = career_map.build_career_map([
        _game("g1", 2024, "2024-05-01", FR_A, [("p2024", "Ann Player", ["ss"])]),
        _game("g2", 2025, "2025-05-01", FR_A, [("p2025", "Ann Player", ["ss"])]),
        _game("g3", 2026, "2026-05-01", FR_A, [("p2026", "Ann Player", ["ss"])]),
    ])
    assert two["assignments"]["p2024"] == three["assignments"]["p2024"]
    assert career_map.mint_career_id(2024, "p2024") == two["assignments"]["p2024"]


def test_an_unlinked_person_still_gets_a_singleton_career():
    """A singleton career is a complete answer, not a missing value."""
    art = career_map.build_career_map([
        _game("g1", 2026, "2026-05-01", FR_A, [("p1", "Solo Player", ["ss"])]),
    ])
    assert art["assignments"]["p1"] == career_map.mint_career_id(2026, "p1")
    assert art["careers"][art["assignments"]["p1"]]["seasons"] == [2026]


# --- the refusals -----------------------------------------------------------


def test_a_franchise_change_between_seasons_is_not_linked():
    """The honest cost of the rule. It is exactly the shape of every
    proven-different pair in the corpus, and nothing separates the two."""
    art = career_map.build_career_map([
        _game("g1", 2025, "2025-05-01", FR_A, [("p2025", "Ann Player", ["ss"])]),
        _game("g2", 2026, "2026-05-01", FR_B, [("p2026", "Ann Player", ["ss"])],
              away_franchise=FR_A),
    ])
    assert art["assignments"]["p2025"] != art["assignments"]["p2026"]
    assert art["league"]["refused_by_reason"]["franchise_changed"] == 1
    assert [u["reason"] for u in art["unlinked"]] == ["franchise_changed"]


def test_position_overlap_alone_never_links():
    """Position overlap fires on ~45% of people who are definitively not each
    other, so it is measured and reported but never acted on."""
    art = career_map.build_career_map([
        _game("g1", 2025, "2025-05-01", FR_A, [("p2025", "Ann Player", ["ss", "2b"])]),
        _game("g2", 2026, "2026-05-01", FR_B, [("p2026", "Ann Player", ["ss", "2b"])],
              away_franchise=FR_A),
    ])
    assert art["assignments"]["p2025"] != art["assignments"]["p2026"]
    assert art["meta"]["evidence"]["signal_strength"]["position_overlap"]["used"] is False


def test_two_people_under_one_name_in_a_season_blocks_the_link():
    """person_map does not merge across teams within a season, so that
    ambiguity is inherited here rather than papered over."""
    art = career_map.build_career_map([
        _game("g1", 2025, "2025-05-01", FR_A, [("p2025a", "Ann Player", ["ss"])]),
        _game("g2", 2025, "2025-05-02", FR_A, [("p2025b", "Ann Player", ["ss"])]),
        _game("g3", 2026, "2026-05-01", FR_A, [("p2026", "Ann Player", ["ss"])]),
    ])
    assert len({art["assignments"][p] for p in ("p2025a", "p2025b", "p2026")}) == 3
    assert art["league"]["refused_by_reason"]["ambiguous_within_season"] == 1


def test_different_names_never_link():
    art = career_map.build_career_map([
        _game("g1", 2025, "2025-05-01", FR_A, [("p2025", "Ann Player", ["ss"])]),
        _game("g2", 2026, "2026-05-01", FR_A, [("p2026", "Anne Player", ["ss"])]),
    ])
    assert art["assignments"]["p2025"] != art["assignments"]["p2026"]
    assert art["league"]["links"] == 0


def test_a_person_with_no_person_id_gets_no_career():
    art = career_map.build_career_map([
        {
            "game_id": "g1", "season": 2026, "date": "2026-05-01",
            "teams": {
                "home": {"team_id": "t1", "name": "H", "franchise_id": FR_A},
                "away": {"team_id": "t2", "name": "A", "franchise_id": FR_B},
            },
            "players": {
                "syn:home:1": {"name": "/", "team_id": "t1", "person_id": None,
                               "positions": []},
            },
        },
    ])
    assert art["assignments"] == {}
    assert art["careers"] == {}


def test_the_career_invariants_fail_loudly_when_violated():
    """Both invariants are unreachable through the public path -- the rule
    refuses an ambiguous season and only links equal names -- so the guard is
    factored out and tested directly rather than by a test that re-implements
    it. It exists to catch a FUTURE change to the rule."""
    with pytest.raises(career_map.InconsistentCareer, match="two persons from one season"):
        career_map._assert_career_consistent(
            [(2025, "pa"), (2025, "pb")], [{"Ann Player"}, {"Ann Player"}]
        )
    with pytest.raises(career_map.InconsistentCareer, match="no shared spelling"):
        career_map._assert_career_consistent(
            [(2025, "pa"), (2026, "pb")], [{"Ann Player"}, {"Bob Other"}]
        )


def test_a_well_formed_career_passes_the_invariants():
    career_map._assert_career_consistent(
        [(2025, "pa"), (2026, "pb")], [{"Ann Player"}, {"Ann Player"}]
    )


def test_members_may_carry_different_spellings_if_one_is_shared():
    """A person can be rendered both "J. Smith" and "Jonathan Smith". Members
    need not carry identical name SETS -- only a shared spelling."""
    career_map._assert_career_consistent(
        [(2025, "pa"), (2026, "pb")],
        [{"J. Smith", "Jonathan Smith"}, {"Jonathan Smith"}],
    )


# --- evidence ---------------------------------------------------------------


def test_corpus_proves_name_alone_would_merge_strangers():
    """The hard ground truth the rule is built on: same display name, same
    date, different franchises -- one person cannot do that."""
    art = career_map.build_career_map(career_map.load_games(GAMES_DIR))
    evidence = art["meta"]["evidence"]["name_alone_is_insufficient"]
    assert evidence["proven_same_name_different_people"] >= 1
    for case in evidence["cases"]:
        assert case["shared_dates"]
        assert len(case["person_ids"]) == 2


def test_corpus_franchise_continuity_outranks_position_overlap():
    """The measured justification for using one signal and not the other. If
    this ever inverts, the rule deserves revisiting -- so it is asserted."""
    art = career_map.build_career_map(career_map.load_games(GAMES_DIR))
    signals = art["meta"]["evidence"]["signal_strength"]
    franchise = signals["franchise_continuity"]["likelihood_ratio"]
    position = signals["position_overlap"]["likelihood_ratio"]
    assert franchise > position
    assert franchise >= 5
    # Position overlap fires on a large share of definitely-unrelated pairs.
    assert signals["position_overlap"]["fires_on_null"] > 0.2


def test_corpus_invariants_hold():
    art = career_map.build_career_map(career_map.load_games(GAMES_DIR))
    for career in art["careers"].values():
        assert len(set(career["seasons"])) == len(career["seasons"])
        assert career["seasons"] == sorted(career["seasons"])
    # Every person lands in exactly one career.
    assert len(art["assignments"]) == art["league"]["persons"]
    assert set(art["assignments"].values()) == set(art["careers"])


def test_corpus_every_career_member_shares_a_spelling():
    """The alias-aware form of "no career mixes people". A person can carry
    more than one display-name spelling (the corpus has real Presto ids
    rendered both `J. Smith` and `Jonathan Smith`), so members are compared on
    their full ALIAS SETS -- checking a single last-wins name per person would
    miss exactly the case this guards."""
    games = career_map.load_games(GAMES_DIR)
    art = career_map.build_career_map(games)
    aliases = {}
    for game in games:
        for entry in game["players"].values():
            if entry.get("person_id"):
                aliases.setdefault(entry["person_id"], set()).add(entry["name"])
    for career in art["careers"].values():
        sets = [aliases[p] for _s, p in career["person_ids"]]
        shared = set(sets[0]).intersection(*sets)
        assert shared, (career["career_id"], [sorted(a) for a in sets])
        assert career["name"] in career["aliases"]


def test_corpus_a_person_with_two_spellings_stays_one_career():
    """Regression: grouping by name registers such a person under BOTH
    spellings, so the same relationship can be reached twice. It must produce
    one career and one link, not two."""
    games = career_map.load_games(GAMES_DIR)
    art = career_map.build_career_map(games)
    aliases = {}
    for game in games:
        for entry in game["players"].values():
            if entry.get("person_id"):
                aliases.setdefault(entry["person_id"], set()).add(entry["name"])
    multi = [p for p, names in aliases.items() if len(names) > 1]
    assert multi, "corpus should still contain a multi-spelling person"
    for person_id in multi:
        career = art["careers"][art["assignments"][person_id]]
        assert len(career["person_ids"]) == len({p for _s, p in career["person_ids"]})
        assert aliases[person_id] <= set(career["aliases"])


# --- determinism ------------------------------------------------------------


def test_build_is_deterministic_and_does_not_mutate_input():
    games = [
        _game("g1", 2025, "2025-05-01", FR_A, [("p2025", "Ann Player", ["ss"])]),
        _game("g2", 2026, "2026-05-01", FR_A, [("p2026", "Ann Player", ["ss"])]),
    ]
    before = copy.deepcopy(games)
    a = career_map.normalize_generated_at(career_map.build_career_map(games))
    b = career_map.normalize_generated_at(career_map.build_career_map(list(reversed(games))))
    assert a == b
    assert games == before


def test_the_null_statistic_is_exact_not_sampled():
    """A sampled null would make the artifact non-deterministic. Two builds of
    the same corpus must report identical signal strength."""
    games = career_map.load_games(GAMES_DIR)
    a = career_map.build_career_map(games)["meta"]["evidence"]["signal_strength"]
    b = career_map.build_career_map(games)["meta"]["evidence"]["signal_strength"]
    assert a == b


def test_corpus_career_ids_satisfy_the_schema_pattern():
    schema = json.loads(GAME_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(
        schema["$defs"]["player_entry"]["properties"]["career_id"]
    )
    art = career_map.build_career_map(career_map.load_games(GAMES_DIR))
    for career_id in art["careers"]:
        validator.validate(career_id)


def test_parse_gives_no_career_to_a_player_with_no_person():
    """A player we could not identify has no career either -- never a guess."""
    assert parse._person_id_for("syn:away:1", None) is None


def test_cli_writes_and_then_reports_no_op(tmp_path):
    out = tmp_path / "career_map.json"
    argv = ["--input", str(GAMES_DIR), "--output", str(out)]
    assert career_map.main(argv) == 0
    assert career_map.main(argv + ["--check-no-commit"]) == 0
