"""Tests for bc_pipeline.team_map -- cross-season franchise identity
(issue #41, team half).

Protected intent: `team_id` is reissued every season, so it cannot carry a
multi-season question about a club. `franchise_id` can. As with the person
map, most of what follows tests the REFUSALS -- a franchise map that
cheerfully links a relocated club to whichever team happens to share players
with it is worse than one that says "unknown".
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bc_pipeline import parse, team_map

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = REPO_ROOT / "games"
GAME_SCHEMA_PATH = REPO_ROOT / "schemas" / "game.schema.json"


def _game(game_id, season, home, away, players=None):
    return {
        "game_id": game_id,
        "season": season,
        "teams": {
            "home": {"team_id": home[0], "name": home[1]},
            "away": {"team_id": away[0], "name": away[1]},
        },
        "players": players or {},
    }


# --- the key ----------------------------------------------------------------


def test_one_franchise_across_seasons_despite_reissued_team_ids():
    art = team_map.build_team_map([
        _game("g1", 2024, ("id2024aaaaaaaaaa", "Boise Hawks"), ("id2024bbbbbbbbbb", "Ogden Raptors")),
        _game("g2", 2025, ("id2025aaaaaaaaaa", "Boise Hawks"), ("id2025bbbbbbbbbb", "Ogden Raptors")),
    ])
    hawks = team_map.mint_franchise_id("Boise Hawks")
    assert art["assignments"]["2024"]["id2024aaaaaaaaaa"] == hawks
    assert art["assignments"]["2025"]["id2025aaaaaaaaaa"] == hawks
    assert art["franchises"][hawks]["seasons"] == [2024, 2025]
    assert art["franchises"][hawks]["team_ids"] == {
        "2024": "id2024aaaaaaaaaa", "2025": "id2025aaaaaaaaaa",
    }
    assert art["league"]["franchises"] == 2


def test_minting_is_a_pure_function_of_the_name():
    a = team_map.mint_franchise_id("Boise Hawks")
    assert a == team_map.mint_franchise_id("Boise Hawks")
    assert a != team_map.mint_franchise_id("Ogden Raptors")
    assert a.startswith("franchise:")
    assert len(a) == len("franchise:") + 16


def test_parse_needs_no_artifact_to_populate_franchise_id():
    """Unlike person_id, franchise_id is a pure function of the name in the
    file -- so there is no artifact dependency and no drift to measure."""
    assert team_map.mint_franchise_id("Boise Hawks").startswith("franchise:")
    assert parse.SCHEMA_VERSION == "1.8.0"


# --- the refusals -----------------------------------------------------------


def test_a_departed_and_an_arrived_team_are_not_linked():
    """2025 lost three clubs and 2026 gained three. Whether any arrival is a
    relocation is not determinable here, so they stay separate franchises and
    the question is left VISIBLE in `continuity`."""
    art = team_map.build_team_map([
        _game("g1", 2025, ("id2025aaaaaaaaaa", "Boise Hawks"), ("id2025bbbbbbbbbb", "Rocky Mountain Vibes")),
        _game("g2", 2026, ("id2026aaaaaaaaaa", "Boise Hawks"), ("id2026bbbbbbbbbb", "Long Beach Coast")),
    ])
    assert art["continuity"]["2025->2026"]["departed"] == ["Rocky Mountain Vibes"]
    assert art["continuity"]["2025->2026"]["arrived"] == ["Long Beach Coast"]
    assert team_map.mint_franchise_id("Rocky Mountain Vibes") != team_map.mint_franchise_id(
        "Long Beach Coast"
    )
    assert art["league"]["franchises"] == 3


def test_two_team_ids_for_one_name_in_a_season_is_refused_loudly():
    """The whole key rests on name <-> team_id being 1:1 within a season. A
    violation means the key is unsafe, so it fails rather than degrading."""
    with pytest.raises(team_map.AmbiguousTeamIdentity, match="not safe"):
        team_map.build_team_map([
            _game("g1", 2026, ("id2026aaaaaaaaaa", "Boise Hawks"), ("id2026bbbbbbbbbb", "Ogden Raptors")),
            _game("g2", 2026, ("id2026ccccccccc1", "Boise Hawks"), ("id2026bbbbbbbbbb", "Ogden Raptors")),
        ])


def test_one_team_id_carrying_two_names_is_refused_loudly():
    with pytest.raises(team_map.AmbiguousTeamIdentity, match="two names"):
        team_map.build_team_map([
            _game("g1", 2026, ("id2026aaaaaaaaaa", "Boise Hawks"), ("id2026bbbbbbbbbb", "Ogden Raptors")),
            _game("g2", 2026, ("id2026aaaaaaaaaa", "Boise Falcons"), ("id2026bbbbbbbbbb", "Ogden Raptors")),
        ])


def test_a_synthetic_team_id_resolves_by_name_and_never_becomes_a_key():
    """A team-site boxscore links only the host's caption, so the opponent
    gets a file-local `syn:team:<side>`. That id denotes a different club in
    every file, so it must never be a join key -- but the NAME is rendered
    correctly, so the franchise still resolves."""
    art = team_map.build_team_map([
        _game("g1", 2026, ("id2026aaaaaaaaaa", "Boise Hawks"), ("syn:team:away", "Ogden Raptors")),
        _game("g2", 2026, ("id2026aaaaaaaaaa", "Boise Hawks"), ("syn:team:away", "Great Falls Voyagers")),
    ])
    assert "syn:team:away" not in art["assignments"]["2026"]
    # Two records: one per (season, syn:team:away, name) triple. Keying by
    # the id alone would have collapsed them and silently lost a franchise.
    assert art["league"]["synthetic_team_records"] == 2
    # Both opponents are still registered as their own franchises.
    for name in ("Ogden Raptors", "Great Falls Voyagers"):
        assert team_map.mint_franchise_id(name) in art["franchises"]


def test_roster_overlap_is_measured_and_reported_as_unreliable():
    """The refusal to infer relocation from shared players is evidence-backed
    and recomputed every run, so it cannot decay into folklore."""
    art = team_map.build_team_map(team_map.load_games(GAMES_DIR))
    signal = art["meta"]["not_attempted"]["roster_signal"]
    assert signal["checkable_season_pairs"] > 0
    # If this ever becomes perfect, the refusal deserves revisiting -- but on
    # the current corpus roster overlap picks the WRONG team on known cases.
    assert signal["top_match_was_correct"] < signal["checkable_season_pairs"]
    assert signal["misidentified"]


# --- determinism ------------------------------------------------------------


def test_build_is_deterministic_and_does_not_mutate_input():
    games = [
        _game("g1", 2024, ("id2024aaaaaaaaaa", "Boise Hawks"), ("id2024bbbbbbbbbb", "Ogden Raptors")),
        _game("g2", 2025, ("id2025aaaaaaaaaa", "Boise Hawks"), ("id2025bbbbbbbbbb", "Ogden Raptors")),
    ]
    before = copy.deepcopy(games)
    a = team_map.normalize_generated_at(team_map.build_team_map(games))
    b = team_map.normalize_generated_at(team_map.build_team_map(list(reversed(games))))
    assert a == b
    assert games == before


# --- the real corpus --------------------------------------------------------


def test_corpus_no_team_id_survives_a_season_and_every_name_does():
    art = team_map.build_team_map(team_map.load_games(GAMES_DIR))
    assert art["league"]["team_ids_preserved_across_seasons"] == 0
    assert art["league"]["multi_season_franchises"] == 12
    assert art["league"]["franchises"] == 15
    for span in art["continuity"].values():
        assert span["team_ids_preserved"] == 0


def test_corpus_franchise_ids_satisfy_the_schema_pattern():
    schema = json.loads(GAME_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(
        schema["$defs"]["team"]["properties"]["franchise_id"]
    )
    art = team_map.build_team_map(team_map.load_games(GAMES_DIR))
    for franchise_id in art["franchises"]:
        validator.validate(franchise_id)


def test_cli_writes_and_then_reports_no_op(tmp_path):
    out = tmp_path / "team_map.json"
    argv = ["--input", str(GAMES_DIR), "--output", str(out)]
    assert team_map.main(argv) == 0
    assert team_map.main(argv + ["--check-no-commit"]) == 0
