"""Tests for bc_pipeline.person_map -- the within-season cross-GAME person id
(issue #41, Gap 2).

Protected intent: `player_id` cannot be joined on across games, because a
synthetic `syn:<side>:<n>` is assigned by boxscore row order and means a
different person in every file. `person_id` is the id that CAN be joined on.
The whole value of the layer is that it never merges two people -- so most of
what follows tests the REFUSALS, not the links.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bc_pipeline import parse, person_map

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = REPO_ROOT / "games"
GAME_SCHEMA_PATH = REPO_ROOT / "schemas" / "game.schema.json"


def _game(game_id: str, season: int, players: dict) -> dict:
    """A minimal game dict carrying only what person_map reads."""
    return {"game_id": game_id, "season": season, "players": players}


def _player(player_id: str, name: str, team_id: str) -> dict:
    return {"player_id": player_id, "name": name, "team_id": team_id}


TEAM_A = "aaaaaaaaaaaaaaaa"
TEAM_B = "bbbbbbbbbbbbbbbb"


# --- the identity rule for real ids ----------------------------------------


def test_real_player_id_is_its_own_person_id():
    """A real Presto id is stable for a whole season, so it needs no
    consolidation and is never absorbed into some other person."""
    art = person_map.build_person_map(
        [_game("g1", 2026, {"realaaaaaaaaaaa1": _player("realaaaaaaaaaaa1", "Ann Real", TEAM_A)})]
    )
    assert "realaaaaaaaaaaa1" in art["persons"]
    # Real ids are NOT restated in `assignments` -- that map is synthetics only.
    assert art["assignments"] == {}


def test_parse_resolves_a_real_id_without_any_person_map():
    """The real-id rule lives in parse, so a fresh parse of a game with no
    artifact available still emits a usable person_id for real players."""
    assert parse._person_id_for("realaaaaaaaaaaa1", None) == "realaaaaaaaaaaa1"


def test_parse_never_falls_back_to_the_synthetic_id_itself():
    """Emitting `syn:away:3` as a person_id would fabricate a join key that
    silently merges strangers across games -- the exact defect this layer
    exists to prevent. An honest null instead."""
    assert parse._person_id_for("syn:away:3", None) is None
    assert parse._person_id_for("syn:away:3", {}) is None
    assert parse._person_id_for("syn:away:3", {"syn:away:3": None}) is None


# --- linking ----------------------------------------------------------------


def test_synthetics_consolidate_onto_a_single_real_anchor():
    art = person_map.build_person_map([
        _game("g1", 2026, {"realaaaaaaaaaaa1": _player("realaaaaaaaaaaa1", "Ann Real", TEAM_A)}),
        _game("g2", 2026, {"syn:away:3": _player("syn:away:3", "Ann Real", TEAM_A)}),
        _game("g3", 2026, {"syn:home:7": _player("syn:home:7", "Ann Real", TEAM_A)}),
    ])
    assert art["assignments"]["g2"]["syn:away:3"] == "realaaaaaaaaaaa1"
    assert art["assignments"]["g3"]["syn:home:7"] == "realaaaaaaaaaaa1"
    assert art["persons"]["realaaaaaaaaaaa1"]["games"] == 3
    assert art["league"]["groups_by_reason"]["real_anchor"] == 1


def test_a_group_with_no_real_id_gets_one_minted_person():
    art = person_map.build_person_map([
        _game("g1", 2026, {"syn:away:1": _player("syn:away:1", "Bob Ghost", TEAM_A)}),
        _game("g2", 2026, {"syn:home:9": _player("syn:home:9", "Bob Ghost", TEAM_A)}),
    ])
    minted = person_map.mint_person_id(2026, TEAM_A, "Bob Ghost")
    assert art["assignments"]["g1"]["syn:away:1"] == minted
    assert art["assignments"]["g2"]["syn:home:9"] == minted
    assert art["persons"][minted]["origin"] == "minted"


def test_a_lone_synthetic_is_still_minted():
    """`syn:away:3` appearing in only one game still needs a person id: the
    value is reused by other people in other games, so leaving it as-is would
    make it unjoinable AND collide."""
    art = person_map.build_person_map(
        [_game("g1", 2026, {"syn:away:3": _player("syn:away:3", "Cal Once", TEAM_A)})]
    )
    assert art["assignments"]["g1"]["syn:away:3"] == person_map.mint_person_id(
        2026, TEAM_A, "Cal Once"
    )


def test_minting_is_a_pure_function_of_the_group_key():
    a = person_map.mint_person_id(2026, TEAM_A, "Bob Ghost")
    assert a == person_map.mint_person_id(2026, TEAM_A, "Bob Ghost")
    assert a != person_map.mint_person_id(2025, TEAM_A, "Bob Ghost")
    assert a != person_map.mint_person_id(2026, TEAM_B, "Bob Ghost")
    assert a != person_map.mint_person_id(2026, TEAM_A, "Bob Ghoul")


def test_minted_ids_live_in_their_own_namespace():
    """A minted id must never be mistakable for a file-local player_id: real
    is bare [a-z0-9]{16}, synthetic is syn:<side>:<n>."""
    minted = person_map.mint_person_id(2026, TEAM_A, "Bob Ghost")
    assert minted.startswith("person:")
    assert not person_map.is_synthetic(minted)
    assert len(minted) == len("person:") + 16


# --- the refusals -----------------------------------------------------------


def test_same_name_on_two_teams_stays_two_persons():
    """The key includes team_id. Two rosters carrying the same name are not
    evidence of one person, and this layer does not attempt the link."""
    art = person_map.build_person_map([
        _game("g1", 2026, {"syn:away:1": _player("syn:away:1", "Dee Twin", TEAM_A)}),
        _game("g2", 2026, {"syn:away:1": _player("syn:away:1", "Dee Twin", TEAM_B)}),
    ])
    assert art["assignments"]["g1"]["syn:away:1"] != art["assignments"]["g2"]["syn:away:1"]
    assert art["meta"]["not_attempted"]["cross_team_within_season"][
        "names_on_more_than_one_team"
    ] == 1


def test_same_name_in_two_seasons_stays_two_persons():
    art = person_map.build_person_map([
        _game("g1", 2025, {"syn:away:1": _player("syn:away:1", "Eve Year", TEAM_A)}),
        _game("g2", 2026, {"syn:away:1": _player("syn:away:1", "Eve Year", TEAM_A)}),
    ])
    assert art["assignments"]["g1"]["syn:away:1"] != art["assignments"]["g2"]["syn:away:1"]
    assert art["meta"]["not_attempted"]["cross_season"]["names_in_more_than_one_season"] == 1


def test_two_ids_in_one_game_refuses_the_merge():
    """One person cannot hold two ids in the same game. Either they are two
    people or the file has a defect -- neither supports a merge."""
    art = person_map.build_person_map([
        _game("g1", 2026, {
            "syn:home:1": _player("syn:home:1", "R. Velazquez", TEAM_A),
            "syn:home:4": _player("syn:home:4", "R. Velazquez", TEAM_A),
        }),
    ])
    assert art["assignments"]["g1"]["syn:home:1"] is None
    assert art["assignments"]["g1"]["syn:home:4"] is None
    assert [u["reason"] for u in art["unlinked"]] == ["same_game_conflict"]


def test_same_game_conflict_beats_an_otherwise_valid_real_anchor():
    """Refusals are evaluated BEFORE the linking rules, so a defect can never
    be merged away by an anchor that happens to exist."""
    art = person_map.build_person_map([
        _game("g1", 2026, {"realaaaaaaaaaaa1": _player("realaaaaaaaaaaa1", "Fay Split", TEAM_A)}),
        _game("g2", 2026, {
            "syn:home:1": _player("syn:home:1", "Fay Split", TEAM_A),
            "syn:home:2": _player("syn:home:2", "Fay Split", TEAM_A),
        }),
    ])
    assert art["assignments"]["g2"] == {"syn:home:1": None, "syn:home:2": None}
    assert art["league"]["groups_by_reason"]["same_game_conflict"] == 1


def test_two_real_ids_leaves_synthetics_unlinked_but_keeps_both_persons():
    art = person_map.build_person_map([
        _game("g1", 2026, {"realaaaaaaaaaaa1": _player("realaaaaaaaaaaa1", "Gus Dup", TEAM_A)}),
        _game("g2", 2026, {"realaaaaaaaaaaa2": _player("realaaaaaaaaaaa2", "Gus Dup", TEAM_A)}),
        _game("g3", 2026, {"syn:away:2": _player("syn:away:2", "Gus Dup", TEAM_A)}),
    ])
    assert art["assignments"]["g3"]["syn:away:2"] is None
    assert art["league"]["groups_by_reason"]["multi_real_id"] == 1
    # Both real ids remain their own persons -- the refusal is about which one
    # the SYNTHETIC belongs to, not about the real ids' own validity.
    assert {"realaaaaaaaaaaa1", "realaaaaaaaaaaa2"} <= set(art["persons"])


def test_the_slash_source_defect_is_never_a_person():
    """StatCrew's `/ for X` line omits the incoming player's name; the parser
    admits the PBP-declared player, but "/" is not somebody."""
    art = person_map.build_person_map([
        _game("g1", 2026, {"syn:away:1": _player("syn:away:1", "/", TEAM_A)}),
        _game("g2", 2026, {"syn:away:6": _player("syn:away:6", "/", TEAM_A)}),
    ])
    assert art["assignments"]["g1"]["syn:away:1"] is None
    assert art["assignments"]["g2"]["syn:away:6"] is None
    assert [u["reason"] for u in art["unlinked"]] == ["non_person_name"]


def test_every_group_is_classified_with_a_known_reason():
    """Closed set: a group is never silently dropped."""
    art = person_map.build_person_map(person_map.load_games(GAMES_DIR))
    assert set(art["league"]["groups_by_reason"]) == set(person_map.REASONS)
    assert sum(art["league"]["groups_by_reason"].values()) == art["league"]["groups"]


# --- determinism / idempotence ---------------------------------------------


def test_build_is_deterministic_modulo_generated_at():
    games = [
        _game("g1", 2026, {"syn:away:1": _player("syn:away:1", "Bob Ghost", TEAM_A)}),
        _game("g2", 2026, {"realaaaaaaaaaaa1": _player("realaaaaaaaaaaa1", "Ann Real", TEAM_A)}),
    ]
    a = person_map.normalize_generated_at(person_map.build_person_map(games))
    b = person_map.normalize_generated_at(person_map.build_person_map(list(reversed(games))))
    assert a == b


def test_build_does_not_mutate_its_input():
    games = [_game("g1", 2026, {"syn:away:1": _player("syn:away:1", "Bob Ghost", TEAM_A)})]
    before = copy.deepcopy(games)
    person_map.build_person_map(games)
    assert games == before


def test_rebuilding_after_person_ids_are_written_back_is_idempotent():
    """The artifact is derived FROM games/** and then written back INTO it.
    That is only safe if the map is a function of fields the write-back does
    not touch. Simulate the round trip and assert the map is unchanged."""
    games = [
        _game("g1", 2026, {"realaaaaaaaaaaa1": _player("realaaaaaaaaaaa1", "Ann Real", TEAM_A)}),
        _game("g2", 2026, {"syn:away:3": _player("syn:away:3", "Ann Real", TEAM_A)}),
        _game("g3", 2026, {"syn:home:1": _player("syn:home:1", "Bob Ghost", TEAM_A)}),
    ]
    first = person_map.build_person_map(games)
    written = copy.deepcopy(games)
    for game in written:
        for pid, entry in game["players"].items():
            entry["person_id"] = parse._person_id_for(
                pid, person_map.assignments_for_game(first, game["game_id"])
            )
    second = person_map.build_person_map(written)
    assert person_map.normalize_generated_at(first) == person_map.normalize_generated_at(second)


def test_assignments_for_game_is_synthetics_only_and_safe_on_a_miss():
    art = person_map.build_person_map([
        _game("g1", 2026, {
            "realaaaaaaaaaaa1": _player("realaaaaaaaaaaa1", "Ann Real", TEAM_A),
            "syn:away:3": _player("syn:away:3", "Bob Ghost", TEAM_A),
        }),
    ])
    assert set(person_map.assignments_for_game(art, "g1")) == {"syn:away:3"}
    assert person_map.assignments_for_game(art, "nope") == {}


# --- the real corpus --------------------------------------------------------


def test_every_person_id_satisfies_the_schema_pattern():
    """person_id is `["string","null"]` with a pattern that admits a real id
    or a minted `person:<16 hex>` -- and deliberately NOT a syn: value."""
    schema = json.loads(GAME_SCHEMA_PATH.read_text(encoding="utf-8"))
    prop = schema["$defs"]["player_entry"]["properties"]["person_id"]
    validator = Draft202012Validator(prop)
    art = person_map.build_person_map(person_map.load_games(GAMES_DIR))
    for person_id in art["persons"]:
        validator.validate(person_id)
    for by_pid in art["assignments"].values():
        for person_id in by_pid.values():
            validator.validate(person_id)
            assert person_id is None or not person_map.is_synthetic(person_id)


def test_corpus_synthetic_records_are_fully_accounted_for():
    """Linked + unlinked must equal the total; nothing may go missing."""
    art = person_map.build_person_map(person_map.load_games(GAMES_DIR))
    league = art["league"]
    assert league["synthetic_linked"] + league["synthetic_unlinked"] == league["synthetic_records"]
    assert sum(league["synthetic_records_by_reason"].values()) == league["synthetic_records"]


def test_corpus_manny_and_marquis_jackson_stay_distinct_people():
    """The standing same-name proof in this corpus. Both are on one roster in
    2025 and again in 2026; a layer that merged them would be wrong."""
    art = person_map.build_person_map(person_map.load_games(GAMES_DIR))
    jacksons = {
        (p["season"], p["name"]): pid
        for pid, p in art["persons"].items()
        if p["name"] in ("Manny Jackson", "Marquis Jackson")
    }
    assert len(set(jacksons.values())) == len(jacksons) >= 2
    for season in {s for s, _ in jacksons}:
        manny = jacksons.get((season, "Manny Jackson"))
        marquis = jacksons.get((season, "Marquis Jackson"))
        if manny and marquis:
            assert manny != marquis


def test_cli_writes_and_then_reports_no_op(tmp_path):
    out = tmp_path / "person_map.json"
    argv = ["--input", str(GAMES_DIR), "--output", str(out)]
    assert person_map.main(argv) == 0
    assert out.exists()
    assert person_map.main(argv + ["--check-no-commit"]) == 0


def test_cli_check_no_commit_reports_changed_when_nothing_is_committed(tmp_path):
    out = tmp_path / "missing.json"
    assert person_map.main(["--input", str(GAMES_DIR), "--output", str(out),
                            "--check-no-commit"]) == 2
    assert not out.exists()
