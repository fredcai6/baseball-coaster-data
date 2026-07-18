"""Tests for bc_pipeline.frequencies: season/league event-frequency
aggregation over games/**'s events[].outcome.type (g1, issue #21).
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bc_pipeline import frequencies

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = REPO_ROOT / "games"
FREQUENCIES_SCHEMA_PATH = REPO_ROOT / "schemas" / "frequencies.schema.json"

# The real committed game used for hand-count reconciliation: 2026 season,
# 129 total events / 89 `plate_appearance` events (>= the required 50), 12
# distinct outcome.type values (>= the required 4) -- qualifies per the
# handoff's close criterion. Picked by scanning games/2026/**/*.json for the
# first file meeting both thresholds.
HAND_COUNT_GAME_PATH = GAMES_DIR / "2026" / "20260519_0ibc.json"


def _load_hand_count_game() -> dict:
    return json.loads(HAND_COUNT_GAME_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Hand-count reconciliation: proves aggregation correct against
# independently hand-tallied ground truth (not merely internally
# self-consistent). The expected counts below were tallied by a small,
# separate, one-off script that loops `events[]` directly with
# collections.Counter -- NOT by calling build_frequencies -- keyed on:
#   - teams: {'away': {'name': 'Billings Mustangs',
#             'team_id': 'tlqlpupbujjreauc'},
#             'home': {'name': 'Idaho Falls Chukars',
#             'team_id': 'gwwjqo5s6nki4x8c'}}
#   - batter xtti55f7ey0u6ih4 == "Cameron Bowen" (per game['players'])
#   - pitcher hq5v434ekgu367lf
# ---------------------------------------------------------------------------


def test_hand_count_reconciliation_all_four_surfaces():
    game = _load_hand_count_game()
    assert game["game_id"] == "20260519_0ibc"
    assert game["season"] == 2026

    pa_events = [e for e in game["events"] if e["kind"] == "plate_appearance"]
    distinct_types = {e["outcome"]["type"] for e in pa_events}
    assert len(pa_events) >= 50, "hand-count game must have >=50 plate_appearance events"
    assert len(distinct_types) >= 4, "hand-count game must have >=4 distinct outcome.type values"

    artifact = frequencies.build_frequencies([game], generated_at=frequencies.NORMALIZED_TIMESTAMP)

    # (a) team tlqlpupbujjreauc's BATTING table -- hand tally: double:3,
    # flyout:8, groundout:6, hit_by_pitch:1, home_run:1, lineout:1, popout:1,
    # single:5, strikeout_looking:1, strikeout_swinging:9, walk:9 (total 45;
    # every other one of the 19 taxonomy types is 0 for this team/game).
    team_batting = artifact["league"]["batting"]["teams"]["tlqlpupbujjreauc"]
    assert team_batting["total_plate_appearances"] == 45
    expected_team_batting_counts = {t: 0 for t in frequencies.OUTCOME_TYPES}
    expected_team_batting_counts.update(
        {
            "double": 3,
            "flyout": 8,
            "groundout": 6,
            "hit_by_pitch": 1,
            "home_run": 1,
            "lineout": 1,
            "popout": 1,
            "single": 5,
            "strikeout_looking": 1,
            "strikeout_swinging": 9,
            "walk": 9,
        }
    )
    assert team_batting["counts"] == expected_team_batting_counts
    # Rate check: walk rate for this team's batting = 9/45 = 0.2.
    assert team_batting["rates"]["walk"] == 9 / 45

    # (b) that SAME team's PITCHING table (what tlqlpupbujjreauc's pitcher(s)
    # allowed) -- hand tally: double:5, flyout:9, grounded_into_double_play:2,
    # groundout:6, home_run:1, lineout:2, single:10, strikeout_swinging:5,
    # walk:4 (total 44).
    team_pitching = artifact["league"]["pitching"]["teams"]["tlqlpupbujjreauc"]
    assert team_pitching["total_plate_appearances"] == 44
    expected_team_pitching_counts = {t: 0 for t in frequencies.OUTCOME_TYPES}
    expected_team_pitching_counts.update(
        {
            "double": 5,
            "flyout": 9,
            "grounded_into_double_play": 2,
            "groundout": 6,
            "home_run": 1,
            "lineout": 2,
            "single": 10,
            "strikeout_swinging": 5,
            "walk": 4,
        }
    )
    assert team_pitching["counts"] == expected_team_pitching_counts
    # Rate check: single-allowed rate = 10/44.
    assert team_pitching["rates"]["single"] == 10 / 44

    # (c) individual player BATTING line: batter xtti55f7ey0u6ih4 -- hand
    # tally: groundout:2, single:1, strikeout_looking:1, strikeout_swinging:1,
    # walk:1 (total 6).
    player_batting = artifact["league"]["batting"]["players"]["xtti55f7ey0u6ih4"]
    assert player_batting["total_plate_appearances"] == 6
    expected_player_batting_counts = {t: 0 for t in frequencies.OUTCOME_TYPES}
    expected_player_batting_counts.update(
        {
            "groundout": 2,
            "single": 1,
            "strikeout_looking": 1,
            "strikeout_swinging": 1,
            "walk": 1,
        }
    )
    assert player_batting["counts"] == expected_player_batting_counts
    # Rate check: groundout rate = 2/6.
    assert player_batting["rates"]["groundout"] == 2 / 6

    # (d) individual player PITCHING line: pitcher hq5v434ekgu367lf -- hand
    # tally: flyout:6, groundout:4, hit_by_pitch:1, home_run:1, popout:1,
    # single:3, strikeout_swinging:6, walk:4 (total 26).
    player_pitching = artifact["league"]["pitching"]["players"]["hq5v434ekgu367lf"]
    assert player_pitching["total_plate_appearances"] == 26
    expected_player_pitching_counts = {t: 0 for t in frequencies.OUTCOME_TYPES}
    expected_player_pitching_counts.update(
        {
            "flyout": 6,
            "groundout": 4,
            "hit_by_pitch": 1,
            "home_run": 1,
            "popout": 1,
            "single": 3,
            "strikeout_swinging": 6,
            "walk": 4,
        }
    )
    assert player_pitching["counts"] == expected_player_pitching_counts
    # Rate check: strikeout_swinging rate = 6/26.
    assert player_pitching["rates"]["strikeout_swinging"] == 6 / 26

    # by_season["2026"] must reproduce the same figures (only game in the
    # aggregation, single season).
    assert artifact["by_season"]["2026"]["batting"]["teams"]["tlqlpupbujjreauc"] == team_batting
    assert artifact["meta"]["games_included"] == {"total": 1, "by_season": {"2026": 1}}


# ---------------------------------------------------------------------------
# Schema validity: the schema itself is valid Draft 2020-12, and a generated
# artifact validates against it.
# ---------------------------------------------------------------------------


def _load_frequencies_schema() -> dict:
    return json.loads(FREQUENCIES_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_self_valid_and_generated_artifact_validates():
    schema = _load_frequencies_schema()
    Draft202012Validator.check_schema(schema)  # raises SchemaError if invalid

    game = _load_hand_count_game()
    artifact = frequencies.build_frequencies([game])

    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(artifact))
    assert errors == [], f"generated artifact failed schema validation: {errors}"


# ---------------------------------------------------------------------------
# scripts/validate_frequencies.py actually executed (subprocess) against a
# generated artifact -- not merely an inline Draft202012Validator call.
# ---------------------------------------------------------------------------

VALIDATE_SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_frequencies.py"


def test_validate_frequencies_script_runs_clean(tmp_path):
    game = _load_hand_count_game()
    artifact = frequencies.build_frequencies([game], generated_at=frequencies.NORMALIZED_TIMESTAMP)

    target = tmp_path / "frequencies.json"
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT_PATH), "--target", str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, (
        f"validate_frequencies.py failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# Determinism + CLI --check-no-commit guard (both branches).
# ---------------------------------------------------------------------------

# A second real, distinct qualifying game (different game_id) used to prove
# the no-commit guard's CHANGED branch against a genuine content difference
# (not merely a different timestamp).
SECOND_GAME_PATH = GAMES_DIR / "2026" / "20260519_dnuq.json"


def _copy_game(dest_dir: Path, game_path: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(game_path, dest_dir / game_path.name)


def test_determinism_two_builds_diff_empty_after_timestamp_normalization():
    game = _load_hand_count_game()

    # Two independent build_frequencies calls with DIFFERENT generated_at
    # values (standing in for two independent runs at two different wall-
    # clock times) -- everything else about the input is identical.
    run_a = frequencies.build_frequencies([game], generated_at="2020-01-01T00:00:00Z")
    run_b = frequencies.build_frequencies([game], generated_at="2021-06-15T12:00:00Z")

    # Without normalization they differ (proves generated_at is the only
    # volatile field, and that normalization is doing real work below).
    assert run_a != run_b
    assert run_a["meta"]["generated_at"] != run_b["meta"]["generated_at"]

    # Once meta.generated_at is normalized on both sides, the two runs over
    # the identical games/** snapshot diff to empty.
    assert frequencies.normalize_generated_at(run_a) == frequencies.normalize_generated_at(run_b)


def test_no_commit_guard_reports_no_op_then_changed(tmp_path):
    games_dir = tmp_path / "games"
    output_path = tmp_path / "artifacts" / "frequencies.json"
    _copy_game(games_dir, HAND_COUNT_GAME_PATH)

    write_rc = frequencies.main(["--input", str(games_dir), "--output", str(output_path)])
    assert write_rc == 0
    assert output_path.exists()

    # Branch 1: corpus unchanged since the committed write -> NO-OP (exit 0).
    noop_rc = frequencies.main(
        ["--input", str(games_dir), "--output", str(output_path), "--check-no-commit"]
    )
    assert noop_rc == 0

    # Branch 2: a real content difference (second distinct game added to the
    # corpus) -> CHANGED (exit 2), not just a timestamp drift.
    _copy_game(games_dir, SECOND_GAME_PATH)
    changed_rc = frequencies.main(
        ["--input", str(games_dir), "--output", str(output_path), "--check-no-commit"]
    )
    assert changed_rc == 2


# ---------------------------------------------------------------------------
# Honest-Null coverage block: a thin/synthetic corpus (hand-built, minimal
# fixture games -- not loaded from games/**) must report its true small
# scale, never padded or fabricated.
# ---------------------------------------------------------------------------


def _pa_event(seq, batting_team, fielding_team, batter_id, pitcher_id, outcome_type):
    return {
        "seq": seq,
        "kind": "plate_appearance",
        "batting_team": batting_team,
        "fielding_team": fielding_team,
        "batter": {"player_id": batter_id, "name_raw": None, "resolved": True},
        "pitcher": {"player_id": pitcher_id, "name_raw": None, "resolved": True},
        "outcome": {
            "type": outcome_type,
            "modifiers": [],
            "fielders": [],
            "outs_recorded": 0,
            "location": None,
        },
    }


def _minimal_game(season, game_id, parser_version, events, events_count, unparsed_count):
    return {
        "game_id": game_id,
        "season": season,
        "meta": {
            "parser_version": parser_version,
            "parse": {"events_count": events_count, "unparsed_count": unparsed_count},
        },
        "events": events,
    }


def test_honest_null_coverage_on_thin_synthetic_corpus():
    game1 = _minimal_game(
        season=2099,
        game_id="synthetic_g1",
        parser_version="9.9.9",
        events=[
            _pa_event(0, "teamA", "teamB", "battA1", "pitchB1", "single"),
            _pa_event(1, "teamA", "teamB", "battA2", "pitchB1", "walk"),
        ],
        events_count=2,
        unparsed_count=1,
    )
    game2 = _minimal_game(
        season=2099,
        game_id="synthetic_g2",
        parser_version="9.9.9",
        events=[
            _pa_event(0, "teamB", "teamA", "battB1", "pitchA1", "strikeout_swinging"),
        ],
        events_count=1,
        unparsed_count=0,
    )

    artifact = frequencies.build_frequencies(
        [game1, game2], generated_at=frequencies.NORMALIZED_TIMESTAMP
    )

    # Honest small-corpus accounting -- never padded.
    assert artifact["meta"]["games_included"] == {"total": 2, "by_season": {"2099": 2}}
    assert artifact["meta"]["coverage"]["total_narrative_lines"] == (2 + 1) + (1 + 0)
    assert artifact["meta"]["coverage"]["total_unparsed_lines"] == 1
    assert artifact["meta"]["coverage"]["unparsed_rate"] == 1 / 4

    # Outcome types never observed in this thin corpus stay 0 -- never
    # fabricated, estimated, or omitted (every one of the 19 keys present).
    team_a_batting = artifact["league"]["batting"]["teams"]["teamA"]
    assert set(team_a_batting["counts"]) == set(frequencies.OUTCOME_TYPES)
    assert team_a_batting["counts"]["home_run"] == 0
    assert team_a_batting["counts"]["double"] == 0
    assert team_a_batting["counts"]["single"] == 1
    assert team_a_batting["counts"]["walk"] == 1
    assert team_a_batting["total_plate_appearances"] == 2
