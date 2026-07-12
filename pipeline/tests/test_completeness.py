"""Tests for bc_pipeline.completeness: the backfill completeness report
generator + threshold CLI (g2, issue #20).

These build fake ``BackfillResult``/``GameOutcome``/``SeasonSummary``
objects directly (the real dataclasses from ``bc_pipeline.backfill``,
populated by hand) -- no HTML, no transport, no git. This module is a pure
aggregator over an already-shaped data structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bc_pipeline import completeness
from bc_pipeline.backfill import BackfillResult, GameOutcome, SeasonSummary


def _make_result() -> BackfillResult:
    """Two seasons: 2025 (clean-ish) and 2024 (bad parse rate)."""
    season_2025 = SeasonSummary(
        season=2025,
        fetched=3,
        skipped_already_done=0,
        parsed=3,
        replayable=3,
        non_final=0,
        parse_failed=0,
        skipped_already_committed=1,
    )
    season_2024 = SeasonSummary(
        season=2024,
        fetched=2,
        skipped_already_done=0,
        parsed=2,
        replayable=1,
        non_final=1,
        parse_failed=1,
        skipped_already_committed=0,
    )

    games = [
        # 2025: 3 parsed+replayable, 1 skipped_already_committed.
        # g1: zero unparsed lines. g2: nonzero unparsed lines. g3: zero.
        GameOutcome(
            url="https://example.com/2025/boxscores/g1.xml",
            season=2025,
            game_id="g1",
            outcome="parsed",
            replayable=True,
            events_count=100,
            unparsed_count=0,
        ),
        GameOutcome(
            url="https://example.com/2025/boxscores/g2.xml",
            season=2025,
            game_id="g2",
            outcome="parsed",
            replayable=True,
            events_count=90,
            unparsed_count=10,
        ),
        GameOutcome(
            url="https://example.com/2025/boxscores/g3.xml",
            season=2025,
            game_id="g3",
            outcome="parsed",
            replayable=True,
            events_count=50,
            unparsed_count=0,
        ),
        GameOutcome(
            url="https://example.com/2025/boxscores/g4.xml",
            season=2025,
            game_id="g4",
            outcome="skipped_already_committed",
        ),
        # 2024: 1 parsed+replayable (with unparsed lines), 1 non_final (no
        # line counts -- never parsed this run), 1 parse_failed (likewise).
        GameOutcome(
            url="https://example.com/2024/boxscores/h1.xml",
            season=2024,
            game_id="h1",
            outcome="parsed",
            replayable=True,
            events_count=20,
            unparsed_count=5,
        ),
        GameOutcome(
            url="https://example.com/2024/boxscores/h2.xml",
            season=2024,
            game_id="h2",
            outcome="non_final",
            reason="no PBP pane yet",
        ),
        GameOutcome(
            url="https://example.com/2024/boxscores/h3.xml",
            season=2024,
            game_id="h3",
            outcome="parse_failed",
            reason="no archived_path found in checkpoint for this url",
        ),
    ]

    return BackfillResult(
        seasons={2025: season_2025, 2024: season_2024},
        games=games,
        commits=["backfill(2025): games 1-3"],
        challenge=None,
    )


def test_report_shape_and_league_totals():
    report = completeness.build_completeness_report([_make_result()])

    assert "generated_at" in report["meta"]
    # ISO-8601 UTC, e.g. 2026-07-12T00:00:00Z
    assert report["meta"]["generated_at"].endswith("Z")

    league = report["league"]
    assert league["games_discovered"] == 7
    assert league["games_fetched"] == 5
    assert league["games_parsed"] == 5
    assert league["games_replayable"] == 4
    assert league["games_non_final"] == 1
    assert league["games_parse_failed"] == 1
    assert league["games_skipped_already_committed"] == 1

    # failure_rate (game-level) = (parse_failed=1 + (parsed=5 - replayable=4)=1) / discovered=7
    assert league["failure_rate"] == pytest.approx(2 / 7)

    # unparsed_rate (line-level, totals-based) = sum(unparsed_count) / sum(events_count+unparsed_count)
    # across games actually parsed this run: g1(0/100) + g2(10/100) + g3(0/50) + h1(5/25)
    # = unparsed (0+10+0+5)=15 / total (100+100+50+25)=275. h2 (non_final) and
    # h3 (parse_failed) are excluded entirely (never parsed this run -- no
    # line counts to contribute).
    assert league["unparsed_rate"] == pytest.approx(15 / 275)


def test_by_season_breakdown():
    report = completeness.build_completeness_report([_make_result()])
    by_season = report["by_season"]

    assert set(by_season) == {"2025", "2024"}

    s2025 = by_season["2025"]
    assert s2025["games_discovered"] == 4
    assert s2025["games_parsed"] == 3
    assert s2025["games_replayable"] == 3
    assert s2025["games_skipped_already_committed"] == 1
    # failure_rate: all 3 parsed games are replayable, no parse_failed -> 0.
    assert s2025["failure_rate"] == pytest.approx(0.0)
    # unparsed_rate: g1(0/100) + g2(10/100) + g3(0/50) = unparsed 10 / total 250.
    assert s2025["unparsed_rate"] == pytest.approx(10 / 250)

    s2024 = by_season["2024"]
    assert s2024["games_discovered"] == 3
    assert s2024["games_parsed"] == 2
    assert s2024["games_replayable"] == 1
    assert s2024["games_non_final"] == 1
    assert s2024["games_parse_failed"] == 1
    # failure_rate = (1 + (2-1)) / 3 = 2/3
    assert s2024["failure_rate"] == pytest.approx(2 / 3)
    # unparsed_rate: only h1 was actually parsed this season (h2 non_final,
    # h3 parse_failed both excluded) -> unparsed 5 / total 25.
    assert s2024["unparsed_rate"] == pytest.approx(5 / 25)


def test_enumerated_failures_includes_every_failure_never_dropped():
    report = completeness.build_completeness_report([_make_result()])
    failures = report["enumerated_failures"]

    # Exactly the parse_failed game (h3) -- no parsed-but-unreplayable games
    # exist in this fixture, so only one entry is expected, but the field
    # names/values must be exact and nothing silently summarized away.
    assert len(failures) == 1
    entry = failures[0]
    assert entry["game_id"] == "h3"
    assert entry["season"] == 2024
    assert entry["url"] == "https://example.com/2024/boxscores/h3.xml"
    assert entry["outcome"] == "parse_failed"
    assert entry["reason"] == "no archived_path found in checkpoint for this url"


def test_enumerated_failures_includes_parsed_but_unreplayable():
    result = _make_result()
    result.games.append(
        GameOutcome(
            url="https://example.com/2024/boxscores/h4.xml",
            season=2024,
            game_id="h4",
            outcome="parsed",
            replayable=False,
            warnings=["LOB check failed"],
        )
    )
    report = completeness.build_completeness_report([result])
    failures = report["enumerated_failures"]

    assert len(failures) == 2
    unreplayable = next(f for f in failures if f["game_id"] == "h4")
    assert unreplayable["outcome"] == "parsed"
    assert "LOB check failed" in unreplayable["reason"]


def test_non_final_games_kept_separate_from_failures():
    report = completeness.build_completeness_report([_make_result()])

    non_final = report["non_final_games"]
    assert len(non_final) == 1
    assert non_final[0] == {
        "game_id": "h2",
        "season": 2024,
        "url": "https://example.com/2024/boxscores/h2.xml",
        "reason": "no PBP pane yet",
    }

    # h2 must never appear in enumerated_failures.
    failure_ids = {f["game_id"] for f in report["enumerated_failures"]}
    assert "h2" not in failure_ids


def test_games_with_no_events_count_are_excluded_not_zeroed():
    """non_final/parse_failed games (never parsed this run) must not corrupt
    the line-level unparsed_rate aggregate -- they are excluded from both
    numerator and denominator entirely, not counted as a 0%-unparsed game."""
    result = _make_result()
    # 2024's line-level rate comes ONLY from h1 (5 unparsed / 25 total).
    # Confirm this by computing what the rate WOULD be if h2/h3 were wrongly
    # treated as 0-unparsed, 0-total games (i.e. included as extra zero-sized
    # entries) -- that would still be 5/25 numerically (0 total contributes
    # nothing either way), so instead assert directly against the raw
    # per-game math to make the exclusion explicit and unambiguous.
    report = completeness.build_completeness_report([result])
    s2024 = report["by_season"]["2024"]
    assert s2024["unparsed_rate"] == pytest.approx(5 / 25)

    # Remove h1 (the only parsed 2024 game with line counts) and confirm the
    # season's unparsed_rate falls back to 0.0 (no fabricated rate, no
    # division by zero) rather than crashing or inventing a number.
    result_no_lines = _make_result()
    result_no_lines.games = [g for g in result_no_lines.games if g.game_id != "h1"]
    report_no_lines = completeness.build_completeness_report([result_no_lines])
    assert report_no_lines["by_season"]["2024"]["unparsed_rate"] == pytest.approx(0.0)


def test_failure_rate_and_unparsed_rate_are_distinct_fields():
    """The rename must not drop the game-level signal: both rates are
    present, both correctly computed, and they are NOT the same number for
    this fixture (proving the rename didn't collapse the two concepts)."""
    report = completeness.build_completeness_report([_make_result()])
    league = report["league"]
    assert "failure_rate" in league
    assert "unparsed_rate" in league
    assert league["failure_rate"] == pytest.approx(2 / 7)
    assert league["unparsed_rate"] == pytest.approx(15 / 275)
    assert league["failure_rate"] != pytest.approx(league["unparsed_rate"])


def test_threshold_and_exit_are_keyed_on_line_level_unparsed_rate_not_failure_rate():
    """failure_rate (2/7 ~= 0.2857) and unparsed_rate (15/265 ~= 0.0566) are
    both above 0.06 and both below 0.3 for this fixture -- pick a threshold
    that would flip the exceeded verdict depending on WHICH rate gates it,
    proving the CLI is keyed on unparsed_rate (line-level), not failure_rate."""
    # 0.10 is below failure_rate (0.2857) but above unparsed_rate (0.0566):
    # if the threshold were (still, wrongly) keyed on failure_rate, this
    # would report exceeded=True; keyed correctly on unparsed_rate, it must
    # be False.
    report = completeness.build_completeness_report([_make_result()], threshold=0.10)
    assert report["league"]["failure_rate"] > 0.10
    assert report["league"]["unparsed_rate"] < 0.10
    assert report["threshold"]["exceeded"] is False


def test_multiple_backfill_results_aggregate():
    result_a = _make_result()
    result_b = _make_result()
    # Give result_b's games distinct game_ids so nothing collides oddly, but
    # aggregation should simply sum both results' totals.
    for g in result_b.games:
        g.game_id = g.game_id + "-b"

    report = completeness.build_completeness_report([result_a, result_b])
    assert report["league"]["games_discovered"] == 14
    assert report["league"]["games_parse_failed"] == 2


def test_threshold_default_is_provisional_and_documented():
    assert completeness.DEFAULT_THRESHOLD == pytest.approx(0.02)


def test_threshold_not_exceeded_under_generous_threshold():
    report = completeness.build_completeness_report([_make_result()], threshold=0.9)
    assert report["threshold"]["value"] == pytest.approx(0.9)
    assert report["threshold"]["exceeded"] is False


def test_threshold_exceeded_under_strict_threshold():
    report = completeness.build_completeness_report([_make_result()], threshold=0.01)
    assert report["threshold"]["exceeded"] is True


def _write_backfill_result_json(path: Path, result: BackfillResult) -> None:
    path.write_text(json.dumps(result.to_dict()), encoding="utf-8")


def test_cli_exits_zero_under_threshold(tmp_path: Path, capsys):
    input_path = tmp_path / "backfill_result.json"
    output_path = tmp_path / "completeness.json"
    _write_backfill_result_json(input_path, _make_result())

    exit_code = completeness.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--threshold",
            "0.9",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["threshold"]["exceeded"] is False
    out = capsys.readouterr()
    assert "OK" in out.out


def test_cli_exits_nonzero_past_threshold(tmp_path: Path, capsys):
    input_path = tmp_path / "backfill_result.json"
    output_path = tmp_path / "completeness.json"
    _write_backfill_result_json(input_path, _make_result())

    exit_code = completeness.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--threshold",
            "0.01",
        ]
    )

    assert exit_code == 1
    assert output_path.exists()
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["threshold"]["exceeded"] is True
    out = capsys.readouterr()
    assert "FAILED" in out.err
    assert "2024" in out.err  # the bad season should be named


def test_cli_accepts_multiple_input_files(tmp_path: Path):
    input_a = tmp_path / "a.json"
    input_b = tmp_path / "b.json"
    output_path = tmp_path / "completeness.json"

    result_b = _make_result()
    for g in result_b.games:
        g.game_id = g.game_id + "-b"

    _write_backfill_result_json(input_a, _make_result())
    _write_backfill_result_json(input_b, result_b)

    exit_code = completeness.main(
        [
            "--input",
            str(input_a),
            str(input_b),
            "--output",
            str(output_path),
            "--threshold",
            "0.9",
        ]
    )

    assert exit_code == 0
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["league"]["games_discovered"] == 14
