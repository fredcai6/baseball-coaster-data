"""Tests for the corpus-wide labeled re-parse driver (issue #40).

`games/**` is write-once: a committed game file changes ONLY in a labeled
`reparse(vX.Y.Z): ...` commit. Both prior re-parses were driven by ad-hoc
scripts that were never committed, so the most consequential operation this
repo performs had no repeatable, tested driver. These tests pin the
behaviour that makes it safe to run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bc_pipeline import reparse


def _make_repo(tmp_path: Path, games: dict) -> Path:
    root = tmp_path / "data"
    (root / ".git").mkdir(parents=True)
    for gid, doc in games.items():
        d = root / "games" / str(doc["season"])
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{gid}.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


def _game(gid, season=2026, unparsed=0, replayable=True):
    """`unparsed` populates the real `unparsed[]` list, not just the meta
    counter -- `serialize.semantic_equal` strips `meta`, so a fixture that
    differed only there would be (correctly) reported as unchanged."""
    return {
        "game_id": gid,
        "season": season,
        "schema_version": "1.6.0",
        "events": [],
        "unparsed": [{"raw": f"line {i}", "reason": "x"} for i in range(unparsed)],
        "meta": {
            "source_url": f"https://x/boxscores/{gid}.xml",
            "fetched_at": "2026-01-01T00:00:00Z",
            "parse": {"unparsed_count": unparsed, "replayable": replayable, "warnings": []},
        },
    }


def test_commit_subject_satisfies_the_write_once_guard():
    """Cross-module contract: the subject this driver produces must be one
    the write-once guard recognizes as a labeled re-parse. If these two ever
    drift, the re-parse commit is rejected by its own CI."""
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "cwo", root / "scripts" / "check_write_once.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    subject = reparse.REPARSE_SUBJECT_TEMPLATE.format(
        version="0.4.0", message="grammar + identity + schema (issue #40)"
    )
    assert guard.REPARSE_SUBJECT.match(subject), subject


def test_partial_coverage_is_refused_by_default(tmp_path, monkeypatch):
    """A PARTIAL re-parse would leave the corpus straddling two parser
    versions with no marker saying which file is which."""
    root = _make_repo(tmp_path, {"g1": _game("g1"), "g2": _game("g2")})
    monkeypatch.setattr(reparse, "_archived_html_by_game_id", lambda cfg: {})
    msgs = []
    result = reparse.run_reparse(
        repo_root=root, config=None, write=False, print_fn=msgs.append
    )
    assert result.missing_archive
    assert result.deltas == []
    assert any("REFUSING" in m for m in msgs)


def test_allow_partial_opts_out_and_is_reported(tmp_path, monkeypatch):
    root = _make_repo(tmp_path, {"g1": _game("g1")})
    monkeypatch.setattr(reparse, "_archived_html_by_game_id", lambda cfg: {})
    result = reparse.run_reparse(
        repo_root=root, config=None, write=False, allow_partial=True,
        print_fn=lambda _m: None,
    )
    assert result.allowed_partial is True
    assert result.summary()["allowed_partial"] is True


def _stub_pipeline(monkeypatch, tmp_path, fresh_for):
    """Point the driver at fake archived HTML and a fake parse+replay."""
    html = tmp_path / "raw.html"
    html.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(
        reparse, "_archived_html_by_game_id", lambda cfg: {g: html for g in fresh_for}
    )
    monkeypatch.setattr(reparse.parse, "parse_game", lambda *a, **k: {})
    monkeypatch.setattr(
        reparse.replay, "replay_game", lambda game, h: fresh_for[_current["gid"]]
    )


_current = {}


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    committed = _game("g1", unparsed=3, replayable=False)
    root = _make_repo(tmp_path, {"g1": committed})
    fresh = _game("g1", unparsed=0, replayable=True)
    html = tmp_path / "raw.html"
    html.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(reparse, "_archived_html_by_game_id", lambda cfg: {"g1": html})
    monkeypatch.setattr(reparse.parse, "parse_game", lambda *a, **k: fresh)
    monkeypatch.setattr(reparse.replay, "replay_game", lambda g, h: fresh)

    before = (root / "games" / "2026" / "g1.json").read_text(encoding="utf-8")
    result = reparse.run_reparse(repo_root=root, config=None, write=False)
    assert result.wrote == 0
    assert (root / "games" / "2026" / "g1.json").read_text(encoding="utf-8") == before
    assert result.deltas[0].changed is True


def test_write_applies_only_changed_games(tmp_path, monkeypatch):
    committed = _game("g1", unparsed=3, replayable=False)
    root = _make_repo(tmp_path, {"g1": committed})
    fresh = _game("g1", unparsed=0, replayable=True)
    html = tmp_path / "raw.html"
    html.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(reparse, "_archived_html_by_game_id", lambda cfg: {"g1": html})
    monkeypatch.setattr(reparse.parse, "parse_game", lambda *a, **k: fresh)
    monkeypatch.setattr(reparse.replay, "replay_game", lambda g, h: fresh)

    result = reparse.run_reparse(repo_root=root, config=None, write=True)
    assert result.wrote == 1
    written = json.loads((root / "games" / "2026" / "g1.json").read_text(encoding="utf-8"))
    assert written["meta"]["parse"]["unparsed_count"] == 0


def test_a_provenance_only_difference_is_not_a_change(tmp_path, monkeypatch):
    """Comparison is SEMANTIC -- meta and every _derived block stripped -- so
    a run that changes nothing but provenance must not churn the corpus."""
    committed = _game("g1")
    root = _make_repo(tmp_path, {"g1": committed})
    fresh = json.loads(json.dumps(committed))
    fresh["meta"]["parsed_at"] = "2099-01-01T00:00:00Z"
    fresh["meta"]["parser_version"] = "9.9.9"
    html = tmp_path / "raw.html"
    html.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(reparse, "_archived_html_by_game_id", lambda cfg: {"g1": html})
    monkeypatch.setattr(reparse.parse, "parse_game", lambda *a, **k: fresh)
    monkeypatch.setattr(reparse.replay, "replay_game", lambda g, h: fresh)

    result = reparse.run_reparse(repo_root=root, config=None, write=True)
    assert result.deltas[0].changed is False
    assert result.wrote == 0


def test_a_game_that_stops_replaying_is_surfaced_at_the_top_level(tmp_path, monkeypatch):
    """The one thing a re-parse must never do quietly."""
    committed = _game("g1", replayable=True)
    root = _make_repo(tmp_path, {"g1": committed})
    fresh = _game("g1", unparsed=1, replayable=False)
    html = tmp_path / "raw.html"
    html.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(reparse, "_archived_html_by_game_id", lambda cfg: {"g1": html})
    monkeypatch.setattr(reparse.parse, "parse_game", lambda *a, **k: fresh)
    monkeypatch.setattr(reparse.replay, "replay_game", lambda g, h: fresh)

    result = reparse.run_reparse(repo_root=root, config=None, write=False)
    assert result.summary()["regressions"] == ["g1"]


def test_a_parse_failure_is_reported_never_swallowed(tmp_path, monkeypatch):
    root = _make_repo(tmp_path, {"g1": _game("g1")})
    html = tmp_path / "raw.html"
    html.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(reparse, "_archived_html_by_game_id", lambda cfg: {"g1": html})

    def boom(*a, **k):
        raise ValueError("bad html")

    monkeypatch.setattr(reparse.parse, "parse_game", boom)
    result = reparse.run_reparse(repo_root=root, config=None, write=False)
    assert result.parse_failed and result.parse_failed[0][0] == "g1"
    assert result.summary()["parse_failed"] == 1


def test_cli_rejects_a_non_semver_version():
    assert reparse.main(["--version", "0.4", "--repo-root", "."]) == 2
