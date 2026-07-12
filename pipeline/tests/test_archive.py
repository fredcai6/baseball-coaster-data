"""Tests for the raw archive writer + checkpoint idempotency index (g3).

Everything here runs against a tmp_path stand-in for the archive root/
checkpoint path -- no live network, no dependency on any real archive
directory existing on this machine, and no filename-glob-based existence
checks (idempotency is checkpoint-keyed by source-url, proven directly
below).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bc_pipeline.archive import (
    archive_result,
    load_checkpoint,
    resolve_archive_root,
    save_checkpoint,
    should_fetch,
    should_fetch_url,
)
from bc_pipeline.config import PipelineConfig
from bc_pipeline.fetcher import FetchResult


def make_config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        archive_root=str(tmp_path / "archive"),
        checkpoint_path=str(tmp_path / "archive" / "checkpoint.json"),
    )


def make_result(url: str = "https://example.test/box/1", body: str = "<html>hello</html>") -> FetchResult:
    return FetchResult(url=url, status_code=200, body=body, fetched_at=1_700_000_000.123456)


# --- Archive writer -----------------------------------------------------


def test_archive_writer_saves_body_to_a_file_under_the_archive_root(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = make_result()

    entry = archive_result(result, config)

    archived_path = Path(entry["archived_path"])
    assert archived_path.exists()
    assert archived_path.read_text(encoding="utf-8") == result.body
    assert archived_path.is_relative_to(resolve_archive_root(config))


def test_archived_filename_contains_url_slug_timestamp_and_content_hash(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = make_result(url="https://example.test/box/42?season=2025")

    entry = archive_result(result, config)

    name = Path(entry["archived_path"]).name
    assert "example.test" in name
    assert "box" in name
    assert "42" in name
    # fetched_at embedded as an integer-microsecond timestamp somewhere in the name.
    assert str(int(result.fetched_at * 1_000_000)) in name
    assert entry["content_hash"] in name


def test_archive_writer_never_overwrites_an_existing_path(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = make_result()
    entry = archive_result(result, config)
    collision_path = Path(entry["archived_path"])
    # Simulate a name collision by trying to write to the exact same computed
    # path a second time (same url/fetched_at/body -> same deterministic name).
    with pytest.raises(FileExistsError):
        archive_result(result, config)
    # The original file must be untouched.
    assert collision_path.read_text(encoding="utf-8") == result.body


# --- Checkpoint: the authoritative idempotency index --------------------


def test_should_fetch_returns_false_for_a_url_marked_done_in_checkpoint_and_true_for_unseen_url(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    resolve_archive_root(config).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "https://example.test/already-done": {
            "archived_path": str(tmp_path / "archive" / "somefile.html"),
            "fetched_at": 1_700_000_000.0,
            "content_hash": "deadbeef",
            "status": "done",
        }
    }
    save_checkpoint(Path(config.checkpoint_path), checkpoint)

    # Entirely a checkpoint lookup -- no archive-directory scan/glob involved.
    assert should_fetch_url(config, "https://example.test/already-done") is False
    assert should_fetch_url(config, "https://example.test/never-seen") is True


def test_should_fetch_is_a_pure_checkpoint_lookup_with_no_filesystem_access() -> None:
    checkpoint = {
        "https://example.test/done": {
            "archived_path": "/wherever/does/not/exist.html",
            "fetched_at": 1.0,
            "content_hash": "abc",
            "status": "done",
        }
    }
    # No archive dir, no checkpoint file on disk at all -- pure dict lookup.
    assert should_fetch(checkpoint, "https://example.test/done") is False
    assert should_fetch(checkpoint, "https://example.test/unseen") is True


def test_checkpoint_is_read_and_written_as_json(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = make_result()

    archive_result(result, config)

    raw = json.loads(Path(config.checkpoint_path).read_text(encoding="utf-8"))
    assert result.url in raw
    assert raw[result.url]["status"] == "done"


# --- Full round trip: archive then confirm idempotency ------------------


def test_round_trip_archive_then_second_should_fetch_check_returns_false(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = make_result(url="https://example.test/box/99")

    assert should_fetch_url(config, result.url) is True  # nothing archived yet

    archive_result(result, config)

    checkpoint = load_checkpoint(Path(config.checkpoint_path))
    assert result.url in checkpoint
    assert checkpoint[result.url]["status"] == "done"

    # Second run against the same checkpoint: fetch nothing new for this URL.
    assert should_fetch_url(config, result.url) is False
    # An unrelated URL is still fetchable.
    assert should_fetch_url(config, "https://example.test/box/100") is True


def test_a_second_archive_run_over_two_urls_only_fetches_the_new_one(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    first = make_result(url="https://example.test/box/1", body="first body")
    archive_result(first, config)

    urls_to_consider = ["https://example.test/box/1", "https://example.test/box/2"]
    to_fetch = [u for u in urls_to_consider if should_fetch_url(config, u)]

    assert to_fetch == ["https://example.test/box/2"]


# --- Archive root path arithmetic ---------------------------------------


def test_configured_archive_root_resolves_outside_the_git_repo_working_tree(tmp_path: Path) -> None:
    # The repo working tree root for this worktree: pipeline/tests/../.. .
    repo_root = Path(__file__).resolve().parents[2]
    config = PipelineConfig(archive_root=str(tmp_path / "bc-raw-archive"))

    resolved = resolve_archive_root(config)

    assert not resolved.is_relative_to(repo_root)
