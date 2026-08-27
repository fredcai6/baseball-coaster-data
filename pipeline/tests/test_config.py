"""Tests for the pipeline config object (g2: config + paced fetcher).

Covers pacing, season-list, and paths as ONE object per the issue text, with
in-code defaults and optional loading from a small JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bc_pipeline.config import PipelineConfig, load_config, normalize_local_path


def test_default_config_has_sane_in_code_defaults() -> None:
    cfg = PipelineConfig()
    assert cfg.min_interval_seconds >= 10
    assert cfg.jitter_seconds >= 0
    assert cfg.seasons == [2026, 2025, 2024]
    assert cfg.archive_root  # non-empty
    assert cfg.checkpoint_path  # non-empty


def test_load_config_without_path_returns_defaults() -> None:
    cfg = load_config(None)
    assert cfg == PipelineConfig()


def test_load_config_from_json_overrides_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text(
        json.dumps(
            {
                "min_interval_seconds": 15,
                "jitter_seconds": 3,
                "seasons": [2025],
                "archive_root": "/srv/raw_archive",
                "checkpoint_path": "/srv/raw_archive/checkpoint.json",
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.min_interval_seconds == 15
    assert cfg.jitter_seconds == 3
    assert cfg.seasons == [2025]
    assert cfg.archive_root == "/srv/raw_archive"
    assert cfg.checkpoint_path == "/srv/raw_archive/checkpoint.json"


def test_load_config_from_json_partial_override_keeps_other_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text(json.dumps({"seasons": [2024]}), encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.seasons == [2024]
    assert cfg.min_interval_seconds == PipelineConfig().min_interval_seconds


def test_config_rejects_min_interval_below_ten() -> None:
    with pytest.raises(ValueError):
        PipelineConfig(min_interval_seconds=5)


# --- Local-path invariant: PC-local, outside the git working tree ----------
#
# The raw archive must not live under this repo's working tree. .gitignore's
# `*.html` rule would keep the raw HTML out of git, but the checkpoint JSON
# beside it would NOT be excluded -- the contract is "outside the repo", not
# "gitignored within the repo".


def test_default_paths_are_absolute_and_outside_the_repo() -> None:
    cfg = PipelineConfig()
    repo_root = Path(__file__).resolve().parents[2]
    for value in (cfg.archive_root, cfg.checkpoint_path):
        assert Path(value).is_absolute()
        assert not Path(value).is_relative_to(repo_root)


def test_checkpoint_default_sits_inside_the_default_archive_root() -> None:
    cfg = PipelineConfig()
    assert Path(cfg.checkpoint_path).parent == Path(cfg.archive_root)


def test_tilde_rooted_path_is_expanded_to_absolute() -> None:
    cfg = PipelineConfig(
        archive_root="~/bc-raw-archive-test",
        checkpoint_path="~/bc-raw-archive-test/checkpoint.json",
    )
    assert cfg.archive_root == str(Path.home() / "bc-raw-archive-test")
    assert Path(cfg.archive_root).is_absolute()


@pytest.mark.parametrize("bad", ["relative/archive", "./archive", "C:/PRograms/bc-raw-archive"])
def test_relative_paths_are_refused(bad: str) -> None:
    """A Windows-style 'C:/...' is RELATIVE on a POSIX host -- exactly how a
    raw archive once materialized at `pipeline/C:/PRograms/...` inside the
    working tree."""
    with pytest.raises(ValueError, match="must be an absolute path"):
        PipelineConfig(archive_root=bad)


def test_path_inside_the_working_tree_is_refused() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="OUTSIDE this git working tree"):
        PipelineConfig(archive_root=str(repo_root / "pipeline" / "raw"))


def test_checkpoint_path_is_policed_too(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="checkpoint_path"):
        PipelineConfig(
            archive_root=str(tmp_path),
            checkpoint_path=str(repo_root / "checkpoint.json"),
        )


def test_normalize_local_path_is_idempotent(tmp_path: Path) -> None:
    once = normalize_local_path(str(tmp_path), "archive_root")
    assert normalize_local_path(once, "archive_root") == once
