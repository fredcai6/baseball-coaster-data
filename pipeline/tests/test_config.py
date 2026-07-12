"""Tests for the pipeline config object (g2: config + paced fetcher).

Covers pacing, season-list, and paths as ONE object per the issue text, with
in-code defaults and optional loading from a small JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bc_pipeline.config import PipelineConfig, load_config


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
                "archive_root": "D:/raw_archive",
                "checkpoint_path": "D:/raw_archive/checkpoint.json",
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.min_interval_seconds == 15
    assert cfg.jitter_seconds == 3
    assert cfg.seasons == [2025]
    assert cfg.archive_root == "D:/raw_archive"
    assert cfg.checkpoint_path == "D:/raw_archive/checkpoint.json"


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
