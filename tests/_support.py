"""Tiny shared test-support helpers: schema/fixture loading + a jsonschema validator.

Kept deliberately small — this is not a test framework, just the handful of paths
and helpers every gate's test module needs so they don't each re-derive them.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "game.schema.json"
SAMPLES_DIR = REPO_ROOT / "tests" / "samples"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def load_schema() -> dict:
    """Load the frozen game schema (schemas/game.schema.json) as a dict."""
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_game(d: dict) -> None:
    """Validate `d` against the frozen game schema; raises jsonschema.ValidationError on failure."""
    schema = load_schema()
    jsonschema.Draft202012Validator(schema).validate(d)


def load_fixture(name: str) -> dict:
    """Load a JSON fixture by filename from tests/fixtures/."""
    path = FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
