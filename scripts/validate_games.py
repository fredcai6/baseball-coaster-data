#!/usr/bin/env python
"""Validate canonical game files against schemas/game.schema.json.

CI guardrail for the write-once `games/**` tier. Steps:
  1. Load `schemas/game.schema.json` and assert it is itself a valid
     Draft 2020-12 schema (`Draft202012Validator.check_schema`).
  2. Validate EVERY `games/**/*.json` AND EVERY `tests/fixtures/**/*.json`
     against it.

Exits 0 when the schema is valid and every discovered file validates —
INCLUDING the bootstrap case where `games/` holds zero game files. Exits
non-zero with a LOUD message naming each offending file (and the jsonschema
or JSON error) on ANY violation. All failures are aggregated and reported;
the run does not stop at the first bad file.

Run:  py scripts/validate_games.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "game.schema.json"
GAMES_DIR = REPO_ROOT / "games"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _discover_targets() -> list[Path]:
    """Every *.json under games/** and tests/fixtures/**, sorted, deduped."""
    targets: set[Path] = set()
    for base in (GAMES_DIR, FIXTURES_DIR):
        if base.is_dir():
            targets.update(base.rglob("*.json"))
    return sorted(targets)


def main() -> int:
    # Step 1 -- load + self-validate the schema.
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: could not load schema {SCHEMA_PATH}: {exc}", file=sys.stderr)
        return 1
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # SchemaError and friends
        print(f"FAIL: {SCHEMA_PATH} is not a valid Draft 2020-12 schema: {exc}",
              file=sys.stderr)
        return 1

    validator = Draft202012Validator(schema)

    # Step 2 -- validate every discovered game / fixture file.
    targets = _discover_targets()
    failures: list[str] = []
    checked = 0

    for path in targets:
        rel = path.relative_to(REPO_ROOT)
        try:
            instance = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{rel}: malformed JSON -- {exc}")
            continue
        errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
        if errors:
            for err in errors:
                loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
                failures.append(f"{rel}: schema violation at {loc} -- {err.message}")
        else:
            checked += 1

    if failures:
        print("=" * 72, file=sys.stderr)
        print(f"VALIDATION FAILED -- {len(failures)} problem(s) across "
              f"{len(targets)} file(s):", file=sys.stderr)
        for f in failures:
            print(f"  [X] {f}", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 1

    print(f"OK: schema self-valid; {checked} game/fixture file(s) validated "
          f"against {SCHEMA_PATH.name} (0 failures).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
