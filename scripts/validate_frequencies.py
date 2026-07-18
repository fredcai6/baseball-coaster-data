#!/usr/bin/env python
"""Validate the generated frequency artifact against schemas/frequencies.schema.json.

Mirrors ``scripts/validate_games.py``'s shape. Steps:
  1. Load ``schemas/frequencies.schema.json`` and assert it is itself a
     valid Draft 2020-12 schema (``Draft202012Validator.check_schema``).
  2. Validate ``artifacts/latest/frequencies.json`` (or ``--target``)
     against it.

Exits 0 when the schema is valid and the target artifact validates. Exits
non-zero with a LOUD message (the jsonschema error) on any violation, or
when the target artifact does not exist yet (nothing to validate is a
reportable condition, not a silent pass).

Wired into `.github/workflows/validate.yml` as a CI step (Admiral fix-now at
the issue-#21 merge, discharging the paired-guard lesson: a new validation
script must run in CI, not just locally). Also runnable manually.

Run:  py scripts/validate_frequencies.py [--target PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "frequencies.schema.json"
DEFAULT_TARGET = REPO_ROOT / "artifacts" / "latest" / "frequencies.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="py scripts/validate_frequencies.py",
        description=__doc__,
    )
    parser.add_argument(
        "--target",
        type=str,
        default=str(DEFAULT_TARGET),
        metavar="PATH",
        help=f"frequency artifact to validate (default: {DEFAULT_TARGET}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    target = Path(args.target)

    # Step 1 -- load + self-validate the schema.
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: could not load schema {SCHEMA_PATH}: {exc}", file=sys.stderr)
        return 1
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # SchemaError and friends
        print(
            f"FAIL: {SCHEMA_PATH} is not a valid Draft 2020-12 schema: {exc}",
            file=sys.stderr,
        )
        return 1

    validator = Draft202012Validator(schema)

    # Step 2 -- validate the target artifact.
    if not target.is_file():
        print(f"FAIL: target artifact does not exist: {target}", file=sys.stderr)
        return 1

    try:
        instance = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: {target}: malformed JSON -- {exc}", file=sys.stderr)
        return 1

    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    if errors:
        print("=" * 72, file=sys.stderr)
        print(f"VALIDATION FAILED -- {len(errors)} problem(s) in {target}:", file=sys.stderr)
        for err in errors:
            loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
            print(f"  [X] {target.name}: schema violation at {loc} -- {err.message}", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 1

    print(f"OK: schema self-valid; {target} validated against {SCHEMA_PATH.name} (0 failures).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
