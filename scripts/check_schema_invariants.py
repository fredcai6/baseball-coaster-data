#!/usr/bin/env python
"""Mechanical guard for schemas/game.schema.json.

Asserts the five FROZEN invariants of the hybrid game schema (B base + 3 C
grafts). Each invariant fails LOUDLY: a clear message on stderr and a non-zero
exit. On full success prints `all invariants passed`.

Run:  py scripts/check_schema_invariants.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "game.schema.json"

# Appendix A — the outcome `type` taxonomy (B §6.4), EXACT. Extended by the
# human-ratified additive MINOR 1.3.0 (foul_out, strikeout); extension of this
# closed set requires this deliberate paired edit alongside the schema enum.
FROZEN_OUTCOME_TYPES = {
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "intentional_walk",
    "hit_by_pitch",
    "strikeout_swinging",
    "strikeout_looking",
    "strikeout",
    "groundout",
    "flyout",
    "lineout",
    "popout",
    "foul_out",
    "fielders_choice",
    "reached_on_error",
    "grounded_into_double_play",
    "sacrifice",
}

SEMVER_PATTERN = r"^[0-9]+\.[0-9]+\.[0-9]+$"


class InvariantError(Exception):
    """A frozen invariant was violated."""


def _fail(msg: str) -> None:
    raise InvariantError(msg)


def _walk(node, path):
    """Yield (node, path) for every dict/list node in the schema tree."""
    yield node, path
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, path + [key])
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, path + [i])


def inv1_no_x_hatch(schema: dict) -> str:
    """No property key named exactly `x` appears anywhere — the C escape hatch is
    forbidden. Check the key maps where a schema names its properties."""
    key_maps = ("properties", "patternProperties", "$defs", "definitions")
    for node, path in _walk(schema, []):
        if not isinstance(node, dict):
            continue
        for map_key in key_maps:
            sub = node.get(map_key)
            if isinstance(sub, dict) and "x" in sub:
                _fail(
                    f"invariant 1 FAILED: forbidden `x` escape-hatch property found "
                    f"under {'/'.join(map(str, path + [map_key]))}"
                )
    return "1. no `x` escape hatch"


def inv2_closed_outcome_taxonomy(schema: dict) -> str:
    """plate_appearance outcome.type enum equals EXACTLY the Appendix A set."""
    try:
        enum = schema["$defs"]["outcome"]["properties"]["type"]["enum"]
    except (KeyError, TypeError):
        _fail("invariant 2 FAILED: cannot locate $defs.outcome.properties.type.enum")
    got = set(enum)
    if got != FROZEN_OUTCOME_TYPES:
        missing = FROZEN_OUTCOME_TYPES - got
        extra = got - FROZEN_OUTCOME_TYPES
        _fail(
            "invariant 2 FAILED: outcome.type enum does not match the frozen §6.4 set "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    if len(enum) != len(FROZEN_OUTCOME_TYPES):
        _fail(f"invariant 2 FAILED: outcome.type enum has duplicates ({len(enum)} members)")
    return "2. closed 6.4 outcome taxonomy (19 members, exact)"


def inv3_no_open_asserted_objects(schema: dict) -> str:
    """No object node carries `additionalProperties: true` anywhere EXCEPT within
    the `_derived` definition subtree (the sole regenerable open cache). A map
    object whose additionalProperties is a *schema* is constrained, not open, and
    is not flagged."""
    for node, path in _walk(schema, []):
        if not isinstance(node, dict):
            continue
        if node.get("additionalProperties") is True:
            if "_derived" not in path:
                _fail(
                    "invariant 3 FAILED: open object (additionalProperties:true) outside "
                    f"the _derived cache at {'/'.join(map(str, path))}"
                )
    return "3. no open asserted objects (additionalProperties:true only inside _derived)"


def inv4_synthetic_pid_admitted(schema: dict) -> str:
    """The `pid` pattern matches BOTH a 16-char id AND `syn:away:1`."""
    try:
        pattern = schema["$defs"]["pid"]["pattern"]
    except (KeyError, TypeError):
        _fail("invariant 4 FAILED: cannot locate $defs.pid.pattern")
    for sample in ("4bs3tvwryvtzrvpa", "syn:away:1", "syn:home:12"):
        if re.fullmatch(pattern, sample) is None:
            _fail(f"invariant 4 FAILED: pid pattern {pattern!r} does not admit {sample!r}")
    # And it must reject an obvious non-pid, so the pattern is not vacuously open.
    if re.fullmatch(pattern, "syn:visitor:1") is not None:
        _fail(f"invariant 4 FAILED: pid pattern {pattern!r} wrongly admits 'syn:visitor:1'")
    return "4. synthetic-pid admitted (16-char id AND syn:<side>:<n>)"


def inv5_semver_schema_version(schema: dict) -> str:
    """Root `schema_version` carries a semver-shaped pattern and matches `1.0.0`
    (and rejects the 2-part `1.0`, proving three-part semver)."""
    try:
        pattern = schema["properties"]["schema_version"]["pattern"]
    except (KeyError, TypeError):
        _fail("invariant 5 FAILED: cannot locate properties.schema_version.pattern")
    if pattern != SEMVER_PATTERN:
        _fail(
            f"invariant 5 FAILED: schema_version pattern {pattern!r} is not the frozen "
            f"semver pattern {SEMVER_PATTERN!r}"
        )
    if re.fullmatch(pattern, "1.0.0") is None:
        _fail(f"invariant 5 FAILED: schema_version pattern {pattern!r} does not match '1.0.0'")
    if re.fullmatch(pattern, "1.0") is not None:
        _fail(f"invariant 5 FAILED: schema_version pattern {pattern!r} wrongly matches 2-part '1.0'")
    return "5. semver schema_version pattern"


CHECKS = (
    inv1_no_x_hatch,
    inv2_closed_outcome_taxonomy,
    inv3_no_open_asserted_objects,
    inv4_synthetic_pid_admitted,
    inv5_semver_schema_version,
)


def main() -> int:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAILED to load {SCHEMA_PATH}: {exc}", file=sys.stderr)
        return 1
    try:
        for check in CHECKS:
            print(f"PASS  {check(schema)}")
    except InvariantError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("all invariants passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
