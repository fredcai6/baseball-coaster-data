"""Canonical serializer + semantic-equality contract for game files.

Protected intent (schemas/game.schema.json `$comment`; DECISION.md sec4): two game
files are "the same game" iff they are deep-equal after deleting the root `meta`
key and EVERY `_derived` block, at any depth. Determinism and clean diffs for
`games/**` depend on this being exactly right — this module is the single place
that contract is implemented; every later gate (parse, replay, reparse-summary)
calls through it rather than re-deriving the rule.
"""
from __future__ import annotations

import copy
import json
from typing import Any

_VOLATILE_KEYS = frozenset({"_derived"})


def _strip_derived(node: Any) -> Any:
    """Recursively drop every `_derived` key at any depth, in a fresh copy."""
    if isinstance(node, dict):
        return {
            key: _strip_derived(value)
            for key, value in node.items()
            if key not in _VOLATILE_KEYS
        }
    if isinstance(node, list):
        return [_strip_derived(item) for item in node]
    return node


def strip_volatile(game: dict) -> dict:
    """Return a copy of `game` with the root `meta` key and every `_derived`
    block (at any depth) removed. Does not mutate the input."""
    without_derived = _strip_derived(game)
    without_derived.pop("meta", None)
    return without_derived


def semantic_equal(a: dict, b: dict) -> bool:
    """True iff `a` and `b` are deep-equal after removing the root `meta` key and
    every `_derived` key at any depth. Does not mutate either input (both are
    deep-copied internally before stripping)."""
    stripped_a = strip_volatile(copy.deepcopy(a))
    stripped_b = strip_volatile(copy.deepcopy(b))
    return stripped_a == stripped_b


def canonical_dumps(game: dict) -> str:
    """Deterministic text form of `game`: stable (sorted) key order, indent=2,
    `ensure_ascii=False`, trailing newline. Idempotent:
    `canonical_dumps(x) == canonical_dumps(json.loads(canonical_dumps(x)))`.
    """
    return json.dumps(game, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
