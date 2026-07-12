"""Tests for the negative-path contract: a page with no PBP panes must NOT
be fabricated into a schema `final` game file.

Protected intent: `NonFinalPageError` (or an equivalent typed refusal) is
the ONLY acceptable outcome for a pre-game/"today" page -- never a guessed
or partially-fabricated `final` dict.
"""
from __future__ import annotations

import pytest

from _support import SAMPLES_DIR

from bc_pipeline import parse


def _load(name: str) -> str:
    path = SAMPLES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


TODAY_HTML = _load("boxscore_20260710_today.html")


def test_non_final_page_raises_non_final_page_error():
    with pytest.raises(parse.NonFinalPageError):
        parse.parse_game(
            TODAY_HTML,
            source_url="https://longbeachcoast.com/sports/bsb/2026/boxscores/20260710_today.xml",
            fetched_at="2026-07-11T00:00:00Z",
        )


def test_non_final_page_never_produces_a_dict():
    # Belt-and-suspenders: the call either raises or (if some future
    # refactor changes the contract) must never return something that
    # looks like a `final` game.
    try:
        result = parse.parse_game(
            TODAY_HTML,
            source_url="https://longbeachcoast.com/sports/bsb/2026/boxscores/20260710_today.xml",
            fetched_at="2026-07-11T00:00:00Z",
        )
    except parse.NonFinalPageError:
        return
    assert result.get("status") != "final"
