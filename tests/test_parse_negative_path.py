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


# ---------------------------------------------------------------------------
# A page can carry PBP panes and still describe no baseball. 20260809_3555 is
# marked final with a 0-0 linescore, twenty boxscore rows totalling zero
# at-bats, and two events. Every replay oracle passed it vacuously.
# ---------------------------------------------------------------------------

import re

FINAL_HTML = _load("boxscore_20260709_final.html")

#: The final sample with every play-by-play narrative cell emptied, panes and
#: tables left intact -- the shape of a page that is "final" and describes
#: nothing.
_EMPTIED = re.sub(
    r'(<td class="text">)(.*?)(</td>)', r"\1\3", FINAL_HTML, flags=re.S
)

#: The same page with the narratives replaced by text no rule can match, so
#: every line lands in `unparsed[]`. This must still parse: a game whose lines
#: all FAILED is a visible failure, not a page to refuse.
_GIBBERISH = re.sub(
    r'(<td class="text">)(.*?)(</td>)',
    r"\1zzzz qqqq wwww\3",
    FINAL_HTML,
    flags=re.S,
)

_URL = "https://longbeachcoast.com/sports/bsb/2026/boxscores/20260709_h94w.xml"


def test_panes_with_no_plate_appearance_are_refused():
    with pytest.raises(parse.NonFinalPageError) as excinfo:
        parse.parse_game(_EMPTIED, source_url=_URL, fetched_at="2026-07-11T00:00:00Z")
    assert "plate appearance" in str(excinfo.value)


def test_a_game_whose_lines_all_failed_is_NOT_refused():
    """The guard requires zero plate appearances AND zero unparsed lines.

    Refusing on the plate-appearance count alone would delete the evidence
    for the very failure worth looking at -- a game the parser could not
    read would vanish instead of being reported.
    """
    game = parse.parse_game(
        _GIBBERISH, source_url=_URL, fetched_at="2026-07-11T00:00:00Z"
    )
    assert game["status"] == "final"
    assert game["unparsed"], "expected every line to land in unparsed[]"
    assert not [e for e in game["events"] if e["kind"] == "plate_appearance"]
