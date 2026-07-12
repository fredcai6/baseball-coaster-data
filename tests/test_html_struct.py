"""Tests for bc_pipeline.html_struct: generic stdlib DOM helpers + PBP-pane
iteration, asserted against the two real archived sample pages.

Protected intent: this module stays GENERIC and must never interpret the
linescore/box tables (that's deliberately duplicated independently in later
gates for spec-D2 independence) — these tests only exercise structural
extraction (panes, cells, strong-wrapping), never row/column semantics.
"""
from __future__ import annotations

from _support import SAMPLES_DIR

from bc_pipeline import html_struct


def _load(name: str) -> str:
    path = SAMPLES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _final_root():
    return html_struct.parse_html(_load("boxscore_20260709_final.html"))


def _today_root():
    return html_struct.parse_html(_load("boxscore_20260710_today.html"))


def test_final_sample_has_pbp_panes():
    root = _final_root()
    assert html_struct.has_pbp_panes(root) is True


def test_final_sample_has_nine_panes():
    root = _final_root()
    panes = html_struct.iter_pbp_panes(root)
    assert len(panes) == 9
    assert sorted(inning for inning, _ in panes) == list(range(1, 10))


def test_inning_one_first_cell_text():
    root = _final_root()
    panes = dict(html_struct.iter_pbp_panes(root))
    cells = panes[1]
    assert len(cells) > 0
    assert cells[0].text == "Isaac Nunez singled to left field (1-1 BS)."
    assert cells[0].is_strong is False


def test_inning_one_scoring_play_is_strong():
    root = _final_root()
    panes = dict(html_struct.iter_pbp_panes(root))
    cells = panes[1]
    matches = [
        c
        for c in cells
        if c.text.startswith("Josh Phillips singled to center field, RBI")
    ]
    assert len(matches) == 1
    assert matches[0].is_strong is True


def test_today_sample_has_no_pbp_panes():
    root = _today_root()
    assert html_struct.has_pbp_panes(root) is False
    assert html_struct.iter_pbp_panes(root) == []


def test_find_by_id_and_find_all_by_class_generic():
    root = _final_root()
    node = html_struct.find_by_id(root, "pbp-inning-1")
    assert node is not None
    assert node.tag == "section"
    text_cells = html_struct.find_all_by_class(node, "text")
    assert all(c.tag in ("td",) for c in text_cells if c.tag)


def test_find_all_and_iter_rows_and_cell_texts_generic():
    root = _final_root()
    node = html_struct.find_by_id(root, "pbp-inning-1")
    tables = html_struct.find_all(node, "table")
    assert len(tables) >= 1
    rows = list(html_struct.iter_rows(tables[0]))
    assert len(rows) > 0
    # First row is the inning-header caption row: a single <td> with no
    # class="text" wrapper, so cell_texts should return exactly one string.
    texts = html_struct.cell_texts(rows[0])
    assert len(texts) == 1
    assert "Inning" in texts[0]


def test_text_of_preserves_internal_whitespace_trims_only_surrounding():
    """text_of trims only leading/trailing whitespace; internal whitespace
    (mid-content newlines/tabs/multi-space runs, e.g. a trailing "(N out)"
    annotation) is preserved verbatim so later gates get the exact narrative."""
    root = _final_root()
    panes = dict(html_struct.iter_pbp_panes(root))
    cells = panes[1]

    # A real cell carrying an internal whitespace run before a "(N out)" tail.
    out_cells = [
        c
        for c in cells
        if c.text.startswith("Kyle Carlson struck out swinging (3-2 BBSBKS).")
    ]
    assert len(out_cells) == 1
    cell = out_cells[0]
    # Surrounding whitespace trimmed: no leading/trailing whitespace.
    assert cell.text == cell.text.strip()
    assert cell.text != ""
    # Internal whitespace PRESERVED verbatim.
    assert "\n" in cell.text
    assert "\t" in cell.text
    assert cell.text.endswith("(1 out)")
    # The internal whitespace run between the play text and the "(N out)"
    # annotation survives unmangled (not collapsed to a single space).
    assert ").\n" in cell.text
