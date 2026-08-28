"""Authored per-line corrections to defective source lines.

The point of this facility is negative as much as positive: it exists so a
one-off scorer error never becomes a reason to WEAKEN a general rule. These
tests therefore care as much about how an erratum FAILS as about how it
applies -- a correction that silently stops matching, or that quietly stops
mattering, would put the corpus back in exactly the position the facility
was built to avoid.
"""

from __future__ import annotations

import copy
import glob
import json
from pathlib import Path

import pytest

from bc_pipeline import errata, parse

REPO_ROOT = Path(__file__).resolve().parents[1]
ERRATA_PATH = REPO_ROOT / "corrections" / "errata.json"


def _line(text, inning=1, half="top", line_index=0):
    return parse.PbpLine(
        inning=inning, half=half, line_index=line_index, text=text, is_strong=False
    )


def _entry(**over):
    base = {
        "erratum_id": "20240101_test-1",
        "game_id": "20240101_test",
        "location": {"inning": 1, "half": "top", "line_index": 0},
        "raw_sha256": errata.line_sha256("A. One grounded out to 2b."),
        "raw_excerpt": "A. One grounded out to 2b.",
        "replace": "A. One",
        "with": "B. Two",
        "class": "misidentified_batter",
        "evidence": "x" * 45,
    }
    base.update(over)
    return base


# --- applying ---------------------------------------------------------------


def test_correction_rewrites_only_the_named_line():
    lines = [_line("A. One grounded out to 2b."), _line("C. Three walked.", line_index=1)]
    out, applied = errata.apply_to_lines([_entry()], lines)
    assert out[0].text == "B. Two grounded out to 2b."
    assert out[1].text == "C. Three walked."  # untouched
    assert len(applied) == 1
    assert applied[0]["erratum_id"] == "20240101_test-1"


def test_correction_preserves_every_other_field_of_the_line():
    # Only `text` may change: inning/half/line_index locate the line, and
    # is_strong is layout the HTML decided. A correction is about what the
    # scorer wrote, not about where the line sits.
    lines = [_line("A. One grounded out to 2b.")]
    out, _ = errata.apply_to_lines([_entry()], lines)
    assert (out[0].inning, out[0].half, out[0].line_index, out[0].is_strong) == (
        1,
        "top",
        0,
        False,
    )


def test_no_errata_is_an_exact_passthrough():
    lines = [_line("A. One grounded out to 2b.")]
    out, applied = errata.apply_to_lines([], lines)
    assert out == lines and applied == []


# --- failing loudly ---------------------------------------------------------
#
# Every one of these means the correction no longer describes the source it
# was authored against. Skipping any of them silently would be a wrong parse
# hiding behind a clean one -- this repository's oldest failure mode.


def test_a_changed_source_line_raises_rather_than_rewriting_it():
    lines = [_line("A. One flied out to cf.")]  # not the text the hash pins
    with pytest.raises(errata.ErratumError, match="has changed since"):
        errata.apply_to_lines([_entry()], lines)


def test_an_absent_replace_substring_raises():
    entry = _entry(replace="Z. Nobody")
    lines = [_line("A. One grounded out to 2b.")]
    with pytest.raises(errata.ErratumError, match="occurs 0 times"):
        errata.apply_to_lines([entry], lines)


def test_an_ambiguous_replace_substring_raises():
    # "A. One" twice in one line does not name a unique edit, so the
    # correction cannot say which occurrence its author meant.
    text = "A. One singled; A. One advanced to second."
    entry = _entry(replace="A. One", raw_sha256=errata.line_sha256(text))
    with pytest.raises(errata.ErratumError, match="occurs 2 times"):
        errata.apply_to_lines([entry], [_line(text)])


def test_a_location_that_does_not_exist_raises():
    entry = _entry(location={"inning": 9, "half": "bottom", "line_index": 4})
    with pytest.raises(errata.ErratumError, match="expected exactly one line"):
        errata.apply_to_lines([entry], [_line("A. One grounded out to 2b.")])


def test_a_duplicate_erratum_id_raises(tmp_path):
    doc = {"schema_version": "1.0.0", "errata": [_entry(), _entry()]}
    p = tmp_path / "errata.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(errata.ErratumError, match="duplicate erratum_id"):
        errata.load(p)


# --- the committed file -----------------------------------------------------


def test_committed_errata_validates_against_its_schema():
    jsonschema = pytest.importorskip("jsonschema")
    doc = json.loads(ERRATA_PATH.read_text(encoding="utf-8"))
    schema = json.loads((REPO_ROOT / "schemas" / "errata.schema.json").read_text("utf-8"))
    jsonschema.validate(doc, schema)


def test_every_committed_erratum_names_a_game_that_exists():
    doc = json.loads(ERRATA_PATH.read_text(encoding="utf-8"))
    for entry in doc["errata"]:
        hits = glob.glob(str(REPO_ROOT / "games" / "*" / f"{entry['game_id']}.json"))
        assert hits, f"{entry['erratum_id']} names unknown game {entry['game_id']}"


def test_every_committed_erratum_is_still_applied_in_its_game_file():
    """The anti-junk-drawer guard.

    An erratum that no longer reaches its game -- because the source moved,
    because the game left the corpus, because someone edited the file and
    never re-parsed -- is dead weight asserting authority it no longer has.
    Each one must be visible in the committed game's `inferred[]`, which is
    the only place its application is recorded.
    """
    doc = json.loads(ERRATA_PATH.read_text(encoding="utf-8"))
    for entry in doc["errata"]:
        path = glob.glob(str(REPO_ROOT / "games" / "*" / f"{entry['game_id']}.json"))[0]
        game = json.loads(Path(path).read_text(encoding="utf-8"))
        ids = [
            i["asserted"].split()[1]
            for i in game.get("inferred", [])
            if i["rule"] == "erratum"
        ]
        assert entry["erratum_id"] in ids, (
            f"{entry['erratum_id']} is committed but does not appear in "
            f"{entry['game_id']}'s inferred[] -- re-parse, or delete it"
        )


def test_no_game_claims_an_erratum_that_is_not_committed():
    """The other direction: a game file may not assert a correction that no
    reviewer ever signed off on."""
    doc = json.loads(ERRATA_PATH.read_text(encoding="utf-8"))
    known = {e["erratum_id"] for e in doc["errata"]}
    for path in glob.glob(str(REPO_ROOT / "games" / "*" / "*.json")):
        game = json.loads(Path(path).read_text(encoding="utf-8"))
        for i in game.get("inferred", []):
            if i["rule"] == "erratum":
                eid = i["asserted"].split()[1]
                assert eid in known, f"{path} asserts unknown erratum {eid}"


def test_every_committed_erratum_still_changes_the_parse():
    """An erratum that changes nothing is not a correction, it is clutter.

    Re-parses each affected game from its archived HTML with and without its
    errata and requires the two to differ. Skipped when the raw archive is
    not on this machine.
    """
    from bc_pipeline import archive
    from bc_pipeline.config import PipelineConfig

    try:
        checkpoint = archive.load_checkpoint(PipelineConfig().checkpoint_path)
    except Exception:  # pragma: no cover - archive absent
        pytest.skip("raw archive not available on this machine")
    if not checkpoint:
        pytest.skip("raw archive not available on this machine")
    by_id = {
        url.rsplit("/", 1)[-1].replace(".xml", ""): e for url, e in checkpoint.items()
    }
    doc = json.loads(ERRATA_PATH.read_text(encoding="utf-8"))
    for game_id in sorted({e["game_id"] for e in doc["errata"]}):
        if game_id not in by_id:
            pytest.skip(f"{game_id} not in the local archive")
        entry = by_id[game_id]
        html = Path(entry["archived_path"]).read_text(encoding="utf-8", errors="replace")
        committed = json.loads(
            Path(glob.glob(str(REPO_ROOT / "games" / "*" / f"{game_id}.json"))[0]).read_text(
                encoding="utf-8"
            )
        )
        kwargs = dict(
            source_url=committed["meta"]["source_url"], fetched_at=entry["fetched_at"]
        )
        with_errata = parse.parse_game(html, **kwargs)
        without = parse.parse_game(html, errata_entries=[], **kwargs)
        assert with_errata["events"] != without["events"], (
            f"the errata for {game_id} change nothing about its events -- "
            f"delete them or fix them"
        )
