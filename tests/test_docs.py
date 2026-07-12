"""Mechanically assert the "documented" half of the fixture-promotion
protocol + README requirement (issue #19 gate g7, critic #7): a doc file
existing is not evidence anyone can trust unless something actually checks
for it. This test is that check.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMOTION_PROTOCOL_PATH = REPO_ROOT / "tests" / "fixtures" / "PROMOTION_PROTOCOL.md"
README_PATH = REPO_ROOT / "README.md"
SYNTHETIC_TAXONOMY_TAIL_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic_taxonomy_tail"


def test_promotion_protocol_doc_exists_and_is_non_trivial():
    assert PROMOTION_PROTOCOL_PATH.exists(), (
        f"fixture-promotion protocol doc missing: {PROMOTION_PROTOCOL_PATH}"
    )
    text = PROMOTION_PROTOCOL_PATH.read_text(encoding="utf-8")
    assert len(text) > 200, "PROMOTION_PROTOCOL.md is suspiciously short/empty"
    assert "unparsed" in text.lower()


def test_readme_contains_parsing_and_replay_section_header():
    text = README_PATH.read_text(encoding="utf-8")
    assert "## Parsing & replay" in text, (
        "README.md is missing the required '## Parsing & replay' section header"
    )


def test_readme_parsing_and_replay_section_references_the_promotion_protocol():
    text = README_PATH.read_text(encoding="utf-8")
    idx = text.index("## Parsing & replay")
    section = text[idx:]
    assert "PROMOTION_PROTOCOL.md" in section, (
        "the README's 'Parsing & replay' section should point at "
        "tests/fixtures/PROMOTION_PROTOCOL.md"
    )


def test_synthetic_taxonomy_tail_exercise_fixture_exists_and_is_labeled():
    assert SYNTHETIC_TAXONOMY_TAIL_DIR.is_dir(), (
        f"missing directory: {SYNTHETIC_TAXONOMY_TAIL_DIR}"
    )
    fixtures = list(SYNTHETIC_TAXONOMY_TAIL_DIR.glob("*.json"))
    assert fixtures, "no promotion-exercise fixture found under synthetic_taxonomy_tail/"
    for path in fixtures:
        text = path.read_text(encoding="utf-8")
        assert "SYNTHETIC" in text, f"{path} is not clearly labeled SYNTHETIC"


def test_readme_did_not_lose_its_pre_existing_content():
    """Guard against an accidental rewrite: the append-only rule (the
    handoff's own constraint) means the original sections must still be
    present verbatim, not just the new one."""
    text = README_PATH.read_text(encoding="utf-8")
    assert "## Repository layout" in text
    assert "## The caller contract" in text
    assert "## Semantic equality" in text
    assert "## License" in text
    # The new section must come AFTER the pre-existing ones (append, not
    # prepend/interleave).
    assert text.index("## Semantic equality") < text.index("## Parsing & replay")
