"""Tests for `scripts/check_write_once.py` -- the games/** write-once guard.

Each test builds a throwaway git repo in `tmp_path`, commits a real game file
into it, then runs the guard over a real commit range. The guard shells out to
git, so exercising it against anything less than an actual repo would test a
mock rather than the contract.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD_PATH = REPO_ROOT / "scripts" / "check_write_once.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("check_write_once", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


GAME = {
    "meta": {
        "generated_at": "2026-07-18T02:23:15Z",
        "parser_version": "0.3.0",
        "source_hash": "abc123",
    },
    "game_id": "20260712_76fp",
    "season": 2026,
    "events": [
        {
            "seq": 1,
            "type": "plate_appearance",
            "outcome": {"type": "single"},
            "_derived": {"bases_after": [1, 0, 0], "outs_after": 0},
        }
    ],
}


def _run(repo: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=repo, check=True, capture_output=True, text=True)


def _rev(repo: Path, ref: str = "HEAD") -> str:
    out = subprocess.run(
        ("git", "rev-parse", ref), cwd=repo, check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


def _write_game(repo: Path, doc: dict) -> Path:
    path = repo / "games" / "2026" / "20260712_76fp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def _commit(repo: Path, subject: str) -> str:
    _run(repo, "add", "-A")
    _run(repo, "commit", "-m", subject)
    return _rev(repo)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A git repo holding one committed game file, plus an empty root commit
    before it so there is always a base to diff against."""
    root = tmp_path / "data"
    root.mkdir()
    _run(root, "init", "-q", "-b", "master")
    _run(root, "config", "user.email", "guard@test.invalid")
    _run(root, "config", "user.name", "Guard Test")
    (root / "README.md").write_text("seed\n", encoding="utf-8")
    _commit(root, "seed: empty corpus")
    _write_game(root, GAME)
    _commit(root, "backfill(2026): games 1-1")
    return root


def _check(repo: Path, base: str, head: str = "HEAD"):
    return guard.check_range(base, head, repo)


def test_added_game_file_is_allowed(repo: Path) -> None:
    """A brand-new game file is the pipeline doing its job, not a rewrite."""
    base = _rev(repo, "HEAD~1")
    failures, _notes, examined = _check(repo, base)
    assert failures == []
    assert examined == 1


def test_in_place_semantic_edit_without_label_fails(repo: Path) -> None:
    base = _rev(repo)
    edited = json.loads(json.dumps(GAME))
    edited["events"][0]["outcome"]["type"] = "double"
    _write_game(repo, edited)
    _commit(repo, "fix: correct an outcome")
    failures, _notes, _examined = _check(repo, base)
    assert len(failures) == 1
    assert "semantically modified in place" in failures[0]
    assert "fix: correct an outcome" in failures[0]


def test_in_place_semantic_edit_in_labeled_reparse_passes(repo: Path) -> None:
    base = _rev(repo)
    edited = json.loads(json.dumps(GAME))
    edited["events"][0]["outcome"]["type"] = "double"
    _write_game(repo, edited)
    _commit(repo, "reparse(v0.4.0): grammar expansion (issue #33)")
    failures, _notes, _examined = _check(repo, base)
    assert failures == []


def test_meta_only_change_passes_unlabeled_with_a_note(repo: Path) -> None:
    """`meta` is provenance, not identity -- a diff confined to it is not a
    rewrite of the game under the README's semantic-equality rule."""
    base = _rev(repo)
    touched = json.loads(json.dumps(GAME))
    touched["meta"]["generated_at"] = "2026-08-27T00:00:00Z"
    _write_game(repo, touched)
    _commit(repo, "chore: re-stamp provenance")
    failures, notes, _examined = _check(repo, base)
    assert failures == []
    assert any("meta/_derived-only" in note for note in notes)


def test_derived_only_change_passes_unlabeled(repo: Path) -> None:
    """`_derived` is a regenerable cache at any depth."""
    base = _rev(repo)
    touched = json.loads(json.dumps(GAME))
    touched["events"][0]["_derived"]["outs_after"] = 1
    _write_game(repo, touched)
    _commit(repo, "chore: regenerate derived cache")
    failures, notes, _examined = _check(repo, base)
    assert failures == []
    assert any("meta/_derived-only" in note for note in notes)


def test_deletion_without_label_fails(repo: Path) -> None:
    base = _rev(repo)
    (repo / "games" / "2026" / "20260712_76fp.json").unlink()
    _commit(repo, "cleanup: drop a game")
    failures, _notes, _examined = _check(repo, base)
    assert len(failures) == 1
    assert "deleted" in failures[0]


def test_deletion_in_labeled_reparse_passes(repo: Path) -> None:
    base = _rev(repo)
    (repo / "games" / "2026" / "20260712_76fp.json").unlink()
    _commit(repo, "reparse(v0.4.0): drop a non-final game")
    failures, _notes, _examined = _check(repo, base)
    assert failures == []


def test_labeled_reparse_followed_by_stray_edit_fails(repo: Path) -> None:
    """The bar is EVERY commit touching the path, not merely the last one --
    a stray edit riding in behind a legitimate re-parse must still be caught.
    """
    base = _rev(repo)
    edited = json.loads(json.dumps(GAME))
    edited["events"][0]["outcome"]["type"] = "double"
    _write_game(repo, edited)
    _commit(repo, "reparse(v0.4.0): grammar expansion (issue #33)")
    edited["events"][0]["outcome"]["type"] = "triple"
    _write_game(repo, edited)
    _commit(repo, "oops: hand-tweak")
    failures, _notes, _examined = _check(repo, base)
    assert len(failures) == 1
    assert "oops: hand-tweak" in failures[0]


def test_range_touching_no_game_files_passes(repo: Path) -> None:
    base = _rev(repo)
    (repo / "README.md").write_text("docs change\n", encoding="utf-8")
    _commit(repo, "docs: tidy")
    failures, notes, examined = _check(repo, base)
    assert failures == []
    assert notes == []
    assert examined == 0


def test_artifacts_are_out_of_scope(repo: Path) -> None:
    """artifacts/** is the mutable tier (caller contract clause 2) -- the
    guard must not police it."""
    base = _rev(repo)
    artifact = repo / "artifacts" / "latest" / "frequencies.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"meta": {}}\n', encoding="utf-8")
    _commit(repo, "refresh: regenerate frequency artifacts")
    failures, _notes, examined = _check(repo, base)
    assert failures == []
    assert examined == 0


def test_zero_base_sha_skips_cleanly(repo: Path, capsys) -> None:
    """`github.event.before` is all-zeros on a branch's first push."""
    code = guard.main(["--base", "0" * 40, "--head", "HEAD", "--repo-root", str(repo)])
    assert code == 0
    assert "SKIP" in capsys.readouterr().out


def test_unparseable_blob_escalates_rather_than_crashing(repo: Path) -> None:
    base = _rev(repo)
    (repo / "games" / "2026" / "20260712_76fp.json").write_text("{not json", encoding="utf-8")
    _commit(repo, "oops: corrupt a game file")
    failures, _notes, _examined = _check(repo, base)
    assert len(failures) == 1
    assert "semantically modified in place" in failures[0]


@pytest.mark.parametrize(
    "subject, expected",
    [
        ("reparse(v0.3.0): replay-validation fixes (issue #31, #32)", True),
        ("reparse(v0.2.0): grammar+identity+schema expansion", True),
        ("reparse(v1.10.2): tail shapes", True),
        ("reparse: unversioned", False),
        ("reparse(v0.3.0):", False),
        ("chore: reparse(v0.3.0) mentioned mid-subject", False),
        ("backfill(2026): games 1-1", False),
    ],
)
def test_reparse_subject_convention(subject: str, expected: bool) -> None:
    assert bool(guard.REPARSE_SUBJECT.match(subject)) is expected
