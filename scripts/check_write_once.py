#!/usr/bin/env python
"""Enforce the `games/**` write-once contract across a commit range.

CI guardrail for caller-contract clause 1 (README, "Three rules"): a final
game file changes only in an explicitly labeled re-parse commit -- it is
never silently edited in place, and never ambiently deleted.

What "changed" means here is the README's **semantic equality** rule: two
game files describe the same game iff they are deep-equal after deleting the
root `meta` block and every `_derived` block at any depth. `meta` is
provenance and `_derived` is a regenerable cache, so a diff confined to
those is a provenance touch, not a rewrite of the game, and is allowed.

Steps:
  1. Resolve a base and head commit (`--base`/`--head`, else BASE_SHA/HEAD_SHA
     in the environment).
  2. `git diff --name-status base..head -- games/` to find every added,
     modified, deleted, or renamed game file.
  3. Additions are always fine. For every modification, compare the base and
     head blobs under semantic equality; a meta/_derived-only diff passes
     with an informational note. For every semantic modification, deletion,
     or rename, require that EVERY non-merge commit in the range touching
     that path is a labeled re-parse commit (subject matching
     `reparse(vX.Y.Z): ...`, the convention set by v0.2.0 and v0.3.0).

Exits 0 when every game-file change in the range is permitted -- INCLUDING
the empty case where the range touches no game files at all. Exits non-zero
with a LOUD message naming each offending file and why on ANY violation. All
failures are aggregated and reported; the run does not stop at the first one.

Run:  python scripts/check_write_once.py --base <sha> --head <sha>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_PREFIX = "games/"

#: A labeled re-parse commit, per the convention established by
#: `reparse(v0.2.0): ...` and `reparse(v0.3.0): ...`. The version is the
#: parser version the corpus was regenerated with.
REPARSE_SUBJECT = re.compile(r"^reparse\(v\d+\.\d+\.\d+\):\s*\S")

#: A zero sha -- what `github.event.before` carries on a branch's first push,
#: where there is no meaningful base to diff against.
ZERO_SHA = re.compile(r"^0{7,40}$")


def _git(*args: str, repo_root: Path = REPO_ROOT) -> str:
    """Run a git command in the repo and return stdout, raising on failure."""
    proc = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def strip_provenance(node):
    """PURE. Return `node` with the root `meta` removed and every `_derived`
    key removed at any depth -- the README's semantic-equality normal form.

    The root `meta` strip is applied by the caller (`_semantic_form`); this
    function handles `_derived` recursively so nested per-event caches are
    normalized away wherever they sit.
    """
    if isinstance(node, dict):
        return {
            key: strip_provenance(value)
            for key, value in node.items()
            if key != "_derived"
        }
    if isinstance(node, list):
        return [strip_provenance(item) for item in node]
    return node


def _semantic_form(raw: str):
    """Parse a game-file blob into its semantic normal form.

    Returns a sentinel string on unparseable JSON rather than raising, so a
    malformed blob compares unequal and escalates into the labeled-re-parse
    requirement instead of crashing the guard.
    """
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"<unparseable: {exc}>"
    if isinstance(doc, dict):
        doc = {key: value for key, value in doc.items() if key != "meta"}
    return strip_provenance(doc)


def _blob_at(rev: str, path: str, repo_root: Path) -> str:
    return _git("show", f"{rev}:{path}", repo_root=repo_root)


def _rev_exists(rev: str, repo_root: Path) -> bool:
    """Is ``rev`` a commit this repository actually has?"""
    proc = subprocess.run(
        ("git", "cat-file", "-e", f"{rev}^{{commit}}"),
        cwd=repo_root, capture_output=True, text=True,
    )
    return proc.returncode == 0


def _changed_game_files(base: str, head: str, repo_root: Path) -> list[tuple[str, str]]:
    """Every (status, path) under games/** changed between base and head.

    Renames are reported as their raw status (`R###`) against the OLD path --
    a renamed game file is a delete of that path as far as write-once goes.
    """
    out = _git(
        "diff", "--name-status", "-z", f"{base}..{head}", "--", GAMES_PREFIX,
        repo_root=repo_root,
    )
    fields = [f for f in out.split("\0") if f]
    changes: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        # Rename/copy statuses carry TWO paths (old, new); everything else one.
        if status[0] in ("R", "C"):
            changes.append((status, fields[index + 1]))
            index += 3
        else:
            changes.append((status, fields[index + 1]))
            index += 2
    return changes


def _commits_touching(base: str, head: str, path: str, repo_root: Path) -> list[tuple[str, str]]:
    """(sha, subject) for every non-merge commit in the range touching `path`."""
    out = _git(
        "log", "--no-merges", "--format=%H%x1f%s", f"{base}..{head}", "--", path,
        repo_root=repo_root,
    )
    commits: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, _, subject = line.partition("\x1f")
        commits.append((sha, subject))
    return commits


def _unlabeled_commits(base: str, head: str, path: str, repo_root: Path) -> list[str]:
    """Subjects of the range's commits touching `path` that are NOT labeled
    re-parse commits. Empty means the change is permitted."""
    return [
        f"{sha[:9]} {subject}"
        for sha, subject in _commits_touching(base, head, path, repo_root)
        if not REPARSE_SUBJECT.match(subject)
    ]


def check_range(base: str, head: str, repo_root: Path = REPO_ROOT) -> tuple[list[str], list[str], int]:
    """Returns (failures, notes, files_examined) for the given commit range."""
    failures: list[str] = []
    notes: list[str] = []
    changes = _changed_game_files(base, head, repo_root)

    for status, path in changes:
        code = status[0]

        if code == "A":
            continue  # A new game file is the pipeline doing its job.

        if code == "M":
            before = _semantic_form(_blob_at(base, path, repo_root))
            after = _semantic_form(_blob_at(head, path, repo_root))
            if before == after:
                notes.append(f"{path}: meta/_derived-only change (semantically identical)")
                continue
            reason = "semantically modified in place"
        elif code == "D":
            reason = "deleted"
        elif code in ("R", "C"):
            reason = f"renamed/copied away (status {status})"
        else:
            reason = f"changed (status {status})"

        unlabeled = _unlabeled_commits(base, head, path, repo_root)
        if unlabeled:
            detail = "; ".join(unlabeled)
            failures.append(
                f"{path}: {reason} outside a labeled re-parse commit -- by: {detail}"
            )

    return failures, notes, len(changes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=os.environ.get("BASE_SHA", ""),
                        help="Base commit of the range (default: $BASE_SHA)")
    parser.add_argument("--head", default=os.environ.get("HEAD_SHA", "HEAD"),
                        help="Head commit of the range (default: $HEAD_SHA, else HEAD)")
    parser.add_argument("--repo-root", default=str(REPO_ROOT),
                        help="Repository root to run git in")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    if not args.base or ZERO_SHA.match(args.base):
        print("SKIP: no base commit to diff against (new branch or unset "
              "--base/$BASE_SHA); write-once is checked on the next push.")
        return 0

    if not _rev_exists(args.base, repo_root):
        # A FORCE-PUSH (rebasing a branch onto a moved master) leaves
        # `github.event.before` pointing at a commit that no longer exists in
        # the repository, and `git diff <gone>..<head>` then fails outright.
        # That is the same situation as a new branch -- there is no base to
        # diff against -- not a contract violation, so it skips for the same
        # reason rather than failing the build.
        print(f"SKIP: base commit {args.base[:9]} is not present in this "
              "repository (force-push rewrote history); write-once is checked "
              "on the next push.")
        return 0

    try:
        failures, notes, examined = check_range(args.base, args.head, repo_root)
    except RuntimeError as exc:
        print("=" * 72, file=sys.stderr)
        print(f"WRITE-ONCE CHECK COULD NOT RUN -- {exc}", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 2

    for note in notes:
        print(f"  [i] {note}")

    if failures:
        print("=" * 72, file=sys.stderr)
        print(f"WRITE-ONCE VIOLATION -- {len(failures)} game file(s) changed "
              f"outside a labeled re-parse commit:", file=sys.stderr)
        for failure in failures:
            print(f"  [X] {failure}", file=sys.stderr)
        print("", file=sys.stderr)
        print("games/** is write-once (README, caller contract clause 1). A final", file=sys.stderr)
        print("game file may only change in a commit whose subject is labeled", file=sys.stderr)
        print("'reparse(vX.Y.Z): ...'. If this is a deliberate corpus regeneration,", file=sys.stderr)
        print("relabel the commit; if it is not, the change should be reverted.", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        return 1

    print(f"OK: {examined} game-file change(s) in {args.base[:9]}..{args.head[:9]} "
          f"honor the write-once contract (0 violations).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
