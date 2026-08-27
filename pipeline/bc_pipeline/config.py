"""Pipeline config: one object covering pacing, season list, and paths (g2).

The issue text asks for "one object (pacing, season list, paths)" -- that
object is :class:`PipelineConfig` below. It has sane in-code defaults so the
pipeline can run with zero configuration, and can also be loaded from a small
JSON file via :func:`load_config` (only the keys present in the file override
the defaults; anything omitted keeps its in-code default).

This module does zero network I/O and zero fetching -- config only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

#: Default seasons to walk, most recent first (matches schedule.DEFAULT_YEARS).
DEFAULT_SEASONS: tuple[int, ...] = (2026, 2025, 2024)

#: Default minimum pacing interval, in seconds, between the START of any two
#: successive calls through the paced-fetcher seam. Grounded in a real WAF
#: trip observed during recon (constraint:waf-pacing) -- must stay >= 10.
DEFAULT_MIN_INTERVAL_SECONDS: float = 12.0

#: Default jitter range, in seconds, added on top of the minimum interval.
DEFAULT_JITTER_SECONDS: float = 3.0

#: This repository's working tree, used by the outside-the-repo invariant
#: below. `.git` may be a directory or (in a worktree) a file; when neither
#: is present we are running from an installed copy rather than a checkout
#: and the invariant has nothing to police.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _repo_working_tree() -> Path | None:
    return _REPO_ROOT if (_REPO_ROOT / ".git").exists() else None


def normalize_local_path(value: str, field_name: str) -> str:
    """Resolve a configured local path to an absolute one, and refuse any
    path that lands inside this git working tree.

    Enforces the launch-order Pre-Ruling ("PC-local, outside the git repo").
    The raw archive must not live under the working tree even though
    .gitignore's `*.html` rule would keep the raw HTML itself out of git --
    the checkpoint JSON alongside it would NOT be excluded, and the contract
    is "outside the repo", not "gitignored within the repo".

    Resolution is also what stops a RELATIVE path from silently taking root
    under whatever the current working directory happens to be, which is how
    a Windows-shaped default (`C:/...`) once materialized a raw archive at
    `pipeline/C:/PRograms/...` inside the tree on a POSIX host.
    """
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        raise ValueError(
            f"{field_name} must be an absolute path (or '~'-rooted); got {value!r}, "
            f"which would resolve against the current working directory and so "
            f"point somewhere different depending on where the pipeline is run "
            f"from. Note that a Windows-style path like 'C:/...' is RELATIVE on "
            f"a POSIX host."
        )
    expanded = expanded.resolve()
    working_tree = _repo_working_tree()
    if working_tree is not None and expanded.is_relative_to(working_tree):
        raise ValueError(
            f"{field_name} must live OUTSIDE this git working tree "
            f"(launch-order Pre-Ruling: 'PC-local, outside the git repo'); "
            f"{value!r} resolves to {expanded}, which is inside {working_tree}. "
            f"A relative path resolves against the current working directory -- "
            f"pass an absolute path, or one starting with '~'."
        )
    return str(expanded)


#: Default PC-local root for raw archived HTML -- deliberately OUTSIDE this
#: git repository, per the Pre-Ruling quoted in `normalize_local_path`, which
#: enforces it for configured overrides too. Home-rooted so it is absolute on
#: the Linux workstation that is now the standing single writer for this repo
#: (see the "Raw archive & fetching" README section); override via --config.
DEFAULT_ARCHIVE_ROOT: str = str(Path.home() / "bc-raw-archive")

#: Default PC-local checkpoint file path (resume progress marker) -- lives
#: alongside the archive, same outside-the-repo rationale.
DEFAULT_CHECKPOINT_PATH: str = str(Path.home() / "bc-raw-archive" / "checkpoint.json")


@dataclass
class PipelineConfig:
    """Pacing, season list, and paths -- the pipeline's single config object.

    Fields:
        min_interval_seconds: Minimum seconds between the start of any two
            successive fetches through the paced-fetcher seam. Must be >= 10
            (constraint:waf-pacing) even though tests substitute a fake clock
            so they never actually wait this long in real time.
        jitter_seconds: Extra random seconds (uniformly distributed between 0
            and this value) added on top of ``min_interval_seconds`` for each
            wait, so fetches aren't perfectly periodic.
        seasons: Season years to walk, in the order they should be walked.
        archive_root: Local filesystem root where raw fetched HTML is
            archived (owned by a later gate; this gate only carries the path).
        checkpoint_path: Local filesystem path to the resume/checkpoint file
            (owned by a later gate; this gate only carries the path).
    """

    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS
    jitter_seconds: float = DEFAULT_JITTER_SECONDS
    seasons: list[int] = field(default_factory=lambda: list(DEFAULT_SEASONS))
    archive_root: str = DEFAULT_ARCHIVE_ROOT
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH

    def __post_init__(self) -> None:
        if self.min_interval_seconds < 10:
            raise ValueError(
                "min_interval_seconds must be >= 10 (constraint:waf-pacing); "
                f"got {self.min_interval_seconds!r}"
            )
        if self.jitter_seconds < 0:
            raise ValueError(f"jitter_seconds must be >= 0; got {self.jitter_seconds!r}")
        self.archive_root = normalize_local_path(self.archive_root, "archive_root")
        self.checkpoint_path = normalize_local_path(self.checkpoint_path, "checkpoint_path")


def load_config(path: str | Path | None) -> PipelineConfig:
    """Load a :class:`PipelineConfig`, applying JSON overrides if ``path`` is given.

    ``path`` may be ``None`` (returns pure in-code defaults) or point at a
    small JSON file containing any subset of the ``PipelineConfig`` field
    names; omitted keys keep their in-code default. Unknown keys in the file
    raise ``TypeError`` (fail fast on typos rather than silently ignoring).
    """
    if path is None:
        return PipelineConfig()

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(PipelineConfig)}
    unknown = set(raw) - known
    if unknown:
        raise TypeError(f"Unknown PipelineConfig field(s) in {path}: {sorted(unknown)}")
    return PipelineConfig(**raw)
