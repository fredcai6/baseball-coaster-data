#!/usr/bin/env python
"""Assert every derived artifact under `artifacts/latest/` is current with `games/**`.

CI guardrail for caller-contract clause 2. `artifacts/**` is the mutable tier,
regenerated FROM `games/**` -- so a change to the corpus that is not followed by
a regeneration leaves a committed artifact quietly describing a corpus that no
longer exists. Nothing caught that:

  `validate_games.py` checks game files against their schema.
  `validate_frequencies.py` checks the frequency artifact against ITS schema.

Both pass on a perfectly well-formed, completely stale artifact. And one went
stale: `artifacts/latest/frequencies.json` was last regenerated at
`refresh: regenerate frequency artifacts` (2026-08-27 08:20) and the corpus was
rewritten hours later by `reparse(v0.4.0)`, which took replayable games from 84
to 999 and changed thousands of events. The artifact kept validating cleanly the
whole time. This script is the missing check.

It regenerates each artifact IN MEMORY and compares against what is committed,
with `meta.generated_at` normalized on both sides (a wall-clock stamp is not a
change) -- reusing each module's own `--check-no-commit` contract rather than
reimplementing the comparison.

Every artifact is checked and ALL failures are reported; the run does not stop
at the first one, matching `check_write_once.py`'s own aggregation style.

Exits 0 when every artifact matches. Exits 1 with a LOUD message naming each
stale artifact and the exact command that regenerates it.

Run:  python scripts/check_artifacts_current.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from bc_pipeline import career_map, frequencies, person_map, team_map  # noqa: E402

#: (module, artifact filename, builder). Each module supplies its own
#: `build_*`/`normalize_generated_at`/`load_games`, so the freshness rule lives
#: with the artifact it belongs to and this script stays a driver.
_ARTIFACTS = (
    ("person_map", "person_map.json", person_map, person_map.build_person_map),
    ("team_map", "team_map.json", team_map, team_map.build_team_map),
    ("career_map", "career_map.json", career_map, career_map.build_career_map),
    ("frequencies", "frequencies.json", frequencies, frequencies.build_frequencies),
)


def check(repo_root: Path) -> list[str]:
    """Return a list of human-readable failures; empty means everything is current."""
    games_dir = repo_root / "games"
    if not games_dir.is_dir():
        return [f"no games/ directory at {games_dir}; nothing to check against"]

    games = frequencies.load_games(games_dir)
    failures: list[str] = []
    for name, filename, module, build in _ARTIFACTS:
        path = repo_root / "artifacts" / "latest" / filename
        if not path.exists():
            failures.append(
                f"{name}: {path} is not committed. Regenerate it:\n"
                f"    PYTHONPATH=pipeline python -m bc_pipeline.{name} "
                f"--input games/ --output artifacts/latest/{filename}"
            )
            continue
        fresh = module.normalize_generated_at(build(games))
        committed = module.normalize_generated_at(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if fresh != committed:
            failures.append(
                f"{name}: {path} is STALE -- it does not match a regeneration from "
                f"the current games/**. Regenerate and commit it:\n"
                f"    PYTHONPATH=pipeline python -m bc_pipeline.{name} "
                f"--input games/ --output artifacts/latest/{filename}"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/check_artifacts_current.py",
        description="Assert artifacts/latest/** is current with games/**.",
    )
    parser.add_argument("--repo-root", default=None, metavar="PATH")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root) if args.repo_root else REPO_ROOT

    failures = check(repo_root)
    if failures:
        print(
            f"STALE ARTIFACTS: {len(failures)} artifact(s) under artifacts/latest/ do "
            "not match a regeneration from the current games/**.",
            file=sys.stderr,
        )
        print(
            "artifacts/** is the mutable tier derived FROM games/** (README caller "
            "contract clause 2). A schema check passes on a perfectly well-formed "
            "stale artifact, which is exactly how one went unnoticed across a "
            "re-parse -- so freshness is checked here.\n",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  [X] {failure}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(_ARTIFACTS)} derived artifact(s) under artifacts/latest/ are current "
        "with games/** (generated_at normalized on both sides)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
