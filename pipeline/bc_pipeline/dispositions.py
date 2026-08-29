"""Terminal dispositions -- the games that are DONE without being clean.

The corpus's goal is not "every game replays". It is that every game the
league published reaches a state a human put it in on purpose: parsed and
replaying, or refused with the refusal disclosed and evidenced. A disclosed
refusal is a done state. This file is what makes that a claim you can check
rather than one you have to take on trust.

WHY IT IS A FILE AND NOT A DOCUMENT

The argument for each of these games already existed. It lived in a handoff
written between sessions -- careful, evidenced, and entirely inert. Nothing
read it and nothing checked it, and it went stale twice: `20240508_04ck` and
`20240528_6w90` were each recorded as unfixable, and each became fixable the
morning an unrelated oracle was corrected. Neither the document nor the test
suite noticed. A human re-reading the document did.

That is the failure this module exists to make impossible, and it is the
same failure the rest of this repository keeps finding in itself: a claim
that was true when measured, believed after it stopped being true. So a
disposition is not prose about a game. It is a record PINNED to the exact
failure it was written against.

THE PIN

`warnings_sha256` is the sha256 of the game's replay warnings, sorted and
newline-joined -- the same discipline as `errata.raw_sha256`, applied to a
different kind of evidence. If an oracle gets better, a rule gets tighter,
or the source moves, the warnings change, the hash stops matching, and
`audit` reports the entry as STALE. The disposition can go out of date; it
cannot go out of date quietly.

The audit runs in both directions, which is the part that matters:

  - a game that fails replay with no disposition is UNDISCLOSED -- the gap
    the file exists to close;
  - a disposition whose pin no longer matches is STALE -- re-read it, the
    game may now be fixable;
  - a disposition for a game that now passes every check is SPENT -- delete
    it, it is an excuse for a problem that no longer exists.

A one-directional check would only have caught the first, and the first is
not the one that bit us.

WHAT IT IS NOT

It is not an excuse. `evidence` must name what in the source makes the game
unrecoverable, and `alternative_scored` must report the fix that was tried
and the number that killed it -- the same bar `errata` sets, for the same
reason. "We looked and could not fix it" tells the next reader nothing
except that they should look again.

And `class` keeps one distinction sharp on purpose: `oracle_residual` means
OUR check is wrong and the source is right. Every other class means the
source failed to write something down. Collapsing the two would let this
file quietly relabel our bugs as the league's, which is the one thing it is
really exposed to.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

#: The committed dispositions file, resolved relative to this package.
DEFAULT_DISPOSITIONS_PATH: Path = (
    Path(__file__).resolve().parents[2] / "corrections" / "dispositions.json"
)

#: States in which a game carries no replay warnings to pin, because it is
#: never handed to the replayer at all. Both are legitimate: the source
#: published no play-by-play, or no game file exists.
NOT_REPLAYED_STATES = frozenset({"committed_no_play_by_play", "not_committed"})

_CACHE: Optional[Dict[str, dict]] = None


class DispositionError(RuntimeError):
    """A disposition does not describe the corpus it was written for.

    Never caught inside this module. Continuing past one would mean scoring
    a game as intentionally-done on the strength of an argument that no
    longer applies to it -- which is exactly the state this file exists to
    detect.
    """


def load(path: Optional[Path] = None) -> Dict[str, dict]:
    """Read the dispositions file and index it by ``game_id``.

    Raises ``DispositionError`` on a duplicate ``game_id``: a game has
    exactly one terminal state, and two entries would mean two different
    answers to the same question with no way to tell which was believed.
    """
    target = Path(path) if path is not None else DEFAULT_DISPOSITIONS_PATH
    if not target.exists():
        return {}
    doc = json.loads(target.read_text(encoding="utf-8"))
    by_game: Dict[str, dict] = {}
    for entry in doc.get("dispositions", []):
        gid = entry["game_id"]
        if gid in by_game:
            raise DispositionError(f"duplicate disposition for {gid!r} in {target}")
        by_game[gid] = entry
    return by_game


def for_game(game_id: str, path: Optional[Path] = None) -> Optional[dict]:
    """The disposition authored for ``game_id``, or None.

    The committed file is read once and cached, so this is cheap to call per
    game across a corpus-wide sweep. Pass ``path`` to bypass the cache.
    """
    global _CACHE
    if path is not None:
        return load(path).get(game_id)
    if _CACHE is None:
        _CACHE = load()
    return _CACHE.get(game_id)


def warnings_sha256(warnings: Iterable[str]) -> str:
    """sha256 of ``warnings``, sorted and newline-joined.

    Sorted because a check's emission order is a parser detail that may
    change without the FAILURE changing, and a pin that trips on a reorder
    would cry wolf until nobody read it. Deduped for the same reason: a
    caller that replays an already-replayed game sees each warning twice,
    and that is an artifact of the caller, not of the game.

    An empty iterable hashes to the empty-string digest, which is the right
    answer for a game that is never replayed -- not a special case.
    """
    joined = "\n".join(sorted(set(warnings)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def audit(
    replay_warnings: Mapping[str, Sequence[str]],
    committed_game_ids: Optional[Iterable[str]] = None,
    path: Optional[Path] = None,
) -> List[dict]:
    """Reconcile the ledger against the corpus, in both directions.

    ``replay_warnings`` maps every COMMITTED, REPLAYED game_id to the
    warnings its replay produced -- empty for a game that passes. A game
    disposed ``committed_no_play_by_play`` is committed but not replayed, so
    it belongs in ``committed_game_ids`` and not in ``replay_warnings``.

    Returns a list of problems, each ``{"game_id", "problem", "detail"}``,
    empty when the ledger and the corpus agree. Nothing raises: the caller
    (a test, a report) decides what a discrepancy means. The four problems:

      ``undisclosed``  a game fails replay and no disposition covers it
      ``stale``        the pinned failure is not the failure observed now
      ``spent``        the game passes every check; the entry is obsolete
      ``absent``       the entry names a game the corpus does not hold
    """
    ledger = load(path)
    committed = (
        set(committed_game_ids)
        if committed_game_ids is not None
        else set(replay_warnings)
    )
    problems: List[dict] = []

    for game_id, warnings in sorted(replay_warnings.items()):
        entry = ledger.get(game_id)
        if not warnings:
            if entry is not None and entry["state"] not in NOT_REPLAYED_STATES:
                problems.append({
                    "game_id": game_id,
                    "problem": "spent",
                    "detail": (
                        "replays clean now; this disposition is an excuse for a "
                        "problem that no longer exists and should be deleted"
                    ),
                })
            continue
        if entry is None:
            problems.append({
                "game_id": game_id,
                "problem": "undisclosed",
                "detail": f"fails replay with no disposition: {'; '.join(sorted(set(warnings)))}",
            })
            continue
        actual = warnings_sha256(warnings)
        if actual != entry["warnings_sha256"]:
            problems.append({
                "game_id": game_id,
                "problem": "stale",
                "detail": (
                    f"pinned {entry['warnings_sha256'][:12]}, observed {actual[:12]} -- "
                    f"the failure has changed since this was authored, so re-read it "
                    f"before trusting it. Observed: {'; '.join(sorted(set(warnings)))}"
                ),
            })

    for game_id, entry in sorted(ledger.items()):
        if entry["state"] == "not_committed":
            if game_id in committed:
                problems.append({
                    "game_id": game_id,
                    "problem": "absent",
                    "detail": "disposed as not_committed, but a game file exists",
                })
        elif game_id not in committed:
            problems.append({
                "game_id": game_id,
                "problem": "absent",
                "detail": f"disposed as {entry['state']}, but no game file exists",
            })

    return problems
