"""Authored corrections to defective SOURCE lines.

Every rule in this parser is scored corpus-wide against cases whose answer
is already known, with its null reported beside it. That discipline is what
makes the rules trustworthy, and it has a hard edge: a phenomenon that
occurs ONCE cannot be scored that way. "100% correct on one case" is not
evidence, and a rule narrow enough to catch only that case is a hardcode
wearing a rule's clothes -- it carries a rule's authority without a rule's
evidence, and the next reader cannot tell the difference.

The alternative that keeps offering itself is worse: loosen a real rule
until the one-off falls inside it. That is exactly the trade `check_lob`
already measured and refused -- 66 recoveries against 287 regressions.
Widening a rule to reach one line puts every line the rule already got
right at risk.

So errata are the pressure valve that keeps the general rules general. A
one-off scorer error goes here, named and evidenced, and the rules stay as
tight as their evidence supports.

ADMISSIBILITY -- both must hold:

  1. No general rule can reach it. Usually that is because the phenomenon
     is essentially unique (N=1 or 2) and so cannot be scored; at higher N,
     a rule is normally the right answer and gets scored like one.

     ONE POPULATION is admitted despite being large, deliberately and with
     the owner's ruling: `inflated_inning_summary_lob`, 34 entries. A source
     Inning Summary reports one more runner left on base than its own half's
     plays put there. Measured, the defect sits at 2.01% of FINAL
     half-innings (29 of 1,445) against 0.030% everywhere else (7 of
     23,486) -- a 67x concentration -- spread evenly over all 12 home clubs
     and all three seasons, which makes it a close-out artifact of the
     scoring software rather than one scorer or a fault of ours. Ruled out
     by measurement: the summary tags are not misaligned by a half (99.80%
     match in place), walk-offs explain only 5, and the play type ending the
     half carries no signal.

     The reason it is errata and not a rule: the only rule available would
     be "in a final half-inning, when the tag exceeds the fold by one,
     believe the fold" -- which fires blind on every future instance,
     including any that turn out to be OUR bug. That is precisely the
     failure this file exists to avoid. Per-line entries pinned to a hash,
     each carrying its own arithmetic, cannot fire on anything but the exact
     lines a human checked.
  2. Other evidence IN THE SAME GAME forces the correction: the batting
     order, the boxscore line, the linescore, an inning summary. The
     `evidence` field must name it, in enough detail that a reader can
     check it without rerunning anything.

HOW IT APPLIES. A correction rewrites the raw PBP line text BEFORE the
grammar sees it, so the corrected line is parsed by exactly the same rules
as every other line in the corpus. There is no erratum code path through
the parser, and no rule is relaxed anywhere. Each application is disclosed
in the game file's ``inferred[]`` under rule ``erratum``, so a consumer
that wants only what the source actually said can drop the events it names.

HOW IT FAILS. Loudly, always. An erratum is pinned to the sha256 of the
verbatim line it was authored against. If the archived source ever changes,
the hash stops matching and the loader RAISES rather than rewriting text
its author never read. Same for a `replace` substring that is absent, or
that occurs more than once and so does not name a unique edit. A silently
skipped correction would be a wrong parse hiding behind a clean one -- this
repository's oldest failure mode -- so nothing here is skipped silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: The committed errata file, resolved relative to this package.
DEFAULT_ERRATA_PATH: Path = Path(__file__).resolve().parents[2] / "corrections" / "errata.json"

_CACHE: Optional[Dict[str, List[dict]]] = None


class ErratumError(RuntimeError):
    """An erratum could not be applied as authored.

    Never caught inside this module. A correction that cannot be applied
    exactly as written is a correction whose author's assumptions no longer
    hold, and continuing past it would rewrite text nobody reviewed.
    """


def load(path: Optional[Path] = None) -> Dict[str, List[dict]]:
    """Read the errata file and index it by ``game_id``.

    Raises ``ErratumError`` on a duplicate ``erratum_id`` -- ids are how a
    correction is referred to in a game file and in review, so a collision
    would make two different edits indistinguishable.
    """
    target = Path(path) if path is not None else DEFAULT_ERRATA_PATH
    if not target.exists():
        return {}
    doc = json.loads(target.read_text(encoding="utf-8"))
    by_game: Dict[str, List[dict]] = {}
    seen: set = set()
    for entry in doc.get("errata", []):
        eid = entry["erratum_id"]
        if eid in seen:
            raise ErratumError(f"duplicate erratum_id {eid!r} in {target}")
        seen.add(eid)
        by_game.setdefault(entry["game_id"], []).append(entry)
    return by_game


def for_game(game_id: str, path: Optional[Path] = None) -> List[dict]:
    """Every erratum authored for ``game_id``, in file order.

    The committed file is read once and cached, so this is cheap to call
    per game across a corpus-wide re-parse. Pass ``path`` to bypass the
    cache in a test.
    """
    global _CACHE
    if path is not None:
        return load(path).get(game_id, [])
    if _CACHE is None:
        _CACHE = load()
    return _CACHE.get(game_id, [])


def line_sha256(text: str) -> str:
    """sha256 of a raw PBP line, whitespace and all.

    Hashing the VERBATIM text rather than a normalized form is deliberate:
    normalization is a parser decision that can change, and an erratum must
    stay pinned to the bytes its author actually read.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_to_lines(entries: Sequence[dict], lines: Sequence) -> Tuple[List, List[dict]]:
    """Apply ``entries`` to ``lines`` (a sequence of ``parse.PbpLine``).

    Returns ``(corrected_lines, applications)``, where each application is
    ``{"erratum_id", "location", "raw", "corrected", "class", "evidence"}``
    ready for disclosure in the game file's ``inferred[]``.

    Raises ``ErratumError`` if an erratum names a location that does not
    exist, a line whose text does not hash to ``raw_sha256``, or a
    ``replace`` substring that is absent from that line or occurs more than
    once. Every one of those means the correction no longer describes the
    source it was written for.
    """
    out = list(lines)
    applications: List[dict] = []
    for entry in entries:
        loc = entry["location"]
        matches = [
            i
            for i, line in enumerate(out)
            if line.inning == loc["inning"]
            and line.half == loc["half"]
            and line.line_index == loc["line_index"]
        ]
        if len(matches) != 1:
            raise ErratumError(
                f"{entry['erratum_id']}: expected exactly one line at "
                f"inning {loc['inning']} {loc['half']} index {loc['line_index']}, "
                f"found {len(matches)}"
            )
        idx = matches[0]
        line = out[idx]
        actual = line_sha256(line.text)
        if actual != entry["raw_sha256"]:
            raise ErratumError(
                f"{entry['erratum_id']}: the source line has changed since this "
                f"erratum was authored (expected sha256 {entry['raw_sha256']}, "
                f"found {actual}). Re-read the line and re-author the correction; "
                f"do not repin the hash without doing so."
            )
        occurrences = line.text.count(entry["replace"])
        if occurrences != 1:
            raise ErratumError(
                f"{entry['erratum_id']}: replace={entry['replace']!r} occurs "
                f"{occurrences} times in the line; a correction must name a "
                f"unique edit"
            )
        corrected = line.text.replace(entry["replace"], entry["with"])
        # PbpLine is frozen, so this is a replacement rather than a mutation
        # -- the original line object is left exactly as the HTML produced it.
        out[idx] = type(line)(
            inning=line.inning,
            half=line.half,
            line_index=line.line_index,
            text=corrected,
            is_strong=line.is_strong,
        )
        applications.append(
            {
                "erratum_id": entry["erratum_id"],
                "location": dict(loc),
                "raw": line.text,
                "corrected": corrected,
                "class": entry["class"],
                "evidence": entry["evidence"],
            }
        )
    return out, applications
