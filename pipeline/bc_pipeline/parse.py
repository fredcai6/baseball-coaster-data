"""parse -- raw boxscore HTML -> a full schema-valid ``final`` game dict.

Orchestrates the three earlier gates (``html_struct`` structural DOM helpers,
``grammar``'s pure-text clause grammar, ``identity``'s per-game player table)
plus a PURE ``build_events`` helper that folds parsed clauses forward into
the schema's ordered ``events[]`` spine, asserting runner ``from``/``to``
primitives by tracking base occupancy as it goes. This module does NOT
compute or stamp ``_derived`` (that is g6's replayer job) and does NOT
validate base-out state -- it only ASSERTS the primitives a human reading the
narrative would assert.

Correctness over coverage: every PBP line becomes an event OR a verbatim
``unparsed[]`` entry -- never dropped, never guessed. A page with no PBP
panes is not a final boxscore at all and must not be fabricated into a
``final`` file; ``NonFinalPageError`` is raised instead.

linescore/box interpretation here is this module's OWN, independent
reading of the same tables the g6 replayer will independently re-derive
(spec D2 independence, per ``html_struct``'s own restraint) -- it is
deliberately NOT a function shared with ``replay.py``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from . import career_map, errata as errata_mod, identity, team_map
from .grammar import (
    BATTER_OUTCOME_CAUSE,
    GrammarMiss,
    RunnerMovement,
    parse_clause_group,
)
from .html_struct import (
    Node,
    find_all,
    find_all_by_class,
    has_pbp_panes,
    is_strong,
    parse_html,
    text_of,
)

PARSER_VERSION = "0.17.0"
SCHEMA_VERSION = "1.11.0"
DERIVED_REPLAYER_VERSION_PLACEHOLDER = "unreplayed"


class NonFinalPageError(Exception):
    """Raised when a page has no PBP panes -- it is not a final boxscore.

    The caller must never fabricate a schema `final` game dict from a page
    like this (e.g. a pre-game/"today" page); this is the negative-path
    contract's typed signal.
    """


def sha256_hex(html: str) -> str:
    """Sha256 of the raw HTML text (utf-8), hex-encoded."""
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def idempotency_key(html: str) -> str:
    """(source hash + parser version) -- a re-parse of identical bytes by the
    same parser version always yields the same key."""
    return f"{sha256_hex(html)}:{PARSER_VERSION}"


# ---------------------------------------------------------------------------
# Small pure data shapes fed into build_events (no HTML/Node types below this
# line -- build_events is a PURE function over plain data + the identity
# player table).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PbpLine:
    """One ordered PBP cell, already split into its half by (impure) HTML
    traversal, ready for the PURE ``build_events`` fold."""

    inning: int
    half: str  # "top" | "bottom"
    line_index: int
    text: str
    is_strong: bool


_DEST_BASE = {"first": 1, "second": 2, "third": 3, "home": 4}

# The raw cell text carries StatCrew's CSS-layout whitespace (embedded
# newlines/tabs between clauses) and, on the last play of a half, a
# trailing "(N out)" annotation that is structurally separate metadata
# (grammar already peels it into `trailing_outs`). `narrative` is the
# human-readable verbatim PLAY text -- this module's own normalization
# decision, deliberately delegated to it by html_struct.text_of's docstring
# ("so the parser can decide its own narrative normalization"): collapse
# whitespace to single spaces, tidy "<space>," artifacts from the source's
# multi-line comma-separated-clause layout, and drop the trailing out-count
# annotation (it is not part of the play's narrative sentence).
_TRAILING_OUT_DISPLAY_RE = re.compile(
    r"^(?P<body>.*\S)\s*\(\s*\d+\s+out\)\s*$", re.DOTALL
)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;])")


def _display_narrative(raw_text: str) -> str:
    collapsed = " ".join(raw_text.split())
    m = _TRAILING_OUT_DISPLAY_RE.fullmatch(collapsed)
    if m:
        collapsed = m.group("body")
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", collapsed).strip()

# Hit-location free text -> fielder position abbreviation(s). Grammar leaves
# `fielders` empty on hit types (single/double/triple/home_run) since the
# fielding chain isn't named on a clean hit -- only the LOCATION is. This is
# this module's own closed mapping (grown from the real sample's observed
# location strings), not a grammar rule.
_LOCATION_FIELDERS: Dict[str, Tuple[str, ...]] = {
    "left field": ("lf",),
    "center field": ("cf",),
    "right field": ("rf",),
    "left center": ("lf", "cf"),
    "right center": ("cf", "rf"),
    "third base": ("3b",),
    "second base": ("2b",),
    "first base": ("1b",),
    "shortstop": ("ss",),
    # Spelled-out battery positions, as a fielder's choice names
    # them ("reached on a fielder's choice to pitcher").
    "pitcher": ("p",),
    "catcher": ("c",),
}
#: A down-the-line hit location names the position it went past, so the
#: fielder falls out of the location itself. Matches the ONE normalized
#: spelling grammar now emits for every hit type ("down the <pos> line" --
#: issue #40); the old pattern was `^the (lf|rf) line$`, which matched only
#: the double rule's spelling and only outfield corners, so a single down
#: the line never got a fielder and "down the 1b line" never matched at all.
_LINE_LOCATION_RE = re.compile(r"^down the (lf|rf|cf|1b|2b|3b|ss) line$")


def _location_to_fielders(location: Optional[str]) -> List[str]:
    if not location:
        return []
    known = _LOCATION_FIELDERS.get(location)
    if known is not None:
        return list(known)
    m = _LINE_LOCATION_RE.match(location)
    if m:
        return [m.group(1)]
    return []


#: StatCrew scorer-correction directives that describe bookkeeping rather
#: than play. "Batter set to X" resets who the software believes is at bat.
_SCORER_DIRECTIVE_RE = re.compile(r"\bBatter set to\b")

_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}


def _last_name_token(full_name: str) -> str:
    """Extract the PBP join token (surname) from a full display name.

    Deliberately re-derives the same "strip a trailing suffix token" policy
    ``identity._derive_last_name`` uses, rather than importing that private
    helper: this module owns its OWN read of the narrative name token
    (grammar's ``name_token`` is the full display name as PBP prints it),
    independent of identity's own DOM-row name parsing.
    """
    tokens = full_name.split()
    if len(tokens) >= 2 and tokens[-1].strip(".").lower() in _SUFFIX_TOKENS:
        # The surname token can carry a trailing comma left over from the
        # "Surname, Suffix" narrative shape (e.g. "Rojas, Jr" splits into
        # ["Rojas,", "Jr"]) -- strip it so the join token matches the
        # roster's comma-free last_name.
        return tokens[-2].rstrip(",")
    return tokens[-1] if tokens else full_name


def _resolve_substitution_pair(
    player_table: identity.PlayerTable,
    in_last: str,
    in_full: str,
    out_last: Optional[str],
    out_full: Optional[str],
    side: str,
) -> Tuple[Optional[str], bool, Optional[str], bool]:
    """Resolve a substitution's (in, out) last names against ONE side.

    ``in_full``/``out_full`` are the FULL pbp name tokens (issue #31 g4) --
    threaded into ``resolve()`` as its ``full_name`` tie-breaker so a
    same-surname collision on this side can still be disambiguated by
    first-initial/first-name; never guesses.

    ``out_last`` is ``None`` for a bare move (no outgoing player named) --
    the out half of the pair is trivially "resolved" (``None``, ``True``),
    mirroring the pre-existing bare-DH-entry convention (never call
    ``resolve()`` on a value that was never there). Returns
    ``(in_pid, in_ok, out_pid, out_ok)``; a caller checks both ``*_ok``
    flags to decide whether THIS side is a clean match.
    """
    in_pid, in_ok = player_table.resolve(in_last, side, in_full)
    if out_last is None:
        return in_pid, in_ok, None, True
    out_pid, out_ok = player_table.resolve(out_last, side, out_full)
    return in_pid, in_ok, out_pid, out_ok


def _side_first_names_agree(
    player_table: identity.PlayerTable,
    side: str,
    pairs: Tuple[Tuple[Optional[str], Optional[str]], ...],
) -> bool:
    """Does EVERY name this substitution line states agree, by first-name
    token, with the player it resolved to on ``side``?

    ``pairs`` is ``((pbp_full_name, resolved_pid), ...)`` for the incoming
    and outgoing halves. A ``None`` pbp token is a half the line never named
    (a bare position move names no outgoing player) and is skipped -- a name
    that was never stated cannot disagree. Requiring ALL stated names to
    agree is what makes this a discriminator rather than a vote: a line whose
    incoming name points at one side and whose outgoing name points at the
    other agrees with NEITHER, and the caller refuses.
    """
    team = player_table.home if side == "home" else player_table.away
    for token, pid in pairs:
        if token is None:
            continue
        entry = team.players.get(pid) if pid is not None else None
        if entry is None or not identity.first_token_agrees(token, entry.name):
            return False
    return True


def build_events(
    lines: List[PbpLine],
    player_table: identity.PlayerTable,
    box_pitching_order: Optional[Dict[str, List[str]]] = None,
    box_batting: Optional[Dict[str, List[dict]]] = None,
) -> Tuple[List[dict], List[dict], Dict[str, List[dict]], List[dict]]:
    """PURE: fold ordered ``PbpLine``s + the identity player table into the
    schema's ``events[]`` spine, an ``unparsed[]`` list, and a per-team-id
    substitutions list (for ``lineups[team].substitutions``).

    Tracks base occupancy (a map base-number -> player_id, reset at the
    start of each half) and per-team pitcher-of-record as it folds forward,
    asserting the runner ``from``/``to`` PRIMITIVES a reader of the
    narrative would assert -- never derived/validated base-out state (that
    is g6 replay's independent job).
    """
    home_id = player_table.home.team_id
    away_id = player_table.away.team_id
    #: The boxscore Pitchers table, in appearance order, per team -- the
    #: evidence the blank-incoming-pitcher rule reads. Distinct from
    #: `pitching_pool` below: that is "every roster row past the ninth
    #: batter", which OMITS a reliever who never took a plate appearance and
    #: therefore has no batting row at all. On the 58 blank-substitution
    #: lines the missing man is very often exactly the reliever being
    #: inferred, so the pool cannot stand in for the real order here.
    box_pitching_order = box_pitching_order or {}

    # Batting order = the first 9 rows of each team's Batters table (dict
    # insertion order == document row order, per identity.py); every row
    # after that is a non-batting-order pitching-staff bookkeeping entry.
    slot_occupant: Dict[str, Dict[int, str]] = {}
    pitching_pool: Dict[str, List[str]] = {}
    current_pitcher: Dict[str, Optional[str]] = {}
    for team in (player_table.home, player_table.away):
        ids = list(team.players.keys())
        batting_ids, pitcher_ids = ids[:9], ids[9:]
        slot_occupant[team.team_id] = {i + 1: pid for i, pid in enumerate(batting_ids)}
        pitching_pool[team.team_id] = pitcher_ids
        current_pitcher[team.team_id] = pitcher_ids[0] if pitcher_ids else None

    events: List[dict] = []
    unparsed: List[dict] = []
    #: Facts this parse ASSERTS that the source line does not state, each with
    #: the rule that concluded it. Every entry is a place the corpus's own
    #: text was defective and a measured rule filled the hole; nothing here is
    #: silent, and a consumer that wants only what the source said can drop
    #: every event these entries name (issue #40).
    inferred: List[dict] = []
    subs_by_team: Dict[str, List[dict]] = {home_id: [], away_id: []}

    base_occ: Dict[int, str] = {}
    # Snapshot of base_occ taken at the START of the line currently being
    # folded -- every runner clause's "from" reads THIS, never the live
    # base_occ, so that multiple clauses on the same line (e.g. two
    # baserunners each advancing one base on the same play) each see the
    # occupancy as it stood before ANY of that line's own movements, not a
    # partially-updated mid-line state (which would otherwise let one
    # clause's write clobber the very entry the next clause needs to read).
    line_snapshot: Dict[int, str] = {}
    # Per-line record of each runner's LATEST within-event destination base,
    # so a second clause for the same runner on one line chains off the first
    # (distinct DIFFERENT runners still read line_snapshot, above). Reset per
    # line.
    event_pos: Dict[str, int] = {}
    cur_half_key: Optional[Tuple[int, str]] = None
    seq = 0

    def _unparsed(line: PbpLine, reason: str) -> None:
        unparsed.append(
            {
                "location": {
                    "inning": line.inning,
                    "half": line.half,
                    "line_index": line.line_index,
                },
                "raw": line.text,
                "reason": reason,
            }
        )

    def _inferred(line: PbpLine, rule: str, statement: str) -> None:
        inferred.append(
            {
                "location": {
                    "inning": line.inning,
                    "half": line.half,
                    "line_index": line.line_index,
                },
                "raw": line.text,
                "rule": rule,
                "asserted": statement,
            }
        )

    def _note_inferred_runners(line: PbpLine, cg) -> None:
        """Surface every runner movement the grammar tagged as inferred.

        Called only once the line has become a real event -- a line that ends
        up in unparsed[] asserts nothing, so it has nothing to disclose.
        """
        for rm in cg.runners:
            if rm.inferred:
                _inferred(line, "doubled_name_scored", rm.inferred)

    _pbp_declared_ids: Dict[Tuple[str, str], str] = {}

    def _declare_from_pbp(name_token: str, side: str) -> Optional[str]:
        """Admit a player the PBP names but the boxscore never lists.

        StatCrew omits an all-zero box row for a player who entered and then
        never batted or reached -- 91% of this population -- and rarely omits
        one who DID record a plate appearance. Either way the narrative is
        authority that the player was in the game, so refusing to admit them
        loses the substitution event itself, not just a stat line.

        The entry is flagged ``box_listed=False`` (schema 1.4.0) so a
        consumer can never mistake "no box row" for "zero stats", and so the
        replay oracle's box-derived checks can see which players it has no
        row to reconcile against.

        Returns the new player_id, or None when the token is unusable.
        """
        display = " ".join((name_token or "").split()).strip().rstrip(".")
        if not display or not _last_name_token(display):
            return None
        team = player_table.home if side == "home" else player_table.away
        existing = _pbp_declared_ids.get((side, display))
        if existing is not None:
            return existing
        n = 1 + max(
            [
                int(pid.rsplit(":", 1)[1])
                for pid in team.players
                if pid.startswith(f"syn:{side}:") and pid.rsplit(":", 1)[1].isdigit()
            ]
            or [0]
        )
        pid = f"syn:{side}:{n}"
        team.players[pid] = identity.PlayerEntry(
            player_id=pid,
            name=display,
            last_name=_last_name_token(display),
            team_id=team.team_id,
            positions=[],
            box_listed=False,
        )
        _pbp_declared_ids[(side, display)] = pid
        return pid

    def _resolve_runner(
        rm: RunnerMovement,
        batting_side: str,
        modifiers: Optional[List[str]],
    ) -> Optional[dict]:
        last = _last_name_token(rm.name_token)
        pid, ok = player_table.resolve(last, batting_side, rm.name_token)
        if not ok:
            return None
        # `from` chains WITHIN an event: if this runner already moved earlier
        # in THIS line, its origin is that prior clause's destination
        # (event_pos), NOT its event-start base -- e.g. "Mata advanced to
        # second on a passed ball, advanced to third" is 1->2 then 2->3, not
        # 1->2 and 1->3. Only the FIRST clause for a runner in an event reads
        # the pre-line occupancy snapshot; falls back to the destination's own
        # base when the runner isn't tracked at all.
        if pid in event_pos:
            from_base = event_pos[pid]
        else:
            from_base = next(
                (b for b, occ_pid in line_snapshot.items() if occ_pid == pid), None
            )
            if from_base is None:
                from_base = _DEST_BASE.get(rm.destination, 0) if rm.destination else 0
        if rm.destination is not None:
            to_base = _DEST_BASE[rm.destination]
        elif rm.out:
            # A retired runner with no named destination (e.g. "out on the
            # play") -- off the bases.
            to_base = -1
        else:
            # No named destination and not retired (e.g. a failed pickoff
            # attempt): the runner simply stays at their current base.
            to_base = from_base
        record = {
            "player_id": pid,
            "from": from_base,
            "to": to_base,
            "cause": rm.cause,
            "out": rm.out,
            "scored": rm.scored,
        }
        if rm.scored:
            record["earned"] = not rm.unearned
            record["rbi"] = bool(
                modifiers and "RBI" in modifiers and rm.cause != "error"
            )
        # Record this runner's within-event position so a later clause for the
        # SAME runner on this line chains off it. -1 (retired) is kept so a
        # subsequent clause doesn't re-place an out runner on a base.
        event_pos[pid] = -1 if rm.out else to_base
        # Cross-event base occupancy is deliberately NOT written here. Writing
        # it per CLAUSE let one runner's transient intermediate stop clobber a
        # DIFFERENT runner's live entry for the same numbered base on the same
        # line: "P. Howard advanced to second, advanced to third ...; T.
        # Fontenot advanced to third, scored ..." parks Howard on third, then
        # Fontenot's own pass THROUGH third overwrites him, and Fontenot's
        # score vacates it -- leaving Howard untracked. The next line's clause
        # for him then falls back to `from` = 4, which replay's clear pass
        # ignores (its _BASE_INDEXES is (1, 2, 3)), so he phantom-occupies
        # third for the rest of the half.
        #
        # See `_apply_occupancy`, called once per LINE against the MERGED
        # records -- the pattern the BATTER path below already established and
        # the runner path never got (issue #33).
        return record

    def _merge_same_runner(records: List[dict]) -> List[dict]:
        """Collapse multiple clauses for the SAME runner in ONE event into a
        single net-path record (first clause's `from` + cause, last clause's
        `to`/`out`/`scored` + per-run flags).

        StatCrew occasionally narrates one runner's advance in two clauses on
        one line ("advanced to second on a passed ball, advanced to third").
        The internal event_pos chain already asserts each hop's true origin,
        but the g6 replayer validates every emitted `from` against the base
        occupancy AS OF THE START of the event (a single frozen array) -- so a
        second emitted entry with `from` = the intermediate base (2), which
        was NOT occupied before the event, reads as an illegal transition. The
        runner only ever HAD one net move this event (1 -> 3), so we emit one
        record for it.

        Cross-event occupancy is applied FROM this merged output, by
        `_apply_occupancy`. It used to be folded hop-by-hop inside
        `_resolve_runner` instead, and that immediacy was the bug: an
        intermediate stop is not an end-of-line position, so writing it could
        evict a different runner genuinely standing there (issue #33)."""
        order: List[str] = []
        grouped: Dict[str, List[dict]] = {}
        for rec in records:
            pid = rec["player_id"]
            if pid not in grouped:
                grouped[pid] = []
                order.append(pid)
            grouped[pid].append(rec)
        merged: List[dict] = []
        for pid in order:
            recs = grouped[pid]
            if len(recs) == 1:
                merged.append(recs[0])
                continue
            first, last = recs[0], recs[-1]
            net = {
                "player_id": pid,
                "from": first["from"],
                "to": last["to"],
                "cause": first["cause"],
                "out": last["out"],
                "scored": last["scored"],
            }
            # Preserve per-run flags from whichever hop carried them (the
            # scoring hop) so a run driven across two clauses keeps earned/rbi.
            for key in ("earned", "rbi"):
                for rec in recs:
                    if key in rec:
                        net[key] = rec[key]
            merged.append(net)
        return merged

    def _apply_occupancy(records: List[dict]) -> None:
        """Fold every named participant's NET end-of-line position into the
        cross-event base occupancy, read from the MERGED records.

        A base a runner merely passes THROUGH on the way to a further
        destination, or to being retired, is never an end-of-line occupant,
        so it must not reach `base_occ` even momentarily. Deferring every
        write until the whole line is merged is what stops two runners whose
        clauses transiently compute the same numbered base from clobbering
        each other.

        This is the batter's own established pattern, generalized to every
        participant. It was previously applied to the batter alone, with a
        comment explaining exactly why reading the merged record matters --
        and the runner path, which needed it for the same reason, wrote
        hop-by-hop instead. One rule in two places, the shape this codebase
        keeps rediscovering (issue #33).
        """
        for rec in records:
            pid = rec["player_id"]
            final_base = rec["to"]
            for base, occupant in list(base_occ.items()):
                if occupant == pid and base != final_base:
                    del base_occ[base]
            if not rec["out"] and final_base not in (-1, 4):
                base_occ[final_base] = pid

    for line in lines:
        half_key = (line.inning, line.half)
        if half_key != cur_half_key:
            base_occ = {}
            cur_half_key = half_key
        batting_side = "away" if line.half == "top" else "home"
        fielding_side = "home" if line.half == "top" else "away"
        batting_team_id = away_id if line.half == "top" else home_id
        fielding_team_id = home_id if line.half == "top" else away_id
        line_snapshot = dict(base_occ)
        event_pos = {}

        cg = parse_clause_group(line.text)
        if isinstance(cg, GrammarMiss):
            _unparsed(line, cg.reason)
            continue

        if cg.kind == "inning_summary":
            events.append(
                {
                    "seq": seq,
                    "inning": line.inning,
                    "half": line.half,
                    "kind": "inning_summary",
                    "batting_team": batting_team_id,
                    "fielding_team": fielding_team_id,
                    "narrative": _display_narrative(line.text),
                    "scoring_play": line.is_strong,
                    "summary": {
                        "R": cg.summary.runs,
                        "H": cg.summary.hits,
                        "E": cg.summary.errors,
                        "LOB": cg.summary.lob,
                    },
                }
            )
            seq += 1
            continue

        if (
            cg.kind == "substitution"
            and cg.substitution is not None
            and cg.substitution.player_in is None
        ):
            # Issue #40, Family B: "/  for R. Bost." -- StatCrew wrote the
            # line with the incoming pitcher's name simply missing. The
            # outgoing name is intact, and the boxscore's Pitchers table is
            # ordered by appearance, so the reliever is the pitcher listed
            # directly after him.
            #
            # Measured, not assumed: run this same rule against the 8,991
            # substitutions in the corpus that DO name both players and it
            # predicts the named reliever in 8,917 of them (99.18%).
            #
            # It fires only when FORCED -- the outgoing name must resolve
            # uniquely AND there must be exactly one successor in the order.
            # When it does not (the outgoing pitcher is last in the order, or
            # his name does not resolve), the line stays in unparsed[]. That
            # refusal is the point: 7 of the corpus's 58 such lines are not
            # determinable and are not guessed at.
            out_full = cg.substitution.player_out
            out_pid, out_ok = player_table.resolve(
                _last_name_token(out_full), fielding_side, out_full
            )
            order = box_pitching_order.get(fielding_team_id) or []
            successor = None
            if out_ok and out_pid in order:
                idx = order.index(out_pid)
                if idx + 1 < len(order):
                    successor = order[idx + 1]
            if successor is None:
                _unparsed(
                    line,
                    "substitution names no incoming player and the boxscore "
                    "pitching order does not force one: "
                    f"out={out_full!r}",
                )
                continue
            _inferred(
                line,
                "blank_incoming_pitcher",
                f"incoming pitcher {successor} read from the boxscore "
                f"pitching order as the successor to {out_pid}",
            )
            forced_sub_pids = (successor, out_pid)
            team = (
                player_table.home
                if fielding_team_id == player_table.home.team_id
                else player_table.away
            )
            entry = team.players.get(successor)
            cg = replace(
                cg,
                substitution=replace(
                    cg.substitution,
                    player_in=(entry.name if entry is not None else successor),
                ),
            )
        else:
            forced_sub_pids = None

        if cg.kind == "substitution":
            # A "pitching" substitution changes the FIELDING side's pitcher
            # (the mound belongs to the side not currently batting). An
            # "offensive" substitution (pinch-run, DH-slot entry -- issue #30
            # g1/g2b) changes the BATTING side's lineup instead -- resolve
            # against the side the grammar's own `kind` names. A hardcoded
            # fielding-side assumption here predates the "offensive" kind and
            # would silently mis-resolve every real pinch-run/DH-slot line
            # against the wrong roster (either a spurious unparsed miss, or
            # worse, a false match against an unrelated same-surname player
            # on the wrong team).
            if cg.substitution.kind == "offensive":
                primary_side, primary_team = batting_side, batting_team_id
                fallback_side, fallback_team = fielding_side, fielding_team_id
            else:
                primary_side, primary_team = fielding_side, fielding_team_id
                fallback_side, fallback_team = batting_side, batting_team_id
            in_full = cg.substitution.player_in
            in_last = _last_name_token(in_full)
            # Bare DH-slot-entry / bare position move (schema 1.2.0, issue
            # #30 g2b / issue #31 g3): the line names only the incoming
            # player. Never guess an outgoing player from a line that does
            # not name one -- `out_last=None` short-circuits both the
            # primary and fallback resolution to (None, True) directly,
            # instead of calling resolve() on a value that was never there.
            out_full = cg.substitution.player_out
            out_last = None if out_full is None else _last_name_token(out_full)
            # REWORK (commander-31, parse.py-assembly design call, responding
            # to the stop condition originally reported here): a real-corpus
            # data quirk (a substitution announcement sometimes logged as a
            # trailing roster-shuffle at a half boundary) means the
            # kind-implied PRIMARY side is sometimes wrong for ANY
            # substitution kind -- see this gate's IMPLEMENTER_RESULT for the
            # full real-data measurement (issue #31/#32, 22 of 47 real
            # "to dh for" lines). Resolve BOTH sides and accept whichever ONE
            # fully resolves both names uniquely. If BOTH sides fully
            # resolve -- a genuine cross-side ambiguity, e.g. a common
            # surname pair that happens to exist on both rosters -- or if
            # NEITHER does, keep the existing honest unparsed[] behavior.
            # Never guesses. Both sides are always checked (not
            # short-circuited on a primary success) specifically so this
            # ambiguous-on-both-sides case is caught rather than silently
            # accepting a coincidentally-matching primary.
            p_in_pid, p_in_ok, p_out_pid, p_out_ok = _resolve_substitution_pair(
                player_table, in_last, in_full, out_last, out_full, primary_side
            )
            f_in_pid, f_in_ok, f_out_pid, f_out_ok = _resolve_substitution_pair(
                player_table, in_last, in_full, out_last, out_full, fallback_side
            )
            primary_full = p_in_ok and p_out_ok
            fallback_full = f_in_ok and f_out_ok
            if primary_full and not fallback_full:
                side, team_id = primary_side, primary_team
                in_pid, in_ok, out_pid, out_ok = p_in_pid, p_in_ok, p_out_pid, p_out_ok
            elif fallback_full and not primary_full:
                side, team_id = fallback_side, fallback_team
                in_pid, in_ok, out_pid, out_ok = f_in_pid, f_in_ok, f_out_pid, f_out_ok
            elif primary_full and fallback_full:
                # BOTH sides resolve every name -- but "this surname exists on
                # both rosters" is NOT the same thing as "the line is
                # ambiguous". Each side resolved to exactly ONE player, so the
                # only open question is WHICH SIDE, and the line itself
                # usually answers it: the PBP names a first initial, and
                # "J. Smith" cannot be Tanner Smith.
                #
                # That evidence was being discarded. `resolve()` threads the
                # full pbp token in as `full_name`, but `_narrow_by_first_token`
                # only consults it on the >= 2-candidates-WITHIN-ONE-SIDE
                # branch -- so when each roster held exactly one Smith, the
                # initial never entered the decision at all and the line was
                # filed as "genuine cross-side ambiguity". It was not
                # ambiguous; the discriminating fact was simply never read.
                #
                # Measured against known answers before use (the discipline
                # DECISION.md sets for every identity signal): over all 14,675
                # substitution lines the corpus already resolves -- where the
                # answer is established independently, by the outgoing name or
                # by the surname resolving on one roster only -- this rule
                # fires on 14,409 and is correct on 14,409. Zero wrong. It
                # declines to fire on the other 266 rather than guessing.
                #
                # Two rival signals were measured on the same known answers
                # and REJECTED, and are recorded here so the choice stays
                # visible: the kind-implied fielding side ("X to 1b" is a
                # defensive move, so X fields) scores 97.8% -- a half-inning
                # boundary roster shuffle is logged in the previous half often
                # enough to break it; boxscore position overlap ("X to lf" and
                # only one candidate is listed at lf) scores 95.0%; and
                # match-tier precedence (an exact surname match outranking the
                # one-edit typo tier) scores 93.3%. All three are below the
                # 98.44% signal issue #40 rejected, so none is used. Only the
                # first-name token is forced.
                primary_agrees = _side_first_names_agree(
                    player_table,
                    primary_side,
                    ((in_full, p_in_pid), (out_full, p_out_pid)),
                )
                fallback_agrees = _side_first_names_agree(
                    player_table,
                    fallback_side,
                    ((in_full, f_in_pid), (out_full, f_out_pid)),
                )
                if primary_agrees and not fallback_agrees:
                    side, team_id = primary_side, primary_team
                    in_pid, in_ok, out_pid, out_ok = (
                        p_in_pid, p_in_ok, p_out_pid, p_out_ok
                    )
                elif fallback_agrees and not primary_agrees:
                    side, team_id = fallback_side, fallback_team
                    in_pid, in_ok, out_pid, out_ok = (
                        f_in_pid, f_in_ok, f_out_pid, f_out_ok
                    )
                else:
                    # The initial does not separate them either -- both
                    # candidates answer to it (two players who share a first
                    # AND last name, one per team), or neither does. This is
                    # the irreducible case. Never guess.
                    side, team_id = primary_side, primary_team
                    in_pid, in_ok, out_pid, out_ok = None, False, None, False
            else:
                # Neither side fully resolves -- never guess.
                side, team_id = primary_side, primary_team
                in_pid, in_ok, out_pid, out_ok = None, False, None, False
            # issue #40: admit a PBP-named player the boxscore never lists,
            # rather than losing the whole substitution.
            #
            # StatCrew omits an all-zero box row for a player who entered and
            # then never batted or reached (91% of this population), and
            # rarely omits one who DID record a plate appearance. Either way
            # the narrative is authority that they played.
            #
            # This reads the PER-SIDE partial results, not in_ok/out_ok --
            # the branch above deliberately zeroes those whenever neither
            # side resolves BOTH names, which is exactly the case here. The
            # resolved name is the anchor for which team this is; that anchor
            # must be unambiguous, so we require the other side to have
            # matched NEITHER name. Without an anchor we still refuse.
            if not in_ok and not out_ok:
                primary_half = p_in_ok != p_out_ok
                fallback_half = f_in_ok != f_out_ok
                anchor = None
                if primary_half and not (f_in_ok or f_out_ok):
                    anchor = (primary_side, primary_team, p_in_pid, p_in_ok, p_out_pid, p_out_ok)
                elif fallback_half and not (p_in_ok or p_out_ok):
                    anchor = (fallback_side, fallback_team, f_in_pid, f_in_ok, f_out_pid, f_out_ok)
                if anchor is not None:
                    a_side, a_team, a_in_pid, a_in_ok, a_out_pid, a_out_ok = anchor
                    missing = cg.substitution.player_out if a_in_ok else cg.substitution.player_in
                    declared = _declare_from_pbp(missing, a_side) if missing else None
                    if declared is not None:
                        side, team_id = a_side, a_team
                        if a_in_ok:
                            in_pid, in_ok = a_in_pid, True
                            out_pid, out_ok = declared, True
                        else:
                            out_pid, out_ok = a_out_pid, True
                            in_pid, in_ok = declared, True

            if forced_sub_pids is not None:
                # The blank-incoming-pitcher branch above already established
                # both ends -- the successor by boxscore order, the outgoing
                # player by a unique name resolution. Bypass the name-based
                # pair resolution rather than round-tripping the successor
                # back through his own name, which would re-introduce exactly
                # the same-surname ambiguity the pid already settles.
                side, team_id = fielding_side, fielding_team_id
                in_pid, in_ok = forced_sub_pids[0], True
                out_pid, out_ok = forced_sub_pids[1], True

            if not out_ok or not in_ok:
                if primary_full and fallback_full:
                    reason = (
                        f"substitution names resolved uniquely on BOTH the "
                        f"{primary_side} and {fallback_side} side and the "
                        f"first-name token does not separate them: "
                        f"out={cg.substitution.player_out!r} "
                        f"in={cg.substitution.player_in!r}"
                    )
                else:
                    reason = (
                        f"substitution names did not resolve uniquely on "
                        f"either the {primary_side} or {fallback_side} "
                        f"side: out={cg.substitution.player_out!r} "
                        f"in={cg.substitution.player_in!r}"
                    )
                _unparsed(line, reason)
                continue
            slot = None
            if out_pid is not None:
                for s, pid in slot_occupant[team_id].items():
                    if pid == out_pid:
                        slot = s
                        break
            # Pitcher-of-record tracking updates regardless of whether the
            # outgoing player holds a batting-order slot (later PAs' `pitcher`
            # field depends on it). A bare DH-slot entry (out_pid is None)
            # names no outgoing player at all, so it is never a pitching
            # change and never updates pitcher-of-record bookkeeping.
            if out_pid is not None and (
                slot is None or out_pid in pitching_pool[team_id] or out_pid == current_pitcher[team_id]
            ):
                current_pitcher[team_id] = in_pid
            if slot is not None:
                slot_occupant[team_id][slot] = in_pid
            # A substitute for a player standing on a base INHERITS that base.
            # `base_occ` is keyed by player_id and is what a later runner
            # clause reads to resolve its own `from` (see the `from_base is
            # None` fallback above). Without this transfer the incoming runner
            # is untracked, that fallback fires, and the clause is emitted with
            # `from` == `to` -- which the g6 replayer then correctly flags as
            # an illegal transition, because the destination base is not
            # occupied at event start. 59 of the 60 illegal_transition
            # warnings across the corpus's clean-parse games trace to exactly
            # this (issue #33).
            #
            # Not gated on substitution.kind: whatever the announcement is
            # called, if the outgoing player is on a base then the incoming
            # player physically takes that base. A pitching change cannot
            # trip this, since a fielding pitcher occupies no base.
            if out_pid is not None and in_pid is not None:
                for _base, _occ_pid in list(base_occ.items()):
                    if _occ_pid == out_pid:
                        base_occ[_base] = in_pid
            # Under a DH rule the pitcher is not in the batting order ->
            # slot=None (schema 1.1.0 made substitution.slot nullable, so
            # this is now a real event, not an unparsed[] residue). `kind`
            # is read from the grammar's own Substitution.kind (issue #30
            # g2b) rather than hardcoded, since the new bare DH-slot-entry
            # row builds kind="offensive" -- stamping "pitching" on it here
            # would contradict the pitcher-of-record bookkeeping just above,
            # which already treats a null player_out as never a pitching
            # change.
            sub_obj = {
                "slot": slot,
                "player_out": out_pid,
                "player_in": in_pid,
                "kind": cg.substitution.kind,
                "after_event_seq": seq - 1 if seq > 0 else 0,
            }
            events.append(
                {
                    "seq": seq,
                    "inning": line.inning,
                    "half": line.half,
                    "kind": "substitution",
                    "batting_team": batting_team_id,
                    "fielding_team": fielding_team_id,
                    "narrative": _display_narrative(line.text),
                    "scoring_play": line.is_strong,
                    "substitution": sub_obj,
                }
            )
            subs_by_team[team_id].append(sub_obj)
            seq += 1
            continue

        if cg.kind == "runner_event":
            runners: List[dict] = []
            ok = True
            for rm in cg.runners:
                rec = _resolve_runner(rm, batting_side, modifiers=None)
                if rec is None:
                    ok = False
                    break
                runners.append(rec)
            if not ok:
                _unparsed(line, "runner clause name did not resolve uniquely")
                continue
            merged_runners = _merge_same_runner(runners)
            _apply_occupancy(merged_runners)
            _note_inferred_runners(line, cg)
            events.append(
                {
                    "seq": seq,
                    "inning": line.inning,
                    "half": line.half,
                    "kind": "runner_event",
                    "batting_team": batting_team_id,
                    "fielding_team": fielding_team_id,
                    "narrative": _display_narrative(line.text),
                    "scoring_play": line.is_strong,
                    "runners": merged_runners,
                }
            )
            seq += 1
            continue

        # plate_appearance
        p = cg.primary
        batter_last = _last_name_token(p.name_token)
        batter_pid, batter_ok = player_table.resolve(batter_last, batting_side, p.name_token)
        if not batter_ok:
            _unparsed(line, f"batter name did not resolve uniquely: {p.name_token!r}")
            continue
        pitcher_pid = current_pitcher.get(fielding_team_id)

        fielders = list(p.fielders)
        if not fielders and p.location:
            fielders = _location_to_fielders(p.location)

        cause, dest_token, out_flag, scored_flag = BATTER_OUTCOME_CAUSE[p.outcome_type]
        to_base = _DEST_BASE[dest_token] if dest_token else -1
        batter_runner: dict = {
            "player_id": batter_pid,
            "from": 0,
            "to": to_base,
            "cause": cause,
            "out": out_flag,
            "scored": scored_flag,
        }
        if scored_flag:
            batter_runner["earned"] = True
            batter_runner["rbi"] = "RBI" in p.modifiers

        runner_records = [batter_runner]
        ok = True
        for rm in cg.runners:
            rec = _resolve_runner(rm, batting_side, modifiers=p.modifiers)
            if rec is None:
                ok = False
                break
            runner_records.append(rec)
        if not ok:
            _unparsed(line, "runner clause name did not resolve uniquely")
            continue

        # Issue #40: an out this line RECORDS but never attributes -- "X
        # reached on a fielder's choice, out at second ss to 2b". Until now a
        # `.*` tail swept that clause into `modifiers` and the out simply
        # vanished from the record (14 corpus lines).
        #
        # The runner is not named, but on a fielder's choice he is forced: an
        # out at second retires the runner who was on first, at third the one
        # on second, at home the one on third. Read from the base occupancy
        # as it stood BEFORE this line (line_snapshot), the same source every
        # runner clause's `from` reads.
        #
        # When that base was empty the play is not a force and the line does
        # not say who was out, so nothing is asserted -- the out is dropped
        # exactly as before, but LOUDLY, as a parse warning rather than
        # silently. Never a guess.
        if p.forced_out_at is not None:
            out_base = _DEST_BASE[p.forced_out_at]

            # The stated base can CONTRADICT the line's own fielding chain,
            # and when it does the chain is right. "out at second ss to 1b"
            # ends the throw at the FIRST BASEMAN, who does not record an out
            # at second; the out was made at first, and on a fielder's choice
            # the runner going to first is the BATTER. So the source's
            # "reached" is the error, not the out.
            #
            # This matters twice over, because the same mislabelled line
            # lands in two different places depending on who happens to be
            # standing on first:
            #
            #   base 1 EMPTY    -- no force exists, so the line was refused
            #                      whole. 4 games, and the refusal looked
            #                      principled ("names no runner").
            #   base 1 OCCUPIED -- far worse. The block below pins the out on
            #                      whoever stood there, and _merge_same_runner
            #                      then folds that fabricated out together
            #                      with the SAME player's explicitly narrated
            #                      safe advance, keeping the out. 6 games
            #                      where the line says "J. Daly advanced to
            #                      second" and the record says Daly was out.
            #                      Three of them parse clean AND replay, so
            #                      nothing anywhere reported it.
            #
            # Measured: 11 lines in the corpus state an out base while their
            # chain ends at 1b, and all 11 are this defect. Every fielder's-
            # choice line whose chain AGREES with its stated base is left
            # alone -- the rule fires on the contradiction, not on the shape.
            chain_end = (p.forced_out_chain or "").split(" to ")[-1].strip()

            # A second reading of the same defect. The unattributed out can
            # name a base at which a runner the line NAMES is already out:
            #
            #   "N. Marcelo reached on a fielder's choice to shortstop, out
            #    at second 1b to 2b; M. O'Hara out at second ss to 2b."
            #
            # Two runners cannot both be retired at second base on one play,
            # so the unattributed out is not a second runner -- it is the
            # batter, and "reached" is again the part that is wrong. One line
            # states it twice over: 20240530_ufps repeats the identical chain
            # ("3b to 2b") in both clauses.
            #
            # 3 corpus lines, and the argument is logical rather than
            # statistical: the base is already spoken for. Each reconciles on
            # TWO independent oracles at once -- the half goes from 2 outs to
            # its expected 3, and the folded LOB goes from 1 to the box's 0,
            # which only happens if the batter is off the bases.
            named_out_at_same_base = any(
                rm.out
                and rm.destination is not None
                and _DEST_BASE.get(rm.destination) == out_base
                for rm in cg.runners
            )

            if named_out_at_same_base or (
                chain_end == "1b" and p.forced_out_at != "first"
            ):
                batter_runner["to"] = -1
                batter_runner["out"] = True
                batter_runner["cause"] = "force_out"
                batter_runner.pop("earned", None)
                batter_runner.pop("rbi", None)
                _inferred(
                    line,
                    "out_base_contradicts_fielding_chain",
                    (
                        f"line says the out was at {p.forced_out_at} but a "
                        "runner it NAMES is already out there, and two "
                        "runners cannot be retired at one base on one play; "
                        "the out is the batter's"
                    )
                    if named_out_at_same_base
                    else (
                        f"line says the out was at {p.forced_out_at} but its "
                        f"own chain {p.forced_out_chain!r} ends at the first "
                        "baseman, who cannot record one there; the out is the "
                        "batter's at first, and no runner is retired"
                    ),
                )
            else:
                forced_pid = line_snapshot.get(out_base - 1)
                if forced_pid is None:
                    _unparsed(
                        line,
                        "line records an out at "
                        f"{p.forced_out_at} but names no runner, and base "
                        f"{out_base - 1} was empty at the start of the play "
                        "so the force does not identify one",
                    )
                    continue
                runner_records.append(
                    {
                        "player_id": forced_pid,
                        "from": out_base - 1,
                        "to": -1,
                        "cause": "force_out",
                        "out": True,
                        "scored": False,
                    }
                )
                base_occ.pop(out_base - 1, None)
                _inferred(
                    line,
                    "unattributed_force_out",
                    f"out at {p.forced_out_at} attributed to {forced_pid}, "
                    f"the runner occupying base {out_base - 1} before the "
                    "play; the line records the out but names no runner",
                )

        runner_records = _merge_same_runner(runner_records)

        # Apply EVERY named participant's base-occupancy update -- batter and
        # runners alike -- from the MERGED records. This subsumes what used to
        # be a batter-only block whose comment already explained why merged
        # records are the right source; the runner path now gets the identical
        # treatment instead of writing hop-by-hop (issue #33).
        _apply_occupancy(runner_records)
        # Schema: "Outs THIS event adds." Read that off the MERGED runner
        # records -- the same primitives g6 replay folds -- not off the
        # primary verb's `out_flag`, which describes the verb rather than the
        # play's outcome. The two disagree whenever a chained clause changes
        # the batter's disposition, in BOTH directions:
        #
        #   "struck out swinging, reached first on a wild pitch" -- a dropped
        #   third strike. The pitcher is credited a strikeout, but the batter
        #   is standing on first and the play adds NO out. `out_flag` said 1.
        #   108 corpus events carried that contradiction: an `outcome` block
        #   claiming an out beside a `runners` entry with `out: false, to: 1`.
        #
        #   "singled to right field, out at second" -- the batter is retired
        #   after reaching. `out_flag` said 0, and the retiring clause merged
        #   into the batter's own record at index 0, which the old
        #   `records[1:]` slice skipped, so the out vanished.
        #
        # Replay never noticed either: `fold_base_out` derives outs from
        # `runners[].out` independently and ignores this field entirely. That
        # is exactly why it could stay wrong -- so it is fixed at the source
        # rather than left as a discrepancy a consumer would have to know
        # about.
        outs_recorded = sum(1 for r in runner_records if r["out"])

        _note_inferred_runners(line, cg)
        events.append(
            {
                "seq": seq,
                "inning": line.inning,
                "half": line.half,
                "kind": "plate_appearance",
                "batting_team": batting_team_id,
                "fielding_team": fielding_team_id,
                "narrative": _display_narrative(line.text),
                "scoring_play": line.is_strong,
                "batter": {
                    "player_id": batter_pid,
                    "name_raw": p.name_token,
                    "resolved": True,
                },
                "pitcher": {
                    "player_id": pitcher_pid,
                    "name_raw": None,
                    "resolved": pitcher_pid is not None,
                },
                "outcome": {
                    "type": p.outcome_type,
                    "modifiers": list(p.modifiers),
                    "fielders": fielders,
                    "outs_recorded": outs_recorded,
                    "location": p.location,
                },
                "count": (
                    {"balls": p.count.balls, "strikes": p.count.strikes}
                    if p.count is not None
                    else None
                ),
                "pitches": p.pitches,
                "runners": runner_records,
            }
        )
        seq += 1

    # PASS 2. `slot_occupant` is seeded above from `list(players)[:9]` -- the
    # same naive slice `_reconstruct_starting_order` measures at 44.3%, and
    # it drives every `substitution.slot`. It has to be seeded BEFORE the fold
    # loop, while the corrected order needs `events`, so the correction can
    # only happen afterwards: recompute the order, then re-walk ONLY the
    # substitution chronology.
    #
    # Nothing else moves. Batter and runner resolution never read
    # `slot_occupant` -- its only readers are the seed above and the
    # substitution branch -- so pass 2 cannot disturb `events`. Measured over
    # the whole corpus: 0 replay-verdict changes, 0 unparsed-count changes,
    # events[] byte-identical apart from `substitution.slot`.
    if box_batting is not None:
        for team in (player_table.home, player_table.away):
            subs = subs_by_team.get(team.team_id, [])
            fixed = _reconstruct_starting_order(
                box_batting.get(team.team_id, []), subs, events, team.team_id
            )
            if fixed is None:
                fixed = list(team.players.keys())[:9]
            occupant = {i + 1: pid for i, pid in enumerate(fixed)}
            for sub in subs:
                out_pid, in_pid = sub.get("player_out"), sub.get("player_in")
                slot = None
                if out_pid is not None:
                    slot = next(
                        (s for s, pid in occupant.items() if pid == out_pid), None
                    )
                if slot is not None:
                    # A player cannot hold two batting slots at once. When
                    # `player_in` already occupies a DIFFERENT one, this line
                    # realigns someone already in the lineup rather than
                    # changing it, and transferring would overwrite his real
                    # slot.
                    #
                    # The guard and the seed MUST land together. Measured on
                    # an oracle that needs no narrative -- a player recorded
                    # entering two different slots in one game, which is
                    # impossible -- the corrected seed ALONE makes things
                    # worse (57 -> 67), because the two defects were
                    # partially masking each other. Together: 57 -> 25, over
                    # 45 -> 21 team-games, while slot coverage RISES from
                    # 3,615 to 4,016 substitutions.
                    if any(
                        s != slot and pid == in_pid for s, pid in occupant.items()
                    ):
                        slot = None
                    else:
                        occupant[slot] = in_pid
                sub["slot"] = slot

    return events, unparsed, subs_by_team, inferred


# ---------------------------------------------------------------------------
# Impure HTML-reading helpers (these call html_struct/identity; build_events
# above never sees a Node).
# ---------------------------------------------------------------------------

_PANE_ID_RE = re.compile(r"^pbp-inning-(\d+)$")
_HALF_RE = re.compile(r"(Top|Bottom) of")


def _iter_halves(root: Node) -> List[PbpLine]:
    """Read every pbp-inning-N pane's half-tables (each captioned "Top of
    Nth"/"Bottom of Nth") in document order, returning the flat ordered
    ``PbpLine`` sequence build_events folds over.

    This is this module's OWN half-split traversal: ``html_struct.
    iter_pbp_panes`` deliberately does not split top/bottom (that is parser
    semantics, per its own docstring), so this walks the pane's nested
    `<table>` elements (one per half) directly with the same generic
    primitives.
    """
    lines: List[PbpLine] = []
    panes = [
        node
        for node in find_all(root, "section")
        if _PANE_ID_RE.match(node.attrs.get("id") or "")
    ]
    for pane in panes:
        inning = int(_PANE_ID_RE.match(pane.attrs["id"]).group(1))
        for table in find_all(pane, "table"):
            headers = find_all(table, "h3")
            if not headers:
                continue
            caption_text = " ".join(text_of(headers[0]).split())
            m = _HALF_RE.search(caption_text)
            if not m:
                continue
            half = "top" if m.group(1) == "Top" else "bottom"
            cells = [
                td
                for td in find_all_by_class(table, "text")
                if td.tag == "td"
            ]
            for idx, td in enumerate(cells):
                text = text_of(td)
                if _SCORER_DIRECTIVE_RE.search(text):
                    # "R. Preece Batter set to A. Albert." -- a StatCrew
                    # scorer CORRECTION directive (the scorer fixing which
                    # batter the software thinks is up), not a description of
                    # anything that happened on the field. Same reasoning as
                    # the empty cell below: nothing to represent as an event
                    # and nothing to preserve verbatim, so it never becomes a
                    # narrative line (issue #40, human-ratified).
                    continue
                if not text.strip():
                    # An EMPTY `td.text` cell is layout, not narrative. It
                    # carries nothing to represent as an event and nothing to
                    # preserve verbatim, so it is skipped here rather than
                    # reaching the grammar and landing in `unparsed[]` as an
                    # "empty clause body" -- 27 such entries across the
                    # archived corpus, 8 of them the sole thing keeping a game
                    # off clean-parse (issue #40).
                    #
                    # This does not weaken the never-drop guarantee in this
                    # module's docstring: that covers PBP narrative lines, and
                    # an empty cell is not one. `line_index` still enumerates
                    # the SOURCE cells, so a skipped cell leaves a visible gap
                    # rather than silently renumbering its neighbours.
                    continue
                lines.append(
                    PbpLine(
                        inning=inning,
                        half=half,
                        line_index=idx,
                        text=text,
                        is_strong=is_strong(td),
                    )
                )
    return lines


def _extract_game_id(source_url: str) -> str:
    m = re.search(r"/boxscores/([A-Za-z0-9_]+)\.xml", source_url)
    if not m:
        raise ValueError(f"could not extract game_id from source_url: {source_url!r}")
    return m.group(1)


_MONTHS = (
    "January February March April May June July August September October "
    "November December"
).split()


def _extract_date_iso(root: Node) -> str:
    nodes = find_all_by_class(root, "date")
    if not nodes:
        raise ValueError("no element with class 'date' found; cannot extract game date")
    raw = " ".join(text_of(nodes[0]).split())
    dt = datetime.strptime(raw, "%B %d, %Y")
    return dt.date().isoformat()


def _find_tables_by_caption(root: Node, marker: str) -> List[Node]:
    tables = []
    for table in find_all(root, "table"):
        captions = find_all(table, "caption")
        if captions and marker in text_of(captions[0]):
            tables.append(table)
    return tables


def _caption_team_name(table: Node) -> str:
    caption = find_all(table, "caption")[0]
    names = find_all_by_class(caption, "team-name")
    return " ".join(text_of(names[0]).split())


def _parse_linescore(root: Node, player_table: identity.PlayerTable) -> dict:
    divs = find_all_by_class(root, "linescore")
    if not divs:
        raise ValueError("no element with class 'linescore' found")
    table = find_all(divs[0], "table")[0]
    home_name = player_table.home.name
    away_name = player_table.away.name

    innings: Dict[str, List[Optional[int]]] = {}
    totals: Dict[str, dict] = {}
    for row in find_all(table, "tr"):
        cells = [
            text_of(c)
            for c in row.children
            if isinstance(c, Node) and c.tag in ("td", "th")
        ]
        if not cells:
            continue
        name = " ".join(cells[0].split())
        if name == home_name:
            side = "home"
        elif name == away_name:
            side = "away"
        else:
            continue
        rest = cells[1:]
        inning_cells, rhe_cells = rest[:-3], rest[-3:]
        innings[side] = [None if c.strip() == "X" else int(c) for c in inning_cells]
        r, h, e = (int(c) for c in rhe_cells)
        totals[side] = {"R": r, "H": h, "E": e}

    return {
        "innings": {"away": innings["away"], "home": innings["home"]},
        "totals": {"away": totals["away"], "home": totals["home"]},
    }


def _row_player_name(row_header: Node) -> Optional[str]:
    names = find_all_by_class(row_header, "player-name")
    if not names:
        return None
    return " ".join(text_of(names[0]).split())


def _row_position(row_header: Node) -> str:
    positions = find_all_by_class(row_header, "position")
    if not positions:
        return ""
    return " ".join(text_of(positions[0]).split()).lower()


def _team_name_to_id(player_table: identity.PlayerTable) -> Dict[str, str]:
    return {
        player_table.home.name: player_table.home.team_id,
        player_table.away.name: player_table.away.team_id,
    }


def _name_to_pid_map(player_table: identity.PlayerTable, team_id: str) -> Dict[str, str]:
    team = player_table.home if team_id == player_table.home.team_id else player_table.away
    return {entry.name: pid for pid, entry in team.players.items()}


def _parse_box_batting(root: Node, player_table: identity.PlayerTable) -> Dict[str, List[dict]]:
    name_to_id = _team_name_to_id(player_table)
    out: Dict[str, List[dict]] = {}
    for table in _find_tables_by_caption(root, "Batters"):
        team_name = _caption_team_name(table)
        team_id = name_to_id.get(team_name)
        if team_id is None:
            continue
        pid_by_name = _name_to_pid_map(player_table, team_id)
        lines: List[dict] = []
        for row in find_all(table, "tr"):
            headers = [th for th in find_all(row, "th") if th.attrs.get("scope") == "row"]
            if not headers:
                continue
            name = _row_player_name(headers[0])
            if name is None:
                continue  # e.g. the "Totals" row
            pid = pid_by_name.get(name)
            if pid is None:
                continue
            cells = [
                text_of(c)
                for c in row.children
                if isinstance(c, Node) and c.tag == "td"
            ]
            if len(cells) < 8:
                continue
            ab, r, h, rbi, bb, so, lob, avg = cells[:8]
            lines.append(
                {
                    "player_id": pid,
                    "pos": _row_position(headers[0]),
                    "AB": int(ab),
                    "R": int(r),
                    "H": int(h),
                    "RBI": int(rbi),
                    "BB": int(bb),
                    "SO": int(so),
                    "LOB": int(lob),
                    "AVG": avg.strip(),
                }
            )
        out[team_id] = lines
    return out


def _parse_box_pitching(root: Node, player_table: identity.PlayerTable) -> Dict[str, List[dict]]:
    name_to_id = _team_name_to_id(player_table)
    out: Dict[str, List[dict]] = {}
    for table in _find_tables_by_caption(root, "Pitchers"):
        team_name = _caption_team_name(table)
        team_id = name_to_id.get(team_name)
        if team_id is None:
            continue
        pid_by_name = _name_to_pid_map(player_table, team_id)
        lines: List[dict] = []
        for row in find_all(table, "tr"):
            headers = [th for th in find_all(row, "th") if th.attrs.get("scope") == "row"]
            if not headers:
                continue
            name = _row_player_name(headers[0])
            if name is None:
                continue
            pid = pid_by_name.get(name)
            if pid is None:
                continue
            cells = [
                text_of(c)
                for c in row.children
                if isinstance(c, Node) and c.tag == "td"
            ]
            if len(cells) < 6:
                continue
            ip, h, r, er, bb, so = cells[:6]
            lines.append(
                {
                    "player_id": pid,
                    "IP": ip.strip(),
                    "H": int(h),
                    "R": int(r),
                    "ER": int(er),
                    "BB": int(bb),
                    "SO": int(so),
                }
            )
        out[team_id] = lines
    return out


def _reconstruct_starting_order(
    box_rows: List[dict],
    subs: List[dict],
    events: List[dict],
    team_id: str,
) -> Optional[List[str]]:
    """The first nine GENUINE starters, from boxscore row order.

    StatCrew nests a slot's substitutes immediately UNDER that slot's
    starter, so taking the first nine rows in document order pulls bench
    players into the starting nine and pushes real starters past index 9.
    Measured against the first nine distinct batters each team's own
    narrative shows, plain `[:9]` slicing agrees on only 1,309 of 2,957
    team-games -- 44.3%. It is not a small error; the field was wrong more
    often than it was right, and nothing validated it.

    A row is a fresh ENTRANT, and so not a starter, only when its player's
    earliest substitution mention that NAMES a player_out is an entry, and
    that entry comes before the player's own first plate appearance. Both
    halves of that test earn their place:

      - A bare `player_out: None` announcement is not evidence of entry. It
        as often means "starter X moves to a different position" as "bench
        player X comes in" (20240821_e0zj: Dondrei Hubbard's bare "to dh"
        line lands AFTER his first plate appearance).
      - A player already batting before his first named entry is a
        defensive swap between two men both already in the lineup, each
        naming the other as outgoing -- not an arrival.

    Scored the same way: 2,860 of 2,952 team-games, 96.9%, against an
    independent third source. The two losses are a disclosed blind spot,
    not a new failure mode: a starter who went 0-for-0 can be omitted from
    the Batters table altogether, leaving no row to keep.

    MEASURED NEGATIVE, so it is not retried: excluding a row whose position
    duplicates the row above it -- on the theory that a substitute inherits
    the position of the man he replaced -- scores only 54.8%. Substitutes
    frequently take a different position.

    Returns None when fewer than nine rows survive, leaving the caller to
    decide rather than guessing.
    """
    first_pa: Dict[str, int] = {}
    for event in events:
        if event["kind"] == "plate_appearance" and event["batting_team"] == team_id:
            first_pa.setdefault(event["batter"]["player_id"], event["seq"])

    first_named_entry: Dict[str, int] = {}
    for sub in subs:
        if sub["player_in"] is not None and sub["player_out"] is not None:
            seq = sub["after_event_seq"]
            pid = sub["player_in"]
            if seq < first_named_entry.get(pid, seq + 1):
                first_named_entry[pid] = seq

    starters = [
        row["player_id"]
        for row in box_rows
        if not (
            row["player_id"] in first_named_entry
            and first_pa.get(row["player_id"], float("inf"))
            >= first_named_entry[row["player_id"]]
        )
    ]
    return starters[:9] if len(starters) >= 9 else None


def _build_lineups(
    player_table: identity.PlayerTable,
    subs_by_team: Dict[str, List[dict]],
    box_batting: Dict[str, List[dict]],
    events: List[dict],
) -> Dict[str, dict]:
    lineups: Dict[str, dict] = {}
    for team in (player_table.home, player_table.away):
        subs = subs_by_team.get(team.team_id, [])
        batting_ids = _reconstruct_starting_order(
            box_batting.get(team.team_id, []), subs, events, team.team_id
        )
        if batting_ids is None:
            # Fewer than nine rows survived -- 5 team-games in 2,968. Keep
            # the old behaviour rather than raise, so the shortfall stays
            # visible in the field instead of killing the parse.
            batting_ids = list(team.players.keys())[:9]
        lineups[team.team_id] = {
            "batting_order": [
                {"slot": i + 1, "player_id": pid} for i, pid in enumerate(batting_ids)
            ],
            "substitutions": subs,
        }
    return lineups


def _person_id_for(
    player_id: str, person_ids: Optional[Dict[str, Optional[str]]]
) -> Optional[str]:
    """The schema 1.7.0 ``person_id`` for one ``player_id`` in THIS game.

    A REAL 16-char Presto id is its own person id: it is already stable for a
    whole season, so it needs no consolidation layer and is never absorbed
    into another person. That rule lives here, in exactly one place, which is
    why ``person_map``'s artifact carries assignments for SYNTHETIC ids only.

    A synthetic ``syn:<side>:<n>`` is per-GAME positional -- the same value is
    a different person in another game -- so it can only be resolved from the
    corpus-level map the caller passes in. With no map supplied, or with this
    id deliberately unlinked in it, the answer is an honest ``None``: never a
    guess, and never the synthetic id itself (which would fabricate a join key
    that silently merges strangers).
    """
    if not player_id.startswith("syn:"):
        return player_id
    if not person_ids:
        return None
    return person_ids.get(player_id)


def _players_table(
    player_table: identity.PlayerTable,
    person_ids: Optional[Dict[str, Optional[str]]] = None,
    career_ids: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, dict]:
    players: Dict[str, dict] = {}
    for team in (player_table.home, player_table.away):
        for pid, entry in team.players.items():
            person_id = _person_id_for(pid, person_ids)
            players[pid] = {
                "player_id": entry.player_id,
                "name": entry.name,
                "last_name": entry.last_name,
                "team_id": entry.team_id,
                "bats_side": entry.bats_side,
                "positions": list(entry.positions),
                "box_listed": entry.box_listed,
                "person_id": person_id,
                # schema 1.9.0. Composed here rather than in either map module
                # because this is the one place a player is already resolved to
                # a person: career_map is keyed by person_id, person_map by
                # player_id, and neither needs to know about the other. A person
                # we could not identify has no career either -- never a guess.
                "career_id": (
                    (career_ids or {}).get(person_id) if person_id else None
                ),
            }
    return players


def parse_game(
    html: str,
    *,
    source_url: str,
    fetched_at: str,
    parsed_at: Optional[str] = None,
    league_id: str = "pioneer",
    provider: str = "prestosports",
    id_overrides: Optional[Dict[Tuple[str, str], str]] = None,
    person_ids: Optional[Dict[str, Optional[str]]] = None,
    career_ids: Optional[Dict[str, Optional[str]]] = None,
    errata_entries: Optional[Sequence[dict]] = None,
) -> dict:
    """Parse raw boxscore HTML into a full schema-valid ``final`` game dict.

    Raises ``NonFinalPageError`` if the page has no PBP panes (the negative-
    path contract) -- never fabricates a `final` file from such a page.

    ``person_ids`` (schema 1.7.0, issue #41) maps this game's SYNTHETIC
    ``player_id``s to their corpus-level ``person_id``, as built by
    ``bc_pipeline.person_map`` and passed in by the re-parse driver. Real ids
    resolve to themselves without it (see ``_person_id_for``); omitting it
    leaves every synthetic player's ``person_id`` null rather than guessing.

    ``career_ids`` (schema 1.9.0) maps a PERSON id to its cross-season
    ``career_id``, as built by ``bc_pipeline.career_map``. Keyed by person
    rather than player because that is the layer above; a player with no
    resolvable ``person_id`` gets no career either.
    """
    root = parse_html(html)
    if not has_pbp_panes(root):
        raise NonFinalPageError(
            "page has no PBP panes (id='pbp-inning-N'); not a final boxscore"
        )

    game_id = _extract_game_id(source_url)
    date_iso = _extract_date_iso(root)
    season = int(date_iso[:4])

    player_table = identity.build_player_table(root, id_overrides=id_overrides)
    lines = _iter_halves(root)
    # Authored corrections to DEFECTIVE SOURCE LINES, applied to the raw line
    # text BEFORE the grammar sees it -- so a corrected line is parsed by
    # exactly the same rules as every other line, and no rule is relaxed
    # anywhere to accommodate a one-off. See bc_pipeline.errata for the
    # admissibility bar. Defaults to the COMMITTED errata file so that every
    # caller (reparse, backfill, fetch) parses a game identically; pass an
    # empty sequence to disable.
    lines, erratum_applications = errata_mod.apply_to_lines(
        errata_mod.for_game(game_id) if errata_entries is None else errata_entries,
        lines,
    )
    # The boxscore is parsed BEFORE the events because build_events needs the
    # Pitchers table's appearance order (issue #40's blank-incoming-pitcher
    # rule). Neither box parser reads events, so the reorder is inert.
    linescore = _parse_linescore(root, player_table)
    box = {
        "batting": _parse_box_batting(root, player_table),
        "pitching": _parse_box_pitching(root, player_table),
    }
    events, unparsed, subs_by_team, inferred = build_events(
        lines,
        player_table,
        box_pitching_order={
            team_id: [row["player_id"] for row in rows]
            for team_id, rows in box["pitching"].items()
        },
        box_batting=box["batting"],
    )
    # Disclose every applied correction alongside the parser's own
    # inferences, under the same contract: a consumer that wants only what
    # the source actually said can drop the events these entries name.
    # Prepended because a correction happened BEFORE any rule fired on the
    # line, so `inferred[]` reads in the order the assertions were made.
    inferred = [
        {
            "location": app["location"],
            "raw": app["raw"],
            "rule": "erratum",
            "asserted": (
                f"erratum {app['erratum_id']} ({app['class']}): read as "
                f"{' '.join(app['corrected'].split())!r}. {app['evidence']}"
            ),
        }
        for app in erratum_applications
    ] + inferred

    lineups = _build_lineups(player_table, subs_by_team, box["batting"], events)
    players = _players_table(player_table, person_ids, career_ids)

    parsed_at_iso = parsed_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "season": season,
        "status": "final",
        "date": date_iso,
        "source": {
            "provider": provider,
            "league_id": league_id,
            "site": urlparse(source_url).netloc,
        },
        "teams": {
            side: {
                "team_id": team.team_id,
                "name": team.name,
                # schema 1.8.0. Unlike person_id, this needs no corpus-level
                # evidence: franchise_id is a pure function of the team name,
                # which is right here in the file. So it is always populated
                # and can never drift out of sync with the registry.
                "franchise_id": team_map.mint_franchise_id(team.name),
            }
            for side, team in (("home", player_table.home), ("away", player_table.away))
        },
        "players": players,
        "linescore": linescore,
        "box": box,
        "lineups": lineups,
        "events": events,
        "unparsed": unparsed,
        "inferred": inferred,
        "meta": {
            "parser_version": PARSER_VERSION,
            "source_url": source_url,
            "source_sha256": sha256_hex(html),
            "fetched_at": fetched_at,
            "parsed_at": parsed_at_iso,
            "derived_replayer_version": DERIVED_REPLAYER_VERSION_PLACEHOLDER,
            "parse": {
                "events_count": len(events),
                "unparsed_count": len(unparsed),
                "inferred_count": len(inferred),
                "replayable": False,
                "warnings": [],
            },
        },
    }
