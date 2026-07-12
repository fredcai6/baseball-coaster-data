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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from . import identity
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

PARSER_VERSION = "0.2.0"
SCHEMA_VERSION = "1.2.0"
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
}
_LINE_LOCATION_RE = re.compile(r"^the (lf|rf) line$")


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


def build_events(
    lines: List[PbpLine], player_table: identity.PlayerTable
) -> Tuple[List[dict], List[dict], Dict[str, List[dict]]]:
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

    def _resolve_runner(
        rm: RunnerMovement,
        batting_side: str,
        modifiers: Optional[List[str]],
    ) -> Optional[dict]:
        last = _last_name_token(rm.name_token)
        pid, ok = player_table.resolve(last, batting_side)
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
        # Update the live cross-event base occupancy: vacate the base this
        # clause left (only if the runner still holds it), and occupy the
        # destination (unless out, scored, or off the bases).
        if from_base in base_occ and base_occ[from_base] == pid:
            del base_occ[from_base]
        if not rm.out and to_base not in (-1, 4):
            base_occ[to_base] = pid
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
        record for it; cross-event occupancy is unaffected (base_occ was
        already folded hop-by-hop inside _resolve_runner)."""
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
                side = batting_side
                team_id = batting_team_id
            else:
                side = fielding_side
                team_id = fielding_team_id
            in_last = _last_name_token(cg.substitution.player_in)
            in_pid, in_ok = player_table.resolve(in_last, side)
            if cg.substitution.player_out is None:
                # Bare DH-slot-entry (schema 1.2.0, issue #30 g2b): the line
                # names only the incoming player. Never guess an outgoing
                # player from a line that does not name one -- emit
                # player_out: None directly, mirroring the p.count is None
                # guard above, instead of calling _last_name_token/resolve()
                # on a value that was never there.
                out_pid, out_ok = None, True
            else:
                out_last = _last_name_token(cg.substitution.player_out)
                out_pid, out_ok = player_table.resolve(out_last, side)
            if not out_ok or not in_ok:
                _unparsed(
                    line,
                    f"substitution names did not resolve uniquely on the "
                    f"{side} side: out={cg.substitution.player_out!r} "
                    f"in={cg.substitution.player_in!r}",
                )
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
                    "runners": _merge_same_runner(runners),
                }
            )
            seq += 1
            continue

        # plate_appearance
        p = cg.primary
        batter_last = _last_name_token(p.name_token)
        batter_pid, batter_ok = player_table.resolve(batter_last, batting_side)
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

        # Apply the batter's own base-occupancy update (runner clauses
        # already updated themselves inside _resolve_runner).
        if not out_flag and to_base not in (-1, 4):
            base_occ[to_base] = batter_pid

        runner_records = _merge_same_runner(runner_records)
        outs_recorded = (1 if out_flag else 0) + sum(
            1 for r in runner_records[1:] if r["out"]
        )

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

    return events, unparsed, subs_by_team


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
                lines.append(
                    PbpLine(
                        inning=inning,
                        half=half,
                        line_index=idx,
                        text=text_of(td),
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


def _build_lineups(
    player_table: identity.PlayerTable, subs_by_team: Dict[str, List[dict]]
) -> Dict[str, dict]:
    lineups: Dict[str, dict] = {}
    for team in (player_table.home, player_table.away):
        batting_ids = list(team.players.keys())[:9]
        lineups[team.team_id] = {
            "batting_order": [
                {"slot": i + 1, "player_id": pid} for i, pid in enumerate(batting_ids)
            ],
            "substitutions": subs_by_team.get(team.team_id, []),
        }
    return lineups


def _players_table(player_table: identity.PlayerTable) -> Dict[str, dict]:
    players: Dict[str, dict] = {}
    for team in (player_table.home, player_table.away):
        for pid, entry in team.players.items():
            players[pid] = {
                "player_id": entry.player_id,
                "name": entry.name,
                "last_name": entry.last_name,
                "team_id": entry.team_id,
                "bats_side": entry.bats_side,
                "positions": list(entry.positions),
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
) -> dict:
    """Parse raw boxscore HTML into a full schema-valid ``final`` game dict.

    Raises ``NonFinalPageError`` if the page has no PBP panes (the negative-
    path contract) -- never fabricates a `final` file from such a page.
    """
    root = parse_html(html)
    if not has_pbp_panes(root):
        raise NonFinalPageError(
            "page has no PBP panes (id='pbp-inning-N'); not a final boxscore"
        )

    game_id = _extract_game_id(source_url)
    date_iso = _extract_date_iso(root)
    season = int(date_iso[:4])

    player_table = identity.build_player_table(root)
    lines = _iter_halves(root)
    events, unparsed, subs_by_team = build_events(lines, player_table)

    linescore = _parse_linescore(root, player_table)
    box = {
        "batting": _parse_box_batting(root, player_table),
        "pitching": _parse_box_pitching(root, player_table),
    }
    lineups = _build_lineups(player_table, subs_by_team)
    players = _players_table(player_table)

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
            "home": {"team_id": player_table.home.team_id, "name": player_table.home.name},
            "away": {"team_id": player_table.away.team_id, "name": player_table.away.name},
        },
        "players": players,
        "linescore": linescore,
        "box": box,
        "lineups": lineups,
        "events": events,
        "unparsed": unparsed,
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
                "replayable": False,
                "warnings": [],
            },
        },
    }
