"""Replay a parsed game to INDEPENDENTLY validate it and stamp the
``_derived`` base-out cache (spec D2 / issue #19 gate g6).

Protected intent: this module is an INDEPENDENT check on ``parse.py``. It
re-derives the linescore/box oracle from the raw HTML with its OWN
table-interpretation code -- it MUST NOT import ``parse`` and MUST NOT reuse
any parser-derived numbers. It may use ONLY ``html_struct``'s GENERIC DOM
helpers (``find_by_id``/``find_all_by_class``/``iter_rows``/``cell_texts``/
``text_of`` and friends); a shared table-reader would let one bug fool both
sides, which defeats the whole point of this gate. A failed check FLAGS the
game (``meta.parse.replayable = False`` + a warning) -- it never raises past
the caller and never silently passes.

Three independent pieces:

* ``extract_oracle(html, game)`` -- re-reads the linescore + box-batting
  tables from raw HTML using only generic ``html_struct`` helpers, keyed by
  the ``player_id``/``team_id`` identity already established in ``game``
  (identity resolution is not the linescore/box *number* the gate protects;
  re-deriving it a second time would require importing ``identity``, which
  is not this gate's job).
* ``fold_base_out(events)`` -- a PURE function (no HTML, no I/O) that folds
  the schema's asserted ``runners[].from``/``to`` primitives forward into
  the regenerable ``_derived`` base-out cache.
* Five ``check_*`` functions, each ``(game, oracle) -> CheckResult`` and
  independently testable, plus ``replay_game`` which wires the oracle, the
  fold, and the five checks together and stamps the result onto a *copy* of
  the input game.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .html_struct import Node, find_all, find_all_by_class, parse_html, text_of

REPLAYER_VERSION = "0.1.0"

_BASE_INDEXES = (1, 2, 3)


# ---------------------------------------------------------------------------
# Independent oracle: raw HTML -> {linescore, box} using ONLY generic
# html_struct helpers + this module's OWN column interpretation. This is a
# deliberately SEPARATE reading of the tables from parse.py's -- see the
# module docstring. It is driven by the table's OWN header-row text (rather
# than a hardcoded column-position assumption) so a reordered column would
# be caught here rather than silently mis-mapped the same way twice.
# ---------------------------------------------------------------------------


def _header_labels(table: Node) -> List[str]:
    """The column labels of ``table``'s header row, in document order."""
    header_cells: List[str] = []
    for node in find_all(table, "tr"):
        cols = [c for c in find_all(node, "th") if c.attrs.get("scope") == "col"]
        if cols:
            header_cells = [" ".join(text_of(c).split()) for c in cols]
            break
    return header_cells


def _row_data_cells(row: Node) -> List[str]:
    """Verbatim text of every direct `<td>` child of ``row``, in order."""
    return [
        text_of(c)
        for c in row.children
        if isinstance(c, Node) and c.tag == "td"
    ]


def _extract_linescore_oracle(root: Node, home_name: str, away_name: str) -> dict:
    divs = find_all_by_class(root, "linescore")
    if not divs:
        raise ValueError("oracle: no element with class 'linescore' found")
    tables = find_all(divs[0], "table")
    if not tables:
        raise ValueError("oracle: linescore div has no table")
    table = tables[0]

    innings: Dict[str, List[Optional[int]]] = {}
    totals: Dict[str, dict] = {}
    for row in find_all(table, "tr"):
        headers = [th for th in find_all(row, "th") if th.attrs.get("scope") == "row"]
        if not headers:
            continue
        name = " ".join(text_of(headers[0]).split())
        if name == home_name:
            side = "home"
        elif name == away_name:
            side = "away"
        else:
            continue
        # Own column interpretation: cells carrying a "score" class token are
        # the per-inning + R/H/E cells (in that order); the last 3 are the
        # totals regardless of how many innings were played (extra innings).
        score_cells = [
            c
            for c in row.children
            if isinstance(c, Node)
            and c.tag == "td"
            and "score" in (c.attrs.get("class") or "").split()
        ]
        texts = [text_of(c) for c in score_cells]
        if len(texts) < 3:
            raise ValueError(f"oracle: linescore row for {name!r} has too few score cells")
        inning_texts, rhe_texts = texts[:-3], texts[-3:]
        innings[side] = [None if t.strip().upper() == "X" else int(t) for t in inning_texts]
        r, h, e = (int(t) for t in rhe_texts)
        totals[side] = {"R": r, "H": h, "E": e}

    if "home" not in innings or "away" not in innings:
        raise ValueError("oracle: linescore did not resolve both home and away rows")

    return {
        "innings": {"away": innings["away"], "home": innings["home"]},
        "totals": {"away": totals["away"], "home": totals["home"]},
    }


def _tables_with_offscreen_label(root: Node, label: str) -> List[Node]:
    """Tables whose `<caption>` contains an `.offscreen` span with exact
    text ``label`` (e.g. "Batters"). Own traversal, independent of parse.py's
    caption-substring marker search."""
    out = []
    for table in find_all(root, "table"):
        captions = find_all(table, "caption")
        if not captions:
            continue
        offscreen = find_all_by_class(captions[0], "offscreen")
        if any(" ".join(text_of(n).split()) == label for n in offscreen):
            out.append(table)
    return out


def _caption_team_name(table: Node) -> Optional[str]:
    captions = find_all(table, "caption")
    if not captions:
        return None
    names = find_all_by_class(captions[0], "team-name")
    if not names:
        return None
    return " ".join(text_of(names[0]).split())


def _extract_box_batting_oracle(
    root: Node,
    name_to_team_id: Dict[str, str],
    pid_by_team_and_name: Dict[str, Dict[str, str]],
) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for table in _tables_with_offscreen_label(root, "Batters"):
        team_name = _caption_team_name(table)
        team_id = name_to_team_id.get(team_name) if team_name else None
        if team_id is None:
            continue
        # The header row's first `<th scope="col">` labels the row-header
        # column itself (e.g. "Hitters") -- it has no corresponding `<td>`
        # data cell (the row header is a `<th>`, not a `<td>`), so it must
        # be dropped before zipping labels against `_row_data_cells`.
        labels = _header_labels(table)[1:]
        pid_by_name = pid_by_team_and_name.get(team_id, {})
        lines: List[dict] = []
        for row in find_all(table, "tr"):
            headers = [th for th in find_all(row, "th") if th.attrs.get("scope") == "row"]
            if not headers:
                continue
            names = find_all_by_class(headers[0], "player-name")
            if not names:
                continue  # e.g. the "Totals" row has no player-name span
            name = " ".join(text_of(names[0]).split())
            pid = pid_by_name.get(name)
            if pid is None:
                continue
            positions = find_all_by_class(headers[0], "position")
            pos = " ".join(text_of(positions[0]).split()).lower() if positions else ""
            cells = _row_data_cells(row)
            by_label = dict(zip(labels, cells))
            try:
                line = {
                    "player_id": pid,
                    "pos": pos,
                    "AB": int(by_label["AB"]),
                    "R": int(by_label["R"]),
                    "H": int(by_label["H"]),
                    "RBI": int(by_label["RBI"]),
                    "BB": int(by_label["BB"]),
                    "SO": int(by_label["SO"]),
                    "LOB": int(by_label["LOB"]),
                    "AVG": by_label["AVG"].strip(),
                }
            except KeyError as exc:
                raise ValueError(f"oracle: box batting header missing column {exc}") from exc
            lines.append(line)
        out[team_id] = lines
    return out


def extract_oracle(html: str, game: dict) -> dict:
    """Independently re-derive ``{linescore, box}`` from raw HTML.

    ``game`` supplies ONLY the identity mapping already established upstream
    (team names/ids, player names/ids) so oracle rows can be keyed the same
    way the schema keys them -- the linescore/box NUMBERS themselves are
    read fresh from the HTML by this module's own code, never taken from
    ``game``.
    """
    root = parse_html(html)
    home = game["teams"]["home"]
    away = game["teams"]["away"]
    name_to_team_id = {home["name"]: home["team_id"], away["name"]: away["team_id"]}

    pid_by_team_and_name: Dict[str, Dict[str, str]] = {home["team_id"]: {}, away["team_id"]: {}}
    for pid, entry in game["players"].items():
        pid_by_team_and_name.setdefault(entry["team_id"], {})[entry["name"]] = pid

    linescore = _extract_linescore_oracle(root, home["name"], away["name"])
    box_batting = _extract_box_batting_oracle(root, name_to_team_id, pid_by_team_and_name)
    return {"linescore": linescore, "box": {"batting": box_batting}}


# ---------------------------------------------------------------------------
# Pure fold: events[] (asserted runner primitives) -> per-event _derived.
# No HTML, no Node types, no I/O below this line.
# ---------------------------------------------------------------------------

_FOLDABLE_KINDS = ("plate_appearance", "runner_event")


def _apply_runners(bases_before: List[bool], runners: List[dict]) -> List[bool]:
    """Fold one event's ``runners[]`` onto ``bases_before``, returning the
    resulting base occupancy.

    Two-pass clear-then-set over the SAME pre-event snapshot (mirroring
    parse.py's own `line_snapshot` protection against one clause clobbering
    another's read of the same line -- re-derived independently here, no
    code shared). A single real narrative occasionally emits TWO runner
    records for the SAME player in one event (e.g. "advanced to third,
    scored on an error" -- the mid-play base and the final resting spot both
    get asserted as separate `runners[]` entries with the same pre-event
    `from`); the LAST such record for a given player is authoritative for
    where they end up, so the set-pass tracks one pending destination per
    player rather than applying every record independently.
    """
    new_bases = list(bases_before)
    for r in runners:
        if r["from"] in _BASE_INDEXES:
            new_bases[r["from"] - 1] = False

    last_to_by_player: Dict[str, int] = {}
    for r in runners:
        if not r["out"]:
            last_to_by_player[r["player_id"]] = r["to"]
    for to in last_to_by_player.values():
        if to in _BASE_INDEXES:
            new_bases[to - 1] = True
    return new_bases


def fold_base_out(events: List[dict]) -> List[dict]:
    """PURE. Folds the asserted ``runners[].from``/``to`` primitives of every
    ``plate_appearance``/``runner_event`` in ``events`` (already sorted by
    ``seq``) forward into a list of ``_derived`` dicts, one per such event,
    in the same relative order.

    Base occupancy resets at the start of each (inning, half). Scores are
    cumulative across the whole game. Within one event, all runner clauses
    read the SAME pre-event base snapshot (a two-pass clear-then-set), so
    multiple runners named on one play never clobber each other -- mirroring
    the same real-world simultaneity the parser's own `line_snapshot`
    independently protects against (see parse.py's `build_events`
    docstring), re-derived here with none of its code shared.
    """
    derived: List[dict] = []
    bases = [False, False, False]
    outs = 0
    away_score = 0
    home_score = 0
    cur_half: Optional[Tuple[int, str]] = None
    pa_counter: Dict[str, int] = {}

    for ev in events:
        if ev.get("kind") not in _FOLDABLE_KINDS:
            continue
        half_key = (ev["inning"], ev["half"])
        if half_key != cur_half:
            bases = [False, False, False]
            outs = 0
            cur_half = half_key

        bases_before = list(bases)
        outs_before = outs
        base_out_state = "".join("1" if b else "0" for b in bases_before) + "|" + str(outs_before)

        runners = ev.get("runners", [])
        new_bases = _apply_runners(bases_before, runners)

        outs_added = sum(1 for r in runners if r["out"])
        runs_on_play = sum(1 for r in runners if r["scored"])

        d = {
            "outs_before": outs_before,
            "bases_before": bases_before,
            "base_out_state": base_out_state,
            "away_score_before": away_score,
            "home_score_before": home_score,
            "outs_after": outs_before + outs_added,
            "bases_after": new_bases,
            "runs_on_play": runs_on_play,
        }
        if ev["kind"] == "plate_appearance":
            batter_pid = ev["batter"]["player_id"]
            pa_counter[batter_pid] = pa_counter.get(batter_pid, 0) + 1
            d["pa_number_of_batter"] = pa_counter[batter_pid]

        if runs_on_play:
            if ev["half"] == "top":
                away_score += runs_on_play
            else:
                home_score += runs_on_play

        bases = new_bases
        outs = outs_before + outs_added
        derived.append(d)

    return derived


# ---------------------------------------------------------------------------
# Five independent checks.
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    ok: bool
    warnings: List[str] = field(default_factory=list)


def _foldable_with_derived(game: dict) -> List[Tuple[dict, dict]]:
    """Zip each foldable event with its freshly-folded ``_derived`` (never
    trusts any ``_derived`` already present on the input -- always refolds)."""
    foldable = [e for e in game["events"] if e.get("kind") in _FOLDABLE_KINDS]
    derived = fold_base_out(game["events"])
    return list(zip(foldable, derived))


def _group_by_half(pairs: List[Tuple[dict, dict]]) -> Dict[Tuple[int, str], List[Tuple[dict, dict]]]:
    groups: Dict[Tuple[int, str], List[Tuple[dict, dict]]] = {}
    for ev, d in pairs:
        groups.setdefault((ev["inning"], ev["half"]), []).append((ev, d))
    return groups


def check_linescore(game: dict, oracle: dict) -> CheckResult:
    """Recompute per-half-inning runs from events; compare to the oracle's
    linescore innings arrays, and the game's final cumulative score to the
    oracle's totals."""
    warnings: List[str] = []
    pairs = _foldable_with_derived(game)
    groups = _group_by_half(pairs)

    max_inning = max((k[0] for k in groups), default=0)
    max_inning = max(max_inning, len(oracle["linescore"]["innings"]["away"]), len(oracle["linescore"]["innings"]["home"]))

    computed = {"away": [], "home": []}
    for side, half in (("away", "top"), ("home", "bottom")):
        for inning in range(1, max_inning + 1):
            key = (inning, half)
            if key not in groups:
                computed[side].append(None)
                continue
            runs = sum(d["runs_on_play"] for _, d in groups[key])
            computed[side].append(runs)

    oracle_innings = oracle["linescore"]["innings"]
    for side in ("away", "home"):
        oi = oracle_innings[side]
        ci = computed[side]
        for i in range(max(len(oi), len(ci))):
            o_val = oi[i] if i < len(oi) else None
            c_val = ci[i] if i < len(ci) else None
            if o_val is None or c_val is None:
                # Unbatted/absent half on either side is a legal non-mismatch
                # as long as the OTHER side agrees it's absent-or-null too;
                # a genuine numeric disagreement is caught by the else branch.
                if o_val is not None and c_val is None:
                    warnings.append(
                        f"linescore: {side} inning {i + 1} has no folded events but "
                        f"oracle expects {o_val}"
                    )
                continue
            if o_val != c_val:
                warnings.append(
                    f"linescore: {side} inning {i + 1} computed {c_val} != oracle {o_val}"
                )

    final_away = groups and sum(
        d["runs_on_play"] for k, evd in groups.items() if k[1] == "top" for _, d in evd
    )
    final_home = groups and sum(
        d["runs_on_play"] for k, evd in groups.items() if k[1] == "bottom" for _, d in evd
    )
    final_away = final_away or 0
    final_home = final_home or 0
    if final_away != oracle["linescore"]["totals"]["away"]["R"]:
        warnings.append(
            f"linescore: computed away final R {final_away} != oracle total "
            f"{oracle['linescore']['totals']['away']['R']}"
        )
    if final_home != oracle["linescore"]["totals"]["home"]["R"]:
        warnings.append(
            f"linescore: computed home final R {final_home} != oracle total "
            f"{oracle['linescore']['totals']['home']['R']}"
        )

    return CheckResult(ok=not warnings, warnings=warnings)


def check_outs_per_half(game: dict, oracle: dict) -> CheckResult:
    """Each half-inning's folded outs must sum to 3, except a legal walk-off
    (game ends on a winning run before the 3rd out) or an unbatted half
    (linescore innings entry is null -- e.g. home doesn't bat the bottom 9th
    when already leading)."""
    warnings: List[str] = []
    pairs = _foldable_with_derived(game)
    groups = _group_by_half(pairs)
    if not groups:
        return CheckResult(ok=True, warnings=[])

    last_half_key = max(groups.keys())

    for key, evd in groups.items():
        total_outs = sum(d["outs_after"] - d["outs_before"] for _, d in evd)
        if total_outs == 3:
            continue

        inning, half = key
        is_last_half = key == last_half_key
        # Unbatted-half exception: linescore records null for this inning
        # on this side.
        side = "away" if half == "top" else "home"
        innings_arr = oracle["linescore"]["innings"][side]
        unbatted = (
            inning - 1 < len(innings_arr) and innings_arr[inning - 1] is None
        )
        # Walk-off exception: bottom half, game-ending, and the batting
        # (home) side's score after the last play exceeds the away side's
        # final total -- the winning run ended the game before 3 outs.
        last_ev, last_d = evd[-1]
        walkoff = (
            half == "bottom"
            and is_last_half
            and (last_d["home_score_before"] + last_d["runs_on_play"])
            > oracle["linescore"]["totals"]["away"]["R"]
        )
        if unbatted or walkoff:
            continue
        warnings.append(
            f"outs_per_half: inning {inning} {half} totals {total_outs} outs "
            "(expected 3, no walk-off/unbatted exception applies)"
        )

    return CheckResult(ok=not warnings, warnings=warnings)


def check_lob(game: dict, oracle: dict) -> CheckResult:
    """Runners left on base, folded from the last play of each half, must
    reconcile with that half's own ``inning_summary`` LOB (the oracle
    already agrees the linescore/box are consistent with these totals;
    this check is purely about the fold agreeing with the box's own
    narrative-embedded summary line for that half)."""
    warnings: List[str] = []
    pairs = _foldable_with_derived(game)
    groups = _group_by_half(pairs)

    for ev in game["events"]:
        if ev.get("kind") != "inning_summary":
            continue
        key = (ev["inning"], ev["half"])
        evd = groups.get(key)
        if not evd:
            continue  # no folded plays this half (shouldn't happen with a summary)
        _, last_d = evd[-1]
        folded_lob = sum(1 for b in last_d["bases_after"] if b)
        summary_lob = ev["summary"]["LOB"]
        if folded_lob != summary_lob:
            warnings.append(
                f"lob: inning {ev['inning']} {ev['half']} folded LOB {folded_lob} != "
                f"inning_summary LOB {summary_lob}"
            )

    return CheckResult(ok=not warnings, warnings=warnings)


def check_pa_counts(game: dict, oracle: dict) -> CheckResult:
    """Per-batter PA count from events must reconcile with what the box
    batting line implies. Formula (documented, since the box schema has no
    HBP/SAC columns of its own): ``events_PA == box.AB + box.BB +
    hbp_events + sac_events``, where ``hbp_events``/``sac_events`` are
    counted directly from this batter's own events (outcome type
    ``hit_by_pitch`` / ``sacrifice``) since the box has no dedicated column
    for them."""
    warnings: List[str] = []
    events_pa: Dict[str, int] = {}
    hbp: Dict[str, int] = {}
    sac: Dict[str, int] = {}
    for ev in game["events"]:
        if ev.get("kind") != "plate_appearance":
            continue
        pid = ev["batter"]["player_id"]
        events_pa[pid] = events_pa.get(pid, 0) + 1
        outcome_type = ev["outcome"]["type"]
        modifiers = ev["outcome"]["modifiers"]
        if outcome_type == "hit_by_pitch":
            hbp[pid] = hbp.get(pid, 0) + 1
        elif outcome_type == "sacrifice" or "SAC" in modifiers:
            # The grammar's closed outcome taxonomy expresses a sac bunt/fly
            # as the underlying batted-ball outcome (e.g. `flyout`) PLUS a
            # "SAC" modifier -- not as the generic `sacrifice` type -- so
            # both spellings must be checked.
            sac[pid] = sac.get(pid, 0) + 1

    for team_id, lines in oracle["box"]["batting"].items():
        for line in lines:
            pid = line["player_id"]
            if pid not in events_pa:
                continue  # box row with no PBP plate appearance in this game slice
            expected = line["AB"] + line["BB"] + hbp.get(pid, 0) + sac.get(pid, 0)
            actual = events_pa[pid]
            if actual != expected:
                warnings.append(
                    f"pa_counts: player {pid} events PA {actual} != box-implied "
                    f"PA {expected} (AB={line['AB']} BB={line['BB']} "
                    f"HBP={hbp.get(pid, 0)} SAC={sac.get(pid, 0)})"
                )

    return CheckResult(ok=not warnings, warnings=warnings)


def check_illegal_transitions(game: dict, oracle: dict) -> CheckResult:
    """The folded base-out sequence has no impossible transitions: outs
    never exceed 3 within a half, a runner never advances from a base that
    is not occupied at the base-out state frozen at event start, and a
    scored/putout runner's ``to`` is internally consistent.

    STRICT ``from``-occupancy: when a runner's ``from`` is a real base
    (1/2/3), that base MUST be occupied in ``bases_before`` (the snapshot
    taken at the START of the event, before any of this event's own
    movements). g5's parser now chains multi-clause same-runner advances
    into a single consistent ``from``/``to`` primitive, so a ``from`` that
    names an unoccupied base is a genuine illegal sequence (a runner
    "advancing from" a base they are not on), not a parser artifact -- there
    is no "previously asserted for this player" escape hatch.
    """
    warnings: List[str] = []
    bases = [False, False, False]
    outs = 0
    cur_half: Optional[Tuple[int, str]] = None

    for ev in game["events"]:
        if ev.get("kind") not in _FOLDABLE_KINDS:
            continue
        half_key = (ev["inning"], ev["half"])
        if half_key != cur_half:
            bases = [False, False, False]
            outs = 0
            cur_half = half_key

        bases_before = list(bases)
        runners = ev.get("runners", [])

        for r in runners:
            # Note: an out runner's `to` is NOT required to be -1 -- e.g. a
            # runner caught stealing (or forced) while attempting a named
            # base legitimately carries that base as `to` even though they
            # never occupy it (see parse.py's `_resolve_runner`: a named
            # destination is asserted whenever the narrative names one,
            # independent of whether the runner is out). Occupancy itself
            # is governed by `out`, not by what `to` says, so there is
            # nothing to validate about `to`'s value when `out` is true.
            if r["scored"] and r["to"] != 4:
                warnings.append(
                    f"illegal_transition: seq {ev.get('seq')} runner {r['player_id']} "
                    f"scored but to={r['to']} (expected 4)"
                )
            if r["out"] and r["scored"]:
                warnings.append(
                    f"illegal_transition: seq {ev.get('seq')} runner {r['player_id']} "
                    "is both out and scored"
                )
            if r["from"] in _BASE_INDEXES and not bases_before[r["from"] - 1]:
                warnings.append(
                    f"illegal_transition: seq {ev.get('seq')} runner {r['player_id']} "
                    f"advances from base {r['from']} which was not occupied at event start"
                )

        new_bases = _apply_runners(bases_before, runners)

        outs_after = outs + sum(1 for r in runners if r["out"])
        if outs_after > 3:
            warnings.append(
                f"illegal_transition: seq {ev.get('seq')} outs_after {outs_after} exceeds 3 "
                f"in inning {ev['inning']} {ev['half']}"
            )

        bases = new_bases
        outs = outs_after

    return CheckResult(ok=not warnings, warnings=warnings)


_CHECKS = (
    ("linescore", check_linescore),
    ("outs_per_half", check_outs_per_half),
    ("lob", check_lob),
    ("pa_counts", check_pa_counts),
    ("illegal_transitions", check_illegal_transitions),
)


def replay_game(game: dict, html: str) -> dict:
    """Independently validate ``game`` against oracles re-derived from
    ``html``, stamp the ``_derived`` base-out cache on every foldable event,
    and set ``meta.parse.replayable``/``meta.parse.warnings``/
    ``meta.derived_replayer_version``.

    Never mutates the input ``game``; returns a fresh copy. NEVER raises
    past the caller -- any internal failure (a check crashing, the oracle
    extraction failing) is caught and turned into ``replayable = False`` +
    a warning, exactly like a failed check. A failed check flags the game;
    it never silently passes.
    """
    out = copy.deepcopy(game)
    out.setdefault("meta", {})
    out["meta"]["derived_replayer_version"] = REPLAYER_VERSION
    out["meta"].setdefault("parse", {})
    existing_warnings = list(out["meta"]["parse"].get("warnings", []))

    try:
        oracle = extract_oracle(html, out)
        derived_list = fold_base_out(out["events"])
        di = 0
        for ev in out["events"]:
            if ev.get("kind") in _FOLDABLE_KINDS:
                ev["_derived"] = derived_list[di]
                di += 1

        new_warnings: List[str] = []
        all_ok = True
        for name, check_fn in _CHECKS:
            result = check_fn(out, oracle)
            if not result.ok:
                all_ok = False
                new_warnings.extend(f"[{name}] {w}" for w in result.warnings)

        out["meta"]["parse"]["replayable"] = all_ok
        out["meta"]["parse"]["warnings"] = existing_warnings + new_warnings
    except Exception as exc:  # noqa: BLE001 -- deliberate: never raise past the caller
        out["meta"]["parse"]["replayable"] = False
        out["meta"]["parse"]["warnings"] = existing_warnings + [
            f"replay failed to complete: {exc!r}"
        ]

    return out
