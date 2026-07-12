"""identity — the per-game player identity table + name-resolution surface.

Builds, from the boxscore HTML, a stable file-local roster for each side (who
the players are: id, display name, last name, team, positions seen this
game) and exposes ``resolve(last_name, side)`` so the parser (a later gate)
can join a bare PBP last name to a ``player_id``.

Protected intent: the event->player join must be TOTAL and honest. Every
player gets a stable id; a bare last name that cannot be uniquely resolved on
its side returns ``resolved=False`` -- never a guess (the caller routes that
line to ``unparsed[]``).

IDENTITY ONLY. This module reads structural facts about *who* the players
are (names, ids, positions) from the boxscore's per-team "Batters" lineup
table. It never reads box STAT numbers (AB/R/H/RBI/...) or the linescore --
those are independently owned by the parser and replayer oracle (spec D2
independence), matching ``html_struct``'s own restraint.

Uses only ``bc_pipeline.html_struct``'s generic DOM helpers; stdlib only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .html_struct import Node, find_all, find_all_by_class, text_of

_HOME_PLAYER_HREF_RE = re.compile(r"players\?id=([a-z0-9]{16})")
_TEAM_HREF_RE = re.compile(r"teams\?id=([a-z0-9]{16})")

# Surname suffixes that ride along after a player's true surname token in
# display names (e.g. "Patrick Roche Jr."). PBP narrative uses the surname,
# not the suffix, as its join token -- so the suffix is stripped when
# deriving ``last_name``, while ``name`` keeps the verbatim display form.
_SUFFIX_TOKENS = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

AWAY_SYN_TEAM_ID = "syn:team:away"


def _normalize_ws(s: str) -> str:
    """Collapse all whitespace runs to single spaces and strip the ends.

    ``html_struct.text_of`` deliberately preserves internal whitespace (it is
    a generic seam); display names and team names are this module's own
    normalization decision to make, same as any other caller of that seam.
    """
    return " ".join(s.split())


def _derive_last_name(display_name: str) -> str:
    """Return the surname token PBP narrative uses.

    Strips a trailing suffix token (Jr., Sr., II, III, IV, V) when present
    and takes the token before it; otherwise the final whitespace-separated
    token is the surname.
    """
    tokens = display_name.split()
    if len(tokens) >= 2 and tokens[-1].strip(".").lower() in {
        t.strip(".") for t in _SUFFIX_TOKENS
    }:
        return tokens[-2]
    return tokens[-1] if tokens else display_name


@dataclass
class PlayerEntry:
    """One player_entry (schema shape): identity home for a single player."""

    player_id: str
    name: str
    last_name: str
    team_id: str
    positions: List[str] = field(default_factory=list)
    bats_side: Optional[str] = None


@dataclass
class TeamIdentity:
    """One side's roster: team_id/display name + players in stable table order."""

    team_id: str
    name: str
    players: "Dict[str, PlayerEntry]" = field(default_factory=dict)


@dataclass
class PlayerTable:
    """Both sides' rosters for one game, plus the resolve() join surface."""

    home: TeamIdentity
    away: TeamIdentity

    def resolve(self, last_name: str, side: str) -> Tuple[Optional[str], bool]:
        """Resolve a bare PBP last name to a player_id on the given side.

        Returns ``(player_id, True)`` on a unique match, else ``(None,
        False)`` -- for no match AND for a same-last-name collision alike.
        Never guesses.
        """
        team = self.home if side == "home" else self.away
        matches = [pid for pid, p in team.players.items() if p.last_name == last_name]
        if len(matches) == 1:
            return matches[0], True
        return None, False


def _find_batters_tables(root: Node) -> List[Node]:
    """Return every ``<table>`` whose caption marks it as a "Batters" table.

    The boxscore renders one such table per side (the per-team lineup, which
    doubles as the batting box -- this module reads only its name/id/position
    cells, never its AB/R/H/... stat columns).
    """
    tables = []
    for table in find_all(root, "table"):
        captions = find_all(table, "caption")
        if not captions:
            continue
        if "Batters" in text_of(captions[0]):
            tables.append(table)
    return tables


def _team_id_and_name(table: Node) -> Tuple[str, str, bool]:
    """Return ``(team_id, team_name, is_home)`` from a Batters table's caption.

    The home team's caption wraps its name in ``<a ...teams?id=X>`` (a real
    Presto team id); the away team's caption is a bare ``<span
    class="team-name">`` with no id -- so this structural distinction is
    also how home vs. away is decided.
    """
    caption = find_all(table, "caption")[0]
    team_name_nodes = find_all_by_class(caption, "team-name")
    node = team_name_nodes[0]
    name = _normalize_ws(text_of(node))
    if node.tag == "a":
        href = node.attrs.get("href") or ""
        m = _TEAM_HREF_RE.search(href)
        if not m:
            raise ValueError(f"home team caption link has no parseable team id: {href!r}")
        return m.group(1), name, True
    return AWAY_SYN_TEAM_ID, name, False


def _data_rows(table: Node) -> List[Node]:
    """Return the table's per-player data rows.

    A data row has a ``<th scope="row">`` header (excluding the ``<thead>``
    column-header row, whose cells are ``<th scope="col">``) AND a
    ``player-name`` node inside it -- which excludes the trailing "Totals"
    summary row (a ``<th scope="row">`` with no player identity at all).
    """
    rows = []
    for tr in find_all(table, "tr"):
        headers = [th for th in find_all(tr, "th") if th.attrs.get("scope") == "row"]
        if headers and find_all_by_class(headers[0], "player-name"):
            rows.append(tr)
    return rows


def _row_identity(row: Node) -> Tuple[Optional[str], str, Optional[str]]:
    """Return ``(source_player_id_or_None, display_name, position_or_None)``
    for one data row.

    ``source_player_id`` is the 16-char Presto id parsed from the row's
    ``<a href=...players?id=X>`` link when present (home team); ``None`` when
    the row instead renders a bare ``<span class="player-name">`` (away
    team, no source id -- gets a synthetic id assigned by the caller).
    """
    row_th = [th for th in find_all(row, "th") if th.attrs.get("scope") == "row"][0]
    name_nodes = find_all_by_class(row_th, "player-name")
    name_node = name_nodes[0]
    name = _normalize_ws(text_of(name_node))

    position_nodes = find_all_by_class(row_th, "position")
    position = _normalize_ws(text_of(position_nodes[0])).lower() if position_nodes else None

    source_id = None
    if name_node.tag == "a":
        href = name_node.attrs.get("href") or ""
        m = _HOME_PLAYER_HREF_RE.search(href)
        source_id = m.group(1) if m else None

    return source_id, name, position


def _build_team_identity(table: Node) -> TeamIdentity:
    team_id, team_name, is_home = _team_id_and_name(table)
    team = TeamIdentity(team_id=team_id, name=team_name)

    syn_counter = 0
    # name -> already-assigned synthetic id, for the away side's dedupe.
    away_ids_by_name: Dict[str, str] = {}

    for row in _data_rows(table):
        source_id, name, position = _row_identity(row)

        if is_home:
            if source_id is None:
                raise ValueError(
                    f"home Batters row for {name!r} has no players?id=... link; "
                    "identity must not invent ids for players missing a source id"
                )
            player_id = source_id
        else:
            if name in away_ids_by_name:
                player_id = away_ids_by_name[name]
            else:
                syn_counter += 1
                player_id = f"syn:away:{syn_counter}"
                away_ids_by_name[name] = player_id

        entry = team.players.get(player_id)
        if entry is None:
            team.players[player_id] = PlayerEntry(
                player_id=player_id,
                name=name,
                last_name=_derive_last_name(name),
                team_id=team_id,
                positions=[position] if position else [],
            )
        elif position and position not in entry.positions:
            entry.positions.append(position)

    return team


def build_player_table(root: Node) -> PlayerTable:
    """Build the per-game identity table for both sides from the boxscore DOM.

    Locates each side's "Batters" lineup table (home vs. away distinguished
    structurally by their caption's team-name link vs. bare span), walks its
    rows in document order, and assigns each player a stable id: the real
    16-char Presto id for the home side, or a deterministic ``syn:away:<n>``
    (numbered by first-appearance row order) for the away side, which has no
    source id.
    """
    tables = _find_batters_tables(root)
    home_team: Optional[TeamIdentity] = None
    away_team: Optional[TeamIdentity] = None

    for table in tables:
        team = _build_team_identity(table)
        if team.team_id == AWAY_SYN_TEAM_ID:
            if away_team is None:
                away_team = team
        else:
            if home_team is None:
                home_team = team

    if home_team is None or away_team is None:
        raise ValueError("could not locate both home and away Batters tables in boxscore")

    return PlayerTable(home=home_team, away=away_team)
