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
table. It never reads box STAT numbers (AB/R/H/RBI/...) -- those are
independently owned by the parser and replayer oracle (spec D2 independence),
matching ``html_struct``'s own restraint. It reads ONE thing from the
linescore: the team-name ROW ORDER (away first, home second), the authoritative
home/away signal -- never the linescore's R/H/E stat numbers, which stay the
parser/replayer domain.

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


def _caption_team_name(table: Node) -> str:
    """Return the normalized team-name text from a Batters table's caption."""
    caption = find_all(table, "caption")[0]
    return _normalize_ws(text_of(find_all_by_class(caption, "team-name")[0]))


def _team_id_and_name(table: Node, side: str) -> Tuple[str, str]:
    """Return ``(team_id, team_name)`` from a Batters table's caption.

    ``team_id`` is the real 16-char Presto id when the caption's team-name is
    an ``<a ...teams?id=X>`` link -- otherwise a file-local ``syn:team:<side>``.
    A league-site page links BOTH teams' captions; a team-site page links only
    the host's -- so the link-vs-span shape is NOT a reliable home/away signal.
    Home vs. away is decided by the linescore row order (see
    ``_linescore_away_home_names``); ``side`` is passed in already resolved.
    """
    caption = find_all(table, "caption")[0]
    node = find_all_by_class(caption, "team-name")[0]
    name = _normalize_ws(text_of(node))
    if node.tag == "a":
        m = _TEAM_HREF_RE.search(node.attrs.get("href") or "")
        if m:
            return m.group(1), name
    return f"syn:team:{side}", name


def _is_int_text(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    return (s[1:] if s[0] == "-" else s).isdigit()


def _linescore_away_home_names(root: Node) -> Tuple[str, str]:
    """Return ``(away_name, home_name)`` from the linescore table's row order.

    The linescore lists the AWAY team's row first and the HOME team's row
    second -- the stable, template-generated scoreboard convention. This is the
    authoritative home/away signal, independent of the batting-table caption's
    link-vs-span shape (which is unreliable: league-site pages link both). Only
    the leading team-name cell and the row ORDER are read here; the R/H/E stat
    numbers stay the parser's / replayer-oracle's independent domain (spec D2).
    Fails loudly (rather than guessing) when the linescore is missing or does
    not yield two team rows.
    """
    divs = find_all_by_class(root, "linescore")
    if not divs:
        raise ValueError("no element with class 'linescore'; cannot resolve home/away")
    table = find_all(divs[0], "table")[0]
    team_names: List[str] = []
    for row in find_all(table, "tr"):
        cells = [
            _normalize_ws(text_of(c))
            for c in row.children
            if isinstance(c, Node) and c.tag in ("td", "th")
        ]
        if not cells:
            continue
        name, rest = cells[0], cells[1:]
        # A team data row: a non-empty leading team name and a trailing R/H/E
        # triple of integers. The inning-number header row (no leading name)
        # is skipped.
        if name and len(rest) >= 3 and all(_is_int_text(x) for x in rest[-3:]):
            team_names.append(name)
    if len(team_names) < 2:
        raise ValueError(
            f"linescore yielded {len(team_names)} team rows, expected >= 2 "
            "(away first, home second); cannot resolve home/away"
        )
    return team_names[0], team_names[1]


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


def _build_team_identity(table: Node, side: str) -> TeamIdentity:
    team_id, team_name = _team_id_and_name(table, side)
    team = TeamIdentity(team_id=team_id, name=team_name)

    syn_counter = 0
    # display name -> already-assigned synthetic id, so an unlinked player who
    # appears in more than one row keeps one stable id (league pages link both
    # teams, so this dedupe only fires on a genuinely id-less row).
    syn_ids_by_name: Dict[str, str] = {}

    for row in _data_rows(table):
        source_id, name, position = _row_identity(row)

        # Per-player id assignment: the real 16-char id when the row links one
        # (league pages link BOTH teams), else a deterministic syn:<side>:<n>
        # (team-site pages don't link the opponent). Never invent a real id.
        if source_id is not None:
            player_id = source_id
        elif name in syn_ids_by_name:
            player_id = syn_ids_by_name[name]
        else:
            syn_counter += 1
            player_id = f"syn:{side}:{syn_counter}"
            syn_ids_by_name[name] = player_id

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

    Home vs. away is resolved from the linescore's row order (away first, home
    second -- see ``_linescore_away_home_names``), NOT from the caption's
    team-name link-vs-span shape, which is unreliable (league-site pages link
    both teams). Each side's "Batters" table is matched to that side by team
    name, and each player gets a stable id: the real 16-char Presto id when the
    row links one, else a deterministic ``syn:<side>:<n>`` (numbered by
    first-appearance row order). Fails loudly (never guesses) when a Batters
    table matches neither side, both tables match one side, or there are not
    exactly two Batters tables.
    """
    away_name, home_name = _linescore_away_home_names(root)
    tables = _find_batters_tables(root)
    if len(tables) != 2:
        raise ValueError(
            f"expected exactly 2 Batters tables, found {len(tables)}"
        )

    by_side: Dict[str, TeamIdentity] = {}
    for table in tables:
        caption_name = _caption_team_name(table)
        if caption_name == away_name:
            side = "away"
        elif caption_name == home_name:
            side = "home"
        else:
            raise ValueError(
                f"Batters table team {caption_name!r} matches neither the "
                f"linescore away team {away_name!r} nor home team {home_name!r}"
            )
        if side in by_side:
            raise ValueError(
                f"two Batters tables both resolved to the {side!r} side "
                f"({away_name!r}/{home_name!r}); ambiguous, refusing to guess"
            )
        by_side[side] = _build_team_identity(table, side)

    if "home" not in by_side or "away" not in by_side:
        raise ValueError("could not resolve both home and away Batters tables in boxscore")

    return PlayerTable(home=by_side["home"], away=by_side["away"])
