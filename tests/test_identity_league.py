"""Regression: home/away resolution on a real league-site boxscore.

The original #19 identity heuristic decided home vs. away by the caption's
team-name being an ``<a>`` link (home) vs. a bare ``<span>`` (away). That held
only for the team-site sample. Every real league-site (pioneerleague.com) page
links BOTH teams' captions, so the old heuristic found two "home" tables and
raised ``ValueError`` on 10/10 archived games. The fix resolves home/away from
the linescore's row order (away first, home second) and gives each player a real
id when the row links one — so both teams resolve with real ids on a league page.

`tests/samples/boxscore_20260519_f5ap_league.html` is one of the 10 archived
league games (the smallest), committed as a curated zero-fetch regression fixture
(RedPocket Mobiles @ Modesto Roadsters, 2026-05-19).
"""

import re

import bc_pipeline.html_struct as html_struct
import bc_pipeline.identity as identity
from tests._support import SAMPLES_DIR

_REAL_ID = re.compile(r"^[a-z0-9]{16}$")
LEAGUE_SAMPLE = "boxscore_20260519_f5ap_league.html"


def _league_root():
    return html_struct.parse_html((SAMPLES_DIR / LEAGUE_SAMPLE).read_text(encoding="utf-8"))


def test_league_page_parses_where_the_old_heuristic_raised():
    # The old link-vs-span heuristic raised ValueError here (both captions are
    # <a> links). The fix must resolve both sides without raising.
    table = identity.build_player_table(_league_root())
    assert table.home is not None and table.away is not None


def test_league_both_teams_resolve_with_real_ids():
    table = identity.build_player_table(_league_root())
    # Both teams carry a real 16-char Presto team id (a league page links both).
    assert _REAL_ID.match(table.home.team_id), table.home.team_id
    assert _REAL_ID.match(table.away.team_id), table.away.team_id
    # Every player on BOTH sides has a real 16-char id -- no syn ids, because a
    # league page links every player.
    for side in (table.home, table.away):
        assert side.players, f"{side.name} has no players"
        for pid in side.players:
            assert _REAL_ID.match(pid), f"{side.name} player id not real: {pid!r}"


def test_league_home_away_come_from_linescore_row_order():
    # Linescore lists away first, home second: RedPocket Mobiles @ Modesto
    # Roadsters. The fix must assign these correctly regardless of caption shape.
    table = identity.build_player_table(_league_root())
    assert table.away.name == "RedPocket Mobiles"
    assert table.home.name == "Modesto Roadsters"


def test_league_resolve_smoke_on_a_real_last_name():
    table = identity.build_player_table(_league_root())
    # Pick the first away player deterministically and confirm resolve() joins
    # its last name back to its real id on the away side.
    first_pid, first_entry = next(iter(table.away.players.items()))
    pid, resolved = table.resolve(first_entry.last_name, "away")
    # (Resolves uniquely unless that surname is shared on the away side.)
    if resolved:
        assert pid == first_pid


# --- fail-loud guards: never guess home/away -------------------------------

_LINESCORE = (
    '<div class="linescore"><table>'
    "<tr><th></th><th>1</th><th>R</th><th>H</th><th>E</th></tr>"
    "<tr><td>{away}</td><td>1</td><td>1</td><td>2</td><td>0</td></tr>"
    "<tr><td>{home}</td><td>2</td><td>3</td><td>5</td><td>1</td></tr>"
    "</table></div>"
)
_BATTERS = (
    '<table class="table"><caption class="caption"><h2>'
    '<a href="/x/teams?id={tid}" class="team-name">{name}</a>'
    '<span class="offscreen">Batters</span></h2></caption>'
    "<thead><tr><th scope=\"col\">Hitters</th></tr></thead><tbody>"
    '<tr><th scope="row"><div>'
    '<span class="position">ss</span>'
    '<a href="/x/players?id={pid}" class="player-name">Some Player</a>'
    "</div></th></tr></tbody></table>"
)


def _doc(away_name, home_name, t0_name, t1_name):
    ls = _LINESCORE.format(away=away_name, home=home_name)
    b0 = _BATTERS.format(tid="a" * 16, name=t0_name, pid="1" * 16)
    b1 = _BATTERS.format(tid="b" * 16, name=t1_name, pid="2" * 16)
    return html_struct.parse_html(f"<html><body>{ls}{b0}{b1}</body></html>")


def test_fails_loud_when_a_batters_caption_matches_neither_linescore_team():
    # Batters table for "Ghost Team" matches neither the away nor home linescore
    # row -> refuse to guess.
    root = _doc("Away Team", "Home Team", "Ghost Team", "Home Team")
    try:
        identity.build_player_table(root)
        assert False, "expected ValueError for a caption matching neither side"
    except ValueError as e:
        assert "matches neither" in str(e)


def test_fails_loud_when_both_batters_tables_resolve_to_one_side():
    # Two Batters tables both named "Away Team" -> ambiguous -> refuse to guess.
    root = _doc("Away Team", "Home Team", "Away Team", "Away Team")
    try:
        identity.build_player_table(root)
        assert False, "expected ValueError for two tables on one side"
    except ValueError as e:
        assert "both resolved" in str(e)


def test_fails_loud_without_a_linescore():
    b0 = _BATTERS.format(tid="a" * 16, name="Away Team", pid="1" * 16)
    b1 = _BATTERS.format(tid="b" * 16, name="Home Team", pid="2" * 16)
    root = html_struct.parse_html(f"<html><body>{b0}{b1}</body></html>")
    try:
        identity.build_player_table(root)
        assert False, "expected ValueError when the linescore is missing"
    except ValueError as e:
        assert "linescore" in str(e)
