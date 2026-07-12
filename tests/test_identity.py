"""Tests for bc_pipeline.identity: the per-game player identity table +
name-resolution surface, asserted against the real archived sample page plus
synthetic ambiguity cases.

Protected intent: the event->player join must be TOTAL and honest. Every
player gets a stable file-local player_id; a bare last name that cannot be
uniquely resolved on its side returns resolved=False -- never a guess. This
gate is IDENTITY only: no box stat numbers (AB/R/H/RBI/...) and no linescore.
"""
from __future__ import annotations

from _support import SAMPLES_DIR

from bc_pipeline import html_struct, identity

HOME_TEAM_ID = "maotayco79j2g2lx"
AWAY_TEAM_ID = "syn:team:away"


def _load(name: str) -> str:
    path = SAMPLES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _final_root():
    return html_struct.parse_html(_load("boxscore_20260709_final.html"))


def _final_table():
    return identity.build_player_table(_final_root())


# --- home team -------------------------------------------------------------


def test_home_table_has_twelve_players_with_sixteen_char_ids():
    table = _final_table()
    assert table.home.team_id == HOME_TEAM_ID
    assert table.home.name == "Long Beach Coast"
    assert len(table.home.players) == 12
    for pid in table.home.players:
        assert len(pid) == 16
        assert pid.isalnum()


def test_home_table_includes_known_players_with_known_ids():
    table = _final_table()
    by_last = {p.last_name: p for p in table.home.players.values()}
    van_deventer = by_last["VanDeventer"]
    assert van_deventer.player_id == "4bs3tvwryvtzrvpa"
    assert table.home.players[van_deventer.player_id] is van_deventer

    pelc = by_last["Pelc"]
    assert pelc.player_id == "3865oyuz5l2pj51r"
    assert pelc.name == "Eddy Pelc"


def test_home_player_id_equals_its_own_key():
    table = _final_table()
    for pid, entry in table.home.players.items():
        assert entry.player_id == pid


def test_home_suffix_name_last_name_is_surname_not_suffix():
    table = _final_table()
    by_last = {p.last_name: p for p in table.home.players.values()}
    assert "Roche" in by_last
    roche = by_last["Roche"]
    assert roche.name == "Patrick Roche Jr."
    assert roche.last_name == "Roche"


# --- away team ---------------------------------------------------------


_EXPECTED_AWAY_FIRST_SIX = [
    ("syn:away:1", "Isaac Nunez", "Nunez", "ss"),
    ("syn:away:2", "Jordan Donahue", "Donahue", "2b"),
    ("syn:away:3", "Josh Phillips", "Phillips", "dh"),
    ("syn:away:4", "Kyle Carlson", "Carlson", "3b"),
    ("syn:away:5", "Christian Castaneda", "Castaneda", "1b"),
    ("syn:away:6", "Andrew Kirchner", "Kirchner", "rf"),
]


def test_away_table_first_six_match_fixture_in_stable_order():
    table = _final_table()
    assert table.away.team_id == AWAY_TEAM_ID
    assert table.away.name == "Yuba-Sutter Freebirds"
    ordered_ids = list(table.away.players.keys())[:6]
    assert ordered_ids == [pid for pid, *_ in _EXPECTED_AWAY_FIRST_SIX]
    for pid, name, last_name, pos in _EXPECTED_AWAY_FIRST_SIX:
        entry = table.away.players[pid]
        assert entry.player_id == pid
        assert entry.name == name
        assert entry.last_name == last_name
        assert pos in entry.positions


def test_away_synthetic_ids_are_deterministic_across_reparses():
    table_a = _final_table()
    table_b = _final_table()
    assert list(table_a.away.players.keys()) == list(table_b.away.players.keys())


# --- resolve() ---------------------------------------------------------


def test_resolve_unique_away_last_name():
    table = _final_table()
    assert table.resolve("Nunez", "away") == ("syn:away:1", True)


def test_resolve_unique_home_last_name():
    table = _final_table()
    pid, resolved = table.resolve("VanDeventer", "home")
    assert resolved is True
    assert pid == "4bs3tvwryvtzrvpa"


def test_resolve_absent_name_is_unresolved():
    table = _final_table()
    pid, resolved = table.resolve("Ohtani", "away")
    assert resolved is False
    assert pid is None


# --- synthetic ambiguity: two same-last-name players on one side -----------

_SYNTHETIC_HTML = """
<html><body>
<div class="linescore"><table>
<tr><th></th><th>1</th><th>2</th><th>R</th><th>H</th><th>E</th></tr>
<tr><td>Synthetic Away</td><td>1</td><td>1</td><td>2</td><td>5</td><td>0</td></tr>
<tr><td>Synthetic Home</td><td>2</td><td>1</td><td>3</td><td>7</td><td>1</td></tr>
</table></div>
<table class="table table-striped striped">
<caption class="caption"><h2>
<a href="/x/teams?id=abcdefgh12345678" class="team-name">Synthetic Home</a>
<span class="offscreen">Batters</span></h2></caption>
<thead><tr><th scope="col" class="text pinned-col col-head">Hitters</th>
<th scope="col" class="col-head">AB</th></tr></thead>
<tbody>
<tr><th scope="row" class="row-head pinned-col text">
<div class="d-flex align-items-center justify-content-start gap-1">
<span class="position small fw-normal text-uppercase">ss</span>
<a href="/x/players?id=aaaaaaaaaaaaaaaa" class="player-name ">John Smith</a>
</div></th><td>4</td></tr>
<tr><th scope="row" class="row-head pinned-col text">
<div class="d-flex align-items-center justify-content-start gap-1">
<span class="position small fw-normal text-uppercase">2b</span>
<a href="/x/players?id=bbbbbbbbbbbbbbbb" class="player-name ">Unique Player</a>
</div></th><td>3</td></tr>
</tbody>
</table>
<table class="table table-striped striped">
<caption class="caption"><h2>
<span class="team-name">Synthetic Away</span>
<span class="offscreen">Batters</span></h2></caption>
<thead><tr><th scope="col" class="text pinned-col col-head">Hitters</th>
<th scope="col" class="col-head">AB</th></tr></thead>
<tbody>
<tr><th scope="row" class="row-head pinned-col text">
<div class="d-flex align-items-center justify-content-start gap-1">
<span class="position small fw-normal text-uppercase">rf</span>
<span class="player-name w-100 ">Bob Smith</span>
</div></th><td>4</td></tr>
<tr><th scope="row" class="row-head pinned-col text">
<div class="d-flex align-items-center justify-content-start gap-1">
<span class="position small fw-normal text-uppercase">lf</span>
<span class="player-name w-100 ">Alice Smith</span>
</div></th><td>3</td></tr>
<tr><th scope="row" class="row-head pinned-col text">
<div class="d-flex align-items-center justify-content-start gap-1">
<span class="position small fw-normal text-uppercase">c</span>
<span class="player-name w-100 ">Carl Jones</span>
</div></th><td>2</td></tr>
</tbody>
</table>
</body></html>
"""


def _synthetic_table():
    root = html_struct.parse_html(_SYNTHETIC_HTML)
    return identity.build_player_table(root)


def test_synthetic_two_same_last_name_collision_is_unresolved():
    table = _synthetic_table()
    # Two "Smith" entries on the away side -> ambiguous -> never guess.
    pid, resolved = table.resolve("Smith", "away")
    assert resolved is False
    assert pid is None


def test_synthetic_unique_last_name_still_resolves_on_ambiguous_side():
    table = _synthetic_table()
    pid, resolved = table.resolve("Jones", "away")
    assert resolved is True
    assert pid is not None


def test_synthetic_home_side_unaffected_by_away_collision():
    table = _synthetic_table()
    pid, resolved = table.resolve("Smith", "home")
    assert resolved is True
    assert pid == "aaaaaaaaaaaaaaaa"


# --- resolve() prefix-match fallback (Family 2: truncated surnames) --------
#
# The historical template narrative sometimes TRUNCATES a surname (e.g.
# "Richardso" for "Richardson"). The exact-match path in resolve() finds no
# candidate for the truncated token, so an ADDITIVE prefix-match fallback
# (real_last_name.startswith(token) or token.startswith(real_last_name))
# runs next -- but ONLY ever returns a result when it finds EXACTLY ONE
# candidate; any remaining ambiguity still returns (None, False), never a
# guess.


def _one_side_table(players: "Dict[str, identity.PlayerEntry]") -> identity.PlayerTable:
    away = identity.TeamIdentity(
        team_id="syn:team:away", name="Synthetic Away", players=players
    )
    home = identity.TeamIdentity(team_id="syn:team:home", name="Synthetic Home", players={})
    return identity.PlayerTable(home=home, away=away)


def test_resolve_prefix_match_resolves_truncated_surname():
    # Real corpus shape: games/2025/20250520_u80r.json roster has
    # "Conner Richardson" (last_name "Richardson"); the narrative substitution
    # line "C. Richardso to p for L. Short." truncates it to "Richardso".
    players = {
        "p1": identity.PlayerEntry(
            player_id="p1",
            name="Conner Richardson",
            last_name="Richardson",
            team_id="syn:team:away",
        ),
    }
    table = _one_side_table(players)
    pid, resolved = table.resolve("Richardso", "away")
    assert resolved is True
    assert pid == "p1"


def test_resolve_prefix_match_deliberately_ambiguous_collision_stays_unresolved():
    # Two players whose truncated forms collide on a shared prefix
    # ("Richardson" and "Richards" both truncate to "Richard...") -- the
    # fallback must correctly refuse to guess, exactly like the exact-match
    # same-last-name collision case above.
    players = {
        "p1": identity.PlayerEntry(
            player_id="p1",
            name="Conner Richardson",
            last_name="Richardson",
            team_id="syn:team:away",
        ),
        "p2": identity.PlayerEntry(
            player_id="p2",
            name="Pat Richards",
            last_name="Richards",
            team_id="syn:team:away",
        ),
    }
    table = _one_side_table(players)
    pid, resolved = table.resolve("Richard", "away")
    assert resolved is False
    assert pid is None


def test_resolve_prefix_match_reverse_direction_token_longer_than_last_name():
    # The reverse direction: a narrative token longer than the roster's
    # last_name (token.startswith(last_name)) -- still resolves uniquely
    # when there is exactly one candidate.
    players = {
        "p1": identity.PlayerEntry(
            player_id="p1", name="Kyle Chi", last_name="Chi", team_id="syn:team:away"
        ),
    }
    table = _one_side_table(players)
    pid, resolved = table.resolve("Chian", "away")
    assert resolved is True
    assert pid == "p1"
