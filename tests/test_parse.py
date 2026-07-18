"""Tests for bc_pipeline.parse: raw boxscore HTML -> full schema-valid
``final`` game dict.

Protected intent: correctness over coverage. Every PBP line becomes an
event OR a verbatim ``unparsed[]`` entry -- never dropped, never guessed.
This gate's close criteria: (1) the full real sample parses schema-valid
with every one of the 122 PBP cells accounted for by events+unparsed, (2)
the parser's own Top-1 (first 9 events, the away half of the 1st) matches
the hand fixture (the strongest oracle) under `serialize.semantic_equal`.
"""
from __future__ import annotations

from _support import SAMPLES_DIR, load_fixture, validate_game

from bc_pipeline import identity, parse, serialize


def _load(name: str) -> str:
    path = SAMPLES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


FINAL_HTML = _load("boxscore_20260709_final.html")
SOURCE_URL = "https://longbeachcoast.com/sports/bsb/2026/boxscores/20260709_h94w.xml"
FETCHED_AT = "2026-07-11T00:00:00Z"


def _parse_final() -> dict:
    return parse.parse_game(FINAL_HTML, source_url=SOURCE_URL, fetched_at=FETCHED_AT)


# --- schema validity + full-sample coverage --------------------------------


def test_full_sample_is_schema_valid():
    game = _parse_final()
    validate_game(game)  # raises on failure


def test_full_sample_events_plus_unparsed_account_for_every_pbp_cell():
    # g4's own real-sample sweep (tests/test_grammar.py) counts exactly 122
    # `<td class="text">` cells across all 9 innings with 0 GrammarMiss.
    # events+unparsed here must reproduce that total: nothing dropped.
    game = _parse_final()
    assert len(game["events"]) + len(game["unparsed"]) == 122


def test_full_sample_has_no_unparsed_residue():
    # Schema 1.1.0 made substitution.slot nullable, so the 5 DH pitching
    # changes (pitcher not in the batting order) are now real substitution
    # events. This game therefore has ZERO unparsed residue.
    game = _parse_final()
    assert len(game["unparsed"]) == 0


def test_full_sample_dh_pitching_subs_are_slotless_events():
    # The 5 "<in> to p for <out>" pitching changes are encoded as
    # substitution events with slot=null (DH: pitcher not in the batting
    # order) and kind="pitching" -- honest, not fabricated.
    game = _parse_final()
    subs = [e for e in game["events"] if e["kind"] == "substitution"]
    assert len(subs) == 5
    for e in subs:
        sub = e["substitution"]
        assert sub["slot"] is None
        assert sub["kind"] == "pitching"
        assert sub["player_out"] and sub["player_in"]
        assert "to p for" in e["narrative"]


def test_full_sample_kind_counts_match_grammar_coverage():
    # g4's coverage evidence: plate_appearance 87, runner_event 13,
    # inning_summary 17, substitution 5 (total 122). All now encoded as events.
    game = _parse_final()
    kinds: dict = {}
    for e in game["events"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    assert kinds == {
        "plate_appearance": 87,
        "runner_event": 13,
        "inning_summary": 17,
        "substitution": 5,
    }


def test_full_sample_linescore_matches_fixture_oracle():
    game = _parse_final()
    fixture = load_fixture("game_20260709_h94w_top1.json")
    assert game["linescore"] == fixture["linescore"]


def test_full_sample_box_batting_first_six_away_rows_match_fixture():
    game = _parse_final()
    fixture = load_fixture("game_20260709_h94w_top1.json")
    away_team_id = game["teams"]["away"]["team_id"]
    assert (
        game["box"]["batting"][away_team_id][:6]
        == fixture["box"]["batting"][away_team_id]
    )


def test_full_sample_header_fields():
    game = _parse_final()
    assert game["game_id"] == "20260709_h94w"
    assert game["season"] == 2026
    assert game["status"] == "final"
    assert game["date"] == "2026-07-09"
    assert game["source"]["provider"] == "prestosports"
    assert game["source"]["site"] == "longbeachcoast.com"
    assert game["teams"]["home"]["team_id"] == "maotayco79j2g2lx"
    assert game["teams"]["home"]["name"] == "Long Beach Coast"
    assert game["teams"]["away"]["name"] == "Yuba-Sutter Freebirds"


def test_full_sample_meta_parse_integrity_signals():
    game = _parse_final()
    meta = game["meta"]
    assert meta["parser_version"] == parse.PARSER_VERSION
    assert meta["source_url"] == SOURCE_URL
    assert meta["source_sha256"] == parse.sha256_hex(FINAL_HTML)
    assert meta["fetched_at"] == FETCHED_AT
    assert meta["derived_replayer_version"] == "unreplayed"
    assert meta["parse"]["events_count"] == len(game["events"])
    assert meta["parse"]["unparsed_count"] == len(game["unparsed"])
    assert meta["parse"]["replayable"] is False


# --- Top-1 agreement (the strongest oracle) ---------------------------------


def test_top1_events_semantic_equal_hand_fixture():
    game = _parse_final()
    fixture = load_fixture("game_20260709_h94w_top1.json")
    top9 = game["events"][:9]
    assert serialize.semantic_equal({"events": top9}, {"events": fixture["events"]})


def test_top1_events_all_belong_to_inning_1_top():
    game = _parse_final()
    for e in game["events"][:9]:
        assert e["inning"] == 1
        assert e["half"] == "top"
    assert game["events"][9]["half"] == "bottom" or game["events"][9]["inning"] == 2


# --- idempotency key ---------------------------------------------------------


def test_idempotency_key_is_hash_plus_parser_version():
    key = parse.idempotency_key(FINAL_HTML)
    assert parse.sha256_hex(FINAL_HTML) in key
    assert parse.PARSER_VERSION in key


# --- count-assembly guard (schema 1.2.0, issue #30 g2b) ---------------------
#
# grammar.py already emits PrimaryClause(count=None, pitches=None) for a
# plate-appearance line whose source PBP row carries no count-tail at all
# (the historical league template -- see tests/test_grammar.py
# test_count_tail_optional_*). Pre-fix, parse.build_events crashed reading
# `p.count.balls` on that None (a pre-existing blocker flagged by g1's
# implementer result, re-confirmed by the synthetic_taxonomy_tail fixtures).
# Schema 1.2.0 makes the event's `count` field nullable so this can be
# encoded as a real event instead.


def _build_events_for_line(text: str):
    away = identity.TeamIdentity(
        team_id="syn:team:away",
        name="Synthetic Away",
        players={
            "syn:away:1": identity.PlayerEntry(
                player_id="syn:away:1",
                name="Kyle Schmack",
                last_name="Schmack",
                team_id="syn:team:away",
                positions=["cf"],
            ),
        },
    )
    home = identity.TeamIdentity(
        team_id="syn:team:home",
        name="Synthetic Home",
        players={
            f"syn:home:{i}": identity.PlayerEntry(
                player_id=f"syn:home:{i}",
                name=f"Home Player {i}",
                last_name=f"Player{i}",
                team_id="syn:team:home",
                positions=["1b"],
            )
            for i in range(1, 10)
        }
        | {
            "syn:home:10": identity.PlayerEntry(
                player_id="syn:home:10",
                name="Jordan Lee",
                last_name="Lee",
                team_id="syn:team:home",
                positions=["p"],
            ),
        },
    )
    player_table = identity.PlayerTable(home=home, away=away)
    line = parse.PbpLine(inning=1, half="top", line_index=0, text=text, is_strong=False)
    return parse.build_events([line], player_table)


def test_count_tail_optional_line_no_longer_crashes_build_events():
    events, unparsed, _subs = _build_events_for_line(
        "Kyle Schmack singled up the middle."
    )
    assert unparsed == []
    assert len(events) == 1
    assert events[0]["kind"] == "plate_appearance"


def test_count_tail_optional_line_emits_count_none_and_pitches_none():
    events, _unparsed, _subs = _build_events_for_line(
        "Kyle Schmack singled up the middle."
    )
    assert events[0]["count"] is None
    assert events[0]["pitches"] is None


# --- DH-slot-bare substitution end-to-end (schema 1.2.0, issue #30 g2b,
# Commander-authorized scope extension closing the m3 stop condition) ------
#
# The new grammar.py STANDALONE_RULES row for "<name> to dh." builds a
# Substitution(player_out=None, ...). Pre-fix, parse.build_events's
# substitution branch called _last_name_token(None) unconditionally and
# crashed. This proves the line now reaches a real events[] entry, not just
# a grammar-level ClauseGroup.


def _build_events_for_dh_slot_bare_line(text: str):
    # half="top" -> the AWAY side is batting (parse.py:345) -- an "offensive"
    # substitution (pinch-run, DH-slot entry) resolves against the BATTING
    # side, so the DH-entering player must be on the away roster here. (A
    # prior version of this fixture put the player on the home/fielding
    # roster, which only "passed" because of a since-fixed bug that resolved
    # every substitution against the fielding side regardless of kind.)
    away = identity.TeamIdentity(
        team_id="syn:team:away",
        name="Synthetic Away",
        players={
            "syn:away:1": identity.PlayerEntry(
                player_id="syn:away:1",
                name="Kyle Schmack",
                last_name="Schmack",
                team_id="syn:team:away",
                positions=["cf"],
            ),
            "syn:away:2": identity.PlayerEntry(
                player_id="syn:away:2",
                name="Cole Robinson",
                last_name="Robinson",
                team_id="syn:team:away",
                positions=["dh"],
            ),
        },
    )
    home = identity.TeamIdentity(
        team_id="syn:team:home",
        name="Synthetic Home",
        players={
            "syn:home:1": identity.PlayerEntry(
                player_id="syn:home:1",
                name="Jordan Lee",
                last_name="Lee",
                team_id="syn:team:home",
                positions=["p"],
            ),
        },
    )
    player_table = identity.PlayerTable(home=home, away=away)
    line = parse.PbpLine(inning=1, half="top", line_index=0, text=text, is_strong=False)
    return parse.build_events([line], player_table)


def test_pinch_run_substitution_resolves_against_batting_side_not_fielding_side():
    # Regression for a real bug found while implementing g2b: the
    # substitution-assembly branch predated the "offensive" Substitution.kind
    # (added for pinch-run in g1) and unconditionally resolved every
    # substitution against the FIELDING side. A real pinch-run line names two
    # players on the BATTING side, so every real pinch-run line silently
    # landed in unparsed[] (or worse, could false-match an unrelated
    # same-surname player on the wrong team) despite grammar.py correctly
    # parsing the clause. half="top" -> away bats; both named players are on
    # the away roster here, matching how the real corpus actually looks.
    away = identity.TeamIdentity(
        team_id="syn:team:away",
        name="Synthetic Away",
        players={
            "syn:away:1": identity.PlayerEntry(
                player_id="syn:away:1",
                name="Pat Smith",
                last_name="Smith",
                team_id="syn:team:away",
                positions=["cf"],
            ),
            "syn:away:2": identity.PlayerEntry(
                player_id="syn:away:2",
                name="Sam Runner",
                last_name="Runner",
                team_id="syn:team:away",
                positions=["pr"],
            ),
        },
    )
    home = identity.TeamIdentity(
        team_id="syn:team:home",
        name="Synthetic Home",
        players={
            "syn:home:1": identity.PlayerEntry(
                player_id="syn:home:1",
                name="Jordan Lee",
                last_name="Lee",
                team_id="syn:team:home",
                positions=["p"],
            ),
        },
    )
    player_table = identity.PlayerTable(home=home, away=away)
    line = parse.PbpLine(
        inning=1,
        half="top",
        line_index=0,
        text="Sam Runner pinch ran for Pat Smith.",
        is_strong=False,
    )
    events, unparsed, _subs = parse.build_events([line], player_table)
    assert unparsed == []
    assert len(events) == 1
    sub = events[0]["substitution"]
    assert sub["kind"] == "offensive"
    assert sub["player_in"] == "syn:away:2"
    assert sub["player_out"] == "syn:away:1"


def test_dh_slot_bare_line_no_longer_crashes_build_events():
    events, unparsed, _subs = _build_events_for_dh_slot_bare_line(
        "Cole Robinson to dh."
    )
    assert unparsed == []
    assert len(events) == 1
    assert events[0]["kind"] == "substitution"


def test_dh_slot_bare_line_emits_player_out_none_and_offensive_kind():
    events, _unparsed, _subs = _build_events_for_dh_slot_bare_line(
        "Cole Robinson to dh."
    )
    sub = events[0]["substitution"]
    assert sub["player_out"] is None
    assert sub["player_in"] == "syn:away:2"
    assert sub["kind"] == "offensive"


def test_dh_slot_bare_event_is_schema_valid():
    fixture = load_fixture("game_20260709_h94w_top1.json")
    events, _unparsed, _subs = _build_events_for_dh_slot_bare_line(
        "Cole Robinson to dh."
    )
    game = dict(fixture)
    game["events"] = fixture["events"] + [events[0]]
    validate_game(game)


# ---------------------------------------------------------------------------
# Family (issue #31, g3) -- substitution/position-move grammar generalized:
# kind branched by matched position (p -> pitching, dh -> offensive, every
# other fielding position -> defensive) on the two-name "<in> to <pos> for
# <out>." row, a new standalone pinch-hit row, and a new guarded bare
# "<name> to <pos>." row. Every shape's kind->side resolution is checked
# against a synthetic roster shaped to match a REAL corpus example of that
# exact shape -- see each test's docstring for the grounding evidence.
# ---------------------------------------------------------------------------


def _entry2(pid, name, last, team_id, positions):
    return identity.PlayerEntry(
        player_id=pid, name=name, last_name=last, team_id=team_id, positions=list(positions)
    )


def _two_team_table(away_players, home_players):
    away = identity.TeamIdentity(team_id="syn:team:away", name="Synthetic Away", players=away_players)
    home = identity.TeamIdentity(team_id="syn:team:home", name="Synthetic Home", players=home_players)
    return identity.PlayerTable(home=home, away=away)


def test_e2e_pitching_sub_resolves_fielding_side():
    # half="top" -> away bats, HOME fields. A pitching change is made by the
    # FIELDING team (kind="pitching" -> parse.py's fielding_side branch,
    # unchanged by this gate). Verified against 21 of 40 sampled real
    # "to p for" corpus lines resolving cleanly under this convention (the
    # remainder are a PRE-EXISTING, unrelated data quirk -- see this gate's
    # IMPLEMENTER_RESULT -- where a trailing substitution announcement is
    # logged at a half boundary describing the NEXT half's roster; those
    # lines were already honest unparsed[] residues before this gate and
    # remain so, never silently mis-resolved).
    table = _two_team_table(
        away_players={"a1": _entry2("a1", "Away Batter", "Batter", "syn:team:away", ["cf"])},
        home_players={
            "h1": _entry2("h1", "Isaiah Williams", "Williams", "syn:team:home", ["p"]),
            "h2": _entry2("h2", "Chase Martinez", "Martinez", "syn:team:home", ["p"]),
        },
    )
    line = parse.PbpLine(
        inning=1, half="top", line_index=0,
        text="Isaiah Williams to p for Chase Martinez.", is_strong=False,
    )
    events, unparsed, _subs = parse.build_events([line], table)
    assert unparsed == []
    sub = events[0]["substitution"]
    assert sub["kind"] == "pitching"
    assert sub["player_in"] == "h1"
    assert sub["player_out"] == "h2"


def test_e2e_two_name_defensive_position_sub_resolves_fielding_side():
    # half="top" -> away bats, HOME fields. A defensive position change is
    # made by the FIELDING team (kind="defensive" -> fielding_side, the same
    # dispatch branch pitching already used). verbatim shape 'B. Lada to ss
    # for B. Marine.' (games/**/*.json unparsed[] entry).
    table = _two_team_table(
        away_players={"a1": _entry2("a1", "Away Batter", "Batter", "syn:team:away", ["cf"])},
        home_players={
            "h1": _entry2("h1", "B. Lada", "Lada", "syn:team:home", ["ss"]),
            "h2": _entry2("h2", "B. Marine", "Marine", "syn:team:home", ["ss"]),
        },
    )
    line = parse.PbpLine(
        inning=1, half="top", line_index=0, text="B. Lada to ss for B. Marine.", is_strong=False,
    )
    events, unparsed, _subs = parse.build_events([line], table)
    assert unparsed == []
    sub = events[0]["substitution"]
    assert sub["kind"] == "defensive"
    assert sub["player_in"] == "h1"
    assert sub["player_out"] == "h2"


def test_e2e_bare_position_move_resolves_fielding_side_player_out_none():
    # half="top" -> away bats, HOME fields. A bare position move (no
    # outgoing player named) resolves against the FIELDING side, kind flat
    # "defensive". This shape was the CLEANEST in real-corpus verification:
    # 0 of 40 sampled real 'to <pos>.' lines mismatched this convention.
    # verbatim shape 'D. Sackett to 3b.' (games/**/*.json unparsed[] entry).
    table = _two_team_table(
        away_players={"a1": _entry2("a1", "Away Batter", "Batter", "syn:team:away", ["cf"])},
        home_players={"h1": _entry2("h1", "D. Sackett", "Sackett", "syn:team:home", ["3b"])},
    )
    line = parse.PbpLine(inning=1, half="top", line_index=0, text="D. Sackett to 3b.", is_strong=False)
    events, unparsed, _subs = parse.build_events([line], table)
    assert unparsed == []
    sub = events[0]["substitution"]
    assert sub["kind"] == "defensive"
    assert sub["player_in"] == "h1"
    assert sub["player_out"] is None


def test_e2e_pinch_hit_resolves_batting_side():
    # half="top" -> AWAY bats. A pinch-hitter enters exactly at their own
    # team's turn to bat (kind="offensive" -> batting_side). This shape was
    # also the cleanest in real-corpus verification: 0 of 40 sampled real
    # 'pinch hit for' lines mismatched this convention. verbatim shape
    # 'S. Wilmer pinch hit for B. Hancock.' (games/**/*.json unparsed[]).
    table = _two_team_table(
        away_players={
            "a1": _entry2("a1", "S. Wilmer", "Wilmer", "syn:team:away", ["ph"]),
            "a2": _entry2("a2", "B. Hancock", "Hancock", "syn:team:away", ["cf"]),
        },
        home_players={"h1": _entry2("h1", "Home Pitcher", "Pitcher", "syn:team:home", ["p"])},
    )
    line = parse.PbpLine(
        inning=1, half="top", line_index=0, text="S. Wilmer pinch hit for B. Hancock.", is_strong=False,
    )
    events, unparsed, _subs = parse.build_events([line], table)
    assert unparsed == []
    sub = events[0]["substitution"]
    assert sub["kind"] == "offensive"
    assert sub["player_in"] == "a1"
    assert sub["player_out"] == "a2"


def test_e2e_two_name_dh_sub_issue_32_resolves_batting_side_when_convention_holds():
    # Issue #32. half="top" -> AWAY bats. kind="offensive" (grammar layer,
    # p->pitching/dh->offensive/else->defensive, unambiguous and NOT
    # affected by the finding below) resolves against batting_side, same
    # convention as pinch-run/pinch-hit/bare-DH-entry. This test proves the
    # convention DOES work end-to-end when it holds -- verbatim shape
    # 'P. DePasqual to dh for J. Impedugli.', confirmed against real game
    # games/2024/20240524_91ql.json where both named players are on the
    # batting team's roster for that half.
    #
    # STOP CONDITION / real-data finding (see this gate's IMPLEMENTER_RESULT
    # for full detail): sampling all 47 real 'to dh for' corpus lines against
    # their actual per-game rosters, only 24/47 (51%) match this
    # batting_side convention: the remaining 22/47 (47%) are a same-team
    # roster reshuffle logged as a trailing announcement at a half boundary
    # (immediately following a defensive substitution for the SAME
    # FIELDING team, moving a player from a fielding position into the DH
    # slot) -- the DH sub genuinely belongs to the FIELDING side in those
    # cases, not the batting side. This is NOT a simple "always resolve
    # fielding instead" fix either (24/47 need batting, 22/47 need
    # fielding) -- resolving it correctly requires a genuine assembly design
    # decision (e.g. try-both-sides-accept-if-unique-on-exactly-one), which
    # is out of this gate's authorized scope per the handoff's explicit stop
    # condition. parse.py is left UNCHANGED; the ~47% of real DH-sub lines
    # that need the fielding side will safely land in unparsed[] (never
    # silently mis-resolved -- see the next test) until Commander/Admiral
    # rules on the design call.
    table = _two_team_table(
        away_players={
            "a1": _entry2("a1", "P. DePasqual", "DePasqual", "syn:team:away", ["dh"]),
            "a2": _entry2("a2", "J. Impedugli", "Impedugli", "syn:team:away", ["dh"]),
        },
        home_players={"h1": _entry2("h1", "Home Pitcher", "Pitcher", "syn:team:home", ["p"])},
    )
    line = parse.PbpLine(
        inning=1, half="top", line_index=0, text="P. DePasqual to dh for J. Impedugli.", is_strong=False,
    )
    events, unparsed, _subs = parse.build_events([line], table)
    assert unparsed == []
    sub = events[0]["substitution"]
    assert sub["kind"] == "offensive"
    assert sub["player_in"] == "a1"
    assert sub["player_out"] == "a2"


def test_e2e_two_name_dh_sub_resolves_via_fielding_side_fallback():
    # REWORK (commander-31, parse.py-assembly design call, responding to the
    # stop condition originally reported here): a real corpus shape where the
    # DH sub genuinely belongs to the FIELDING side (verbatim 'J. McLaughli
    # to dh for A. Sczepkows.', games/2024/20240613_516b.json -- both players
    # confirmed on that game's AWAY/fielding roster during the "bottom" half
    # this line was logged in). Under the OLD batting_side-only convention,
    # this line could not resolve at all (neither name exists on the assumed
    # batting side) and landed in unparsed[]. The try-both-sides fallback
    # (parse.py's substitution branch) now tries the fielding side when the
    # kind-implied primary (batting, since kind="offensive") fails, and
    # accepts it because BOTH names resolve uniquely there. `kind` stays
    # "offensive" regardless of which side it resolved on -- kind is
    # position semantics (grammar layer), side is roster membership
    # (assembly layer); the two are deliberately decoupled.
    table = _two_team_table(
        away_players={
            "a1": _entry2("a1", "J. McLaughli", "McLaughlin", "syn:team:away", ["dh", "cf"]),
            "a2": _entry2("a2", "A. Sczepkows", "Sczepkowski", "syn:team:away", ["dh", "cf"]),
        },
        home_players={"h1": _entry2("h1", "Home Pitcher", "Pitcher", "syn:team:home", ["p"])},
    )
    # half="bottom" -> batting_side="home" (primary, fails), fielding_side
    # ="away" (fallback, both names resolve uniquely there -> accepted).
    line = parse.PbpLine(
        inning=9, half="bottom", line_index=0, text="J. McLaughli to dh for A. Sczepkows.", is_strong=False,
    )
    events, unparsed, _subs = parse.build_events([line], table)
    assert unparsed == []
    assert len(events) == 1
    sub = events[0]["substitution"]
    assert sub["kind"] == "offensive"
    assert sub["player_in"] == "a1"
    assert sub["player_out"] == "a2"


def test_e2e_dh_sub_ambiguous_on_both_sides_stays_unparsed_never_guesses():
    # REWORK companion (b): the try-both-sides fallback must NEVER create a
    # wrong resolution. Construct a roster where BOTH the batting side (the
    # kind-implied primary) AND the fielding side (the fallback) each have a
    # UNIQUE, FULL match for the two DH-sub names -- a plausible real-world
    # scenario (a common surname pair existing on both rosters). Since both
    # sides fully resolve, this is a genuine cross-side ambiguity: the
    # fallback logic checks BOTH sides (never short-circuits on a primary
    # success) specifically so it can detect this and refuse to guess, same
    # as the pre-rework "never guess" doctrine.
    table = _two_team_table(
        away_players={
            "a1": _entry2("a1", "J. Smith", "Smith", "syn:team:away", ["dh"]),
            "a2": _entry2("a2", "T. Jones", "Jones", "syn:team:away", ["dh"]),
        },
        home_players={
            "h1": _entry2("h1", "J. Smith", "Smith", "syn:team:home", ["dh"]),
            "h2": _entry2("h2", "T. Jones", "Jones", "syn:team:home", ["dh"]),
        },
    )
    # half="top" -> batting_side="away" (primary, fully resolves: both Smith
    # and Jones are on the away roster) AND fielding_side="home" (fallback,
    # ALSO fully resolves: both Smith and Jones are on the home roster too)
    # -- ambiguous across both sides.
    line = parse.PbpLine(
        inning=1, half="top", line_index=0, text="J. Smith to dh for T. Jones.", is_strong=False,
    )
    events, unparsed, _subs = parse.build_events([line], table)
    assert events == []
    assert len(unparsed) == 1
    assert "cross-side ambiguity" in unparsed[0]["reason"]


def test_count_tail_optional_event_is_schema_valid():
    # Embed the count-less event into the frozen hand fixture (which supplies
    # every other required top-level field) and validate the whole file.
    fixture = load_fixture("game_20260709_h94w_top1.json")
    events, _unparsed, _subs = _build_events_for_line(
        "Kyle Schmack singled up the middle."
    )
    game = dict(fixture)
    game["events"] = fixture["events"] + [events[0]]
    validate_game(game)


def test_idempotency_key_changes_with_different_html():
    assert parse.idempotency_key(FINAL_HTML) != parse.idempotency_key(FINAL_HTML + " ")


# --- _last_name_token: narrative-name join tokenizer (Family 2) -------------


def test_last_name_token_strips_trailing_comma_before_suffix():
    # "Rojas, Jr" -> tokens ["Rojas,", "Jr"]: "Jr" is recognized as a
    # trailing suffix, but the returned surname token must NOT retain the
    # comma left dangling from the narrative's "Surname, Suffix" shape.
    assert parse._last_name_token("Rojas, Jr") == "Rojas"


def test_last_name_token_suffix_without_comma_unaffected():
    assert parse._last_name_token("Patrick Roche Jr.") == "Roche"


def test_last_name_token_plain_name_unaffected():
    assert parse._last_name_token("J. McLaughli") == "McLaughli"
