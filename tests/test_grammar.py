"""Tests for bc_pipeline.grammar: the pure-text closed PBP clause grammar.

Protected intent: grammar.py is PURE text -> structured clause data. No HTML,
no player ids, no base-out state. An unrecognized clause returns a
``GrammarMiss`` -- never a guess, never an exception. These tests cover the
5 resistant shapes named in the handoff verbatim, a taxonomy-coverage check
(rule-table outcome/cause sets == the frozen schema's 17/12 enums), a
GrammarMiss smoke test, and a full real-sample sweep (every ``<td
class="text">`` cell of the archived final boxscore) asserting 0
GrammarMiss -- the x2 spike proved 100% coverage is achievable on this game.
"""
from __future__ import annotations

from _support import SAMPLES_DIR, load_schema

from bc_pipeline import grammar, html_struct
from bc_pipeline.grammar import (
    BATTER_OUTCOME_CAUSE,
    PRIMARY_RULES,
    RUNNER_RULES,
    STANDALONE_RULES,
    ClauseGroup,
    GrammarMiss,
    parse_clause_group,
)


def _load(name: str) -> str:
    path = SAMPLES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _all_sample_cells():
    root = html_struct.parse_html(_load("boxscore_20260709_final.html"))
    panes = html_struct.iter_pbp_panes(root)
    cells = []
    for _inning, pane_cells in panes:
        cells.extend(c.text for c in pane_cells)
    return cells


# ---------------------------------------------------------------------------
# The 5 resistant shapes (verbatim from the handoff)
# ---------------------------------------------------------------------------


def test_resistant_shape_1_groundout_force_chain_and_runner_out():
    line = (
        "Isaac Nunez grounded out 3b to 2b to 1b (1-0 B); "
        "Cam Yuran out at second 2b to 1b. (3 out)"
    )
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    assert result.kind == "plate_appearance"
    p = result.primary
    assert p.name_token == "Isaac Nunez"
    assert p.outcome_type == "groundout"
    assert p.fielders == ["3b", "2b", "1b"]
    assert p.count.balls == 1 and p.count.strikes == 0
    assert p.pitches == "B"
    assert len(result.runners) == 1
    r = result.runners[0]
    assert r.name_token == "Cam Yuran"
    assert r.cause == "force_out"
    assert r.out is True
    assert result.trailing_outs == 3


def test_resistant_shape_2_groundout_unassisted():
    line = "Josh Phillips grounded out to 1b unassisted (0-1 K). (2 out)"
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    p = result.primary
    assert p.outcome_type == "groundout"
    assert p.fielders == ["1b"]
    assert "unassisted" in p.modifiers
    assert result.trailing_outs == 2


def test_resistant_shape_3_reached_on_error_an_error_wording():
    line = (
        "Emilio Corona reached first on an error by 3b (3-2 BFKBB); "
        "Cuba Bess advanced to third."
    )
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    p = result.primary
    assert p.outcome_type == "reached_on_error"
    assert p.fielders == ["3b"]
    assert "error" in p.modifiers


def test_resistant_shape_4_hit_by_pitch_rbi():
    line = (
        "Patrick Roche Jr. hit by pitch, RBI (2-0 BH); "
        "Eddy Pelc advanced to second; Anthony Mata advanced to third; "
        "Johnny Pappas scored."
    )
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    p = result.primary
    assert p.outcome_type == "hit_by_pitch"
    assert "RBI" in p.modifiers


def test_resistant_shape_5_compound_advance_then_score_unearned():
    line = (
        "Eddy Pelc reached on a fielder's choice (0-2 KF); "
        "Anthony Mata out at second 2b unassisted; "
        "Johnny Pappas advanced to third, scored on an error by 2b, unearned."
    )
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    assert result.primary.outcome_type == "fielders_choice"
    # 2 runner clauses -> 3 runner movements (compound clause yields 2)
    names = [r.name_token for r in result.runners]
    assert names == ["Anthony Mata", "Johnny Pappas", "Johnny Pappas"]
    advance_move, score_move = result.runners[1], result.runners[2]
    assert advance_move.cause == "advance"
    assert advance_move.destination == "third"
    assert advance_move.scored is False
    assert score_move.cause == "error"
    assert score_move.destination == "home"
    assert score_move.scored is True
    assert score_move.unearned is True


# ---------------------------------------------------------------------------
# GrammarMiss: never a guess, never an exception
# ---------------------------------------------------------------------------


def test_nonsense_line_is_grammar_miss_not_exception():
    result = parse_clause_group("The umpire ordered a rain delay for weather.")
    assert isinstance(result, GrammarMiss)
    assert result.raw == "The umpire ordered a rain delay for weather."
    assert result.reason


def test_grammar_miss_on_unrecognized_runner_clause():
    line = "Isaac Nunez singled to left field (1-1 BS); Cam Yuran teleported home."
    result = parse_clause_group(line)
    assert isinstance(result, GrammarMiss)


# ---------------------------------------------------------------------------
# Taxonomy coverage: rule-table sets == the frozen schema enums
# ---------------------------------------------------------------------------


def test_primary_rules_cover_all_17_outcomes():
    schema = load_schema()
    enum = set(schema["$defs"]["outcome"]["properties"]["type"]["enum"])
    assert len(enum) == 17
    covered = {outcome_type for _regex, outcome_type, _extractor in PRIMARY_RULES}
    assert covered == enum


def test_runner_rules_cover_all_12_causes():
    schema = load_schema()
    enum = set(schema["$defs"]["runner"]["properties"]["cause"]["enum"])
    assert len(enum) == 12
    covered: set = set()
    for _regex, causes, _builder in RUNNER_RULES:
        covered |= set(causes)
    covered |= {cause for _regex, cause, _builder in STANDALONE_RULES if cause}
    covered |= {cause for cause, _dest, _out, _scored in BATTER_OUTCOME_CAUSE.values()}
    assert covered == enum


# ---------------------------------------------------------------------------
# Pitches: None (never "") for count-only PAs
# ---------------------------------------------------------------------------


def test_pitches_none_for_count_only_pa():
    line = "Cuba Bess singled to center field (0-0)."
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    assert result.primary.pitches is None
    assert result.primary.count.balls == 0
    assert result.primary.count.strikes == 0


def test_pitches_string_when_sequence_present():
    line = "Isaac Nunez singled to left field (1-1 BS)."
    result = parse_clause_group(line)
    assert result.primary.pitches == "BS"


# ---------------------------------------------------------------------------
# Standalone clause-group kinds
# ---------------------------------------------------------------------------


def test_standalone_failed_pickoff():
    result = parse_clause_group("Isaac Nunez Failed pickoff attempt.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "runner_event"
    assert len(result.runners) == 1
    r = result.runners[0]
    assert r.name_token == "Isaac Nunez"
    assert r.cause == "pickoff"
    assert r.out is False
    assert r.scored is False


def test_standalone_stolen_base():
    result = parse_clause_group("Eddy Pelc stole second.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "runner_event"
    r = result.runners[0]
    assert r.cause == "stolen_base"
    assert r.destination == "second"


def test_standalone_substitution():
    result = parse_clause_group("Isaiah Williams to p for Chase Martinez.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "substitution"
    assert result.substitution.player_in == "Isaiah Williams"
    assert result.substitution.player_out == "Chase Martinez"


def test_standalone_inning_summary():
    line = (
        "Inning Summary: \n        \t\t\t                                    "
        "\t\t\t                            1 Runs\n                            "
        "                        ,             \t\t\t                            "
        "3 Hits\n                                                                    "
        ",             \t\t\t                            0 Errors\n                 "
        "                                                   ,             \t\t\t     "
        "                       2 LOB"
    )
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    assert result.kind == "inning_summary"
    assert result.summary.runs == 1
    assert result.summary.hits == 3
    assert result.summary.errors == 0
    assert result.summary.lob == 2


# ---------------------------------------------------------------------------
# General shapes not among the 5 resistant ones, for broad confidence
# ---------------------------------------------------------------------------


def test_double_down_the_line_location():
    line = "Jacob Jablonski doubled down the lf line, RBI (2-2 BFSB); Cuba Bess scored."
    result = parse_clause_group(line)
    p = result.primary
    assert p.outcome_type == "double"
    assert "RBI" in p.modifiers


def test_home_run():
    line = "Christian Castaneda homered to left field, RBI (1-0 B)."
    result = parse_clause_group(line)
    p = result.primary
    assert p.outcome_type == "home_run"
    assert p.location == "left field"
    assert "RBI" in p.modifiers


def test_intentional_walk_not_confused_with_plain_walk():
    line = "Jacob Jablonski was intentionally walked (3-0 BBBB)."
    result = parse_clause_group(line)
    assert result.primary.outcome_type == "intentional_walk"


def test_plain_walk():
    line = "Cooper Vest walked (3-2 BBBKKFB)."
    result = parse_clause_group(line)
    assert result.primary.outcome_type == "walk"


def test_grounded_into_double_play():
    line = (
        "Christian Castaneda grounded into double play p to 2b to 1b (0-0); "
        "Kyle Carlson out on the play; Josh Phillips advanced to third."
    )
    result = parse_clause_group(line)
    p = result.primary
    assert p.outcome_type == "grounded_into_double_play"
    assert p.fielders == ["p", "2b", "1b"]
    causes = [r.cause for r in result.runners]
    assert causes == ["putout", "advance"]


def test_fielders_choice_with_runner_out():
    line = (
        "Emilio Corona reached on a fielder's choice (1-2 KBS); "
        "Cuba Bess out at second 3b to 2b."
    )
    result = parse_clause_group(line)
    assert result.primary.outcome_type == "fielders_choice"
    assert result.runners[0].cause == "force_out"


def test_wild_pitch_and_balk_and_passed_ball_causes():
    assert parse_clause_group(
        "Jordan Donahue advanced to third on a wild pitch."
    ).runners[0].cause == "wild_pitch"
    assert parse_clause_group(
        "Jordan Donahue advanced to second on a balk."
    ).runners[0].cause == "balk"
    assert parse_clause_group(
        "Anthony Mata advanced to second on a passed ball, advanced to third."
    ).runners[0].cause == "passed_ball"


def test_compound_double_advance_second_movement_is_plain_advance():
    line = "Anthony Mata advanced to second on a passed ball, advanced to third."
    result = parse_clause_group(line)
    assert len(result.runners) == 2
    assert result.runners[0].cause == "passed_ball"
    assert result.runners[0].destination == "second"
    assert result.runners[1].cause == "advance"
    assert result.runners[1].destination == "third"


def test_scored_on_wild_pitch_clause():
    line = (
        "Cuba Bess advanced to second; Cooper Vest advanced to third; "
        "Patrick Roche Jr. scored on a wild pitch."
    )
    result = parse_clause_group(line)
    last = result.runners[-1]
    assert last.cause == "wild_pitch"
    assert last.scored is True


def test_triple_rule_exists_even_though_absent_from_sample():
    line = "Some Player tripled to left center (1-0 B)."
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    assert result.primary.outcome_type == "triple"
    assert result.primary.location == "left center"


def test_caught_stealing_rule_exists_even_though_absent_from_sample():
    line = "Some Runner caught stealing second."
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    assert result.runners[0].cause == "caught_stealing"
    assert result.runners[0].out is True


def test_sacrifice_bunt_out_at_first():
    line = (
        "Cam Yuran out at first p to 2b, SAC (0-0); "
        "Josh Lopez advanced to second; Jackson Mayo advanced to third."
    )
    result = parse_clause_group(line)
    p = result.primary
    assert p.outcome_type == "sacrifice"
    assert p.fielders == ["p", "2b"]
    assert "SAC" in p.modifiers


def test_strikeout_looking():
    line = "Johnny Pappas struck out looking (0-2 KFK)."
    result = parse_clause_group(line)
    assert result.primary.outcome_type == "strikeout_looking"


def test_flyout_sac_rbi_modifiers():
    line = "Cooper Vest flied out to cf, SAC, RBI (2-2 FSBB); Anthony Mata scored."
    result = parse_clause_group(line)
    p = result.primary
    assert p.outcome_type == "flyout"
    assert p.fielders == ["cf"]
    assert "SAC" in p.modifiers and "RBI" in p.modifiers


def test_lineout_and_popout():
    lo = parse_clause_group("Josh Lopez lined out to 2b (0-0).")
    assert lo.primary.outcome_type == "lineout"
    po = parse_clause_group("Jordan Donahue popped up to 2b (0-1 F).")
    assert po.primary.outcome_type == "popout"


# ---------------------------------------------------------------------------
# Family 1 -- count-tail-optional PRIMARY_RULES fallback (issue #30 g1).
# Real corpus lines where the primary clause has NO "(balls-strikes ...)"
# tail at all; PRIMARY_RULES is now tried directly against the bare clause,
# emitting count=None, pitches=None, rather than only trying RUNNER_RULES.
# ---------------------------------------------------------------------------


def test_count_tail_optional_walked():
    # verbatim games/2025/20250522_o568.json unparsed[] entry
    result = parse_clause_group("E. Scavotto walked; M. Piotrowsk advanced to second")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "plate_appearance"
    p = result.primary
    assert p.outcome_type == "walk"
    assert p.name_token == "E. Scavotto"
    assert p.count is None
    assert p.pitches is None
    assert len(result.runners) == 1
    assert result.runners[0].name_token == "M. Piotrowsk"


def test_count_tail_optional_struck_out_swinging():
    # verbatim games/2025/20250520_yrgi.json unparsed[] entry (trailing
    # whitespace/tab-run before the "(1 out)" trailer, per Family 3).
    line = (
        "C. Booth struck out swinging.\n"
        "                                                                "
        "                                                \t\t\t\t\t(1 out)"
    )
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    p = result.primary
    assert p.outcome_type == "strikeout_swinging"
    assert p.count is None
    assert p.pitches is None
    assert result.trailing_outs == 1


def test_count_tail_optional_flied_out_to_cf():
    # verbatim games/2025/20250520_4bkm.json unparsed[] entry.
    line = (
        "T. Specht flied out to cf.\n"
        "                                                                "
        "                                                \t\t\t\t\t(3 out)"
    )
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    p = result.primary
    assert p.outcome_type == "flyout"
    assert p.fielders == ["cf"]
    assert p.count is None
    assert result.trailing_outs == 3


def test_count_tail_optional_singled_to_center_field():
    # verbatim games/2025/20250520_4bkm.json unparsed[] entry.
    result = parse_clause_group("R. Kuntz singled to center field.")
    assert isinstance(result, ClauseGroup)
    p = result.primary
    assert p.outcome_type == "single"
    assert p.location == "center field"
    assert p.count is None
    assert p.pitches is None


def test_count_tail_optional_falls_back_to_runner_only_when_no_primary_matches():
    # No count-tail AND no PRIMARY_RULES row matches -> still falls back to
    # the runner-only path exactly as before this gate's change.
    result = parse_clause_group("Isaac Nunez Failed pickoff attempt.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "runner_event"


def test_count_tail_optional_still_misses_when_nothing_matches():
    result = parse_clause_group("The umpire ordered a rain delay for weather.")
    assert isinstance(result, GrammarMiss)


# ---------------------------------------------------------------------------
# Family 2 -- STANDALONE_RULES rows: pinch-run substitution, and (schema
# 1.2.0, issue #30 g2b) the bare DH-slot-entry "<name> to dh." shape, now
# that substitution.player_out is nullable.
# ---------------------------------------------------------------------------


def test_standalone_pinch_run_substitution():
    # verbatim games/2026/20260519_0ibc.json unparsed[] entry.
    result = parse_clause_group("Bodee Wright pinch ran for Pat Mills.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "substitution"
    assert result.substitution.player_in == "Bodee Wright"
    assert result.substitution.player_out == "Pat Mills"
    assert result.substitution.kind == "offensive"


def test_standalone_pitching_substitution_kind_still_pitching():
    # Regression: the pre-existing pitching-sub row's `kind` is now an
    # explicit field rather than an absent one -- must stay "pitching".
    result = parse_clause_group("Isaiah Williams to p for Chase Martinez.")
    assert result.substitution.kind == "pitching"


def test_dh_slot_bare_shape_parses_to_a_substitution_with_player_out_none():
    # Schema 1.2.0 (issue #30) made substitution.player_out nullable, so the
    # bare DH-slot-entry line -- naming only the incoming player -- is now a
    # real offensive substitution event, never a guessed outgoing player.
    # verbatim games/2026/20260519_0ibc.json unparsed[] entry (pre-fix).
    result = parse_clause_group("Cole Robinson to dh.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "substitution"
    assert result.substitution.player_in == "Cole Robinson"
    assert result.substitution.player_out is None
    assert result.substitution.kind == "offensive"


# ---------------------------------------------------------------------------
# Family 9 (issue #31, g3) -- substitution/position-move grammar: the
# two-name "<in> to <pos> for <out>." shape generalized to any fielding
# position (including #32's "to dh for"), a standalone pinch-hit row, and a
# guarded bare "<name> to <pos>." position-move row. `kind` is asserted for
# every shape -- the prior hardcoded kind="pitching" would ship silently
# wrong for anything but a pitching change, and nothing else catches that.
# ---------------------------------------------------------------------------


def test_two_name_position_sub_defensive_kind():
    # verbatim games/**/*.json unparsed[] entry shape: 'B. Lada to ss for
    # B. Marine.' -- a non-p/non-dh fielding position -> kind="defensive".
    result = parse_clause_group("B. Lada to ss for B. Marine.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "substitution"
    assert result.substitution.player_in == "B. Lada"
    assert result.substitution.player_out == "B. Marine"
    assert result.substitution.kind == "defensive"


def test_two_name_position_sub_second_base_defensive_kind():
    # Second real-corpus position, to guard against an off-by-one in the
    # position alternation (only testing "ss" would not catch a regex that
    # accidentally only covers one token).
    result = parse_clause_group("J. Leslie to 2b for B. Lada.")
    assert result.substitution.kind == "defensive"


def test_two_name_position_sub_pitching_kind_unchanged():
    # Regression: the original p-only shape must still emit kind="pitching"
    # after the regex is generalized to accept other positions.
    result = parse_clause_group("Isaiah Williams to p for Chase Martinez.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "substitution"
    assert result.substitution.kind == "pitching"


def test_two_name_dh_sub_is_issue_32_offensive_kind():
    # Issue #32: the two-name DH sub ('<in> to dh for <out>.') was previously
    # an intentional GrammarMiss (see the pre-fix regression test this one
    # replaces). It is now covered by the same generalized _SUBSTITUTION_RE
    # as every other position, with kind="offensive" (the DH is a
    # batting-lineup slot, not a fielding position -- the #30 bare "to dh"
    # convention). verbatim games/2024/20240524_91ql.json unparsed[] entry.
    result = parse_clause_group("P. DePasqual to dh for J. Impedugli.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "substitution"
    assert result.substitution.player_in == "P. DePasqual"
    assert result.substitution.player_out == "J. Impedugli"
    assert result.substitution.kind == "offensive"


def test_standalone_pinch_hit_substitution():
    # verbatim games/**/*.json unparsed[] entry shape: 'S. Wilmer pinch hit
    # for B. Hancock.'
    result = parse_clause_group("S. Wilmer pinch hit for B. Hancock.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "substitution"
    assert result.substitution.player_in == "S. Wilmer"
    assert result.substitution.player_out == "B. Hancock"
    assert result.substitution.kind == "offensive"


def test_bare_position_move_is_defensive_kind_player_out_none():
    # verbatim games/**/*.json unparsed[] entry shape: 'D. Sackett to 3b.'
    result = parse_clause_group("D. Sackett to 3b.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "substitution"
    assert result.substitution.player_in == "D. Sackett"
    assert result.substitution.player_out is None
    assert result.substitution.kind == "defensive"


def test_bare_position_move_second_position():
    # verbatim games/**/*.json unparsed[] entry shape: 'B. Trammell to lf.'
    result = parse_clause_group("B. Trammell to lf.")
    assert result.substitution.player_in == "B. Trammell"
    assert result.substitution.kind == "defensive"


def test_bare_position_move_ordered_after_dh_slot_bare():
    # "to dh." must still route through _DH_SLOT_BARE_RE (kind="offensive"),
    # never the new bare-position-move row (which excludes "dh" from its own
    # position alternation, but this also proves ordering never matters --
    # the two rows are already mutually exclusive by token set).
    result = parse_clause_group("Cole Robinson to dh.")
    assert result.substitution.kind == "offensive"


def test_bare_position_move_does_not_swallow_a_fielding_assist_chain():
    # CRITICAL false-positive guard (class 1): a naive ".+? to <pos>\.?$"
    # bare-move regex would catastrophically misparse a real multi-clause
    # runner-event line whose trailing clause is a fielding ASSIST CHAIN
    # ("3b to c" means a throw from third base to the catcher, not a
    # substitution). This exact line is verbatim real-corpus shape
    # (games/**/*.json unparsed[]): semicolon-joined clauses ending "... out
    # at home 3b to c." Still a GrammarMiss (this shape family is not part
    # of this gate's scope) -- the point of this test is that it must NEVER
    # become a false substitution.
    line = (
        "A. Davis reached on a fielder's choice; P. Harden advanced to "
        "second; M. Jefferson advanced to third; B. Burckel out at home "
        "3b to c."
    )
    result = parse_clause_group(line)
    assert not (isinstance(result, ClauseGroup) and result.kind == "substitution")


def test_bare_position_move_does_not_swallow_a_flyout_to_position():
    # CRITICAL false-positive guard (class 2, found while implementing this
    # gate -- the semicolon guard above does NOT catch this class): a
    # single-clause plate-appearance line ending "... to <pos>." is
    # structurally identical to a genuine bare position-move's tail. Without
    # the Title-Case name-token guard, "T. Specht flied out to cf." would be
    # misparsed as a substitution (name="T. Specht flied out", pos="cf"),
    # silently stealing this line from PRIMARY_RULES' pre-existing flyout
    # row. Regression-guards test_count_tail_optional_flied_out_to_cf and
    # test_popped_out_to (both pre-existing, Family/1 tests) against this
    # exact collision.
    result = parse_clause_group("T. Specht flied out to cf.")
    assert result.kind == "plate_appearance"
    assert result.primary.outcome_type == "flyout"


def test_bare_position_move_handles_real_corpus_name_edge_cases():
    # Real corpus names include a comma+suffix ("Last, Jr"), a parenthetical
    # nickname, and a curly-quote apostrophe -- none of which a naive
    # [A-Z][\w'-]* per-token char class would accept. Confirms the
    # broadened _NAME_TOKEN character class recovers all of them.
    for line, expected_name in [
        ("Allen, Jr to c.", "Allen, Jr"),
        ("Charles (CJ) Dean to 1b.", "Charles (CJ) Dean"),
        ("Michael O’Hara to cf.", "Michael O’Hara"),
    ]:
        result = parse_clause_group(line)
        assert isinstance(result, ClauseGroup), line
        assert result.kind == "substitution"
        assert result.substitution.player_in == expected_name
        assert result.substitution.kind == "defensive"


# ---------------------------------------------------------------------------
# Family 3 -- extended `singled` coverage: bare, up the middle, through the
# (left|right) side.
# ---------------------------------------------------------------------------


def test_singled_bare_no_location():
    # verbatim-shape games/2026 unparsed[] entry (real corpus: 'Kyle Schmack
    # singled (0-0).').
    result = parse_clause_group("Kyle Schmack singled (0-0).")
    p = result.primary
    assert p.outcome_type == "single"
    assert p.location is None
    assert p.count.balls == 0 and p.count.strikes == 0


def test_singled_up_the_middle():
    # verbatim games/2026/20260519_dpzk.json unparsed[] entry.
    result = parse_clause_group("Kyle Schmack singled up the middle (2-0 BB).")
    p = result.primary
    assert p.outcome_type == "single"
    assert p.location == "up the middle"
    assert p.count.balls == 2 and p.count.strikes == 0
    assert p.pitches == "BB"


def test_singled_through_the_right_side():
    # verbatim games/2026/20260604_427j.json (or sibling) unparsed[] entry.
    result = parse_clause_group(
        "Garret Ostrander singled through the right side (2-2 BKFB)."
    )
    p = result.primary
    assert p.outcome_type == "single"
    assert p.location == "right side"


def test_singled_through_the_left_side():
    result = parse_clause_group("K. Dugan singled through the left side, RBI.")
    p = result.primary
    assert p.outcome_type == "single"
    assert p.location == "left side"
    assert "RBI" in p.modifiers
    # No count-tail on this real corpus shape -- exercises Family 1 too.
    assert p.count is None


def test_singled_up_the_middle_no_count_tail_at_all():
    # verbatim games/2024/20240521_7sf7.json unparsed[] entry -- exercises
    # Family 1 (no count-tail) and Family 3 (up the middle) together.
    result = parse_clause_group("B. Blackford singled up the middle.")
    p = result.primary
    assert p.outcome_type == "single"
    assert p.location == "up the middle"
    assert p.count is None


# ---------------------------------------------------------------------------
# Family 4 -- trailing whitespace/tab-run tolerance before fullmatch
# anchoring, verified beyond the trailing "(N out)" trailer already covered
# above (test_count_tail_optional_struck_out_swinging /
# test_count_tail_optional_flied_out_to_cf).
# ---------------------------------------------------------------------------


def test_whitespace_tab_runs_do_not_break_matching_and_narrative_is_untouched():
    line = (
        "Josh Phillips grounded out to 1b unassisted (0-1 K).\n"
        "            \t\t\t\t(2 out)"
    )
    result = parse_clause_group(line)
    assert isinstance(result, ClauseGroup)
    assert result.primary.outcome_type == "groundout"
    assert result.trailing_outs == 2
    # GrammarMiss.raw always preserves the exact original text -- proven
    # here via the nonsense-line miss test elsewhere; here we additionally
    # confirm a WHITESPACE-only variant of an already-passing line changes
    # nothing about the extracted fields (matching tolerance only).
    clean = parse_clause_group(
        "Josh Phillips grounded out to 1b unassisted (0-1 K). (2 out)"
    )
    assert result.primary == clean.primary
    assert result.trailing_outs == clean.trailing_outs


def test_grammar_miss_preserves_verbatim_raw_with_embedded_whitespace():
    line = "Some nonsense.\n            \t\t\t\t(1 out)"
    result = parse_clause_group(line)
    assert isinstance(result, GrammarMiss)
    assert result.raw == line


# ---------------------------------------------------------------------------
# Real-sample coverage: every <td class="text"> cell of the archived final
# boxscore must parse -- 0 GrammarMiss.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Family 5 (issue #31, g2) -- bare long hits (no location), a generic
# closed-alternation multi-modifier tail (RBI / N RBI / bunt / SAC /
# ground-rule / unearned) on single/double/triple/home_run, and "down the
# X line" location wording extended to triple/home_run (double already had
# it) -- all confirmed against real corpus unparsed[] entries.
# ---------------------------------------------------------------------------


def test_bare_doubled_no_location():
    # verbatim games/2026 unparsed[] entry: "A. Fogel doubled."
    result = parse_clause_group("A. Fogel doubled.")
    p = result.primary
    assert p.outcome_type == "double"
    assert p.location is None
    assert p.modifiers == []


def test_bare_tripled_no_location():
    result = parse_clause_group("C. Thompson tripled")
    p = result.primary
    assert p.outcome_type == "triple"
    assert p.location is None


def test_bare_homered_no_location():
    result = parse_clause_group("C. Thompson homered")
    p = result.primary
    assert p.outcome_type == "home_run"
    assert p.location is None


def test_home_run_with_location_and_multi_rbi():
    # verbatim games/2025 unparsed[] entry: "M. Moralez homered to left
    # field, 2 RBI". N-RBI carries the literal "2 RBI" token for fidelity
    # PLUS a bare "RBI" element so the existing `"RBI" in modifiers` check
    # in parse.py keeps matching regardless of N.
    result = parse_clause_group("M. Moralez homered to left field, 2 RBI")
    p = result.primary
    assert p.outcome_type == "home_run"
    assert p.location == "left field"
    assert "RBI" in p.modifiers
    assert "2 RBI" in p.modifiers


def test_home_run_bare_multi_rbi_no_location():
    # verbatim games/2025 unparsed[] entry: "J. Jablonski homered, 2 RBI".
    result = parse_clause_group("J. Jablonski homered, 2 RBI")
    p = result.primary
    assert p.outcome_type == "home_run"
    assert p.location is None
    assert "RBI" in p.modifiers
    assert "2 RBI" in p.modifiers


def test_home_run_four_rbi_grand_slam():
    # verbatim games/2025 unparsed[] entry: "... homered to catcher, 4 RBI".
    result = parse_clause_group("Z. Player homered to catcher, 4 RBI")
    p = result.primary
    assert p.outcome_type == "home_run"
    assert "RBI" in p.modifiers
    assert "4 RBI" in p.modifiers


def test_single_multi_rbi_with_location():
    # verbatim games/2025 unparsed[] entry: "K. Willman singled, 2 RBI" /
    # "E. McCabe singled up the middle, 2 RBI" / "J. Day singled to left
    # field, 2 RBI".
    bare = parse_clause_group("K. Willman singled, 2 RBI")
    assert bare.primary.outcome_type == "single"
    assert "RBI" in bare.primary.modifiers and "2 RBI" in bare.primary.modifiers

    middle = parse_clause_group("E. McCabe singled up the middle, 2 RBI")
    assert middle.primary.location == "up the middle"
    assert "RBI" in middle.primary.modifiers

    loc = parse_clause_group("J. Day singled to left field, 2 RBI")
    assert loc.primary.location == "left field"
    assert "RBI" in loc.primary.modifiers and "2 RBI" in loc.primary.modifiers


def test_walked_rbi_still_not_confused_with_hit_modifier_tail():
    # A walk's modifier tail is intentionally NOT the generic hit alternation
    # (a walk can never carry SAC/bunt/ground-rule) -- covered separately in
    # Family 6 below; this test only guards single's own tail doesn't regress.
    result = parse_clause_group("K. Dugan singled through the left side, RBI.")
    assert "RBI" in result.primary.modifiers


def test_singled_bunt_modifier():
    # verbatim games corpus: "B. Skinner singled, bunt."
    result = parse_clause_group("B. Skinner singled, bunt.")
    p = result.primary
    assert p.outcome_type == "single"
    assert p.modifiers == ["bunt"]


def test_singled_location_bunt_and_rbi_combined():
    # verbatim games corpus: "... singled to first base, bunt, RBI".
    result = parse_clause_group("N. Player singled to first base, bunt, RBI")
    p = result.primary
    assert p.outcome_type == "single"
    assert p.location == "first base"
    assert p.modifiers == ["bunt", "RBI"]


def test_doubled_ground_rule_modifier():
    # verbatim games corpus: "... doubled down the lf line, ground-rule".
    result = parse_clause_group("N. Player doubled down the lf line, ground-rule")
    p = result.primary
    assert p.outcome_type == "double"
    assert p.location == "the lf line"
    assert "ground-rule" in p.modifiers


def test_doubled_ground_rule_then_multi_rbi():
    # verbatim games corpus: "... doubled down the rf line, ground-rule, 2
    # RBI" -- modifier ORDER as narrated (ground-rule before RBI) preserved.
    result = parse_clause_group(
        "N. Player doubled down the rf line, ground-rule, 2 RBI"
    )
    p = result.primary
    assert p.modifiers == ["ground-rule", "2 RBI", "RBI"]


def test_tripled_down_the_line_location():
    # verbatim games corpus: "... tripled down the lf line, RBI" -- triple
    # previously only accepted "to LOC", never "down the X line".
    result = parse_clause_group("N. Player tripled down the lf line, RBI")
    p = result.primary
    assert p.outcome_type == "triple"
    assert p.location == "the lf line"
    assert "RBI" in p.modifiers


def test_home_run_down_the_line_location():
    # verbatim games corpus: "... homered down the lf line, 3 RBI".
    result = parse_clause_group("N. Player homered down the lf line, 3 RBI")
    p = result.primary
    assert p.outcome_type == "home_run"
    assert p.location == "the lf line"
    assert "3 RBI" in p.modifiers


def test_home_run_unearned_then_rbi_order_variant():
    # verbatim games corpus shows BOTH orderings in the wild: "unearned, 3
    # RBI" and "3 RBI, unearned" -- the closed alternation must accept both.
    a = parse_clause_group("N. Player homered to left field, unearned, 3 RBI")
    assert a.primary.modifiers == ["unearned", "3 RBI", "RBI"]
    b = parse_clause_group("N. Player homered to center field, 3 RBI, unearned")
    assert b.primary.modifiers == ["3 RBI", "RBI", "unearned"]


# ---------------------------------------------------------------------------
# Family 6 (issue #31, g2) -- walked, RBI (bare walk stays untouched) and a
# NEW "popped out to <pos>" row (ADDED alongside the existing "popped up to"
# row, never merged into it) -- confirmed against real corpus unparsed[]
# entries.
# ---------------------------------------------------------------------------


def test_walked_rbi():
    # verbatim games corpus: "C. Brady walked, RBI; E. Diaz advanced to
    # second; G. Kueber advanced to third; K. Santiago scored."
    result = parse_clause_group(
        "C. Brady walked, RBI; E. Diaz advanced to second; "
        "G. Kueber advanced to third; K. Santiago scored."
    )
    p = result.primary
    assert p.outcome_type == "walk"
    assert p.modifiers == ["RBI"]
    assert len(result.runners) == 3


def test_walked_rbi_with_count_tail():
    # verbatim games corpus: "Darryl Jackson walked, RBI (3-2 BBBFFB)".
    result = parse_clause_group("Darryl Jackson walked, RBI (3-2 BBBFFB)")
    p = result.primary
    assert p.outcome_type == "walk"
    assert p.modifiers == ["RBI"]
    assert p.count.balls == 3 and p.count.strikes == 2


def test_bare_walk_still_no_modifiers():
    # Regression: the pre-existing bare-walk shape must stay unaffected by
    # the broadened row.
    result = parse_clause_group("Cooper Vest walked (3-2 BBBKKFB).")
    p = result.primary
    assert p.outcome_type == "walk"
    assert p.modifiers == []


def test_popped_out_to():
    # verbatim games corpus: "L. Barns popped out to 1b."
    result = parse_clause_group("L. Barns popped out to 1b.")
    p = result.primary
    assert p.outcome_type == "popout"
    assert p.fielders == ["1b"]
    assert p.modifiers == []


def test_popped_out_to_bunt():
    # verbatim games corpus: "C. Villafuer popped out to c, bunt."
    result = parse_clause_group("C. Villafuer popped out to c, bunt.")
    p = result.primary
    assert p.outcome_type == "popout"
    assert p.fielders == ["c"]
    assert p.modifiers == ["bunt"]


def test_popped_up_to_row_still_unaffected():
    # Regression: the pre-existing "popped up to" row/wording must stay
    # exactly as it was -- the new row is an ADDITION, not a merge.
    result = parse_clause_group("Jordan Donahue popped up to 2b (0-1 F).")
    assert result.primary.outcome_type == "popout"
    assert result.primary.modifiers == []


# ---------------------------------------------------------------------------
# Family 7 (issue #31, g2) -- bare "NAME out at first CHAIN" (batter thrown
# out, no sacrifice comma) -> groundout, and the "reached first on a
# fielding/throwing error by X" wording variants of the existing
# reached_on_error row.
# ---------------------------------------------------------------------------


def test_bare_out_at_first_chain():
    # verbatim games/2025 unparsed[] entry: "A. Fernandez out at first 1b to
    # ss; H. Hall advanced to second on a fielder's choice; J. Lynch
    # advanced to third." -- the "on a fielder's choice" runner clause is a
    # separate, unrelated gap (out of scope here), so exercise the primary
    # alone via a line whose runner clauses are already-supported shapes.
    result = parse_clause_group(
        "J. Kalafut out at first 1b to p; T. Darden advanced to second; "
        "N. Marcello advanced to third."
    )
    assert isinstance(result, ClauseGroup)
    p = result.primary
    assert p.outcome_type == "groundout"
    assert p.fielders == ["1b", "p"]
    assert len(result.runners) == 2


def test_out_at_first_sac_row_still_wins_when_sac_present():
    # Regression: the existing "out at first CHAIN, SAC" sacrifice row must
    # still win -- the new bare row must never shadow it.
    line = (
        "Cam Yuran out at first p to 2b, SAC (0-0); "
        "Josh Lopez advanced to second; Jackson Mayo advanced to third."
    )
    result = parse_clause_group(line)
    p = result.primary
    assert p.outcome_type == "sacrifice"
    assert "SAC" in p.modifiers


def test_out_at_first_does_not_swallow_struck_out_compound():
    # verbatim games corpus: "K. Jimenez struck out swinging, out at first c
    # to 1b." -- a DIFFERENT (dropped-third-strike) narrative shape, not one
    # of this gate's 10 target families and out of scope for this gate. The
    # new bare row's negative lookahead must NOT match this by treating
    # "K. Jimenez struck out swinging," as if it were a player name (which
    # would misfile it as a groundout plate_appearance) -- it must stay
    # exactly the SAME pre-existing (unrelated, out-of-scope) shape the
    # RUNNER_RULES "out at base" fallback already produced before this
    # gate's change: a runner_event, never a plate_appearance/groundout.
    result = parse_clause_group("K. Jimenez struck out swinging, out at first c to 1b.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "runner_event"
    assert result.primary is None


def test_out_at_first_does_not_swallow_picked_off_compound():
    # verbatim games corpus: "D. Covino picked off, out at first p to 1b to
    # ss." -- also out of scope; same non-hijack guarantee as above.
    result = parse_clause_group("D. Covino picked off, out at first p to 1b to ss.")
    assert isinstance(result, ClauseGroup)
    assert result.kind == "runner_event"
    assert result.primary is None


def test_reached_first_on_a_fielding_error():
    # verbatim games corpus: "T. Lomack reached first on a fielding error by
    # 2b; B. Evans advanced to second; L. Fennelly advanced to third."
    result = parse_clause_group(
        "T. Lomack reached first on a fielding error by 2b; "
        "B. Evans advanced to second; L. Fennelly advanced to third."
    )
    p = result.primary
    assert p.outcome_type == "reached_on_error"
    assert p.fielders == ["2b"]
    assert "error" in p.modifiers
    assert len(result.runners) == 2


def test_reached_first_on_a_throwing_error():
    # verbatim games corpus: "A. Fernandez reached first on a throwing error
    # by ss."
    result = parse_clause_group("A. Fernandez reached first on a throwing error by ss.")
    p = result.primary
    assert p.outcome_type == "reached_on_error"
    assert p.fielders == ["ss"]
    assert "error" in p.modifiers


def test_reached_first_on_a_fielding_error_rbi_tail():
    # verbatim games corpus: "R. Major reached first on a fielding error by
    # 2b, RBI; ..." -- the pre-existing wildcard tail-capture behavior is
    # UNCHANGED, only the error-phrase wording is broadened.
    result = parse_clause_group(
        "R. Major reached first on a fielding error by 2b, RBI; "
        "R. Gonzalez advanced to second."
    )
    p = result.primary
    assert p.outcome_type == "reached_on_error"
    assert "error" in p.modifiers
    assert "RBI" in p.modifiers


def test_reached_first_on_an_error_wording_still_works():
    # Regression: the pre-existing "an error" (no fielding/throwing) wording
    # must stay unaffected by the broadened alternation.
    line = (
        "Emilio Corona reached first on an error by 3b (3-2 BFKBB); "
        "Cuba Bess advanced to third."
    )
    result = parse_clause_group(line)
    p = result.primary
    assert p.outcome_type == "reached_on_error"
    assert p.fielders == ["3b"]


# ---------------------------------------------------------------------------
# Family 8 (issue #31, g2) -- "flied/lined into double play <chain>" ->
# EXISTING flyout/lineout types (never a new flied_into_double_play /
# lined_into_double_play type). Chain covers a multi-hop fielding chain, a
# single bare fielder, and a single fielder + "unassisted" -- all seen in
# the real corpus.
# ---------------------------------------------------------------------------


def test_flied_into_double_play_multi_hop_chain():
    # verbatim games corpus: "J. Lynch flied into double play ss to c;
    # X. Washingto advanced to second on the throw; M. Backstrom out on the
    # play." -- the trailing runner clause exercises the pre-existing
    # RUNNER_RULES "out on the play" row (not re-implemented here).
    result = parse_clause_group(
        "J. Lynch flied into double play ss to c; "
        "M. Backstrom out on the play."
    )
    p = result.primary
    assert p.outcome_type == "flyout"
    assert p.fielders == ["ss", "c"]
    assert p.modifiers == []
    assert result.runners[0].cause == "putout"
    assert result.runners[0].out is True


def test_flied_into_double_play_single_bare_fielder():
    # verbatim games corpus: "... flied into double play lf; ..."
    result = parse_clause_group("N. Player flied into double play lf.")
    p = result.primary
    assert p.outcome_type == "flyout"
    assert p.fielders == ["lf"]


def test_lined_into_double_play_multi_hop_chain():
    # verbatim games corpus: "M. Yonamine lined into double play 2b to ss;
    # D. Poteet out on the play."
    result = parse_clause_group(
        "M. Yonamine lined into double play 2b to ss; D. Poteet out on the play."
    )
    p = result.primary
    assert p.outcome_type == "lineout"
    assert p.fielders == ["2b", "ss"]


def test_lined_into_double_play_unassisted():
    # verbatim games corpus: "P. Howard lined into double play ss
    # unassisted; R. Gill out on the play."
    result = parse_clause_group(
        "P. Howard lined into double play ss unassisted; R. Gill out on the play."
    )
    p = result.primary
    assert p.outcome_type == "lineout"
    assert p.fielders == ["ss"]
    assert p.modifiers == ["unassisted"]


def test_grounded_into_double_play_row_still_unaffected():
    # Regression: the pre-existing "grounded into double play" row/type must
    # stay exactly as before -- this gate never touches it.
    line = (
        "Christian Castaneda grounded into double play p to 2b to 1b (0-0); "
        "Kyle Carlson out on the play; Josh Phillips advanced to third."
    )
    result = parse_clause_group(line)
    p = result.primary
    assert p.outcome_type == "grounded_into_double_play"
    assert p.fielders == ["p", "2b", "1b"]


def test_real_sample_zero_grammar_miss():
    cells = _all_sample_cells()
    assert len(cells) == 122
    misses = []
    outcome_counts: dict[str, int] = {}
    for text in cells:
        result = parse_clause_group(text)
        if isinstance(result, GrammarMiss):
            misses.append(result)
        elif result.kind == "plate_appearance":
            outcome_counts[result.primary.outcome_type] = (
                outcome_counts.get(result.primary.outcome_type, 0) + 1
            )
    assert misses == [], [m.reason for m in misses]
    assert sum(outcome_counts.values()) > 0
