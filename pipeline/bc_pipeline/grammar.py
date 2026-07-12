"""grammar -- the pure-text closed PBP clause grammar.

Turns one StatCrew play-by-play narrative *line* (a single ``<td
class="text">`` cell's verbatim text -- html_struct's job, not this
module's) into structured clause data, or a ``GrammarMiss``. PURE TEXT IN,
STRUCTURED DATA OUT: no HTML, no player-id resolution, no base-out state, no
event assembly. Those are later gates' jobs (g3/g5/g6).

The representation is a clause-splitter (peel the trailing ``(N out)``
trailer, then split the remainder on ``;`` into one PRIMARY clause plus zero
or more RUNNER clauses) feeding ORDERED regex rule tables -- never a single
mega-regex, never recursive descent. Each table row is
``(regex, outcome_type-or-cause, small extractor)``; the first row whose
regex fullmatches wins. Coverage grows by ADDING rows, never by loosening an
existing one into a catch-all.

CLOSED TAXONOMY (schema-frozen, never extended here): 17 outcome types
(``$defs.outcome.properties.type.enum``), 12 runner causes
(``$defs.runner.properties.cause.enum``). A clause the tables cannot match
returns a ``GrammarMiss`` carrying the reason and the verbatim source line,
so the caller (g5) can preserve it in ``unparsed[]`` -- never a guess, never
an exception.

Design note on ``BATTER_OUTCOME_CAUSE``: two of the 12 runner causes
(``batted_ball``, ``fielders_choice``) never appear in narrative RUNNER
clause text -- they describe the *batter's own* base-reaching movement on a
hit or a fielder's choice, which the schema's fixture shows as a runner
record synthesized from the PRIMARY outcome, not parsed from a distinct
clause. Emitting that record here (into ``ClauseGroup.runners``) would
contradict the handoff's own resistant-shape assertions (e.g. resistant
shape 1 names exactly ONE runner clause for a groundout with a force play).
So this module exposes ``BATTER_OUTCOME_CAUSE`` as a separate, static,
outcome_type -> (cause, destination, out, scored) mapping table for g5 to
consult when it assembles the full event's runners[] -- it is a real
rule table (deterministic, closed-taxonomy, no state), just not one that
``parse_clause_group`` applies to ``ClauseGroup.runners`` itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import re

# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Count:
    balls: int
    strikes: int


@dataclass(frozen=True)
class PrimaryClause:
    name_token: str
    outcome_type: str
    fielders: List[str]
    location: Optional[str]
    modifiers: List[str]
    count: Optional[Count]
    pitches: Optional[str]


@dataclass(frozen=True)
class RunnerMovement:
    name_token: str
    cause: str
    destination: Optional[str]
    out: bool
    scored: bool
    unearned: bool = False


@dataclass(frozen=True)
class InningSummary:
    runs: int
    hits: int
    errors: int
    lob: int


@dataclass(frozen=True)
class Substitution:
    player_in: str
    player_out: str
    # One of the schema's closed `substitution.kind` enum values
    # ("offensive", "defensive", "pitching") -- which side/role the
    # substitution applies to. g5 (parse.py) currently resolves every
    # substitution's names against the FIELDING side and hardcodes
    # `kind: "pitching"` on the emitted event (its only STANDALONE_RULES row
    # used to be the "<in> to p for <out>" pitching-change shape, so that was
    # always correct); wiring g5 to read this field instead, and to resolve
    # an "offensive" substitution's names against the BATTING side, is a
    # separate gate's job (identity/name-resolution call sites), not this
    # module's -- this field only carries the honest answer forward.
    kind: str


@dataclass(frozen=True)
class ClauseGroup:
    """One parsed narrative line.

    ``kind`` is one of ``plate_appearance``, ``runner_event``,
    ``substitution``, ``inning_summary``. Only the fields relevant to that
    kind are populated; the rest stay at their default (``None``/empty).
    """

    kind: str
    primary: Optional[PrimaryClause] = None
    runners: Tuple[RunnerMovement, ...] = ()
    trailing_outs: Optional[int] = None
    summary: Optional[InningSummary] = None
    substitution: Optional[Substitution] = None


@dataclass(frozen=True)
class GrammarMiss:
    """An unrecognized clause. Never a guess, never an exception."""

    raw: str
    reason: str


# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------

# The trailing "(N out)" trailer sits at the very end of the cell text,
# separated from the sentence by an arbitrary run of whitespace (StatCrew
# renders it via CSS layout, not narrative prose) -- DOTALL so "." in the
# body can span the embedded newlines/tabs.
_TRAILING_OUT_RE = re.compile(
    r"^(?P<body>.*\S)\s*\(\s*(?P<n>\d+)\s+out\)\s*$", re.DOTALL
)

# The primary clause's own trailing "(balls-strikes [pitchseq])" parenthetical.
# pitches is None (not "") when the letter-sequence group doesn't participate --
# StatCrew omits it for a first-pitch ball in play.
_COUNT_TAIL_RE = re.compile(
    r"^(?P<rest>.+?)\s*\((?P<balls>\d+)-(?P<strikes>\d+)"
    r"(?:\s+(?P<pitches>[BFKSH]+))?\)$"
)

_CAUSEPHRASE = {
    "wild pitch": "wild_pitch",
    "passed ball": "passed_ball",
    "balk": "balk",
}
_DEST_ALT = r"(?:second|third|home)"

# Any run of plain spaces/tabs/carriage-returns/newlines, for the MATCHING
# path only (see _normalize_ws).
_WS_RUN_RE = re.compile(r"[ \t\r\n]+")


def _normalize_ws(text: str) -> str:
    """Collapse any run of whitespace/tab/newline characters to a single
    space and strip the ends, for use ONLY on the text fed to a rule table's
    ``fullmatch`` -- never on the verbatim line stored in ``GrammarMiss.raw``
    or surfaced downstream as the event's narrative (that always comes from
    the caller's own untouched copy of the original line, never from this
    module's internal working copy). StatCrew renders trailing "(N out)"
    trailers (and, on some rows, the inter-clause boundary) via CSS layout
    padding rather than narrative prose, so a run of tabs/newlines there is
    layout noise, not meaningful content -- collapsing it to one space never
    changes what a rule table's regex needs to see.
    """
    return _WS_RUN_RE.sub(" ", text).strip()


def _modifiers_from_tail(tail: str) -> List[str]:
    """Extract comma/space-separated modifier tokens from a verb-phrase tail.

    Handles both "X unassisted" (no comma) and ", RBI" / ", SAC, RBI" (comma
    separated) shapes uniformly.
    """
    tail = tail.strip()
    if tail.startswith(","):
        tail = tail[1:]
    tail = tail.strip()
    if not tail:
        return []
    return [t.strip() for t in tail.split(",") if t.strip()]


def _split_chain(chain: str) -> List[str]:
    return [tok for tok in chain.split(" to ") if tok]


# ---------------------------------------------------------------------------
# PRIMARY_RULES -- ordered (regex, outcome_type, extractor) rows.
# Each regex is matched (fullmatch) against the primary clause text with its
# trailing "(balls-strikes ...)" already stripped off. Extractor takes the
# match and returns (name_token, fielders, location, modifiers).
# ---------------------------------------------------------------------------

Extractor = Callable[[re.Match], Tuple[str, List[str], Optional[str], List[str]]]
PrimaryRule = Tuple[re.Pattern, str, Extractor]


def _x_sacrifice(m: re.Match):
    return (m.group("name"), _split_chain(m.group("chain")), None, ["SAC"])


def _x_grounded_into_double_play(m: re.Match):
    return (m.group("name"), _split_chain(m.group("chain")), None, [])


def _x_groundout_chain(m: re.Match):
    return (m.group("name"), _split_chain(m.group("chain")), None, [])


def _x_groundout_single(m: re.Match):
    return (
        m.group("name"),
        [m.group("f")],
        None,
        _modifiers_from_tail(m.group("tail")),
    )


def _x_flyout(m: re.Match):
    return (
        m.group("name"),
        [m.group("f")],
        None,
        _modifiers_from_tail(m.group("tail")),
    )


def _x_lineout(m: re.Match):
    return (m.group("name"), [m.group("f")], None, [])


def _x_popout(m: re.Match):
    return (m.group("name"), [m.group("f")], None, [])


def _x_fielders_choice(m: re.Match):
    return (m.group("name"), [], None, _modifiers_from_tail(m.group("tail")))


def _x_reached_on_error(m: re.Match):
    mods = ["error"] + _modifiers_from_tail(m.group("tail"))
    return (m.group("name"), [m.group("f")], None, mods)


def _x_single(m: re.Match):
    if m.group("loc") is not None:
        loc = m.group("loc")
    elif m.group("middle") is not None:
        loc = "up the middle"
    elif m.group("side") is not None:
        loc = f"{m.group('side')} side"
    else:
        loc = None
    mods = [m.group("mod")] if m.group("mod") else []
    return (m.group("name"), [], loc, mods)


def _x_double(m: re.Match):
    loc = m.group("loc") if m.group("loc") is not None else m.group("loc2")
    mods = [m.group("mod")] if m.group("mod") else []
    return (m.group("name"), [], loc, mods)


def _x_triple(m: re.Match):
    mods = [m.group("mod")] if m.group("mod") else []
    return (m.group("name"), [], m.group("loc"), mods)


def _x_home_run(m: re.Match):
    mods = [m.group("mod")] if m.group("mod") else []
    return (m.group("name"), [], m.group("loc"), mods)


def _x_walk(m: re.Match):
    return (m.group("name"), [], None, [])


def _x_intentional_walk(m: re.Match):
    return (m.group("name"), [], None, [])


def _x_hit_by_pitch(m: re.Match):
    mods = [m.group("mod")] if m.group("mod") else []
    return (m.group("name"), [], None, mods)


def _x_strikeout_swinging(m: re.Match):
    return (m.group("name"), [], None, [])


def _x_strikeout_looking(m: re.Match):
    return (m.group("name"), [], None, [])


PRIMARY_RULES: List[PrimaryRule] = [
    (
        re.compile(
            r"^(?P<name>.+?) out at first "
            r"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*), SAC$"
        ),
        "sacrifice",
        _x_sacrifice,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) grounded into double play "
            r"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)+)$"
        ),
        "grounded_into_double_play",
        _x_grounded_into_double_play,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) grounded out "
            r"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)+)$"
        ),
        "groundout",
        _x_groundout_chain,
    ),
    (
        re.compile(r"^(?P<name>.+?) grounded out to (?P<f>[a-z0-9]+)(?P<tail>.*)$"),
        "groundout",
        _x_groundout_single,
    ),
    (
        re.compile(r"^(?P<name>.+?) flied out to (?P<f>[a-z0-9]+)(?P<tail>.*)$"),
        "flyout",
        _x_flyout,
    ),
    (
        re.compile(r"^(?P<name>.+?) lined out to (?P<f>[a-z0-9]+)$"),
        "lineout",
        _x_lineout,
    ),
    (
        re.compile(r"^(?P<name>.+?) popped up to (?P<f>[a-z0-9]+)$"),
        "popout",
        _x_popout,
    ),
    (
        re.compile(r"^(?P<name>.+?) reached on a fielder's choice(?P<tail>.*)$"),
        "fielders_choice",
        _x_fielders_choice,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) reached first on an error by "
            r"(?P<f>[a-z0-9]+)(?P<tail>.*)$"
        ),
        "reached_on_error",
        _x_reached_on_error,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) singled(?:"
            r" to (?P<loc>[a-z][a-z ]*?)"
            r"|(?P<middle> up the middle)"
            r"| through the (?P<side>left|right) side"
            r")?(?:, (?P<mod>RBI))?$"
        ),
        "single",
        _x_single,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) doubled (?:to (?P<loc>[a-z][a-z ]*?)"
            r"|down (?P<loc2>[a-z][a-z ]*?))(?:, (?P<mod>RBI))?$"
        ),
        "double",
        _x_double,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) tripled to (?P<loc>[a-z][a-z ]*?)(?:, (?P<mod>RBI))?$"
        ),
        "triple",
        _x_triple,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) homered to (?P<loc>[a-z][a-z ]*?)(?:, (?P<mod>RBI))?$"
        ),
        "home_run",
        _x_home_run,
    ),
    (
        re.compile(r"^(?P<name>.+?) was intentionally walked$"),
        "intentional_walk",
        _x_intentional_walk,
    ),
    (re.compile(r"^(?P<name>.+?) walked$"), "walk", _x_walk),
    (
        re.compile(r"^(?P<name>.+?) hit by pitch(?:, (?P<mod>RBI))?$"),
        "hit_by_pitch",
        _x_hit_by_pitch,
    ),
    (
        re.compile(r"^(?P<name>.+?) struck out swinging$"),
        "strikeout_swinging",
        _x_strikeout_swinging,
    ),
    (
        re.compile(r"^(?P<name>.+?) struck out looking$"),
        "strikeout_looking",
        _x_strikeout_looking,
    ),
]


# ---------------------------------------------------------------------------
# RUNNER_RULES -- ordered (regex, cause, builder) rows for post-';' runner
# clauses (or the standalone "stole"/"caught stealing" phrasing when it
# appears mid-PA). Builder returns a single RunnerMovement OR a list of
# RunnerMovement (the one genuinely compound shape: advance-then-score).
# ---------------------------------------------------------------------------

RunnerBuilder = Callable[[re.Match], Union[RunnerMovement, List[RunnerMovement]]]
# The 2nd element is the tuple of causes THIS rule can actually emit (more
# than one for the two compound rows, which each produce two movements with
# different causes) -- used verbatim by the taxonomy-coverage test.
RunnerRule = Tuple[re.Pattern, Tuple[str, ...], RunnerBuilder]


def _b_compound_advance_scored_error(m: re.Match):
    name = m.group("name")
    return [
        RunnerMovement(
            name_token=name,
            cause="advance",
            destination=m.group("dest1"),
            out=False,
            scored=False,
        ),
        RunnerMovement(
            name_token=name,
            cause="error",
            destination="home",
            out=False,
            scored=True,
            unearned=bool(m.group("unearned")),
        ),
    ]


def _b_compound_double_advance(m: re.Match):
    name = m.group("name")
    cause1 = _CAUSEPHRASE[m.group("causephrase")]
    return [
        RunnerMovement(
            name_token=name,
            cause=cause1,
            destination=m.group("dest1"),
            out=False,
            scored=False,
        ),
        RunnerMovement(
            name_token=name,
            cause="advance",
            destination=m.group("dest2"),
            out=False,
            scored=(m.group("dest2") == "home"),
        ),
    ]


def _b_advance_on_causephrase(m: re.Match):
    cause = _CAUSEPHRASE[m.group("causephrase")]
    return RunnerMovement(
        name_token=m.group("name"),
        cause=cause,
        destination=m.group("dest"),
        out=False,
        scored=False,
    )


def _b_advance_on_error(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="error",
        destination=m.group("dest"),
        out=False,
        scored=False,
        unearned=bool(m.group("unearned")),
    )


def _b_advance_plain(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="advance",
        destination=m.group("dest"),
        out=False,
        scored=False,
    )


def _b_scored_on_causephrase(m: re.Match):
    cause = _CAUSEPHRASE[m.group("causephrase")]
    return RunnerMovement(
        name_token=m.group("name"),
        cause=cause,
        destination="home",
        out=False,
        scored=True,
    )


def _b_scored_on_error(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="error",
        destination="home",
        out=False,
        scored=True,
        unearned=bool(m.group("unearned")),
    )


def _b_scored_plain(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="advance",
        destination="home",
        out=False,
        scored=True,
        unearned=bool(m.group("unearned")),
    )


def _b_stole(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="stolen_base",
        destination=m.group("dest"),
        out=False,
        scored=False,
    )


def _b_caught_stealing(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="caught_stealing",
        destination=m.group("dest"),
        out=True,
        scored=False,
    )


def _b_out_on_the_play(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="putout",
        destination=None,
        out=True,
        scored=False,
    )


def _b_out_at_base(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="force_out",
        destination=m.group("base"),
        out=True,
        scored=False,
    )


def _b_pickoff(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="pickoff",
        destination=None,
        out=False,
        scored=False,
    )


RUNNER_RULES: List[RunnerRule] = [
    (
        re.compile(
            rf"^(?P<name>.+?) advanced to (?P<dest1>{_DEST_ALT}), "
            rf"scored on an error by (?P<f>[a-z0-9]+)"
            rf"(?:, (?P<unearned>unearned))?$"
        ),
        ("advance", "error"),
        _b_compound_advance_scored_error,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) advanced to (?P<dest1>{_DEST_ALT}) on a "
            rf"(?P<causephrase>wild pitch|passed ball|balk), "
            rf"advanced to (?P<dest2>{_DEST_ALT})$"
        ),
        ("wild_pitch", "passed_ball", "balk", "advance"),
        _b_compound_double_advance,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) advanced to (?P<dest>{_DEST_ALT}) on a "
            rf"(?P<causephrase>wild pitch|passed ball|balk)$"
        ),
        ("wild_pitch", "passed_ball", "balk"),
        _b_advance_on_causephrase,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) advanced to (?P<dest>{_DEST_ALT}) on an error by "
            rf"(?P<f>[a-z0-9]+)(?:, (?P<unearned>unearned))?$"
        ),
        ("error",),
        _b_advance_on_error,
    ),
    (
        re.compile(rf"^(?P<name>.+?) advanced to (?P<dest>{_DEST_ALT})$"),
        ("advance",),
        _b_advance_plain,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) scored on a "
            r"(?P<causephrase>wild pitch|passed ball|balk)$"
        ),
        ("wild_pitch", "passed_ball", "balk"),
        _b_scored_on_causephrase,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) scored on an error by "
            r"(?P<f>[a-z0-9]+)(?:, (?P<unearned>unearned))?$"
        ),
        ("error",),
        _b_scored_on_error,
    ),
    (
        re.compile(r"^(?P<name>.+?) scored(?:, (?P<unearned>unearned))?$"),
        ("advance",),
        _b_scored_plain,
    ),
    (
        re.compile(rf"^(?P<name>.+?) stole (?P<dest>{_DEST_ALT})$"),
        ("stolen_base",),
        _b_stole,
    ),
    (
        re.compile(rf"^(?P<name>.+?) caught stealing (?P<dest>{_DEST_ALT})$"),
        ("caught_stealing",),
        _b_caught_stealing,
    ),
    (
        re.compile(r"^(?P<name>.+?) out on the play$"),
        ("putout",),
        _b_out_on_the_play,
    ),
    (
        re.compile(r"^(?P<name>.+?) out at (?P<base>first|second|third|home)\b.*$"),
        ("force_out",),
        _b_out_at_base,
    ),
    (
        re.compile(r"^(?P<name>.+?) Failed pickoff attempt$"),
        ("pickoff",),
        _b_pickoff,
    ),
]


# ---------------------------------------------------------------------------
# STANDALONE_RULES -- whole-line shapes that are neither a PA nor a bare
# sequence of runner-movement clauses: an inning-recap line, a pitching
# substitution line ("<in> to p for <out>."), and a pinch-run substitution
# line ("<in> pinch ran for <out>."). (Whole-line runner-movement-only shapes
# -- "X stole second.", "X Failed pickoff attempt.", "X advanced to Y on a
# wild pitch." and multi-clause variants thereof -- fall out of the SAME
# RUNNER_RULES table via the no-count-tail fallback path below; they need no
# separate regex here.)
#
# NOT covered here: the bare "<name> to dh." DH-slot-entry shape (no outgoing
# player named in the text at all) -- the schema's substitution shape
# requires a non-nullable `player_out`, and this module has no honest way to
# supply one from this single line alone (see the module's stop-condition
# note near BATTER_OUTCOME_CAUSE... actually see g1's implementer result: a
# reported blocker, not implemented). The "<in> to dh for <out>." variant
# (both names present) DOES fit this table's shape but was not requested by
# this gate's authorized scope, so it is intentionally left unimplemented too.
# ---------------------------------------------------------------------------

StandaloneBuilder = Callable[[re.Match, Optional[int]], ClauseGroup]
StandaloneRule = Tuple[re.Pattern, Optional[str], StandaloneBuilder]

_INNING_SUMMARY_RE = re.compile(
    r"^Inning Summary:\s*(?P<r>\d+)\s*Runs\s*,\s*(?P<h>\d+)\s*Hits\s*,\s*"
    r"(?P<e>\d+)\s*Errors\s*,\s*(?P<lob>\d+)\s*LOB\s*$"
)
_SUBSTITUTION_RE = re.compile(r"^(?P<in>.+?) to p for (?P<out>.+?)\.?$")
_PINCH_RUN_RE = re.compile(r"^(?P<in>.+?) pinch ran for (?P<out>.+?)\.?$")


def _build_inning_summary(m: re.Match, trailing_outs: Optional[int]) -> ClauseGroup:
    return ClauseGroup(
        kind="inning_summary",
        summary=InningSummary(
            runs=int(m.group("r")),
            hits=int(m.group("h")),
            errors=int(m.group("e")),
            lob=int(m.group("lob")),
        ),
        trailing_outs=trailing_outs,
    )


def _build_substitution(m: re.Match, trailing_outs: Optional[int]) -> ClauseGroup:
    return ClauseGroup(
        kind="substitution",
        substitution=Substitution(
            player_in=m.group("in"), player_out=m.group("out"), kind="pitching"
        ),
        trailing_outs=trailing_outs,
    )


def _build_pinch_run(m: re.Match, trailing_outs: Optional[int]) -> ClauseGroup:
    return ClauseGroup(
        kind="substitution",
        substitution=Substitution(
            player_in=m.group("in"), player_out=m.group("out"), kind="offensive"
        ),
        trailing_outs=trailing_outs,
    )


STANDALONE_RULES: List[StandaloneRule] = [
    (_INNING_SUMMARY_RE, None, _build_inning_summary),
    (_SUBSTITUTION_RE, None, _build_substitution),
    (_PINCH_RUN_RE, None, _build_pinch_run),
]


# ---------------------------------------------------------------------------
# BATTER_OUTCOME_CAUSE -- static outcome_type -> (cause, destination, out,
# scored) mapping for the batter's OWN base-reaching movement. See the
# module docstring for why this lives here as data, not something
# `parse_clause_group` emits into `ClauseGroup.runners` itself.
# ---------------------------------------------------------------------------

BATTER_OUTCOME_CAUSE: Dict[str, Tuple[str, Optional[str], bool, bool]] = {
    "single": ("batted_ball", "first", False, False),
    "double": ("batted_ball", "second", False, False),
    "triple": ("batted_ball", "third", False, False),
    "home_run": ("batted_ball", "home", False, True),
    "walk": ("advance", "first", False, False),
    "intentional_walk": ("advance", "first", False, False),
    "hit_by_pitch": ("advance", "first", False, False),
    "reached_on_error": ("error", "first", False, False),
    "fielders_choice": ("fielders_choice", "first", False, False),
    "strikeout_swinging": ("putout", None, True, False),
    "strikeout_looking": ("putout", None, True, False),
    "groundout": ("putout", None, True, False),
    "flyout": ("putout", None, True, False),
    "lineout": ("putout", None, True, False),
    "popout": ("putout", None, True, False),
    "grounded_into_double_play": ("putout", None, True, False),
    "sacrifice": ("putout", None, True, False),
}


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def _match_runner_clauses(
    clauses: List[str], raw_line: str
) -> Union[List[RunnerMovement], GrammarMiss]:
    """Match each of ``clauses`` (untrimmed clause strings) against
    RUNNER_RULES in order, in a single pass. Returns the flattened list of
    ``RunnerMovement`` on full success, or a ``GrammarMiss`` (never raises)
    citing the first clause that matches no row.

    Shared by both the runner-only standalone path and the trailing runner
    clauses of a plate appearance -- one table, one matching loop, applied to
    whichever clause list the caller has.
    """
    runners: List[RunnerMovement] = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        matched = False
        for regex, _causes, builder in RUNNER_RULES:
            rm = regex.fullmatch(clause)
            if rm:
                result = builder(rm)
                if isinstance(result, list):
                    runners.extend(result)
                else:
                    runners.append(result)
                matched = True
                break
        if not matched:
            return GrammarMiss(
                raw=raw_line,
                reason=f"runner clause not recognized: {clause!r}",
            )
    return runners


def parse_clause_group(line: str) -> Union[ClauseGroup, GrammarMiss]:
    """Parse one verbatim PBP narrative line into a ``ClauseGroup``.

    Never raises on unrecognized input -- returns a ``GrammarMiss`` with a
    reason and the untouched original ``line`` instead.
    """
    raw_line = line
    m = _TRAILING_OUT_RE.fullmatch(line)
    if m:
        body = m.group("body")
        trailing_outs: Optional[int] = int(m.group("n"))
    else:
        body = line
        trailing_outs = None

    # Whitespace/tab-run normalization is for THIS matching path only --
    # `raw_line` (stored verbatim on a GrammarMiss, and never touched again
    # here) is the caller's own untouched copy; the narrative shown
    # downstream always comes from the caller's original line, never from
    # this normalized working copy.
    stripped = _normalize_ws(body)

    for regex, _label, builder in STANDALONE_RULES:
        sm = regex.fullmatch(stripped)
        if sm:
            return builder(sm, trailing_outs)

    parts = [p.strip() for p in stripped.split(";")]
    if not parts or not parts[0]:
        return GrammarMiss(raw=raw_line, reason="empty clause body")

    parts[-1] = parts[-1].rstrip(".").strip()
    primary_raw = parts[0]

    tail_m = _COUNT_TAIL_RE.fullmatch(primary_raw)
    if not tail_m:
        # No PA count-tail on the first clause. Tried in order:
        #  (a) the primary clause is still a recognized PA verb, just with no
        #      observed count at all -- StatCrew omits the WHOLE count-tail
        #      for some rows, not just the pitch-sequence letters (that case
        #      is `pitches is None` below with a real Count) -- emit
        #      count=None, pitches=None rather than mis-count it as 0-0.
        #  (b) failing that, this may still be a standalone runner-event line
        #      (e.g. "X advanced to second on a balk.", "X stole second.",
        #      "X Failed pickoff attempt.", or several such clauses chained
        #      with ';'). Every part must match a RUNNER_RULES row for this
        #      to count -- otherwise it's a genuine miss.
        primary: Optional[PrimaryClause] = None
        for regex, outcome_type, extractor in PRIMARY_RULES:
            pm = regex.fullmatch(primary_raw)
            if pm:
                name, fielders, location, modifiers = extractor(pm)
                primary = PrimaryClause(
                    name_token=name,
                    outcome_type=outcome_type,
                    fielders=fielders,
                    location=location,
                    modifiers=modifiers,
                    count=None,
                    pitches=None,
                )
                break

        if primary is not None:
            runners_or_miss = _match_runner_clauses(parts[1:], raw_line)
            if isinstance(runners_or_miss, GrammarMiss):
                return runners_or_miss
            return ClauseGroup(
                kind="plate_appearance",
                primary=primary,
                runners=tuple(runners_or_miss),
                trailing_outs=trailing_outs,
            )

        # No PRIMARY_RULES row matched the (count-tail-less) primary clause
        # either -- fall back to trying the WHOLE clause group as a bare
        # sequence of runner-movement clauses.
        runner_only = _match_runner_clauses(parts, raw_line)
        if isinstance(runner_only, GrammarMiss):
            return GrammarMiss(
                raw=raw_line,
                reason=(
                    "no count-tail on primary clause, primary verb not "
                    "recognized without a count either, and clause did not "
                    f"match any runner rule: {runner_only.reason}"
                ),
            )
        if not runner_only:
            return GrammarMiss(raw=raw_line, reason="empty clause body")
        return ClauseGroup(
            kind="runner_event",
            runners=tuple(runner_only),
            trailing_outs=trailing_outs,
        )
    rest = tail_m.group("rest").strip()
    balls = int(tail_m.group("balls"))
    strikes = int(tail_m.group("strikes"))
    pitches = tail_m.group("pitches")

    primary = None
    for regex, outcome_type, extractor in PRIMARY_RULES:
        pm = regex.fullmatch(rest)
        if pm:
            name, fielders, location, modifiers = extractor(pm)
            primary = PrimaryClause(
                name_token=name,
                outcome_type=outcome_type,
                fielders=fielders,
                location=location,
                modifiers=modifiers,
                count=Count(balls=balls, strikes=strikes),
                pitches=pitches,
            )
            break

    if primary is None:
        return GrammarMiss(
            raw=raw_line, reason=f"primary verb not recognized: {rest!r}"
        )

    runners_or_miss = _match_runner_clauses(parts[1:], raw_line)
    if isinstance(runners_or_miss, GrammarMiss):
        return runners_or_miss

    return ClauseGroup(
        kind="plate_appearance",
        primary=primary,
        runners=tuple(runners_or_miss),
        trailing_outs=trailing_outs,
    )
