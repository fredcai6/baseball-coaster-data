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

CLOSED TAXONOMY (schema-frozen, never extended here): 19 outcome types
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

from dataclasses import dataclass, field, replace
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
    #: The base of an out recorded on this play whose RUNNER the line never
    #: names -- "X reached on a fielder's choice, out at second ss to 2b".
    #: The out is real and was being silently discarded into `modifiers` by a
    #: `.*` catch-all; parse.py attributes it to the forced runner from base
    #: occupancy, or refuses (issue #40).
    forced_out_at: Optional[str] = None
    #: The fielding chain thrown to record that out, verbatim ("ss to 1b").
    #: Captured because its TERMINUS says which base the out was actually
    #: recorded at, and the source's stated base contradicts it on real
    #: lines -- "out at second ss to 1b" ends at the first baseman, who
    #: cannot record an out at second. Without this the contradiction is
    #: invisible and the out gets pinned on whoever happens to occupy the
    #: base the source misnamed (issue #33).
    forced_out_chain: Optional[str] = None


@dataclass(frozen=True)
class RunnerMovement:
    name_token: str
    cause: str
    destination: Optional[str]
    out: bool
    scored: bool
    unearned: bool = False
    #: Set ONLY when this movement is not what the source line says, but what
    #: a measured rule concluded it must mean -- see _b_source_defect_scored.
    #: Every consumer that writes a game file surfaces this in the file's
    #: top-level `inferred[]`, so no inferred fact is ever silent (issue #40).
    inferred: Optional[str] = None


@dataclass(frozen=True)
class InningSummary:
    runs: int
    hits: int
    errors: int
    lob: int


@dataclass(frozen=True)
class Substitution:
    #: None when the source line omits the incoming player entirely -- see
    #: _BLANK_SUB_RE. The caller (parse.py) is what fills it in, and records
    #: the inference; the grammar never guesses a name.
    player_in: Optional[str]
    player_out: Optional[str]
    # One of the schema's closed `substitution.kind` enum values
    # ("offensive", "defensive", "pitching") -- which side/role the
    # substitution applies to. Every STANDALONE_RULES builder assigns this
    # from the matched shape: a pitcher-position two-name/bare sub is
    # "pitching", a dh-slot two-name/bare sub is "offensive" (the DH is a
    # batting-lineup slot, not a fielding position), a pinch-hit/pinch-run is
    # "offensive", and a bare or two-name move to any other fielding position
    # is "defensive". g5 (parse.py) reads this field to resolve the
    # substitution's names against the correct side (offensive -> batting,
    # else -> fielding) -- see parse.py's substitution-assembly branch.
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
    # An illegal pitch is enforced as a balk with runners on -- the runners
    # are awarded the base either way -- so it takes the `balk` cause rather
    # than the generic `advance`, which would lose the fact that a pitching
    # infraction caused the movement. The verbatim narrative preserves the
    # scorer's own wording, so the distinction is never destroyed.
    "illegal pitch": "balk",
}
_DEST_ALT = r"(?:second|third|home)"

#: The error-cause phrase, as ONE fragment shared by every row that accepts
#: it. StatCrew writes three spellings -- "an error by F", "a throwing error
#: by F", "a fielding error by F" -- plus a bare "the error" back-reference.
#: Before issue #40 the "advanced to D on ..." rows accepted all three while
#: the "scored on ..." rows accepted only the plain one, so 78 lines across
#: 72 games failed on a variant their sibling row already handled. A single
#: fragment is what keeps the two from drifting apart again.
_ERROR_BY = (
    # "a muffed throw by 1b" is the scorer's other spelling for an error
    # charged to a fielder, and unearned-run bookkeeping treats it
    # identically. The spellings are alternated INSIDE the "... by <f>" shape
    # so they share one fielder group -- a second capture name would have to
    # be read by every builder that reads this fragment, which is how the
    # spelling lists diverged in the first place.
    r"(?:(?:an? (?:throwing |fielding )?error|a muffed throw) by (?P<f>[a-z0-9]+)"
    r"|the error)"
    # ", assist by c" -- the fielder who threw to the one who erred. Part of
    # the error PHRASE, so it belongs in this shared fragment rather than on
    # any single row: the corpus writes it after "reached first on a fielding
    # error by 1b" AND after "advanced to second on an error by 2b", and a
    # row-local copy would accept it on whichever row happened to be edited
    # and reject its sibling. That divergence is this file's recurring bug
    # (see the handoff's "one list in two places").
    r"(?:, assist by (?P<assist>[a-z0-9]+))?"
)

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


# A closed alternation of the modifier tokens a hit-type hit_run_batted_ball
# verb (single/double/triple/home_run) can carry, in StatCrew's own
# comma-separated tail -- "RBI, N RBI, bunt, SAC, ground-rule, unearned" per
# the real corpus (issue #31 g2). Order matters: "\d+ RBI" must be tried
# before the bare "RBI" alternative so a token like "2 RBI" isn't split into
# an unmatched "2" plus a bare "RBI". NEVER a `.*` catch-all -- an unlisted
# token (e.g. the batter's own trailing self-advance clause, "advanced to
# second on an error by X") deliberately does NOT match here, so a line
# carrying one stays a clean GrammarMiss rather than silently discarding
# structured movement data into a junk modifier string.
#: StatCrew writes both "unearned" and "team unearned" -- the latter means
#: the run is unearned for the TEAM though charged to the pitcher. Both are
#: the same fact for this schema's boolean, so both are accepted.
_UNEARNED_TAIL = r"(?:, (?P<unearned>(?:team )?unearned))?"

#: A clause that already records a run. Two readers: the repeated-subject
#: correction (is a re-emitted name redundant, or is it the missing verb?)
#: and the primary-chain modifier lift (does "unearned" belong to a run?).
_SCORED_TOKEN_RE = re.compile(r"\bscored\b")


#: Closed alternation of every modifier token the corpus writes in a
#: comma-separated verb tail. Widened at issue #40 from a census of the
#: tails appearing after OUT-type verbs on unparsed lines: "sacrifice fly"
#: (22 lines), "inside the park" (7), "on appeal" (5), "team unearned" (1).
#: Still closed literals, never a `.*` catch-all, so an unlisted tail keeps
#: failing loud. "team unearned" precedes "unearned" so the longer token
#: wins, for the same reason "\d+ RBI" precedes "RBI".
_HIT_MOD_TOKEN = (
    r"\d+ RBI|RBI|bunt|SAC|ground-rule|team unearned|unearned"
    r"|sacrifice fly|inside the park|on appeal|obstruction|interference"
)
_HIT_MOD_TAIL = rf"(?P<mods>(?:, (?:{_HIT_MOD_TOKEN}))*)"

#: An OUT-type verb can name a location after its fielder, exactly as a hit
#: verb does -- "lined out to cf to left center", "lined out to p to first
#: base", "popped up to 3b down the 3b line" -- or mark the play unassisted.
#: Optional, so the bare "VERB to F" form is unaffected (issue #40).
_OUT_LOC_SUFFIX = (
    r"(?:(?: to (?P<loc>[a-z][a-z0-9 ]*?)"
    r"| down (?P<loc2>[a-z][a-z0-9 ]*?))"
    r"|(?P<unassisted> unassisted))?"
)


def _expand_rbi_modifiers(tokens: List[str]) -> List[str]:
    """A "N RBI" (N >= 2) token keeps its own literal text (fidelity: the
    count is real information) but ALSO gets a bare "RBI" element appended
    right after it, so the pre-existing `"RBI" in modifiers` exact-match
    check (parse.py's per-run rbi-flag assembly, and the identical `"SAC" in
    modifiers` pattern in replay.py's check_pa_counts) keeps matching
    regardless of N -- that boolean check is the code's own existing
    convention for modifier membership, and it is exact-match, not
    substring, so "2 RBI" alone would silently break it. A bare "RBI" token
    is left untouched (no duplication).
    """
    out: List[str] = []
    for tok in tokens:
        out.append(tok)
        if re.fullmatch(r"\d+ RBI", tok):
            out.append("RBI")
    return out


def _hit_modifiers_from_tail(tail: Optional[str]) -> List[str]:
    return _expand_rbi_modifiers(_modifiers_from_tail(tail or ""))


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
    return _x_into_double_play(m)


def _x_into_double_play(m: re.Match):
    mods = ["unassisted"] if m.group("unassisted") else []
    mods += _hit_modifiers_from_tail(m.groupdict().get("mods"))
    return (m.group("name"), _split_chain(m.group("chain")), None, mods)


def _x_groundout_chain(m: re.Match):
    mods = _hit_modifiers_from_tail(m.groupdict().get("mods"))
    return (m.group("name"), _split_chain(m.group("chain")), None, mods)


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


def _x_foul_out(m: re.Match):
    return (
        m.group("name"),
        [m.group("f")],
        None,
        _modifiers_from_tail(m.group("tail")),
    )


def _x_lineout(m: re.Match):
    loc = m.group("loc")
    if loc is None:
        loc = _normalize_down_location(m.group("loc2"))
    mods = _hit_modifiers_from_tail(m.group("mods"))
    if m.group("unassisted"):
        mods = ["unassisted"] + mods
    return (m.group("name"), [m.group("f")], loc, mods)


def _x_popout(m: re.Match):
    loc = m.group("loc")
    if loc is None:
        loc = _normalize_down_location(m.group("loc2"))
    mods = _hit_modifiers_from_tail(m.group("mods"))
    if m.group("unassisted"):
        mods = ["unassisted"] + mods
    return (m.group("name"), [m.group("f")], loc, mods)


def _x_popout_out_to(m: re.Match):
    mods = [m.group("mod")] if m.group("mod") else []
    return (m.group("name"), [m.group("f")], None, mods)


def _x_fielders_choice(m: re.Match):
    """"X reached on a fielder's choice[, out at BASE CHAIN][, RBI]".

    The out clause is captured STRUCTURALLY, not swept into `modifiers` by a
    `.*` tail. That tail was the codebase's own documented anti-pattern
    (_HIT_MOD_TAIL's comment warns against exactly it) and it dropped a real
    out on 14 corpus lines: the play recorded an out, the record said
    otherwise, and nothing downstream could tell (issue #40).
    """
    return (
        m.group("name"),
        [],
        m.group("loc") or m.group("middle") and "up the middle",
        _hit_modifiers_from_tail(m.group("mods")),
        m.group("out_base"),
        m.group("out_chain"),
    )


def _x_reached_on_interference(m: re.Match):
    """"X reached on catcher's interference" -- the batter is awarded first
    because the catcher interfered with the swing. No error is charged and,
    critically, it is NOT an at-bat, which is why it needs its own type
    rather than reached_on_error or fielders_choice (issue #40). The catcher
    is the responsible fielder, so `fielders` carries "c"."""
    mods = _hit_modifiers_from_tail(m.group("mods"))
    return (m.group("name"), ["c"], None, mods)


def _x_infield_fly(m: re.Match):
    """"X infield fly to ss." -- the infield fly rule: with runners on and
    fewer than two out, the batter is declared out on a catchable infield
    pop-up whether or not it is caught.

    Given its own type rather than folded into `popout`/`flyout` because the
    out does NOT depend on the catch, which is exactly what those two types
    assert. The fielder is preserved, same no-defensive-info-loss
    requirement that shaped `foul_out` at 1.3.0 (issue #40)."""
    return (m.group("name"), [m.group("f")], None, [])


def _x_batter_interference(m: re.Match):
    """"X out on batter's interference" -- the batter is retired for
    interfering with the catcher's play. An out with no batted ball, which
    no existing type covers."""
    return (m.group("name"), [], None, [])


def _x_reached_on_error(m: re.Match):
    mods = ["error"] + _hit_modifiers_from_tail(m.group("mods"))
    return (m.group("name"), [m.group("f")], m.group("loc"), mods)


def _x_single(m: re.Match):
    if m.group("loc") is not None:
        loc = m.group("loc")
    elif m.group("middle") is not None:
        loc = "up the middle"
    elif m.group("side") is not None:
        loc = f"{m.group('side')} side"
    elif m.groupdict().get("line") is not None:
        # "down the lf line" / "down the rf line" -- 35 lines in the 2026
        # slice alone, the largest single location gap (issue #40).
        loc = f"down the {m.group('line')} line"
    else:
        loc = None
    mods = _hit_modifiers_from_tail(m.group("mods"))
    return (m.group("name"), [], loc, mods)


def _x_double(m: re.Match):
    loc = m.group("loc")
    if loc is None and m.group("loc2") is not None:
        # The double rule captures "down <X>" WITHOUT the preposition, while
        # the single rule captures "down the <X> line" WITH it -- two
        # different strings for the same physical location, which a consumer
        # joining on location would trip over. Normalized here onto the
        # single rule's form, which is also parallel to "up the middle"
        # (the other location that keeps its preposition). Issue #40.
        loc = _normalize_down_location(m.group("loc2"))
    if loc is None and m.groupdict().get("middle2") is not None:
        loc = "up the middle"
    elif loc is None and m.groupdict().get("side2") is not None:
        loc = f"{m.group('side2')} side"
    mods = _hit_modifiers_from_tail(m.group("mods"))
    return (m.group("name"), [], loc, mods)


def _normalize_down_location(loc: Optional[str]) -> Optional[str]:
    """Give every hit type ONE spelling of a down-the-line location.

    The `down (?P<loc2>...)` capture drops the preposition ("the lf line")
    while the single rule's own row keeps it ("down the lf line") -- two
    strings for the same physical location, which a consumer joining on
    location would trip over. Normalized onto the form that keeps it,
    parallel to "up the middle", the other location carrying a preposition.
    Issue #40.
    """
    if loc is None or loc.startswith("down "):
        return loc
    return f"down {loc}"


def _x_triple(m: re.Match):
    loc = m.group("loc")
    if loc is None:
        loc = _normalize_down_location(m.group("loc2"))
    mods = _hit_modifiers_from_tail(m.group("mods"))
    return (m.group("name"), [], loc, mods)


def _x_home_run(m: re.Match):
    loc = m.group("loc")
    if loc is None:
        loc = _normalize_down_location(m.group("loc2"))
    mods = _hit_modifiers_from_tail(m.group("mods"))
    return (m.group("name"), [], loc, mods)


def _x_walk(m: re.Match):
    mods = _expand_rbi_modifiers([m.group("mod")]) if m.group("mod") else []
    return (m.group("name"), [], None, mods)


def _x_intentional_walk(m: re.Match):
    return (m.group("name"), [], None, [])


def _x_hit_by_pitch(m: re.Match):
    mods = [m.group("mod")] if m.group("mod") else []
    return (m.group("name"), [], None, mods)


def _x_strikeout_swinging(m: re.Match):
    return (m.group("name"), [], None, _hit_modifiers_from_tail(m.groupdict().get("mods")))


def _x_strikeout_looking(m: re.Match):
    return (m.group("name"), [], None, _hit_modifiers_from_tail(m.groupdict().get("mods")))


def _x_strikeout(m: re.Match):
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
        # Bare "NAME out at first CHAIN" (the batter grounds into a force
        # play and is thrown out at first, no sacrifice comma) -- ordered
        # right AFTER the ", SAC" row above so that row still wins when
        # present. The negative lookaheads guard against two UNRELATED
        # compound narrative shapes found in the real corpus that also
        # contain the literal " out at first " substring later in the same
        # clause -- "NAME struck out swinging, out at first C to 1B"
        # (dropped-third-strike thrown out) and "NAME picked off, out at
        # first C to 1B" (pickoff throw) -- neither is one of this gate's 10
        # target shapes; without the guard, `.+?`'s non-greedy name group
        # would swallow "NAME struck out swinging," whole as if it were a
        # player name, misfiling a strikeout as a groundout.
        re.compile(
            rf"^(?!.*\bstruck out\b)(?!.*\bpicked off\b)"
            rf"(?P<name>.+?) out at first "
            rf"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*){_HIT_MOD_TAIL}$"
        ),
        "groundout",
        _x_groundout_chain,
    ),
    (
        re.compile(
            # Chain quantifier is `*` (not `+`) with an optional
            # " unassisted" suffix, matching the flied/lined-into-double-play
            # rows below: the corpus writes "grounded into double play 2b
            # unassisted" and a bare single fielder too (issue #40).
            r"^(?P<name>.+?) grounded into double play "
            rf"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*)"
            rf"(?P<unassisted> unassisted)?{_HIT_MOD_TAIL}$"
        ),
        "grounded_into_double_play",
        _x_grounded_into_double_play,
    ),
    (
        # "flied/lined into double play <chain>" map to the EXISTING
        # flyout/lineout types -- never a new flied_into_double_play /
        # lined_into_double_play type. Unlike "grounded into double play"
        # (always a multi-hop chain in the sample), the real corpus shows
        # this verb pair with a bare single fielder, a multi-hop chain, OR a
        # single fielder + " unassisted" -- so the chain quantifier is `*`
        # (zero or more additional hops) with a separate optional
        # " unassisted" suffix, rather than requiring `+`.
        re.compile(
            r"^(?P<name>.+?) flied into double play "
            rf"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*)"
            rf"(?P<unassisted> unassisted)?{_HIT_MOD_TAIL}$"
        ),
        "flyout",
        _x_into_double_play,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) lined into double play "
            rf"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*)"
            rf"(?P<unassisted> unassisted)?{_HIT_MOD_TAIL}$"
        ),
        "lineout",
        _x_into_double_play,
    ),
    (
        # "fouled into double play" -> the EXISTING foul_out type, the same
        # verb-names-the-batted-ball rule the flied/lined rows follow.
        re.compile(
            r"^(?P<name>.+?) fouled into double play "
            rf"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*)"
            rf"(?P<unassisted> unassisted)?{_HIT_MOD_TAIL}$"
        ),
        "foul_out",
        _x_into_double_play,
    ),
    (
        # A triple play retires three; the verb still names the batted ball,
        # so the outcome_type stays the batted-ball one.
        re.compile(
            r"^(?P<name>.+?) lined into triple play "
            rf"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*)"
            rf"(?P<unassisted> unassisted)?{_HIT_MOD_TAIL}$"
        ),
        "lineout",
        _x_into_double_play,
    ),
    (
        # Same treatment as the flied/lined rows directly above: the verb
        # names the batted ball, so it maps to the EXISTING popout type
        # rather than a new popped_into_double_play one (issue #40).
        re.compile(
            r"^(?P<name>.+?) popped into double play "
            rf"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*)"
            rf"(?P<unassisted> unassisted)?{_HIT_MOD_TAIL}$"
        ),
        "popout",
        _x_into_double_play,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) grounded out "
            rf"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)+){_HIT_MOD_TAIL}$"
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
        # A foul fly ball caught for an out -- distinct verb token
        # ("fouled") from "flied"/"lined"/"grounded"/"popped", so no
        # collision risk with any other row. Mirrors _x_flyout's tail
        # handling exactly: a "sacrifice fly, RBI" tail carries its
        # modifiers under the SAME outcome_type "foul_out" (never a
        # separate "sacrifice" type), matching the existing flyout
        # convention (test_flyout_sac_rbi_modifiers).
        re.compile(r"^(?P<name>.+?) fouled out to (?P<f>[a-z0-9]+)(?P<tail>.*)$"),
        "foul_out",
        _x_foul_out,
    ),
    (
        # The modifier tail is ZERO-or-more, so the bare "NAME lined out to F"
        # form matches exactly as before -- this row is purely additive
        # (issue #40; 51 unparsed lines carried ", RBI" / ", sacrifice fly,
        # RBI" / ", SAC, RBI" tails this row had no way to accept).
        re.compile(rf"^(?P<name>.+?) lined out to (?P<f>[a-z0-9]+)"
            rf"{_OUT_LOC_SUFFIX}{_HIT_MOD_TAIL}$"),
        "lineout",
        _x_lineout,
    ),
    (
        re.compile(rf"^(?P<name>.+?) popped up to (?P<f>[a-z0-9]+)"
            rf"{_OUT_LOC_SUFFIX}{_HIT_MOD_TAIL}$"),
        "popout",
        _x_popout,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) popped out to (?P<f>[a-z0-9]+)(?:, (?P<mod>bunt))?$"
        ),
        "popout",
        _x_popout_out_to,
    ),
    (
        # Curly and straight apostrophes both occur in the corpus.
        re.compile(
            rf"^(?P<name>.+?) reached on catcher['\u2019]s interference{_HIT_MOD_TAIL}$"
        ),
        "reached_on_interference",
        _x_reached_on_interference,
    ),
    (
        re.compile(r"^(?P<name>.+?) out on batter['\u2019]s interference$"),
        "batter_interference",
        _x_batter_interference,
    ),
    (
        # `_OUT_LOC_SUFFIX` for parity with every other out verb: the corpus
        # writes "infield fly to 3b down the lf line" exactly as it writes
        # "popped up to 3b down the 3b line", and this row alone rejected it.
        re.compile(
            rf"^(?P<name>.+?) infield fly to (?P<f>[a-z0-9]+){_OUT_LOC_SUFFIX}$"
        ),
        "infield_fly",
        _x_infield_fly,
    ),
    (
        re.compile(
            # The fielder is named in words ("to shortstop", "to second
            # base") on 200+ corpus lines; the old `.*` tail swept that into
            # `modifiers` as a junk string instead of recovering the fielder.
            rf"^(?P<name>.+?) reached on a fielder's choice"
            rf"(?: to (?P<loc>[a-z][a-z ]*?)|(?P<middle> up the middle))?"
            rf"(?:, out at (?P<out_base>first|second|third|home)"
            rf"(?: (?P<out_chain>[a-z0-9]+(?: to [a-z0-9]+)*))?)?{_HIT_MOD_TAIL}$"
        ),
        "fielders_choice",
        _x_fielders_choice,
    ),
    (
        # "first" is optional (the "reached on a fielding error by X"
        # no-first wording). The error spelling comes from `_ERROR_BY`: this
        # row used to carry its own inline "(?:an error|a fielding error|a
        # throwing error)" list -- a THIRD copy of a list that already
        # existed twice -- so "reached first on a muffed throw by 1b" failed
        # here while the sibling runner rows using the shared fragment
        # accepted it.
        #
        # The tail is `_HIT_MOD_TAIL`, the same CLOSED alternation every
        # other verb row uses. It was an unrestricted `(?P<tail>.*)`, the
        # only such tail in the table, and it quietly ate real content: 248
        # events across 228 games had movement clauses stored as MODIFIER
        # STRINGS -- "advanced to second", "advanced to third on an error by
        # lf", "out at second 2b to ss", even a location ("to right center")
        # and a runner's name. Those runners never moved in `runners[]`, so
        # their advances and runs simply were not in the record.
        #
        # Because `_match_primary_whole` is tried before `_match_primary_chain`,
        # a greedy tail here also PRE-EMPTED the chain path that exists to
        # parse exactly these continuations. Closing the tail lets that path
        # see the line.
        # The optional " to <loc>" is the same batted-ball location every hit
        # row records ("reached first on an error by cf TO RIGHT CENTER").
        # The old unrestricted tail swallowed it into `modifiers` as the
        # string "to right center"; closing the tail without capturing it
        # would have made those lines unparseable instead. A location is
        # information, so it goes where the hit rows put theirs.
        re.compile(
            rf"^(?P<name>.+?) reached (?:first )?on "
            rf"{_ERROR_BY}(?: to (?P<loc>[a-z][a-z ]*?))?{_HIT_MOD_TAIL}$"
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
            r"| down the (?P<line>[a-z0-9]{2}) line"
            rf")?{_HIT_MOD_TAIL}$"
        ),
        "single",
        _x_single,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) doubled(?: to (?P<loc>[a-z][a-z ]*?)"
            r"| down (?P<loc2>[a-z][a-z0-9 ]*?)"
            r"|(?P<middle2> up the middle)"
            r"| through the (?P<side2>left|right) side"
            rf")?{_HIT_MOD_TAIL}$"
        ),
        "double",
        _x_double,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) tripled(?: to (?P<loc>[a-z][a-z ]*?)"
            rf"| down (?P<loc2>[a-z][a-z0-9 ]*?))?{_HIT_MOD_TAIL}$"
        ),
        "triple",
        _x_triple,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) homered(?: to (?P<loc>[a-z][a-z ]*?)"
            rf"| down (?P<loc2>[a-z][a-z0-9 ]*?))?{_HIT_MOD_TAIL}$"
        ),
        "home_run",
        _x_home_run,
    ),
    (
        re.compile(r"^(?P<name>.+?) was intentionally walked$"),
        "intentional_walk",
        _x_intentional_walk,
    ),
    (
        # A bases-loaded walk forces in a run, so the tail can be "N RBI"
        # and not just a bare "RBI" (issue #40).
        re.compile(r"^(?P<name>.+?) walked(?:, (?P<mod>\d+ RBI|RBI))?$"),
        "walk",
        _x_walk,
    ),
    (
        re.compile(r"^(?P<name>.+?) hit by pitch(?:, (?P<mod>RBI))?$"),
        "hit_by_pitch",
        _x_hit_by_pitch,
    ),
    (
        re.compile(rf"^(?P<name>.+?) struck out swinging{_HIT_MOD_TAIL}$"),
        "strikeout_swinging",
        _x_strikeout_swinging,
    ),
    (
        re.compile(rf"^(?P<name>.+?) struck out looking{_HIT_MOD_TAIL}$"),
        "strikeout_looking",
        _x_strikeout_looking,
    ),
    (
        # Bare "NAME struck out" (no swinging/looking qualifier) --
        # ordered AFTER the swinging/looking rows above (the trailing $
        # anchor already prevents a collision with either, since both
        # carry extra trailing text after "out", but the explicit
        # ordering keeps the intent visible).
        re.compile(r"^(?P<name>.+?) struck out$"),
        "strikeout",
        _x_strikeout,
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


def _b_advance_on_fielders_choice(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="fielders_choice",
        destination=m.group("dest"),
        out=False,
        scored=False,
    )


def _b_scored_on_throw(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="advance",
        destination="home",
        out=False,
        scored=True,
        unearned=bool(m.groupdict().get("unearned")),
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
        # The sibling scored-on-error row already carried this tail; the
        # causephrase row did not, so every "scored on a wild pitch,
        # unearned" line was a miss (issue #40).
        unearned=bool(m.groupdict().get("unearned")),
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
    dest = m.group("dest")
    return RunnerMovement(
        name_token=m.group("name"),
        cause="stolen_base",
        destination=dest,
        out=False,
        # A steal of HOME is a run. This builder hardcoded scored=False, so
        # "X stole home" folded to runs_on_play=0 and the run vanished from
        # the linescore -- computed one FEWER run than the box, for that
        # inning and therefore for the final total too. Every other builder
        # already ties `scored` to a home destination; this was the only one
        # that did not (issue #40).
        scored=(dest == "home"),
        unearned=bool(m.groupdict().get("unearned")),
    )


def _b_caught_stealing(m: re.Match):
    return RunnerMovement(
        name_token=m.group("name"),
        cause="caught_stealing",
        destination=m.group("dest"),
        out=True,
        scored=False,
    )


def _b_caught_stealing_chain(m: re.Match):
    """"X caught stealing c to ss, double play" -- the corpus's OTHER caught
    stealing phrasing: a fielding chain instead of a destination base, with
    an optional ", double play" tail when the strikeout ahead of it made the
    second out. The base the runner was thrown out at is not written, so
    `destination` stays None rather than being guessed (issue #40)."""
    return RunnerMovement(
        name_token=m.group("name"),
        cause="caught_stealing",
        destination=None,
        out=True,
        scored=False,
    )


def _b_out_on_double_play(m: re.Match):
    """"X out on double play 2b to ss" -- a RUNNER retired as the second out
    of a double play, always written as its own semicolon clause after the
    batter's. Distinct from the primary-clause "grounded into double play"
    row, which describes the BATTER's batted ball; this one is the runner
    and carries only the fielding chain, so it belongs in RUNNER_RULES
    beside "out on the play" (issue #40)."""
    return RunnerMovement(
        name_token=m.group("name"),
        cause="putout",
        destination=None,
        out=True,
        scored=False,
    )


def _b_source_defect_scored(m: re.Match):
    """"M. Moralez M. Moralez." -- the runner's name written TWICE where the
    verb belongs. Verbatim in StatCrew's own HTML, so it is a source defect,
    not an extraction bug; the run itself is real and simply has no verb.

    That the missing verb is "scored" is not a guess. Take every half-inning
    whose ONLY unparsed line carries this shape, and compare the linescore
    oracle's run total for that half-inning against the runs the PARSED
    events assert. In 54 of 54 such cases the inning reconciles exactly when
    this clause is counted as a run, and in 0 of 54 when it is not -- an
    independent oracle, not a proxy, with the null measured alongside it.

    The movement is tagged `inferred` so it surfaces in the game file's
    top-level `inferred[]` rather than passing as something the line said.
    """
    return RunnerMovement(
        name_token=m.group("name"),
        cause="advance",
        destination="home",
        out=False,
        scored=True,
        unearned=bool(m.group("unearned")),
        inferred=(
            "source line names the runner twice with no verb; read as "
            "'scored' (linescore oracle reconciles 54/54, null 0/54)"
        ),
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


def _b_picked_off(m: re.Match):
    """A bare "X picked off" -- the runner is RETIRED.

    Distinct from `_b_pickoff` below, which serves "X Failed pickoff
    attempt": a throw over that the runner survives. Same word, opposite
    outcome, so they cannot share a builder.
    """
    return RunnerMovement(
        name_token=m.group("name"),
        cause="pickoff",
        destination=None,
        out=True,
        scored=False,
    )


def _c_picked_off(m: "re.Match", name_token: str) -> RunnerMovement:
    """"..., picked off" trailing an already-stated out ("X out at first p to
    1b to ss, picked off") -- the narrative names the same retirement twice,
    so this records the CAUSE without double-counting the out."""
    return RunnerMovement(
        name_token=name_token,
        cause="pickoff",
        destination=None,
        out=False,
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
            rf"scored on {_ERROR_BY}"
            rf"(?:, (?P<unearned>(?:team )?unearned))?$"
        ),
        ("advance", "error"),
        _b_compound_advance_scored_error,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) advanced to (?P<dest1>{_DEST_ALT}) on a "
            rf"(?P<causephrase>wild pitch|passed ball|balk|illegal pitch), "
            rf"advanced to (?P<dest2>{_DEST_ALT})$"
        ),
        ("wild_pitch", "passed_ball", "balk", "advance"),
        _b_compound_double_advance,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) advanced to (?P<dest>{_DEST_ALT}) on a "
            rf"(?P<causephrase>wild pitch|passed ball|balk|illegal pitch)$"
        ),
        ("wild_pitch", "passed_ball", "balk"),
        _b_advance_on_causephrase,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) advanced to (?P<dest>{_DEST_ALT}) on "
            rf"{_ERROR_BY}"
            rf"(?:, (?P<unearned>(?:team )?unearned))?$"
        ),
        ("error",),
        _b_advance_on_error,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) advanced to (?P<dest>{_DEST_ALT}) on the throw$"
        ),
        ("advance",),
        _b_advance_plain,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) advanced to (?P<dest>{_DEST_ALT}) on a "
            rf"fielder's choice(?: to (?P<fc_fielder>[a-z][a-z ]*?))?$"
        ),
        ("fielders_choice",),
        _b_advance_on_fielders_choice,
    ),
    (
        re.compile(rf"^(?P<name>.+?) scored on the throw{_UNEARNED_TAIL}$"),
        ("advance",),
        _b_scored_on_throw,
    ),
    (
        # `_UNEARNED_TAIL` mirrors this row's CONTINUATION_RULES twin (and
        # every `scored` row): "A. Shaver advanced to third, unearned" is a
        # whole runner clause, not a continuation, so it is matched here.
        re.compile(rf"^(?P<name>.+?) advanced to (?P<dest>{_DEST_ALT}){_UNEARNED_TAIL}$"),
        ("advance",),
        _b_advance_plain,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) scored on a "
            r"(?P<causephrase>wild pitch|passed ball|balk|illegal pitch)"
            r"(?:, (?P<unearned>(?:team )?unearned))?$"
        ),
        ("wild_pitch", "passed_ball", "balk"),
        _b_scored_on_causephrase,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) scored on {_ERROR_BY}"
            rf"(?:, (?P<unearned>(?:team )?unearned))?$"
        ),
        ("error",),
        _b_scored_on_error,
    ),
    (
        re.compile(r"^(?P<name>.+?) scored(?:, (?P<unearned>(?:team )?unearned))?$"),
        ("advance",),
        _b_scored_plain,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) stole (?P<dest>{_DEST_ALT})"
            rf"(?:, (?P<unearned>(?:team )?unearned))?$"
        ),
        ("stolen_base",),
        _b_stole,
    ),
    (
        # "X scored, advanced on an error by cf" / "X scored, on a wild
        # pitch" -- StatCrew's inverted phrasing, naming the run first and
        # the mechanism after. Same fact as the "scored on ..." rows above,
        # so they build the same record; kept as their own rows because the
        # comma makes them a different shape, not a different event.
        re.compile(
            rf"^(?P<name>.+?) scored, advanced on {_ERROR_BY}{_UNEARNED_TAIL}$"
        ),
        ("error",),
        _b_scored_on_error,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) scored, on a "
            rf"(?P<causephrase>wild pitch|passed ball|balk|illegal pitch){_UNEARNED_TAIL}$"
        ),
        ("wild_pitch", "passed_ball", "balk"),
        _b_scored_on_causephrase,
    ),
    (
        re.compile(rf"^(?P<name>.+?) caught stealing (?P<dest>{_DEST_ALT})$"),
        ("caught_stealing",),
        _b_caught_stealing,
    ),
    (
        re.compile(
            rf"^(?P<name>.+?) caught stealing "
            rf"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*)(?:, double play)?$"
        ),
        ("caught_stealing",),
        _b_caught_stealing_chain,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) out on double play "
            r"(?P<chain>[a-z0-9]+(?: to [a-z0-9]+)*)$"
        ),
        ("putout",),
        _b_out_on_double_play,
    ),
    (
        re.compile(
            r"^(?P<name>.+?) out on the play(?:, (?:interference|caught stealing))?$"
        ),
        ("putout",),
        _b_out_on_the_play,
    ),
    (
        re.compile(r"^(?P<name>.+?) out at (?P<base>first|second|third|home)\b.*$"),
        ("force_out",),
        _b_out_at_base,
    ),
    (
        re.compile(r"^(?P<name>.+?) picked off$"),
        ("pickoff",),
        _b_picked_off,
    ),
    (
        re.compile(r"^(?P<name>.+?) Failed pickoff attempt$"),
        ("pickoff",),
        _b_pickoff,
    ),
]


# ---------------------------------------------------------------------------
# STANDALONE_RULES -- whole-line shapes that are neither a PA nor a bare
# sequence of runner-movement clauses: an inning-recap line, a two-name
# position-move substitution line ("<in> to <pos> for <out>." -- <pos> is any
# fielding position token, including "dh"), a pinch-run substitution line
# ("<in> pinch ran for <out>."), a pinch-hit substitution line ("<in> pinch
# hit for <out>."), and a bare (no outgoing player named) position-move line
# ("<name> to <pos>."). (Whole-line runner-movement-only shapes -- "X stole
# second.", "X Failed pickoff attempt.", "X advanced to Y on a wild pitch."
# and multi-clause variants thereof -- fall out of the SAME RUNNER_RULES
# table via the no-count-tail fallback path below; they need no separate
# regex here.)
#
# kind assignment (schema's closed `substitution.kind` enum -- see
# `Substitution.kind`'s docstring): the matched <pos> token decides it.
# "p" -> "pitching" (the mound), "dh" -> "offensive" (the DH is a
# batting-lineup slot, not a fielding position -- issue #30's convention),
# every other fielding position -> "defensive". Pinch-run/pinch-hit are
# always "offensive" (they name a batting-order substitution outright, no
# position token to branch on).
#
# The bare "<name> to dh." DH-slot-entry shape (no outgoing player named at
# all) predates this gate (schema 1.2.0, issue #30 g2b, `player_out`
# nullable). `_POSITION_MOVE_BARE_RE` below covers the SAME bare shape for
# every OTHER fielding position (never "dh" -- that is `_DH_SLOT_BARE_RE`'s
# job, and this row is ordered after it so "to dh" keeps its existing
# offensive handling) -- kind="defensive" flat, not branched by position,
# because a bare move never names an outgoing player to disambiguate a true
# pitching change from a defensive repositioning, and (per parse.py's
# substitution assembly) "defensive" and "pitching" both resolve against the
# same fielding side, so the flat label loses no assembly correctness.
#
# GUARD: `_POSITION_MOVE_BARE_RE`'s captured name group is a Title-Case NAME
# TOKEN pattern, not a bare ".+?" -- a naive ".+? to <pos>\.?$" catastrophically
# false-matches TWO real narrative shapes that end in the exact same "to
# <pos>." tail a genuine bare substitution does: (1) multi-clause
# runner-event lines ending in a fielding ASSIST-CHAIN notation, e.g. "...;
# B. Burckel out at home 3b to c." (the trailing "3b to c" is a throw chain,
# not a substitution -- 54 real corpus lines), and (2) single-clause
# plate-appearance narrative text that itself ends "... to <pos>.", e.g.
# "T. Specht flied out to cf.", "L. Barns popped out to 1b.", and the
# "out at first"-guarded compounds ("K. Jimenez struck out swinging, out at
# first c to 1b."). See the detailed comment on `_NAME_TOKEN`/
# `_POSITION_MOVE_BARE_RE` below for the guard that eliminates both classes.
# ---------------------------------------------------------------------------

StandaloneBuilder = Callable[[re.Match, Optional[int]], ClauseGroup]
StandaloneRule = Tuple[re.Pattern, Optional[str], StandaloneBuilder]

_INNING_SUMMARY_RE = re.compile(
    r"^Inning Summary:\s*(?P<r>\d+)\s*Runs\s*,\s*(?P<h>\d+)\s*Hits\s*,\s*"
    r"(?P<e>\d+)\s*Errors\s*,\s*(?P<lob>\d+)\s*LOB\s*$"
)
# Fielding position tokens as they appear in StatCrew narrative text (never
# "dh" here -- dh is handled separately below, both as a two-name kind=
# offensive branch on `_SUBSTITUTION_RE` and as its own bare
# `_DH_SLOT_BARE_RE` row).
_FIELD_POS_TOKENS = r"1b|2b|3b|ss|lf|cf|rf|c|p"
_SUBSTITUTION_RE = re.compile(
    rf"^(?P<in>.+?) to (?P<pos>{_FIELD_POS_TOKENS}|dh) for (?P<out>.+?)\.?$"
)
_PINCH_RUN_RE = re.compile(r"^(?P<in>.+?) pinch ran for (?P<out>.+?)\.?$")
_PINCH_HIT_RE = re.compile(r"^(?P<in>.+?) pinch hit for (?P<out>.+?)\.?$")
_DH_SLOT_BARE_RE = re.compile(r"^(?P<in>.+?) to dh\.?$")
# `_POSITION_MOVE_BARE_RE`'s name group is a NAME TOKEN pattern, not a bare
# ".+?" -- unlike every other STANDALONE_RULES row, this one's trailing
# shape ("to <pos>.") COLLIDES with real plate-appearance narrative text:
# "T. Specht flied out to cf.", "L. Barns popped out to 1b.", and the
# "out at first"-guarded compounds ("K. Jimenez struck out swinging, out at
# first c to 1b.") all end in the exact same "to <pos>." tail a genuine bare
# substitution does. A plain "no semicolon" guard (which is enough for the
# OTHER false-positive class -- multi-clause fielding-assist chains, see the
# module note above) does NOT catch these: they are single-clause narrative
# text, comma-joined, with no semicolon at all. The only reliable
# discriminator is that a real player-name token is Title Case throughout
# (StatCrew's own convention -- "F. Last", "First Last", plus the observed
# real-corpus edge cases "Last, Jr", "First (Nickname) Last", curly-quote
# apostrophes) while every PA/runner verb phrase is lowercase prose. Each
# whitespace-separated word in the name must therefore start with an
# uppercase letter or "(" -- this single rule independently eliminates BOTH
# false-positive classes (verified against the full corpus: 0 of the known
# false positives match, all 1221 legitimate real bare-move lines still do,
# including the 11 edge-case names above that a naive [A-Z][\w'-]* char
# class would have wrongly dropped).
_NAME_TOKEN = r"[A-Z(][A-Za-z0-9.'’(),-]*"
_POSITION_MOVE_BARE_RE = re.compile(
    rf"^(?P<name>{_NAME_TOKEN}(?:\s+{_NAME_TOKEN})*) to (?P<pos>{_FIELD_POS_TOKENS})\.?$"
)


def _substitution_kind_for_pos(pos: str) -> str:
    """p -> pitching, dh -> offensive (batting-lineup slot), else defensive."""
    if pos == "p":
        return "pitching"
    if pos == "dh":
        return "offensive"
    return "defensive"


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
            player_in=m.group("in"),
            player_out=m.group("out"),
            kind=_substitution_kind_for_pos(m.group("pos")),
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


def _build_pinch_hit(m: re.Match, trailing_outs: Optional[int]) -> ClauseGroup:
    return ClauseGroup(
        kind="substitution",
        substitution=Substitution(
            player_in=m.group("in"), player_out=m.group("out"), kind="offensive"
        ),
        trailing_outs=trailing_outs,
    )


def _build_dh_slot_bare(m: re.Match, trailing_outs: Optional[int]) -> ClauseGroup:
    # Bare DH-slot-entry: the line names only the incoming player. Schema
    # 1.2.0 (issue #30) made substitution.player_out nullable so this is a
    # real "offensive" lineup-slot activation, same kind convention as the
    # pinch-run row above -- never guess an outgoing player from a line that
    # does not name one.
    return ClauseGroup(
        kind="substitution",
        substitution=Substitution(
            player_in=m.group("in"), player_out=None, kind="offensive"
        ),
        trailing_outs=trailing_outs,
    )


def _build_position_move_bare(m: re.Match, trailing_outs: Optional[int]) -> ClauseGroup:
    # Bare defensive-position-move: the line names only the player taking the
    # field at the new position, never an outgoing player -- same
    # never-guess convention as the bare DH-slot row. kind is flat
    # "defensive" (see the module-level note above this table for why it is
    # not branched by position here).
    return ClauseGroup(
        kind="substitution",
        substitution=Substitution(
            player_in=m.group("name"), player_out=None, kind="defensive"
        ),
        trailing_outs=trailing_outs,
    )


#: A pitching change whose INCOMING player StatCrew left blank: the line is
#: verbatim "/  for R. Bost." (or "/ to p for R. Bost."), the name before the
#: slash simply missing. Every one of the 54 resolvable outgoing players in
#: the corpus is a box pitcher, which is what identifies the shape as a
#: pitching change; parse.py fills the incoming name from the box pitching
#: order and records the inference (issue #40).
_BLANK_SUB_RE = re.compile(r"^/\s*(?:to p )?for (?P<out>.+?)\.?$")


def _build_blank_sub(m: re.Match, trailing_outs: Optional[int]) -> ClauseGroup:
    return ClauseGroup(
        kind="substitution",
        substitution=Substitution(
            player_in=None, player_out=m.group("out"), kind="pitching"
        ),
        trailing_outs=trailing_outs,
    )


STANDALONE_RULES: List[StandaloneRule] = [
    (_BLANK_SUB_RE, None, _build_blank_sub),
    (_INNING_SUMMARY_RE, None, _build_inning_summary),
    (_SUBSTITUTION_RE, None, _build_substitution),
    (_PINCH_RUN_RE, None, _build_pinch_run),
    (_PINCH_HIT_RE, None, _build_pinch_hit),
    (_DH_SLOT_BARE_RE, None, _build_dh_slot_bare),
    (_POSITION_MOVE_BARE_RE, None, _build_position_move_bare),
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
    # Awarded first base; no error is charged, so a plain advance.
    "reached_on_interference": ("advance", "first", False, False),
    # Retired without a batted ball.
    "batter_interference": ("putout", None, True, False),
    # Declared out under the infield fly rule; the catch is irrelevant.
    "infield_fly": ("putout", None, True, False),
    "fielders_choice": ("fielders_choice", "first", False, False),
    "strikeout_swinging": ("putout", None, True, False),
    "strikeout_looking": ("putout", None, True, False),
    "groundout": ("putout", None, True, False),
    "flyout": ("putout", None, True, False),
    "lineout": ("putout", None, True, False),
    "popout": ("putout", None, True, False),
    "grounded_into_double_play": ("putout", None, True, False),
    "sacrifice": ("putout", None, True, False),
    "foul_out": ("putout", None, True, False),
    "strikeout": ("putout", None, True, False),
}


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CONTINUATION_RULES -- issue #40.
#
# A comma-joined fragment that continues the SAME runner's movement with the
# name elided ("... , advanced to third", "... , out at second ss to 2b",
# "... , scored, unearned"). RUNNER_RULES enumerates specific (first, second)
# PAIRS -- e.g. there is a compound row for `advanced to D1 on a (wild pitch|
# passed ball|balk), advanced to D2` but none for the error cause -- while
# the corpus uses the cross product of a handful of lead verbs and a handful
# of continuations. Where no pair row exists, a lead-anchored row's greedy
# `(?P<name>.+?)` absorbs the unmatched lead clause instead of failing, so
# the line parses into a plausible-but-wrong record whose only symptom is a
# name that never resolves. These rows let the chain be parsed compositionally
# instead.
#
# Every row is anchored with NO name group: the name is inherited from the
# lead clause by `_match_clause_chain`.
# ---------------------------------------------------------------------------

def _c_out_at(m: "re.Match", name_token: str) -> RunnerMovement:
    return RunnerMovement(
        name_token=name_token,
        cause="force_out",
        destination=m.group("base"),
        out=True,
        scored=False,
    )


def _c_advance_on_error(m: "re.Match", name_token: str) -> RunnerMovement:
    return RunnerMovement(
        name_token=name_token,
        cause="error",
        destination=m.group("dest"),
        out=False,
        scored=False,
        unearned=bool(m.group("unearned")),
    )


def _c_advance_on_causephrase(m: "re.Match", name_token: str) -> RunnerMovement:
    return RunnerMovement(
        name_token=name_token,
        cause=_CAUSEPHRASE[m.group("causephrase")],
        destination=m.group("dest"),
        out=False,
        scored=False,
    )


def _c_advance_on_fielders_choice(m: "re.Match", name_token: str) -> RunnerMovement:
    return RunnerMovement(
        name_token=name_token,
        cause="fielders_choice",
        destination=m.group("dest"),
        out=False,
        scored=False,
    )


def _c_scored_on_throw(m: "re.Match", name_token: str) -> RunnerMovement:
    return RunnerMovement(
        name_token=name_token,
        cause="advance",
        destination="home",
        out=False,
        scored=True,
        unearned=bool(m.groupdict().get("unearned")),
    )


def _c_advance_plain(m: "re.Match", name_token: str) -> RunnerMovement:
    return RunnerMovement(
        name_token=name_token,
        cause="advance",
        destination=m.group("dest"),
        out=False,
        scored=False,
    )


def _c_scored(m: "re.Match", name_token: str) -> RunnerMovement:
    return RunnerMovement(
        name_token=name_token,
        cause="advance",
        destination="home",
        out=False,
        scored=True,
        unearned=bool(m.group("unearned")),
    )


def _c_scored_on_error(m: "re.Match", name_token: str) -> RunnerMovement:
    return RunnerMovement(
        name_token=name_token,
        cause="error",
        destination="home",
        out=False,
        scored=True,
        unearned=bool(m.group("unearned")),
    )


def _c_failed_pickoff(m: "re.Match", name_token: str) -> RunnerMovement:
    """"..., failed pickoff attempt" -- a throw over that the runner SURVIVES.
    Mirrors the standalone `_b_pickoff` row exactly (out=False), never the
    `_b_picked_off` one that retires him (issue #40)."""
    return RunnerMovement(
        name_token=name_token,
        cause="pickoff",
        destination=None,
        out=False,
        scored=False,
    )


def _c_reached_on_error(m: "re.Match", name_token: str) -> RunnerMovement:
    """Dropped third strike, batter SAFE: "struck out swinging, reached first
    on an error by c".

    Sibling of the `reached first on a (wild pitch|passed ball|balk)` row
    below -- the same play, differing only in whether the scorer charged an
    error. That row existed and this one did not, so the error spelling was
    unparsed while the passed-ball spelling parsed (17 corpus lines).

    The batter is NOT out: the strikeout is credited to the pitcher, but the
    play records no out. `_merge_same_runner` takes this clause's `out=False`
    as the batter's net disposition, and `outs_recorded` is summed from the
    merged records, so the event correctly adds zero outs."""
    return RunnerMovement(
        name_token=name_token,
        cause="error",
        destination=m.group("dest"),
        out=False,
        scored=False,
        unearned=bool(m.groupdict().get("unearned")),
    )


def _c_reached_on_fielders_choice(m: "re.Match", name_token: str) -> RunnerMovement:
    """"..., reached on a fielder's choice" with no destination named -- a
    batter reaching can only be reaching FIRST, which is why the corpus omits
    it (the sibling runner row spells its destination out because a RUNNER
    could be advancing to any base)."""
    return RunnerMovement(
        name_token=name_token,
        cause="fielders_choice",
        destination="first",
        out=False,
        scored=False,
    )


def _c_retired_after_strikeout(m: "re.Match", name_token: str) -> RunnerMovement:
    """Dropped third strike, batter RETIRED: "struck out swinging, grounded
    out to c unassisted" -- strike three gets away and the catcher throws him
    out (18 corpus lines, the largest single unparsed shape left).

    Exactly ONE out is recorded, not two. The batter already carries the
    strikeout's own `out=True` at record 0; this clause merges into that same
    record rather than appending a second one, and `outs_recorded` sums the
    MERGED records. Emitting this as a separate runner entry would have
    double-counted the out and broken `outs_per_half` on every one of these
    lines."""
    return RunnerMovement(
        name_token=name_token,
        cause="putout",
        destination=None,
        out=True,
        scored=False,
    )


def _c_stole_continuation(m: "re.Match", name_token: str) -> RunnerMovement:
    """"advanced to second, stole third" -- a steal chained onto an earlier
    movement by the same runner. Mirrors the standalone `_b_stole` row's
    cause exactly, so the two spellings of one event agree."""
    return RunnerMovement(
        name_token=name_token,
        cause="stolen_base",
        destination=m.group("dest"),
        out=False,
        scored=False,
    )


CONTINUATION_RULES: List[Tuple["re.Pattern", Callable]] = [
    (
        re.compile(r"^[Ff]ailed pickoff attempt$"),
        _c_failed_pickoff,
    ),
    # "out at second ss to 2b" / "out at home 3b to c" -- the trailing token
    # run is a throw chain, deliberately not captured (RUNNER_RULES' own
    # out-at row treats it the same way).
    (
        re.compile(r"^out at (?P<base>first|second|third|home)\b.*$"),
        _c_out_at,
    ),
    (
        re.compile(
            rf"^scored on {_ERROR_BY}{_UNEARNED_TAIL}$"
        ),
        _c_scored_on_error,
    ),
    (
        re.compile(rf"^scored{_UNEARNED_TAIL}$"),
        _c_scored,
    ),
    (
        re.compile(
            rf"^advanced to (?P<dest>{_DEST_ALT}) on "
            rf"{_ERROR_BY}{_UNEARNED_TAIL}$"
        ),
        _c_advance_on_error,
    ),
    (
        re.compile(
            rf"^advanced to (?P<dest>{_DEST_ALT}) on a "
            rf"(?P<causephrase>wild pitch|passed ball|balk|illegal pitch)$"
        ),
        _c_advance_on_causephrase,
    ),
    (
        re.compile(r"^picked off$"),
        _c_picked_off,
    ),
    (
        # Dropped third strike: the batter strikes out but REACHES on the
        # ball getting away -- "struck out swinging, reached first on a
        # passed ball". The strikeout stands as the primary outcome (it is a
        # strikeout in the box); this records the batter reaching anyway.
        # `_DEST_ALT` deliberately omits "first" -- a runner cannot ADVANCE
        # to first -- but a dropped third strike is precisely a batter
        # REACHING first, so this row needs its own alternation.
        re.compile(
            r"^reached (?P<dest>first|second|third|home) on a "
            r"(?P<causephrase>wild pitch|passed ball|balk|illegal pitch)$"
        ),
        _c_advance_on_causephrase,
    ),
    (
        # Dropped third strike, batter safe on an ERROR rather than on the
        # ball merely getting away -- the sibling of the row directly above.
        # Same `first|second|third|home` alternation and for the same reason:
        # `_DEST_ALT` omits "first" because a runner cannot advance to it,
        # but a batter reaching on a dropped third strike does exactly that.
        re.compile(
            rf"^reached (?P<dest>first|second|third|home) on "
            rf"{_ERROR_BY}{_UNEARNED_TAIL}$"
        ),
        _c_reached_on_error,
    ),
    (
        re.compile(r"^reached on a fielder's choice$"),
        _c_reached_on_fielders_choice,
    ),
    (
        # "struck out, out on double play c to ss" -- strike three plus a
        # runner retired on the throw. This clause retires the BATTER, who is
        # already out on the strikeout, so it merges into his existing record
        # and adds nothing; the second out belongs to the runner named in the
        # line's own following clause. The trailing "c to ss" throw chain is
        # deliberately not captured, exactly as the `out at <base>` row above
        # treats its own chain.
        re.compile(r"^out on double play\b.*$"),
        _c_retired_after_strikeout,
    ),
    (
        # Dropped third strike, batter RETIRED at the plate: "..., grounded
        # out to c unassisted". The out verb is spelled with the full
        # out-verb alternation rather than just "grounded" so a sibling
        # spelling cannot fail on a shape its neighbour accepts.
        re.compile(
            r"^(?:grounded|flied|lined|popped|fouled) out to "
            r"(?P<f>[a-z0-9]+)(?: unassisted)?$"
        ),
        _c_retired_after_strikeout,
    ),
    (
        # "advanced to second on the throw" -- the batter/runner takes an
        # extra base while a throw goes elsewhere. Modelled as a plain
        # advance: no error was charged and no distinct cause exists for it
        # in the closed taxonomy.
        re.compile(rf"^advanced to (?P<dest>{_DEST_ALT}) on the throw$"),
        _c_advance_plain,
    ),
    (
        re.compile(rf"^advanced to (?P<dest>{_DEST_ALT}) on a fielder's choice"
        rf"(?: to (?P<fc_fielder>[a-z][a-z ]*?))?$"),
        _c_advance_on_fielders_choice,
    ),
    (
        re.compile(rf"^scored on the throw{_UNEARNED_TAIL}$"),
        _c_scored_on_throw,
    ),
    (
        # `_UNEARNED_TAIL` for parity with the `scored` row above: the corpus
        # writes ", unearned" after a plain advance too ("advanced to third,
        # unearned"), and without it the token was left dangling and the whole
        # chain failed. Nothing reads the flag on a non-scoring advance -- the
        # schema records `earned` only on a run -- so this accepts the token
        # rather than asserting anything new from it.
        re.compile(rf"^advanced to (?P<dest>{_DEST_ALT}){_UNEARNED_TAIL}$"),
        _c_advance_plain,
    ),
    (
        # "..., stole third" continuing a runner who already moved this play.
        re.compile(rf"^stole (?P<dest>{_DEST_ALT})$"),
        _c_stole_continuation,
    ),
]


#: Ordered LAST so no real-verb row can ever be shadowed by it: only a
#: clause that matched nothing else is read as the doubled-name defect.
RUNNER_RULES.append(
    (
        re.compile(
            r"^(?P<name>.+?) (?P=name)"
            r"(?:, (?P<unearned>(?:team )?unearned))?$"
        ),
        ("advance",),
        _b_source_defect_scored,
    )
)


#: Verb tokens that can never appear inside a player's NAME. A rule whose
#: `(?P<name>.+?)` capture contains one of these -- or a comma -- has greedily
#: absorbed an unmatched lead clause rather than matched a name (issue #40).
#: Treating that as a non-match is what lets `_match_clause_chain` get a look
#: at the line; without this guard the greedy row always wins first and the
#: chain path is unreachable.
_NAME_VERB_RE = re.compile(
    r"\b(?:advanced|scored|stole|singled|doubled|tripled|homered|walked|"
    r"struck|grounded|flied|lined|popped|fouled|reached|picked|caught|"
    r"sacrificed|hit|out)\b"
)


#: Genuine comma-bearing name suffixes ("Cobb, Jr") -- a comma alone is not
#: evidence of a swallowed clause.
_NAME_SUFFIX_RE = re.compile(r",\s*(?:Jr|Sr|II|III|IV|V)\.?$", re.IGNORECASE)


def _name_capture_is_swallowed(name: str) -> bool:
    """True when a `name` capture is really an unmatched lead clause.

    Fail-loud posture: a wrong-but-plausible parse is worse than an honest
    `unparsed[]` entry, because its only downstream symptom is a name that
    never resolves -- which reads as an identity problem and gets triaged as
    one. (It did: issue #33 classified 1,318 such lines as irreducible
    identity ambiguity when 97.5% were this.)
    """
    if _NAME_VERB_RE.search(name):
        return True
    return "," in _NAME_SUFFIX_RE.sub("", name)


def _match_whole_clause(
    clause: str, *, guard: bool = True
) -> Optional[List[RunnerMovement]]:
    """Match ``clause`` against RUNNER_RULES, or None. Preserves the exact
    pre-#40 semantics -- the chaining path below is only ever reached when
    this returns None, so no line that parsed before can parse differently."""
    for regex, _causes, builder in RUNNER_RULES:
        m = regex.fullmatch(clause)
        if not m:
            continue
        if guard and "name" in (m.groupdict() or {}):
            captured = m.group("name")
            if captured and _name_capture_is_swallowed(captured):
                continue
        result = builder(m)
        return result if isinstance(result, list) else [result]
    return None


def _match_continuations(
    rest: str, name_token: str, *, exclude: Tuple[Callable, ...] = ()
) -> Optional[List[RunnerMovement]]:
    """Parse ``rest`` as one or more comma-joined continuation fragments, all
    inheriting ``name_token``. Returns None if any fragment is unrecognized.

    Fragments are consumed LONGEST-FIRST because a continuation can itself
    contain a comma ("scored, unearned"); a shortest-first walk would match
    the bare "scored" row and then choke on a dangling "unearned".
    """
    if not rest:
        return []
    parts = rest.split(", ")
    for k in range(len(parts), 0, -1):
        frag = ", ".join(parts[:k])
        for regex, builder in CONTINUATION_RULES:
            if builder in exclude:
                continue
            m = regex.fullmatch(frag)
            if not m:
                continue
            tail = _match_continuations(
                ", ".join(parts[k:]), name_token, exclude=exclude
            )
            if tail is None:
                continue
            return [builder(m, name_token)] + tail
    return None


def _match_clause_chain(clause: str) -> Optional[List[RunnerMovement]]:
    """Parse ``clause`` as a lead RUNNER_RULES clause plus name-elided
    continuations. Returns None when no such split parses cleanly.

    The lead is tried LONGEST-FIRST so a lead row that legitimately contains
    a comma (e.g. the `advanced to D, scored on an error by f` compound row)
    still wins over a shorter split.
    """
    parts = clause.split(", ")
    if len(parts) < 2:
        return None
    for k in range(len(parts) - 1, 0, -1):
        lead = _match_whole_clause(", ".join(parts[:k]))
        if not lead:
            continue
        tail = _match_continuations(", ".join(parts[k:]), lead[-1].name_token)
        if tail is not None:
            return lead + tail
    return None


#: Clauses that are RECOGNISED but assert no state change, so they produce no
#: RunnerMovement at all rather than a made-up one (issue #40):
#:
#:   "X did not advance"        -- StatCrew spelling out that a runner held.
#:   "X Dropped foul ball, E5"  -- a foul pop-up dropped for an error. The
#:                                 plate appearance is still live; no runner
#:                                 moves and no out is made.
#:
#: Emitting nothing is the honest record for both. Neither loses information:
#: the verbatim line is still the event's narrative, and the charged error
#: (E5) is not a fact anything derives from the play-by-play -- team errors
#: come from the inning-summary oracle (parse.py's `"E": cg.summary.errors`),
#: never from these lines.
#: Tails that carry no base-state assertion and so are stripped before a
#: clause is matched: "assist by 2b" is a fielding credit (nothing in the
#: schema records assists -- the fielding chain that IS recorded comes from
#: the verb's own chain group), and a trailing ", did not advance" is the
#: same held-runner statement _NO_MOVEMENT_RE recognises standalone. Applied
#: repeatedly so a clause can carry both (issue #40).
#: "interference"/"obstruction" join the list: both name why an award was
#: made, not a base-state change, and the runner's own movement clause
#: already carries where he ended up.
#:
#: Anchored with a LOOKAHEAD at a comma-or-end rather than at end-of-string,
#: so a no-state token is stripped wherever it appears. It has to be: the
#: corpus writes "advanced to third, scored, interference, unearned", and an
#: end-only strip left "interference" sitting between "scored" and its own
#: "unearned" -- the pair no longer matched the `scored, unearned` row, the
#: bare `scored` row won instead, and the run was recorded EARNED on a line
#: that says unearned. Stripping mid-clause keeps the two halves adjacent.
_NO_STATE_TAIL_RE = re.compile(
    r",\s*(?:assist by [a-z0-9]+|did not advance|interference|obstruction"
    # See `_strip_no_state_tails` for why ", caught stealing" is here.
    # ", caught stealing" trailing a MOVEMENT clause annotates the play; it
    # does not retire the runner here. StatCrew writes the out on its own
    # following line -- "C. Hanson advanced to second on an error by 2b,
    # assist by p, caught stealing." is followed by "C. Hanson out at third c
    # to ss, caught stealing." Recording an out on both put four outs in an
    # inning. The `out on the play, caught stealing` row already treats the
    # same suffix as adding nothing; this keeps the two readings agreed.
    r"|caught stealing)"
    r"(?=,|$)"
)


def _strip_no_state_tails(clause: str) -> str:
    while True:
        stripped = _NO_STATE_TAIL_RE.sub("", clause).strip()
        if stripped == clause:
            return clause
        clause = stripped


_NO_MOVEMENT_RE = re.compile(
    r"^(?P<name>.+?) (?:did not advance"
    r"|[Dd]ropped foul ball, E\d)$"
)


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
        if _NO_MOVEMENT_RE.fullmatch(clause):
            continue
        clause = _strip_no_state_tails(clause)
        # issue #40, in strict precedence order:
        #   1. a whole-clause RUNNER_RULES match whose name looks like a name;
        #   2. a lead clause plus name-elided continuations.
        #
        # There is deliberately no third, UNGUARDED pass. One existed while
        # the chaining rules were being built, so that no line which parsed
        # before #40 could stop parsing; it also meant the guard never
        # actually rejected anything -- a swallowed name capture just lost a
        # round and then won the next one, which is not failing loud. With
        # the compound rows now covering the shapes it was protecting, its
        # removal costs exactly nothing: measured over all 1,484 games, the
        # clean-parse count, the replayable count and the unparsed-line count
        # are identical with it and without it, with zero regressions.
        movements = _match_whole_clause(clause)
        if movements is None:
            movements = _match_clause_chain(clause)
        if movements is not None:
            runners.extend(movements)
        else:
            return GrammarMiss(
                raw=raw_line,
                reason=f"runner clause not recognized: {clause!r}",
            )
    return runners


#: Tokens that are PRIMARY-clause modifiers rather than runner movement --
#: they may trail a chained continuation ("singled to left field, advanced to
#: second on an error by lf, RBI") and belong on the PrimaryClause, not on a
#: RunnerMovement.
#: Derived from _HIT_MOD_TOKEN rather than re-listing the tokens: this regex
#: had drifted into its own stale copy of the list, so a chained batter
#: self-advance carrying a token the tail rules already accepted still
#: failed here ("singled to right field, advanced to second, obstruction").
#: One source, no drift (issue #40).
_MODIFIER_ONLY_RE = re.compile(rf"^(?:{_HIT_MOD_TOKEN})$")


def _widen(extracted):
    """Extractors return ``(name, fielders, location, modifiers)``; the
    fielder's-choice one appends a fifth and sixth element -- the base of an
    out whose runner the line never names, and the fielding chain thrown to
    record it. Pad the short form rather than touching every other
    extractor."""
    return tuple(extracted) + (None,) * (6 - len(extracted))


def _match_primary_whole(rest: str):
    """Match ``rest`` against PRIMARY_RULES. Returns
    ``((name, fielders, location, modifiers), outcome_type)`` or None."""
    for regex, outcome_type, extractor in PRIMARY_RULES:
        pm = regex.fullmatch(rest)
        if not pm:
            continue
        name = pm.groupdict().get("name")
        if name and _name_capture_is_swallowed(name):
            continue
        return extractor(pm), outcome_type
    return None


def _match_primary_chain(rest: str):
    """Parse ``rest`` as a PRIMARY_RULES lead plus name-elided continuations
    describing the BATTER's own further movement (issue #40).

    StatCrew narrates a batter who keeps running as one clause chain --
    "A.J. Shaver singled to right field, advanced to second" -- and
    PRIMARY_RULES only ever matched the whole thing or nothing. 470 of the
    531 remaining `primary verb not recognized` lines in the 2026 slice are
    this shape.

    Returns ``((name, fielders, location, modifiers), outcome_type,
    batter_movements)`` or None. The batter's movements are emitted as
    ordinary RunnerMovement records keyed to the batter's own name, which
    parse.py's `_merge_same_runner` then folds into one net-path record --
    the same machinery that already handles a runner narrated across two
    clauses.
    """
    parts = rest.split(", ")
    if len(parts) < 2:
        return None
    for k in range(len(parts) - 1, 0, -1):
        head = _match_primary_whole(", ".join(parts[:k]))
        if head is None:
            continue
        name, fielders, location, modifiers, _fo, _fc = _widen(head[0])
        outcome_type = head[1]
        tail_parts = list(parts[k:])
        # Modifier tokens belong to the PRIMARY, not to a movement -- and
        # they are lifted out from ANYWHERE in the chain, not just off the
        # end. The corpus interleaves them: "singled to shortstop, advanced
        # to second on a throwing error by 1b, RBI, advanced to third". An
        # end-only pop left "RBI" sitting in the middle of the movement
        # chain, where no continuation row matches it, so the whole line
        # failed on a token the primary tail already knew how to read.
        #
        # This can only ever ADD parses: a chain carrying a mid-chain
        # modifier returns None today, so no line that parses now takes this
        # path. Relative order is preserved for `_expand_rbi_modifiers`.
        # ... except "unearned" when the chain also carries a RUN. There the
        # token belongs to the run, not the primary: `scored, unearned` is a
        # PAIR that the `^scored<unearned tail>$` continuation row matches as
        # one fragment, and lifting the second half away from the first
        # leaves the bare `scored` row to win, recording as EARNED a run the
        # line says is unearned. `earned` is a real field on the emitted
        # runner record, so this is a wrong fact, not a lost one.
        scoring_tail = any(_SCORED_TOKEN_RE.search(part) for part in tail_parts)

        def _liftable(part: str) -> bool:
            if not _MODIFIER_ONLY_RE.fullmatch(part):
                return False
            return not (scoring_tail and part in ("unearned", "team unearned"))

        trailing_mods: List[str] = [p for p in tail_parts if _liftable(p)]
        tail_parts = [p for p in tail_parts if not _liftable(p)]
        if not tail_parts:
            # Nothing but modifiers followed -- that is a plain hit-tail the
            # existing _HIT_MOD_TAIL should have taken. Decline rather than
            # duplicate its job.
            continue
        # A pickoff retires a BASERUNNER, never the batter. Allowing it as a
        # primary continuation let "X out at first p to 1b, picked off" -- a
        # runner picked off and thrown out -- be claimed by the primary chain
        # as a batter groundout, which both lost an out and invented a plate
        # appearance. 18 games regressed on exactly this.
        movements = _match_continuations(
            ", ".join(tail_parts), name, exclude=(_c_picked_off,)
        )
        if movements is None:
            continue
        mods = list(modifiers) + _expand_rbi_modifiers(trailing_mods)
        return (name, fielders, location, mods), outcome_type, movements
    return None


#: See `parse_clause_group` for what this removes and why it is safe.
_SPLICED_FOUL_BALL_RE = re.compile(r"\s*Dropped foul ball, E\d+,?\s*(?=[^\s.])")


def _parse_clause_group(line: str) -> Union[ClauseGroup, GrammarMiss]:
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
    # A "Dropped foul ball, E<n>" event StatCrew spliced INTO a plate
    # appearance's line instead of emitting it on its own:
    #
    #   "Enzo Apodaca Dropped foul ball, E2, homered to right field (2-2 ...)"
    #   "Wesley Mitchell walkedDropped foul ball, E3 (3-2 BKBBFFB)."
    #   "A. Fernandez Dropped foul ball, E3struck out swinging. (2 out)"
    #
    # -- note the two with no separator at all. The standalone spelling is
    # already recognised as asserting nothing (`_NO_MOVEMENT_RE`): the plate
    # appearance is still live, no runner moves, no out is made, and the
    # charged error is not a fact anything derives from the play-by-play
    # (team errors come from the inning-summary oracle). Removing the spliced
    # copy therefore loses nothing and lets the real outcome be read.
    #
    # The lookahead is what keeps the STANDALONE line intact: there the
    # fragment is followed only by the closing period, so nothing is
    # stripped and `_NO_MOVEMENT_RE` still sees the line it expects.
    stripped = _normalize_ws(_SPLICED_FOUL_BALL_RE.sub(" ", stripped))

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
        nocount_batter_movements: List[RunnerMovement] = []
        whole_nc = _match_primary_whole(primary_raw)
        if whole_nc is None:
            # issue #40: the batter's own trailing self-advance chain, on the
            # NO-COUNT path too. StatCrew omits the whole count-tail for many
            # rows -- overwhelmingly so in the historical template -- and
            # wiring the chain only into the count-tail branch left every
            # count-less line to fall through to RUNNER_RULES, whose greedy
            # name capture then swallowed the lead clause. That is why the
            # 2025 slice gained so much less from the first pass.
            chained_nc = _match_primary_chain(primary_raw)
            if chained_nc is not None:
                extracted, outcome_type, nocount_batter_movements = chained_nc
                whole_nc = (extracted, outcome_type)
        if whole_nc is not None:
            (
                name,
                fielders,
                location,
                modifiers,
                forced_out_at,
                forced_out_chain,
            ) = _widen(whole_nc[0])
            outcome_type = whole_nc[1]
            primary = PrimaryClause(
                name_token=name,
                outcome_type=outcome_type,
                fielders=fielders,
                location=location,
                modifiers=modifiers,
                count=None,
                pitches=None,
                forced_out_at=forced_out_at,
                forced_out_chain=forced_out_chain,
            )

        if primary is not None:
            runners_or_miss = _match_runner_clauses(parts[1:], raw_line)
            if isinstance(runners_or_miss, GrammarMiss):
                return runners_or_miss
            runners_or_miss = list(nocount_batter_movements) + list(runners_or_miss)
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
        if not runner_only and not all(
            _NO_MOVEMENT_RE.fullmatch(part.rstrip(".").strip())
            for part in parts
            if part.strip()
        ):
            # Genuinely nothing matched. A line whose every clause IS a
            # recognized no-movement clause ("X Dropped foul ball, E5.")
            # falls through instead, as a runner_event asserting nothing --
            # recognized-but-empty is a different fact from unrecognized,
            # and only the latter belongs in unparsed[] (issue #40).
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
    batter_movements: List[RunnerMovement] = []
    whole = _match_primary_whole(rest)
    if whole is None:
        # issue #40: the batter's own trailing self-advance chain.
        chained = _match_primary_chain(rest)
        if chained is not None:
            extracted, outcome_type, batter_movements = chained
            whole = (extracted, outcome_type)
    if whole is not None:
        (
            name,
            fielders,
            location,
            modifiers,
            forced_out_at,
            forced_out_chain,
        ) = _widen(whole[0])
        outcome_type = whole[1]
        primary = PrimaryClause(
            name_token=name,
            outcome_type=outcome_type,
            fielders=fielders,
            location=location,
            modifiers=modifiers,
            count=Count(balls=balls, strikes=strikes),
            pitches=pitches,
            forced_out_at=forced_out_at,
            forced_out_chain=forced_out_chain,
        )

    if primary is None:
        if _NO_MOVEMENT_RE.fullmatch(rest):
            # A recognized no-movement line that happens to carry a pitch
            # count ("X Dropped foul ball, E5 (3-1 BBKB)"). The count is NOT
            # carried onto the record: it is a mid-plate-appearance count,
            # and there is no plate-appearance outcome here to attach it to
            # -- inventing one is exactly the silent-wrong-parse failure
            # mode issue #40 exists to remove.
            return ClauseGroup(
                kind="runner_event", runners=(), trailing_outs=trailing_outs
            )
        return GrammarMiss(
            raw=raw_line, reason=f"primary verb not recognized: {rest!r}"
        )

    runners_or_miss = _match_runner_clauses(parts[1:], raw_line)
    if isinstance(runners_or_miss, GrammarMiss):
        return runners_or_miss
    runners_or_miss = list(batter_movements) + list(runners_or_miss)

    return ClauseGroup(
        kind="plate_appearance",
        primary=primary,
        runners=tuple(runners_or_miss),
        trailing_outs=trailing_outs,
    )


def _promote_repeated_subject(line: str) -> Tuple[str, frozenset]:
    """Rewrite the StatCrew defect that re-emits a clause's own subject as a
    bare token, and say which subjects were promoted to a RUN.

    Two positions of one defect. `_b_source_defect_scored` handles the case
    where the doubled name is the WHOLE clause ("M. Moralez M. Moralez.") --
    the verb was lost, and 1.10.0 measured that the missing verb is "scored"
    (linescore oracle reconciles 54/54, null 0/54). This handles the case
    where the name is re-emitted as a TRAILING token on a clause that does
    have a verb:

        "T. Sheehan advanced to third on an error by 3b, T. Sheehan, unearned"

    and it means the same thing. Measured the same way, against the same
    independent oracle:

      * clause does NOT already say "scored" -- the re-emitted name IS the
        missing run. All four such games are short by exactly one run in
        exactly that inning (`linescore: home inning 5 computed 0 != oracle
        1`), and one shows both halves of the signature at once, a missing
        run AND an extra runner left on base. 4 of 4.
      * clause ALREADY says "scored" -- the name is redundant and asserts
        nothing new; dropping it is correct. Three such lines, no linescore
        discrepancy in any of their innings.

    So the reading is conditional, and this returns the promoted subjects so
    the caller can tag the resulting movement `inferred` -- it belongs in the
    game's top-level `inferred[]` under the same `doubled_name_scored` rule
    name as its sibling, never passing as something the line said in words.

    An earlier pass here dropped the token unconditionally, on the reasoning
    that a clause with a verb needs no help. That was wrong, and it was wrong
    in the falsifiable direction: the runners were left stranded and the
    linescore check failed on precisely those games. The oracle caught it.
    """
    promoted = set()
    segments = []
    for segment in line.split(";"):
        verb = _NAME_VERB_RE.search(segment)
        if verb is None or verb.start() == 0:
            segments.append(segment)
            continue
        subject = segment[: verb.start()].strip()
        # A subject carrying its own comma ("Cobb, Jr") would be split by the
        # substitution below; leave it alone rather than risk a real name.
        if not subject or "," in subject:
            segments.append(segment)
            continue
        repeated = re.compile(rf",\s*{re.escape(subject)}(?=,|\s*\.?\s*$)")
        if not repeated.search(segment):
            segments.append(segment)
            continue
        if _SCORED_TOKEN_RE.search(segment):
            segments.append(repeated.sub("", segment))
        else:
            segments.append(repeated.sub(", scored", segment))
            promoted.add(subject)
    return ";".join(segments), frozenset(promoted)


_PROMOTED_SCORE_NOTE = (
    "source line re-emits the runner's own name as a bare token on a clause "
    "that records no run; read as 'scored' (linescore oracle reconciles 4/4 "
    "in the affected innings, and the same defect standalone measured 54/54)"
)


def parse_clause_group(line: str) -> Union[ClauseGroup, GrammarMiss]:
    """Parse one verbatim PBP narrative line into a ``ClauseGroup``.

    Never raises on unrecognized input -- returns a ``GrammarMiss`` with a
    reason and the untouched original ``line`` instead.

    Wraps the matcher with the repeated-subject source-defect correction, so
    that defect is handled in exactly ONE place for both the primary clause
    and the runner clauses (it occurs in both).
    """
    rewritten, promoted = _promote_repeated_subject(line)
    result = _parse_clause_group(rewritten)
    if isinstance(result, GrammarMiss):
        # Report the caller's own line, never this function's working copy.
        return result if rewritten == line else GrammarMiss(
            raw=line, reason=result.reason
        )
    if not promoted:
        return result
    runners = tuple(
        replace(rm, inferred=_PROMOTED_SCORE_NOTE)
        if (rm.scored and not rm.inferred and rm.name_token in promoted)
        else rm
        for rm in result.runners
    )
    return replace(result, runners=runners)
