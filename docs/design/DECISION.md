# Design Decision — game-file JSON Schema (issue #16, epic #15)

## 1. Convergence verdict (human, fredcai6, 2026-07-11)

> **Hybrid: Candidate B as the base, plus three grafts from Candidate C.** B's asserted-primitives
> spine + regenerable `_derived` cache; grafting C's (1) player-ref objects so a future
> cross-season `person_id` lands in one place, (2) `source.league_id`, (3) `meta.parse` integrity
> block (replayable / unparsed_count / warnings).

Human standing guidance:

> "if we learn any lessons we shouldn't feel bad to go back and update this with more knowledge" —
> schema evolution via additive MINOR bumps and labeled re-parse commits is expected and welcome.

## 2. The three candidates

- **A — MINIMAL-INTERFACE.** A single `events[]` spine; the other blocks are verification oracles
  and initial conditions rather than parallel truths.
- **B — ANALYSIS-CALLER-FIRST** *(chosen base)*. Asserted primitives lifted directly from the
  source, plus a regenerable `_derived` cache that carries base-out state for fast analysis.
- **C — MAX-FORWARD-FLEXIBILITY.** State-snapshot events and a graftable `source` block, designed
  to absorb future needs without breaking write-once files.

Full texts are the sibling `SCHEMA_CANDIDATE_A.md`, `SCHEMA_CANDIDATE_B.md`, and
`SCHEMA_CANDIDATE_C.md` files in this directory.

## 3. The precise hybrid definition (what the schema implements)

- Top-level shape, the `players` table, `linescore`, `box`, `lineups`, `events[]` (the common
  envelope, `kind`, `runners[]` as from→to primitives with causes, the closed outcome taxonomy
  per Candidate B §6.4), `unparsed[]`, the `_derived` cache semantics (excluded from semantic
  equality; never hand-authored; CI-stamped), and the reference materializer: **all from
  Candidate B.**
- **GRAFT 1 — player refs are C-style objects** but keep B's field names `{player_id, name_raw,
  resolved}`. A future cross-season `person_id` is an additive MINOR field on the players-table
  entry (the per-game identity home).
- **GRAFT 2 — a root `source` object** `{provider: "prestosports", league_id: "pioneer", site}`.
  B's `meta.source_url` / `source_sha256` stay in `meta` as provenance.
- **GRAFT 3 — `meta.parse` integrity block from C:** `{events_count, unparsed_count, replayable,
  warnings[]}`.
- **Versioning:** B's semver `schema_version` at root, with C's MAJOR.MINOR additive-evolution
  rules (§5 below).
- **NOT grafted:** C's `x` escape hatches and C's open `type` enum. The taxonomy stays
  **CLOSED** — unknown lines go to `unparsed[]`, loudly.

## 4. Semantic-equality definition (caller-visible contract)

Two files are the same game **iff** they are deep-equal after deleting `meta` and every `_derived`
block. (This also appears in the README and in the schema's root `$comment`.)

## 5. Schema evolution rules (MAJOR.MINOR)

- **MINOR = additive only:** a new optional property, a new stat column, a new `_derived` key.
  Old external readers ignore unknown fields and keep working. No file is invalidated.
- **MAJOR = anything else** (remove/rename/retype a field, change a unit, redefine a closed enum,
  change a value's meaning). A MAJOR bump is the ONLY thing that may invalidate an existing file,
  and it is exactly the "labeled re-parse commit" case. Later MINOR / labeled-re-parse updates are
  EXPECTED and welcome — do not over-freeze.

## 6. B/C conflict log (resolved under the Admiral's prefer-B pre-ruling)

- **Where B and C conflict on a detail the hybrid definition does not settle, prefer B.**
- **Synthetic player-id fallback (forced by real data + zero-fetch).** Candidate B assumed the
  file is parsed from the pioneerleague.com copy, which links BOTH teams' players with 16-char
  Presto ids. The only on-disk sample (`boxscore_20260709_final.html`) is the **team-site**
  (longbeachcoast.com) copy: it links only the home team's players (12 ids); away batters render
  as plain text with NO id. With zero network fetches allowed, the schema must be able to encode
  this real game. Resolution: the `player_id` (file-local join key) pattern admits BOTH a 16-char
  Presto id AND a file-local synthetic `syn:<side>:<n>` (adopted from Candidate A). This keeps the
  event→player join TOTAL and records identity honestly (a `syn:` prefix signals "no source id"),
  WITHOUT weakening the schema. A future `person_id` (cross-season identity) still lands on the
  players-table entry as an additive MINOR field. This refinement was surfaced to the Admiral for
  ratification (join-key format is caller-visible).

## 7. Schema evolution addendum

- **2026-07-12 — `schema_version` 1.0.0 → 1.1.0 (additive MINOR).** `$defs.substitution.slot`
  made nullable (`["integer","null"]`, min/max still constrain the integer branch); `null` means
  the substitute is not in the batting order — a DH-game pitching change. **Why:** the 1.0.0 shape
  required `slot: 1-9`, but a pitching change in a DH game (effectively every Pioneer League game)
  puts a new pitcher on the mound with no batting-order slot. Under 1.0.0 those lines could not be
  honestly encoded and fell to `unparsed[]` (issue #19 found 5 in the sole sample). Ratified by the
  human via the Admiral (issue #19 float). Additive-only: every existing 1.0.0 file (all integer
  slots) still validates under 1.1.0; only new files use `null`. This is the fixture-promotion
  protocol's first real exercise (unparsed line → schema/rule lands → real event); the sample
  re-parses from events 117 / unparsed 5 to events 122 / unparsed 0.
- **2026-07-12 — `schema_version` 1.1.0 → 1.2.0 (additive MINOR).** `$defs.count` (the
  `plate_appearance` event's `count` field) and `$defs.substitution.player_out` both made nullable.
  `count: null` means the source PBP line carries no count-tail at all (the historical league
  template omits it for some rows, not just the pitch-sequence letters — distinct from the
  pre-existing `pitches: null` case, which is a real 0-0 count with no observed pitch sequence).
  `player_out: null` means a bare DH-slot-entry line (`"<name> to dh."`) names only the incoming
  player, with no honest way to supply an outgoing one. **Why:** both shapes were falling to
  `unparsed[]` under 1.1.0 (count-less plate-appearance lines caused a `parse.build_events` crash
  once grammar-level support landed in g1/g2; the DH-slot-bare line had no grammar rule at all
  because the schema could not yet encode it). Ratified by the human via the Admiral (issue #30
  float, this is not the implementer's own decision). Additive-only: every existing 1.1.0 file
  (`count` and `player_out` always present as objects/strings, never null) still validates under
  1.2.0; only new parses may emit `null` for either field. The two-name `"<in> to dh for <out>."`
  variant remains intentionally unimplemented (out of this gate's authorized scope) and still falls
  to `unparsed[]` unchanged.
- **2026-07-17 — `schema_version` 1.2.0 → 1.3.0 (additive MINOR).** `$defs.outcome.properties.type.enum`
  gains `"foul_out"` and `"strikeout"` (closed taxonomy 17 → 19). `foul_out` (`"<name> fouled out to
  <pos>."`) is a foul fly ball caught for an out, `outs_recorded=1`, `fielders=["<pos>"]` populated
  exactly like `flyout`/`popout` (a human hard requirement: no defensive-info loss — the position
  chain is preserved for offense/defense analysis, verified infield AND outfield). `strikeout` is a
  bare `"<name> struck out."` carrying no swinging/looking qualifier, `fielders=[]`. **Why:** both were
  high-frequency `unparsed[]` residues under 1.2.0 (`foul_out` alone: 881 lines / 531 games / 42% of
  the corpus) that mapped to no existing outcome type and could not be shoehorned into `popout`/`flyout`
  without a position-based judgment the source never states. Ratified by the human via the Admiral
  (issue #31 float). Additive-only: every existing 1.2.0 file still validates under 1.3.0. Landed in the
  issue #31 labeled re-parse (`reparse(v0.3.0)`), which also implemented the two-name
  `"<in> to dh for <out>."` DH-sub variant noted above as unimplemented (issue #32, now covered:
  47/47 grammar-parse, 44/47 resolve end-to-end via a try-both-sides identity resolution).

## 8. Sibling artifact — frequency schema (issue #21, epic #15)

`game.schema.json` stayed **frozen at 1.3.0** for issue #21 (per the launch order's pre-ruling) — the
season team/player event-frequency artifact got its OWN new schema (`schemas/frequencies.schema.json`,
Draft 2020-12), not a graft onto the game schema. Two design decisions this run made under its own
"design latitude" grant (File Ownership: "your design"), recorded here as the durable design record since
no Cartographer architecture map exists for this repo:

- **Batting + pitching split, both team and player level.** `frequencies.py` aggregates
  `events[].outcome.type` (the same closed 19-type taxonomy as `game.schema.json`) into TWO sub-tables per
  key — `batting` (keyed by `batting_team`/`batter.player_id`, offense) and `pitching` (keyed by
  `fielding_team`/`pitcher.player_id`, what that team/player faced) — at both the team level and the
  player level. Chosen over a batting-only design because it costs nothing extra (same single pass over
  `events[]`, just two keyings) and matches how a real box-score/stat page presents both sides. Stays
  strictly CONTEXT-FREE (still a direct `outcome.type` count, just keyed two ways) — no base-state/
  run-expectancy/LOB/win-probability derivation, which remains out of scope (roadmap #26).
- **One combined artifact file.** `artifacts/latest/frequencies.json` — a single file with `league`/
  `by_season` nesting mirroring `completeness.json`'s existing shape — rather than per-season files.
  Keeps the derived tier's whole v1 surface at two sibling top-level files (`completeness.json`,
  `frequencies.json`), consistent and easy for issue #22's site read path to consume.
- **Rate definition**: `rate = outcome_type_count / total_plate_appearances_for_that_key` (batting: PAs
  that team/player batted in; pitching: PAs that team/player faced) — documented in `frequencies.py`'s own
  module docstring, the artifact's authoritative source.

Both decisions were cold-critic-reviewed at plan time (`.agent-work/epic-15/commander-21/
PLAN_RIGOR_RECORD.md`) and independently re-verified by a reviewer against a hand-count on a real game file
different from the implementer's own (`.agent-work/epic-15/commander-21/crew-handoffs/
g1-review-result.md`) before being adopted.
