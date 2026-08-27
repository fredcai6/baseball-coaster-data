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
- **2026-08-27 — `schema_version` 1.5.0 → 1.6.0 (additive MINOR).** `$defs.outcome.properties.type.enum`
  gains `infield_fly` (closed taxonomy 21 → 22). **Why:** under the infield fly rule, with runners on
  and fewer than two out, the batter is declared out on a catchable infield pop-up **whether or not it
  is caught**. Folding it into `popout` or `flyout` would assert the opposite — that the out depended
  on the catch — which is exactly what those two types mean. 19 lines across 19 games, every one of
  them the sole blocker on an otherwise clean-parse game. `fielders` carries the named position, the
  same no-defensive-info-loss requirement that shaped `foul_out` at 1.3.0. Implied runner primitive is
  `("putout", None, True, False)`, identical to the other batter-retired types, so `check_pa_counts`
  needs no change: the batter is charged an at-bat and is already inside `box.AB`. Human-ratified
  ("okay with schema update for infield fly", issue #40). Additive-only: every existing 1.5.0 file
  still validates under 1.6.0. Lands in the next labeled re-parse.

- **2026-08-27 — `schema_version` 1.4.0 → 1.5.0 (additive MINOR).** `$defs.outcome.properties.type.enum`
  gains `reached_on_interference` and `batter_interference` (closed taxonomy 19 → 21). **Why:** neither
  maps onto an existing type. *Catcher's interference* awards the batter first base with **no error
  charged**, and is a plate appearance that is **not an at-bat** — `reached_on_error` and
  `fielders_choice` misattribute the cause *and* are at-bats. *Batter's interference* is an out with
  **no batted ball**, which nothing in the taxonomy covers. 25 lines across 25 games, 14 of them the
  sole blocker on an otherwise clean-parse game. `fielders` carries `"c"` on
  `reached_on_interference` (the catcher is the responsible fielder), preserving the same
  no-defensive-info-loss requirement that shaped `foul_out` at 1.3.0.

  **Paired oracle-definition change, deliberately separate:** `replay.check_pa_counts`'s formula
  becomes `events_PA == box.AB + box.BB + hbp_events + sac_events + interference_events`. Catcher's
  interference sits in neither AB nor BB, so without the new term every such plate appearance would
  fail by exactly one. `batter_interference` is deliberately NOT in the formula — the batter is
  retired, which IS charged as an at-bat, so it is already inside `box.AB`. Issue #33 requires oracle
  definition changes be deliberate and separately tested; this one is both.

  A runner retired on an interference play (`"C. Booth out on the play, interference."`) needs **no**
  new cause — `putout` already fits — so `$defs.runner.properties.cause` stays frozen at 12.

  Human-ratified via the Admiral (issue #40). Additive-only: every existing 1.4.0 file still validates
  under 1.5.0. Lands in the next labeled re-parse.

- **2026-08-27 — `schema_version` 1.3.0 → 1.4.0 (additive MINOR).** `$defs.player_entry.properties`
  gains `box_listed` (boolean, not required). **False** marks a player who appears ONLY in the PBP
  narrative and has no row in the boxscore's Batters/Pitchers tables. **Why:** StatCrew omits an
  all-zero box row for a player who entered and then never batted or reached — 91% of this
  population, measured over 633 games — and rarely omits one who DID record a plate appearance
  (e.g. `J. Kennedy grounded out to p.` with `Kennedy` nowhere in the box). Previously such a
  substitution line failed name resolution and landed in `unparsed[]`, losing the whole
  substitution EVENT and not merely a stat line. The PBP is authority that the player was in the
  game, so they are now admitted to `players` with a synthetic pid, anchored by the other name in
  the same substitution resolving cleanly on exactly one side (never guessed — without that anchor
  the line still fails loud). `box_listed=False` is what keeps this honest: a consumer can never
  mistake "no box row" for "zero stats", and the replay oracle's box-derived checks can see which
  players they have no row to reconcile against. Absent means True, so every pre-1.4.0 file is
  boxscore-derived by definition. Human-ratified: "im okay with our derived box being better than
  the real one … maybe we want to have an 'invisible' bit to make it clear when we're on this
  corner case" (issue #40). Additive-only: every existing 1.3.0 file still validates under 1.4.0.
  Lands in the next labeled re-parse.

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

## 9. `person_id` — within-season cross-game identity (issue #41 Gap 2, epic #15)

Schema **1.7.0** (additive MINOR): `$defs.player_entry` gains optional, nullable `person_id`.

### The problem the field exists to solve

`player_id` is FILE-LOCAL. A real 16-char Presto id happens to be stable for a whole season, but a
synthetic `syn:<side>:<n>` is assigned by boxscore ROW ORDER, so the same value denotes a different
person in every file. Measured on the 1,484-game corpus: **`syn:away:8` is bound to 99 distinct
display names in 2026 alone.** Any consumer joining on `player_id` across games silently mixes people
together — and 10.0% of player records (4,189) carry a synthetic id, 29.5% of the 2026 season.

### The key: `(season, team_id, name)`

`team_id` is the decision that makes this tractable, and it was not obvious from issue #41's framing.
It is **always a real Presto id, never synthetic** — 0 synthetic team ids in the corpus — even on the
team-site pages whose *player* rows carry no ids at all. Adding it to the key collapses the apparent
ambiguity: the **120** `(season, name)` pairs holding more than one real id drop to **4** once team is
included. The other 116 were the same name on different teams, which is a different question (see
"Not attempted").

Note this makes the `M. Jackson` collision a non-threat here. `Manny Jackson` and `Marquis Jackson`
are distinct display names holding distinct real ids on one roster; the ambiguity that motivated so
much of #31 lives in **PBP narrative resolution** (`identity.resolve`), not in the identity table.

### Assignment rules

A **real** `player_id` is its own `person_id`, unconditionally — it is already the stable within-season
key, and it is never absorbed into another person. Only **synthetics** resolve through their group:

| classification | rule | groups | synthetic records |
|---|---|---|---|
| `real_anchor` | group holds exactly one real id → that id is canonical | 1,781 | 3,342 |
| `minted` | group holds no real id → mint `person:<16 hex>` from the key | 168 | 835 |
| `multi_real_id` | ≥2 real ids → UNLINKED, which one is not determined | 4 | 0 |
| `same_game_conflict` | two of the group's ids occur in one game → UNLINKED | 1 | 4 |
| `non_person_name` | the "name" is `/` (the `/ for X` source defect) → UNLINKED | 7 | 8 |

**Refusals are evaluated before the linking rules**, so a defect can never be merged away by an anchor
that happens to exist. Result: **4,177 of 4,189 synthetic records (99.7%) linked**, 12 unlinked, every
one carrying a reason. Consistent with the standing doctrine — link on strong evidence, enumerate the
rest, never guess; an unlinked player is a measured negative, not a failure.

A lone synthetic still gets minted rather than left alone: `syn:away:3` appearing in only one game is
still a value other people hold in other games, so leaving it would be both unjoinable and colliding.

### Where it lives, and why both places

The authority is **`artifacts/latest/person_map.json`** (mutable tier, caller-contract clause 2),
regenerable like `frequencies.json`. The `person_id` on `player_entry` is a **materialized copy**
refreshed at re-parse time. That split is deliberate: `games/**` is write-once, so a mapping that will
be revised as evidence improves cannot be *owned* by a game file — but a consumer joining across games
should not have to load a 3.8 MB side artifact to do it.

The consequence is stated in the schema description rather than left implicit: a game added since the
last re-parse may carry `null` until the artifact is regenerated and the corpus re-parsed.

The artifact is derived FROM `games/**` and written back INTO it. That is safe only because the map is
a function of fields a re-parse does not change (season, team_id, name, player_id) — proven by a unit
idempotence test and, in practice, by `--check-no-commit` reporting NO-OP when regenerated against the
rewritten corpus.

### Minted id format

`person:` + first 16 hex of `sha256("<season>\x1f<team_id>\x1f<name>")`. A pure function of the group
key, so regeneration reproduces it on any machine with no counter or ordering dependency. The
`person:` prefix keeps it out of both other id namespaces (real is bare `[a-z0-9]{16}`, synthetic is
`syn:<side>:<n>`), so a minted id can never be mistaken for a `player_id`. Hashed rather than spelled
out to keep it short enough to sit in every `player_entry`; the artifact always carries the full key
alongside, so the mapping stays invertible for a human.

### Not attempted, and reported as measured negatives

- **Cross-team within a season** — 134 `(season, name)` pairs sit on more than one team. Not linked:
  nothing in the identity table separates a mid-season move from two people with the same name.
- **Cross-season (Gap 1)** — not attempted. PrestoSports reissues every player id AND every team id
  each season, so no id-based signal survives a season boundary and team continuity is unavailable as
  corroboration. `person_id` is explicitly within-season.

Both are emitted in `meta.not_attempted` so a consumer reads a number rather than inferring silence.

### Version-history gap

`game.schema.json`'s `$comment` VERSION HISTORY runs 1.0.0 → 1.3.0 and then jumps to this entry:
**1.4.0 (`box_listed`), 1.5.0 (interference) and 1.6.0 (`infield_fly`) landed without a history
entry.** Recorded here rather than reconstructed, since the reconstruction would not be evidence.
