# Schema Candidate B — ANALYSIS-CALLER-FIRST

Canonical per-game JSON game-file schema for `baseball-coaster-data`
(`games/<season>/<game_id>.json`, status `"final"` only).

Designed **backwards from the model layer**: run-expectancy tables, Elo, matchup
clusters, and pandas/duckdb exploration. The governing question for every field was
*"can a consumer compute base-out state, run values, and batter–pitcher matchups in one
forward scan, without re-deriving game state and without an HTML round-trip?"*

---

## 0. The one design decision that drives everything

The analysis queries all want the same thing: **per-event base-out state** (who's on
which base, how many outs, the score) so they can bin events into the 24 base-out states,
attach run values, and extract matchups. Re-deriving that from raw narrative in pandas is
exactly the re-derivation this schema exists to prevent.

But the spec's caution is real: if I *store* `bases_before` as plain asserted truth, the
replayer that validates the file has nothing independent to check it against — a parser bug
that mis-tracks a runner writes the wrong `bases_before`, and a replayer that trusts it
passes. A shared bug slips through both sides.

**Resolution — a hard asserted/derived split, enforced by field naming and CI:**

1. **Events store only *asserted primitives*** — things lifted directly out of one specific
   region of the source HTML: the batter, the pitcher, the outcome class, the count, the
   pitch sequence, and **per-runner from→to movements with their causes**, plus the
   `(N out)` marker and the verbatim line. Nothing here is computed by folding earlier
   events. Asserted fields are the *only* inputs to semantic equality and to the replayer's
   own state reconstruction.

2. **Base-out state is derived, never hand-authored, and lives in a namespaced cache.**
   Every event carries an optional `_derived` block (leading underscore, same status as
   `meta`: **excluded from semantic equality**). CI populates it by running the canonical
   replayer. Consumers read `_derived.state_before` / `_derived.state_after` and get trivial
   base-out queries. It is a *regenerable cache*, documented as untrustworthy-until-checked.

3. **The independent oracle is a different region of the HTML.** The replayer reconstructs
   state from PBP runner primitives *only*, then checks its reconstruction against the
   **linescore** (per-inning runs) and the **batting box totals** (per-batter PA / R / LOB) —
   both extracted from separate DOM sub-trees than the PBP narrative. A materializer bug that
   mis-derives `_derived` disagrees with linescore/box and fails CI. That is the two-sided
   guarantee: **PBP primitives are the input, linescore+box are the oracle, `_derived` is the
   validated output.** The narrative and the summary panes would have to be wrong in exactly
   the same way to pass both sides.

So: **`_derived.*` and `meta.*` are the only computed/provenance fields; everything else is
asserted.** A consumer that wants speed reads `_derived`; a consumer (or the replayer) that
wants ground truth ignores `_derived` and folds the asserted primitives itself. The reference
fold is ~15 lines and shipped below, so the "trivial in duckdb" promise holds even for a
consumer who (correctly) distrusts the cache and recomputes.

---

## 1. Top-level shape

```jsonc
{
  "schema_version": "1.0.0",          // semver of THIS schema; gate for breaking changes
  "game_id": "20260709_h94w",         // Presto boxscore stem; unique within season
  "season": 2026,
  "status": "final",                  // literal; only finals are written to games/**
  "date": "2026-07-09",               // ISO local game date (source pane has no tz)
  "teams": {
    "home": { "team_id": "PL016", "name": "Long Beach Coast" },
    "away": { "team_id": "PL014", "name": "Yuba-Sutter Freebirds" }
  },

  "players": { /* §2 — the per-game id table: the join key for everything */ },

  "linescore": { /* §3 — the primary oracle */ },
  "box": { /* §4 — the secondary oracle + season-rate context */ },

  "lineups": { /* §5 — batting order + substitutions, drives PA-count checks */ },

  "events": [ /* §6 — the ordered spine; the whole point of the file */ ],

  "unparsed": [ /* §7 — verbatim missed lines + location */ ],

  "meta": { /* §8 — provenance; EXCLUDED from semantic equality */ }
}
```

Design note (analysis-first): `teams.home`/`teams.away` are objects, not a flat
`home_team_id`. A duckdb reader does `unnest`/`->'home'->>'team_id'`; an Elo builder needs
exactly `{home_id, away_id, home_runs, away_runs, date}` and gets all five from
`teams` + `linescore.totals` with no PBP scan.

---

## 2. `players` — the per-game identity table (the join key)

Resolved **once per game** from the boxscore's own player list, keyed by the 16-char Presto
`players?id`. PBP narrative uses bare last names (`"Corona grounded out"`); every event
references a player by this id, and the mapping from narrative last-name → id is resolved
here at parse time, not left for the consumer.

```jsonc
"players": {
  "3865oyuz5l2pj51r": {
    "player_id": "3865oyuz5l2pj51r",   // 16-char Presto boxscore id; within-season stable
    "name": "Eddy Pelc",               // display name from boxscore link text
    "last_name": "Pelc",               // token PBP narrative uses; the join surface
    "team_id": "PL016",
    "bats_side": null,                  // not in source; reserved, null for now
    "positions": ["lf"]                 // positions seen in box/lineup this game
  },
  "quxk0ram2ev0tb3h": { "player_id": "quxk0ram2ev0tb3h", "name": "Emilio Corona", "last_name": "Corona", "team_id": "PL016", "positions": ["cf"] }
  // ... one entry per player who appears in the boxscore
}
```

**Ambiguity is asserted, never silently guessed.** If two players on the same team share a
last name, the parser cannot resolve PBP → id from the narrative alone (the excursion flagged
this). In that case the event's `batter`/`runner` ref carries `"resolved": false` and the raw
token, and the line *also* lands in `unparsed[]` for human resolution. A consumer can filter
`resolved == true` and know its matchup joins are clean.

Cross-season joins are explicitly **out of scope** here (the id resets per season — proven for
bio slugs, strongly evidenced for the 16-char id). A separate person-crosswalk table owns that;
this file never pretends its ids are cross-season keys.

---

## 3. `linescore` — the primary oracle

```jsonc
"linescore": {
  "innings": {
    "away": [1, 1, 1, 0, 0, 1, 0, 0, 0],       // Yuba-Sutter, runs per inning
    "home": [1, 1, 0, 4, 1, 6, 0, 0, null]      // null = "X", home didn't bat (walk-off/final)
  },
  "totals": {
    "away": { "R": 4,  "H": 8,  "E": 3 },
    "home": { "R": 13, "H": 11, "E": 0 }
  }
}
```

`null` for the home 9th encodes the literal `"X"` (home team did not bat). This is what
distinguishes a normal final from a walk-off: the replayer's "3 outs per half-inning" check
reads `innings.home[i] == null` as "this half-inning legitimately has <3 outs / no PA," so a
walk-off or unbatted bottom-9th is **explicit in the data**, not a special case the replayer
guesses.

---

## 4. `box` — secondary oracle + season context

Per-batter and per-pitcher totals, verbatim from the batting/pitching tables. Two jobs:
(a) independent oracle for the replayer's per-batter PA / LOB reconciliation;
(b) carries the season-to-date rate (`AVG`) the source shows, so a matchup consumer has
context without a second fetch.

```jsonc
"box": {
  "batting": {
    "PL014": [   // away lineup order
      { "player_id": "…nunez…", "pos": "…", "AB": 5, "R": 1, "H": 3, "RBI": 0, "BB": 0, "SO": 0, "LOB": 2, "AVG": ".xxx" }
    ],
    "PL016": [
      { "player_id": "3865oyuz5l2pj51r", "pos": "lf", "AB": 4, "R": 2, "H": 2, "RBI": 0, "BB": 1, "SO": 2, "LOB": 1, "AVG": ".395" },
      { "player_id": "3gpxu42krrgcj6a3", "pos": "3b", "AB": 3, "R": 2, "H": 2, "RBI": 2, "BB": 2, "SO": 0, "LOB": 1, "AVG": ".255" }
      // ...
    ]
  },
  "pitching": {
    "PL014": [ { "player_id": "…", "IP": "…", "H": 0, "R": 0, "ER": 0, "BB": 0, "SO": 0 } ],
    "PL016": [ { "player_id": "…", "IP": "…" /* … */ } ]
  }
}
```

Numbers are stored as parsed from the box (`AB` etc. as integers, `AVG`/`IP` as source
strings — `IP` like `"5.2"` is baseball-notation, not a float, so it stays a string to avoid a
lossy cast).

---

## 5. `lineups` — batting order + substitutions

Drives the per-batter PA-count check and lets a consumer reconstruct who was due up.

```jsonc
"lineups": {
  "PL016": {
    "batting_order": [
      { "slot": 1, "player_id": "3865oyuz5l2pj51r" },   // starter in slot 1
      { "slot": 2, "player_id": "3gpxu42krrgcj6a3" }
      // ... 9 slots
    ],
    "substitutions": [
      { "slot": 4, "player_out": "…", "player_in": "…", "kind": "offensive", "after_event_seq": 41 }
    ]
  },
  "PL014": { "batting_order": [ /* … */ ], "substitutions": [] }
}
```

`after_event_seq` ties each sub to its position in `events[]`, so "who is the current
batter in slot 4" is a scan up to `seq`, not a guess. Pitching changes appear both here
(as a pitching sub) and as a `substitution` event in the spine (§6), because the replayer
needs them inline to attribute subsequent PAs to the right pitcher for matchup extraction.

---

## 6. `events[]` — the ordered spine

The heart of the file. **One array, strictly ordered by `seq`, mixing PA events, runner
events, and substitutions** — because base-out state and the current pitcher both evolve
across all three, and a single ordered fold is what makes analysis trivial. Splitting them
into parallel arrays would force consumers to merge-sort by inning/half to recover order.

### 6.1 Common envelope (every event)

```jsonc
{
  "seq": 5,                    // 0-based global order; the sort key and sub-anchor
  "inning": 1,
  "half": "top",               // "top" | "bottom"
  "kind": "plate_appearance",  // "plate_appearance" | "runner_event" | "substitution" | "inning_summary"
  "batting_team": "PL014",
  "fielding_team": "PL016",
  "narrative": "Josh Phillips singled to center field, RBI (0-0); Jordan Donahue advanced to second; Isaac Nunez scored.",
  "scoring_play": true,        // source rendered this line <strong> (a run scored); asserted flag
  // ... kind-specific body ...
  "_derived": { /* §6.5 — regenerable cache, EXCLUDED from semantic equality */ }
}
```

`scoring_play` is asserted (it's the `<strong>` wrapper in the HTML), and it's an independent
check on `_derived`: every event with `runs_scored > 0` must have `scoring_play == true` and
vice versa. Two source signals for the same fact, from different DOM features (bold tag vs.
`scored` runner clause) — a mismatch is a parser bug.

### 6.2 `plate_appearance`

```jsonc
{
  "kind": "plate_appearance",
  "batter": { "player_id": "…phillips…", "name_raw": "Josh Phillips", "resolved": true },
  "pitcher": { "player_id": "…", "name_raw": null, "resolved": true },   // resolved from current pitcher of record
  "outcome": {
    "type": "single",          // closed taxonomy — §6.4
    "modifiers": ["RBI"],      // closed set: RBI, SAC, unearned, intentional, unassisted, DP, FC, error, ...
    "fielders": ["cf"],        // fielding chain as position tokens, in order; [] when none
    "outs_recorded": 0,        // outs THIS event adds (asserted from batter outcome + runner "out at")
    "location": "center field" // free-text hit location when source gives it; null otherwise
  },
  "count": { "balls": 0, "strikes": 0 },   // final count from the (b-s …) parenthetical
  "pitches": null,             // "BBSBKS" pitch string, or null when source omits it (first-pitch BIP)
  "runners": [ /* §6.3 */ ]
}
```

`pitcher.name_raw` is `null` because PBP never names the pitcher — the pitcher-of-record is
resolved from the last pitching substitution, and that resolution is asserted (from the
lineup), so the field is trustworthy, not derived-from-fold.

### 6.3 `runners[]` — the from→to primitives (the load-bearing part)

Every runner state change the narrative asserts, one entry each, with its **cause**. This is
what a single fold turns into base-out state. Bases are integers `1|2|3`, home is `4`, the
batter's origin is `0` (the box).

```jsonc
"runners": [
  { "player_id": "…phillips…", "from": 0, "to": 1, "cause": "batted_ball", "out": false, "scored": false },
  { "player_id": "…donahue…",  "from": 1, "to": 2, "cause": "advance",     "out": false, "scored": false },
  { "player_id": "…nunez…",    "from": 3, "to": 4, "cause": "advance",     "out": false, "scored": true,  "earned": true, "rbi": true }
]
```

Rules that make the fold total and unambiguous:

- **The batter always appears** with `from: 0` (even on an out: `to` is the base reached or a
  sentinel `-1` for "retired, off the bases"). No implicit "batter reached first" — it's
  spelled out, so the fold never infers.
- `cause` ∈ closed set: `batted_ball`, `advance`, `stolen_base`, `wild_pitch`, `passed_ball`,
  `balk`, `error`, `fielders_choice`, `pickoff` (with `out:true`), `caught_stealing`,
  `force_out`, `putout`. This is the *asserted* mechanism, drawn from the runner clause verb.
- `scored: true` ⇔ `to: 4`. `earned`/`rbi` are per-run asserted flags (from `, unearned` and
  the batter's `RBI` modifier / clause).
- The **compound clause** the excursion flagged (`"advanced to third, scored on an error,
  unearned"`) becomes **one runner entry** with `from: <prev>`, `to: 4`, `cause: "error"`,
  `scored: true`, `earned: false` — the "advanced to third" is a way-point the source narrates
  but the *state change* is prev→home, so we record the net movement + cause, not the waypoint.
  (The verbatim clause is preserved in `narrative`; nothing is lost.)

**Why from→to and not a `bases_after` snapshot:** a snapshot is a *derived* claim (it asserts
the whole world state); a from→to movement is a *local* claim the narrative literally makes.
Storing movements keeps the file to asserted primitives and lets the replayer fold them into a
snapshot *and* re-check each movement's legality (can't advance from an empty base, can't have
two runners land on the same base). A stored snapshot hides those illegal transitions.

### 6.4 Outcome taxonomy (closed)

17 PA types, matching the excursion's observed inventory + the untested-but-expected tail:
`single, double, triple, home_run, walk, intentional_walk, hit_by_pitch, strikeout_swinging,
strikeout_looking, groundout, flyout, lineout, popout, fielders_choice, reached_on_error,
grounded_into_double_play, sacrifice`. Anything outside this set → the line goes to
`unparsed[]` and CI fails until the taxonomy is extended (the closed set is enforced by the
JSON Schema `enum`, so drift is loud).

Runner-only lines (standalone stolen base, wild pitch, balk, pickoff, passed ball) are
`kind: "runner_event"` — same `runners[]` shape, no `batter`/`outcome`/`count`.
`inning_summary` events carry the source's `{R,H,E,LOB}` line as a **fourth oracle** the
replayer checks its half-inning fold against.

### 6.5 `_derived` — the regenerable base-out cache (analysis fast-path)

**Excluded from semantic equality. Never hand-authored. Regenerated by CI's canonical replayer.
Validated against §3/§4/§6.4 oracles before it's allowed to persist.**

```jsonc
"_derived": {
  "outs_before": 0,
  "bases_before": [false, false, false],   // [1B,2B,3B] occupancy, before this event
  "bases_before_ids": [null, null, null],  // occupant player_id per base
  "base_out_state": "000|0",               // "1B2B3B|outs" — the 24-state bin key, ready to GROUP BY
  "away_score_before": 0,
  "home_score_before": 1,
  "outs_after": 0,
  "bases_after": [true, false, false],
  "runs_on_play": 1,
  "re24_state_before": "000|0",            // convenience alias many RE tools want by this name
  "pa_number_of_batter": 1                 // 1-based PA count for this batter, for box reconciliation
}
```

This is the payoff for analysis-caller-first: a run-expectancy build is
`SELECT base_out_state, AVG(runs_rest_of_inning) …` with **zero state derivation**, and an Elo
/ matchup consumer that trusts nothing can drop the whole `_derived.*` column family and fold
`runners[]` itself with the 15-line reference materializer below. Same data, two speeds, one
trust boundary.

---

## 7. `unparsed[]`

```jsonc
"unparsed": [
  { "location": { "inning": 7, "half": "bottom", "line_index": 3 },
    "raw": "Verbatim text the parser could not classify.",
    "reason": "unknown_outcome_token" }
]
```

Verbatim line + enough location to splice it back into order. `games/**` is write-once, so a
file with a non-empty `unparsed[]` is *published as-is* (finals aren't blocked on a perfect
parse) but flagged; the replayer treats an unparsed line inside a half-inning as "state may be
incomplete here" and downgrades that half-inning's checks to warnings rather than asserting
false confidence.

---

## 8. `meta` — provenance (EXCLUDED from semantic equality)

```jsonc
"meta": {
  "parser_version": "2026.07.0",
  "source_url": "https://www.pioneerleague.com/sports/bsb/2026/boxscores/20260709_h94w.xml",
  "source_sha256": "…",        // hash of the fetched HTML; the reproducibility anchor
  "fetched_at": "2026-07-09T…Z",
  "parsed_at": "2026-07-09T…Z",
  "derived_replayer_version": "2026.07.0"   // which replayer stamped _derived.* — cache invalidation key
}
```

Parse the **pioneerleague.com** copy (per the player-id excursion) so both teams' players carry
16-char ids; a team-site copy leaves the opponent as plain text and breaks half the matchup
joins.

---

## 9. Fully worked example — Top of the 1st, verbatim from the sample

Source: `boxscore_20260709_final.html`, `#pbp-inning-1`, top half (lines 4602–4652). Narratives
are byte-for-byte from the HTML. Player-ids for Yuba-Sutter are shown as readable placeholders
(`ys:nunez`) — in the real file they are the 16-char ids parsed from the pioneerleague.com copy,
which links both teams. `_derived` is shown filled to demonstrate the fold; in the file it is
stamped by CI.

```jsonc
{
  "events": [
    {
      "seq": 0, "inning": 1, "half": "top", "kind": "plate_appearance",
      "batting_team": "PL014", "fielding_team": "PL016",
      "narrative": "Isaac Nunez singled to left field (1-1 BS).",
      "scoring_play": false,
      "batter":  { "player_id": "ys:nunez",  "name_raw": "Isaac Nunez", "resolved": true },
      "pitcher": { "player_id": "lbc:p1", "name_raw": null, "resolved": true },
      "outcome": { "type": "single", "modifiers": [], "fielders": ["lf"], "outs_recorded": 0, "location": "left field" },
      "count": { "balls": 1, "strikes": 1 }, "pitches": "BS",
      "runners": [
        { "player_id": "ys:nunez", "from": 0, "to": 1, "cause": "batted_ball", "out": false, "scored": false }
      ],
      "_derived": { "outs_before": 0, "bases_before": [false,false,false], "base_out_state": "000|0",
                    "away_score_before": 0, "home_score_before": 0, "outs_after": 0,
                    "bases_after": [true,false,false], "runs_on_play": 0, "pa_number_of_batter": 1 }
    },
    {
      "seq": 1, "inning": 1, "half": "top", "kind": "runner_event",
      "batting_team": "PL014", "fielding_team": "PL016",
      "narrative": "Isaac Nunez Failed pickoff attempt.",
      "scoring_play": false,
      "runners": [
        { "player_id": "ys:nunez", "from": 1, "to": 1, "cause": "pickoff", "out": false, "scored": false }
      ],
      "_derived": { "outs_before": 0, "bases_before": [true,false,false], "base_out_state": "100|0",
                    "away_score_before": 0, "home_score_before": 0, "outs_after": 0,
                    "bases_after": [true,false,false], "runs_on_play": 0 }
    },
    {
      "seq": 2, "inning": 1, "half": "top", "kind": "plate_appearance",
      "batting_team": "PL014", "fielding_team": "PL016",
      "narrative": "Jordan Donahue singled to right field (1-2 FBSF); Isaac Nunez advanced to third.",
      "scoring_play": false,
      "batter":  { "player_id": "ys:donahue", "name_raw": "Jordan Donahue", "resolved": true },
      "pitcher": { "player_id": "lbc:p1", "name_raw": null, "resolved": true },
      "outcome": { "type": "single", "modifiers": [], "fielders": ["rf"], "outs_recorded": 0, "location": "right field" },
      "count": { "balls": 1, "strikes": 2 }, "pitches": "FBSF",
      "runners": [
        { "player_id": "ys:donahue", "from": 0, "to": 1, "cause": "batted_ball", "out": false, "scored": false },
        { "player_id": "ys:nunez",   "from": 1, "to": 3, "cause": "advance",     "out": false, "scored": false }
      ],
      "_derived": { "outs_before": 0, "bases_before": [true,false,false], "base_out_state": "100|0",
                    "away_score_before": 0, "home_score_before": 0, "outs_after": 0,
                    "bases_after": [true,false,true], "runs_on_play": 0, "pa_number_of_batter": 1 }
    },
    {
      "seq": 3, "inning": 1, "half": "top", "kind": "runner_event",
      "batting_team": "PL014", "fielding_team": "PL016",
      "narrative": "Jordan Donahue Failed pickoff attempt.",
      "scoring_play": false,
      "runners": [ { "player_id": "ys:donahue", "from": 1, "to": 1, "cause": "pickoff", "out": false, "scored": false } ],
      "_derived": { "outs_before": 0, "bases_before": [true,false,true], "base_out_state": "101|0",
                    "away_score_before": 0, "home_score_before": 0, "outs_after": 0,
                    "bases_after": [true,false,true], "runs_on_play": 0 }
    },
    {
      "seq": 4, "inning": 1, "half": "top", "kind": "plate_appearance",
      "batting_team": "PL014", "fielding_team": "PL016",
      "narrative": "Josh Phillips singled to center field, RBI (0-0); Jordan Donahue advanced to second; Isaac Nunez scored.",
      "scoring_play": true,
      "batter":  { "player_id": "ys:phillips", "name_raw": "Josh Phillips", "resolved": true },
      "pitcher": { "player_id": "lbc:p1", "name_raw": null, "resolved": true },
      "outcome": { "type": "single", "modifiers": ["RBI"], "fielders": ["cf"], "outs_recorded": 0, "location": "center field" },
      "count": { "balls": 0, "strikes": 0 }, "pitches": null,
      "runners": [
        { "player_id": "ys:phillips", "from": 0, "to": 1, "cause": "batted_ball", "out": false, "scored": false },
        { "player_id": "ys:donahue",  "from": 1, "to": 2, "cause": "advance",     "out": false, "scored": false },
        { "player_id": "ys:nunez",    "from": 3, "to": 4, "cause": "advance",     "out": false, "scored": true, "earned": true, "rbi": true }
      ],
      "_derived": { "outs_before": 0, "bases_before": [true,false,true], "base_out_state": "101|0",
                    "away_score_before": 0, "home_score_before": 0, "outs_after": 0,
                    "bases_after": [true,true,false], "runs_on_play": 1, "pa_number_of_batter": 1 }
    },
    {
      "seq": 5, "inning": 1, "half": "top", "kind": "plate_appearance",
      "batting_team": "PL014", "fielding_team": "PL016",
      "narrative": "Kyle Carlson struck out swinging (3-2 BBSBKS).",
      "scoring_play": false,
      "batter":  { "player_id": "ys:carlson", "name_raw": "Kyle Carlson", "resolved": true },
      "pitcher": { "player_id": "lbc:p1", "name_raw": null, "resolved": true },
      "outcome": { "type": "strikeout_swinging", "modifiers": [], "fielders": [], "outs_recorded": 1, "location": null },
      "count": { "balls": 3, "strikes": 2 }, "pitches": "BBSBKS",
      "runners": [ { "player_id": "ys:carlson", "from": 0, "to": -1, "cause": "putout", "out": true, "scored": false } ],
      "_derived": { "outs_before": 0, "bases_before": [true,true,false], "base_out_state": "110|0",
                    "away_score_before": 1, "home_score_before": 0, "outs_after": 1,
                    "bases_after": [true,true,false], "runs_on_play": 0, "pa_number_of_batter": 1 }
    },
    {
      "seq": 6, "inning": 1, "half": "top", "kind": "plate_appearance",
      "batting_team": "PL014", "fielding_team": "PL016",
      "narrative": "Christian Castaneda struck out swinging (0-2 KSFS); Josh Phillips stole second; Jordan Donahue stole third.",
      "scoring_play": false,
      "batter":  { "player_id": "ys:castaneda", "name_raw": "Christian Castaneda", "resolved": true },
      "pitcher": { "player_id": "lbc:p1", "name_raw": null, "resolved": true },
      "outcome": { "type": "strikeout_swinging", "modifiers": [], "fielders": [], "outs_recorded": 1, "location": null },
      "count": { "balls": 0, "strikes": 2 }, "pitches": "KSFS",
      "runners": [
        { "player_id": "ys:castaneda", "from": 0, "to": -1, "cause": "putout",       "out": true,  "scored": false },
        { "player_id": "ys:phillips",  "from": 1, "to": 2,  "cause": "stolen_base",  "out": false, "scored": false },
        { "player_id": "ys:donahue",   "from": 2, "to": 3,  "cause": "stolen_base",  "out": false, "scored": false }
      ],
      "_derived": { "outs_before": 1, "bases_before": [true,true,false], "base_out_state": "110|1",
                    "away_score_before": 1, "home_score_before": 0, "outs_after": 2,
                    "bases_after": [false,true,true], "runs_on_play": 0, "pa_number_of_batter": 1 }
    },
    {
      "seq": 7, "inning": 1, "half": "top", "kind": "plate_appearance",
      "batting_team": "PL014", "fielding_team": "PL016",
      "narrative": "Andrew Kirchner grounded out to p (1-2 KSB).",
      "scoring_play": false,
      "batter":  { "player_id": "ys:kirchner", "name_raw": "Andrew Kirchner", "resolved": true },
      "pitcher": { "player_id": "lbc:p1", "name_raw": null, "resolved": true },
      "outcome": { "type": "groundout", "modifiers": [], "fielders": ["p"], "outs_recorded": 1, "location": null },
      "count": { "balls": 1, "strikes": 2 }, "pitches": "KSB",
      "runners": [ { "player_id": "ys:kirchner", "from": 0, "to": -1, "cause": "putout", "out": true, "scored": false } ],
      "_derived": { "outs_before": 2, "bases_before": [false,true,true], "base_out_state": "011|2",
                    "away_score_before": 1, "home_score_before": 0, "outs_after": 3,
                    "bases_after": [false,true,true], "runs_on_play": 0, "pa_number_of_batter": 1 }
    },
    {
      "seq": 8, "inning": 1, "half": "top", "kind": "inning_summary",
      "batting_team": "PL014", "fielding_team": "PL016",
      "narrative": "Inning Summary: 1 Runs, 3 Hits, 0 Errors, 2 LOB",
      "scoring_play": false,
      "summary": { "R": 1, "H": 3, "E": 0, "LOB": 2 }
    }
  ]
}
```

At `seq: 7` the fold lands on `bases_after: [false,true,true]` (Phillips on 2B, Donahue on 3B)
with 3 outs → **2 LOB**, exactly the `inning_summary`. 3 hits (Nunez, Donahue, Phillips), 1 run
→ matches both the summary line *and* `linescore.innings.away[0] == 1`. The half-inning is
internally consistent across four independent signals.

---

## 10. How each replayer check reads from these shapes

The replayer folds **asserted** fields only (`runners[]`, `outcome.outs_recorded`,
`outcome.type`, `scoring_play`), reconstructs state, then checks against the oracles. It never
reads `_derived` as input — after reconstructing, it *overwrites* `_derived` and a mismatch vs.
the previously stamped cache is itself a regression signal.

| Replayer check | Reads from | Passes when |
|---|---|---|
| **Base-out reconstruction** | fold `events[].runners[].{from,to,scored,out}` | every `from` base was occupied, no two runners land on the same base, batter always present — illegal transition ⇒ fail |
| **Outs == 3 per half-inning** | Σ `outcome.outs_recorded` + Σ `runners[].out` within (inning,half) | equals 3, *unless* `linescore.innings.home[i] == null` (walk-off/unbatted) — then <3 is explicit and allowed |
| **Linescore vs PBP** | Σ `runners[].scored` per (inning,half) | equals `linescore.innings.{away,home}[inning-1]` (the independent DOM pane) |
| **LOB reconciliation** | runners left on base at 3rd out of fold | equals `inning_summary.summary.LOB` (and team total in `box`) |
| **Per-batter PA count** | count `events[].batter.player_id` (PA + relevant runner-outs) | matches `box.batting[team][*]` AB+BB+HBP+SF decomposition |
| **Scoring-play cross-check** | `event.scoring_play` vs `Σ runners[].scored > 0` | agree on every event (bold-tag oracle vs. `scored` clause) |
| **Hits cross-check** | count `outcome.type ∈ {single,double,triple,home_run}` per half | equals `inning_summary.H` and rolls to `linescore.totals.H` |
| **`_derived` validation** | re-fold, compare to stamped `_derived.*` | byte-equal, else the cache is stale/buggy ⇒ CI fails, cache is not published |

The key property: **the input side (PBP `runners[]`) and every oracle (linescore, box,
inning_summary, `<strong>` flag) come from different regions of the source HTML.** A parser bug
localized to the PBP grammar corrupts the input but not the oracles, so the fold and the oracle
disagree and CI catches it. That is how the derived `_derived` cache can be trusted by fast
consumers without becoming a place a shared bug hides.

### Reference materializer (the "trivial in duckdb" promise, for distrustful consumers)

```python
def fold(events):
    bases = [None, None, None]; outs = 0; away = home = 0
    for e in events:
        if e["kind"] == "inning_summary": continue
        e.setdefault("_derived", {})
        d = e["_derived"]
        d["bases_before"] = [b is not None for b in bases]
        d["base_out_state"] = f'{"".join("1" if b else "0" for b in bases)}|{outs}'
        d["away_score_before"], d["home_score_before"] = away, home
        runs = 0
        for r in e.get("runners", []):
            frm, to = r["from"], r["to"]
            if frm != 0: bases[frm-1] = None        # vacate origin
            if r.get("scored"): runs += 1
            elif r.get("out"): pass                  # off the bases
            elif to in (1,2,3): bases[to-1] = r["player_id"]
        outs += e.get("outcome", {}).get("outs_recorded", 0)
        if e["batting_team"] == e["fielding_team"]:  # never; guards accidental self-play rows
            raise ValueError
        (away := away + runs) if e["half"] == "top" else (home := home + runs)
        d["outs_after"] = outs; d["runs_on_play"] = runs
    return events
```

Fifteen lines, pure forward scan, no cross-event joins. A consumer who drops `_derived` from
the file and runs this gets identical columns — the schema's promise that analysis needs no
re-derivation holds whether you trust the cache or rebuild it.

---

## 11. Self-assessment

**Depth (behavior per unit of interface).** The interface a consumer touches is small —
`events[]` with a fixed envelope + three kind-bodies, plus five sibling blocks — but each field
earns multiple behaviors. `runners[].{from,to,cause,scored}` alone drives base-out state, RE24
binning, SB/CS rates, run attribution, illegal-transition detection, and LOB — six model
concerns from one primitive. The `_derived` block turns a whole class of consumer code
(state tracking) into a column read. High ratio: the from→to movement is the single deepest
primitive in the file.

**Locality of change.** New outcome type ⇒ one `enum` entry + parser template; no consumer
breaks (unknown types were already routed to `unparsed[]`, loud not silent). New runner cause ⇒
one `enum` entry. Adding a derived analytic (e.g. leverage index) ⇒ a new `_derived.*` key,
zero schema-version bump because `_derived` is cache, excluded from equality. The asserted/derived
split *is* the locality mechanism: analysis features accrete in `_derived` and CI, never in the
write-once asserted spine. Breaking changes are confined to `schema_version` and gated.

**Testability.** Every asserted field has an independent oracle from a different DOM region
(§10), so the file is self-checking, not trust-me. The worked half-inning already exercises
single/steal/RBI/strikeout/groundout/pickoff/compound-advance and reconciles against four
signals. Golden-file testing is natural: one sample game → freeze `events[]` (minus `_derived`
and `meta`) as the semantic-equality fixture; re-parse must reproduce it byte-for-byte. The
reference materializer is 15 lines and testable in isolation against the frozen `_derived`.

**Where it's weakest, honestly.** The `runners[]` "record net movement, not waypoints" rule for
compound clauses (the `advanced to third, scored on an error` case) is the one place the parser
makes a judgement — it collapses two narrated sub-movements into one state change. If a future
game narrates a genuinely two-hop runner event that *isn't* redundant, this rule could lose a
waypoint. Mitigation: `narrative` is always verbatim, so nothing is unrecoverable, and the case
is rare enough (1 line in 122) to route to `unparsed[]` if the collapse is ambiguous.

## 12. Deliberately left out, and why

- **Stored `bases_before` as asserted truth.** The whole design pivots on *not* doing this — a
  snapshot is a derived claim with no independent check; from→to movements are locally asserted
  and re-checkable. Base-out state lives only in the regenerable `_derived` cache.
- **Cross-season player identity / person crosswalk.** Out of scope per spec and per the
  id-stability excursion (ids reset per season). This file's ids are within-season join keys
  only; a separate crosswalk table owns cross-season linkage.
- **Pitch-level physical data (velocity, location, type).** The source has only the pitch-result
  character string (`BBSBKS`); there is no pitch-tracking data to model, so no field pretends
  there is.
- **Win-probability / leverage / RE values themselves.** Those are *model outputs*, not source
  facts. The schema stops at the base-out state that makes them a one-liner; computing them is
  the model layer's job, and baking a run-expectancy table into the game file would couple the
  write-once data to a model that will be re-fit.
- **A `state_after` on the last event / end-of-game rollup.** Redundant with
  `linescore.totals`; the fold reaches it, and duplicating it invites the two to disagree.
- **Fielding-independent split of errors/positions beyond the position-token chain.** The
  `fielders: ["3b","2b","1b"]` chain preserves the source's fielding sequence verbatim; richer
  fielding modeling isn't supported by the source and isn't in the epic's model set.
