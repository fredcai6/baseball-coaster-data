# Schema Candidate A — MINIMAL-INTERFACE

Per-game game-file schema for `baseball-coaster-data`, path `games/<season>/<game_id>.json`, status `"final"` only.

**Constraint owned:** the smallest interface a caller must learn that still satisfies the contract. Every field justifies its existence; fewer, deeper structures beat optional sprawl.

---

## 1. The design in one paragraph

There is exactly **one spine: `events[]`**. Everything a consumer wants about what happened in the game is reconstructed by folding `events[]` over an initial state. The other top-level blocks are not parallel truths — they are **verification oracles** and **initial conditions**, and each exists only because the replayer needs an *independently extracted* copy to check the fold against:

- `players{}` — the roster join table (initial condition + the one identity key a caller learns).
- `lineups{}` — the *starting* batting order and starting battery per team (initial condition; every later change is an `event`, never restated here).
- `linescore{}` and the `inning_summary` events — two **independently sourced** run/hit/error/LOB panes the replayer cross-checks the fold against. They are stored *because* they come from different DOM regions than the play narrative, so agreement is real evidence.
- `unparsed[]` — verbatim lines the parser could not classify, with location.
- `meta{}` — provenance, **excluded from semantic-equality**.

A caller who only wants "what happened" learns **two shapes**: the `event` envelope and the `players` map. Everything else is there for the CI validator / replayer, not for the everyday reader. That is the minimal interface: the deep spine carries the game; the shallow oracles exist to prove the spine is right.

### Deliberate consequence
No box-stat total, no score, no "runners on base" snapshot is stored as first-class truth. AB/R/H/RBI/BB/SO, running score, base state, per-batter PA counts are **all derived** from `events[]`. The box totals that *do* appear (inside `players{}`) are tagged as an oracle pane, not a source — see §2.4. This is the single biggest minimal-interface bet: **one source of truth, many cheap independent checks.**

---

## 2. Annotated JSON Schema (draft 2020-12)

Prose annotations are `// …`. This is the normative shape; the data-repo ships this as the single source of truth its CI validates every file against.

```jsonc
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://baseball-coaster/schemas/game.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "game_id", "season", "date",
               "status", "teams", "players", "lineups",
               "linescore", "events", "unparsed", "meta"],

  "properties": {

    "schema_version": { "const": 1 },        // integer; bump on breaking change

    "game_id":  { "type": "string", "pattern": "^[0-9]{8}_[0-9a-z]{4}$" },
                                             // Presto slug, e.g. "20260709_h94w" — the file's identity
    "season":   { "type": "integer", "minimum": 1900 },   // e.g. 2026
    "date":     { "type": "string", "format": "date" },   // "2026-07-09", game-local
    "status":   { "const": "final" },        // this schema describes final games only

    // ---- teams: the two sides, minimal. away = visitor bats top. ----
    "teams": {
      "type": "object",
      "additionalProperties": false,
      "required": ["away", "home"],
      "properties": {
        "away": { "$ref": "#/$defs/team" },
        "home": { "$ref": "#/$defs/team" }
      }
    },

    // ---- players: THE join table. Keyed by file-local pid. ----
    // One key the caller learns. event.batter / event.pitcher / runner.pid
    // and lineups all reference these keys and nothing else.
    "players": {
      "type": "object",
      "propertyNames": { "$ref": "#/$defs/pid" },
      "additionalProperties": { "$ref": "#/$defs/player" },
      "minProperties": 1
    },

    // ---- lineups: INITIAL condition only. Deltas live in events[]. ----
    "lineups": {
      "type": "object",
      "additionalProperties": false,
      "required": ["away", "home"],
      "properties": {
        "away": { "$ref": "#/$defs/lineup" },
        "home": { "$ref": "#/$defs/lineup" }
      }
    },

    // ---- linescore: ORACLE pane #1 (independent DOM region). ----
    "linescore": {
      "type": "object",
      "additionalProperties": false,
      "required": ["away", "home"],
      "properties": {
        "away": { "$ref": "#/$defs/linescore_row" },
        "home": { "$ref": "#/$defs/linescore_row" }
      }
    },

    // ---- events: THE SPINE. Ordered, replay-complete. ----
    "events": {
      "type": "array",
      "items": { "$ref": "#/$defs/event" }
      // MUST be in game order. events[i].seq == i (0-based) — see $defs/event.
    },

    // ---- unparsed: verbatim misses, so no data is silently dropped. ----
    "unparsed": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["location", "text"],
        "properties": {
          "location": { "type": "string" },   // e.g. "pbp-inning-6/half=top/row=12"
          "text":     { "type": "string" }     // the line, byte-verbatim
        }
      }
      // Empty array = clean parse. CI can gate: length==0 required to publish,
      // or allow-with-warning — policy, not schema.
    },

    // ---- meta: provenance. EXCLUDED from semantic equality. ----
    "meta": {
      "type": "object",
      "additionalProperties": false,
      "required": ["parser_version", "source_url", "source_sha256", "parsed_at"],
      "properties": {
        "parser_version": { "type": "string" },              // semver of the parser
        "source_url":     { "type": "string", "format": "uri" },
        "source_sha256":  { "type": "string", "pattern": "^[0-9a-f]{64}$" },
        "parsed_at":      { "type": "string", "format": "date-time" },
        "source_fetched_at": { "type": "string", "format": "date-time" }
      }
      // Two files are "the same game" iff they match on everything EXCEPT meta.
      // The equality checker drops this key before comparing.
    }
  },

  "$defs": {

    "pid": {
      // File-local player key. Equals the 16-char Presto boxscore id when the
      // source exposed it; else a synthetic deterministic key "syn:<side>:<n>".
      // ONE reference type across the whole file.
      "type": "string",
      "pattern": "^([0-9a-z]{16}|syn:(home|away):[0-9]{1,2})$"
    },

    "team": {
      "type": "object",
      "additionalProperties": false,
      "required": ["id", "name"],
      "properties": {
        "id":   { "type": "string" },   // Presto 16-char teamId, e.g. "maotayco79j2g2lx"
        "name": { "type": "string" }    // "Long Beach Coast"
      }
    },

    "player": {
      "type": "object",
      "additionalProperties": false,
      "required": ["side", "name", "presto_id"],
      "properties": {
        "side":      { "enum": ["home", "away"] },
        "name":      { "type": "string" },        // "Isaac Nunez", as printed in the box
        "presto_id": {                            // the VERIFIED 16-char id, or null
          "type": ["string", "null"],
          "pattern": "^[0-9a-z]{16}$"
        },
        "last_name": { "type": "string" }         // the PBP surname key, e.g. "Nunez".
                                                  // Present so the replayer can audit the
                                                  // PBP-surname -> pid resolution the parser did.
      }
      // NOTE: no AB/R/H/... here by default. Box totals are an oracle, carried
      // separately and optionally — see box_totals below and §2.4.
    },

    "lineup": {
      // Initial condition. Batting order 1..9 of pids; starting pitcher.
      // Every substitution after first pitch is an event, never edited in here.
      "type": "object",
      "additionalProperties": false,
      "required": ["order", "starting_pitcher"],
      "properties": {
        "order": {
          "type": "array",
          "minItems": 9, "maxItems": 9,
          "items": { "$ref": "#/$defs/pid" }   // slot i (0-based) = batting position i+1
        },
        "starting_pitcher": { "$ref": "#/$defs/pid" }
      }
    },

    "linescore_row": {
      "type": "object",
      "additionalProperties": false,
      "required": ["innings", "runs", "hits", "errors"],
      "properties": {
        "innings": {
          "type": "array",
          "items": { "type": ["integer", "null"] }  // runs per inning; null = "X" (not batted)
        },
        "runs":   { "type": "integer" },
        "hits":   { "type": "integer" },
        "errors": { "type": "integer" }
      }
    },

    // ===== THE EVENT ENVELOPE — one shape, closed type enum. =====
    "event": {
      "type": "object",
      "additionalProperties": false,
      "required": ["seq", "inning", "half", "type", "verbatim"],
      "properties": {

        "seq":     { "type": "integer", "minimum": 0 },  // == array index; explicit so a
                                                         // single event is self-locating in logs
        "inning":  { "type": "integer", "minimum": 1 },
        "half":    { "enum": ["top", "bottom"] },

        // Closed taxonomy. ~17 PA outcomes + summaries/subs/runner/game events.
        "type": {
          "enum": [
            // -- plate-appearance outcomes (carry `pa`) --
            "single", "double", "triple", "home_run",
            "walk", "intentional_walk", "hit_by_pitch",
            "strikeout", "groundout", "flyout", "lineout", "popout",
            "fielders_choice", "reached_on_error",
            "grounded_into_double_play", "sacrifice", "fielded_out_other",
            // -- non-PA --
            "runner_event",      // SB/CS/WP/PB/balk/pickoff advance — no batter result
            "substitution",      // carries `sub`
            "inning_summary",    // ORACLE pane #2, carries `summary`
            "no_play"            // failed pickoff attempt etc. — kept for line fidelity
          ]
        },

        "verbatim": { "type": "string" },   // the narrative line, byte-exact. Always present.
                                            // The audit backstop: any field can be re-derived
                                            // from this by hand.

        // outs standing AFTER this event, IFF the narrative stated "(N out)".
        // null when the line carried no out marker. Verbatim-grounded, NOT computed —
        // that is why it is a real independent oracle for the outs==3 / illegal-transition checks.
        "outs_after": { "type": ["integer", "null"], "minimum": 0, "maximum": 3 },

        // present only on PA-outcome types
        "pa":      { "$ref": "#/$defs/pa" },
        // present only on type == substitution
        "sub":     { "$ref": "#/$defs/sub" },
        // present only on type == inning_summary
        "summary": { "$ref": "#/$defs/summary" },

        // runner movements this event caused. Absent/empty when none.
        // Shared shape across PA and runner_event types — one thing to learn.
        "runners": {
          "type": "array",
          "items": { "$ref": "#/$defs/runner_move" }
        }
      },

      // Structural coupling: the optional block that appears is fixed by `type`.
      "allOf": [
        { "if":   { "properties": { "type": { "const": "substitution" } } },
          "then": { "required": ["sub"] } },
        { "if":   { "properties": { "type": { "const": "inning_summary" } } },
          "then": { "required": ["summary"] } }
      ]
    },

    // plate-appearance detail
    "pa": {
      "type": "object",
      "additionalProperties": false,
      "required": ["batter", "pitcher", "balls", "strikes"],
      "properties": {
        "batter":  { "$ref": "#/$defs/pid" },
        "pitcher": { "$ref": "#/$defs/pid" },
        "balls":   { "type": "integer", "minimum": 0, "maximum": 4 },  // final count
        "strikes": { "type": "integer", "minimum": 0, "maximum": 3 },
        "pitches": {                          // pitch-char sequence when present.
          "type": ["string", "null"],         // null on count-only "(0-0)" first-pitch-in-play.
          "pattern": "^[BFKSH]*$"             // closed alphabet observed: Ball/Foul/Kalled/Swing/Hbp
        },
        "rbi": { "type": "integer", "minimum": 0, "maximum": 4 }
                                              // from ", RBI" / ", N RBI". Per-event RBI
                                              // attribution — the advanced-stat value-add.
        // fielders (e.g. "3b to 2b to 1b") intentionally NOT structured — see §7.
      }
    },

    // one runner's state change
    "runner_move": {
      "type": "object",
      "additionalProperties": false,
      "required": ["pid", "to"],
      "properties": {
        "pid": { "$ref": "#/$defs/pid" },
        "to":  { "enum": ["first", "second", "third", "home", "out"] },
        // when to == "home": earned-run flag from ", unearned" narrative. Default true.
        "earned": { "type": "boolean" },
        // how the move happened, closed set, for CS/pickoff/error/wp attribution.
        "how": { "enum": [
          "hit", "walk", "advance", "steal", "caught_stealing",
          "picked_off", "wild_pitch", "passed_ball", "balk",
          "error", "fielders_choice", "force", "other"
        ] }
      }
    },

    "sub": {
      "type": "object",
      "additionalProperties": false,
      "required": ["player_in", "role"],
      "properties": {
        "player_in":  { "$ref": "#/$defs/pid" },
        "player_out": { "$ref": "#/$defs/pid" },   // may be absent (e.g. new pitcher, defensive)
        "role": { "enum": ["pitcher", "pinch_hitter", "pinch_runner", "fielder"] },
        "position": { "type": "string" }           // "p", "cf", ... when stated
      }
    },

    "summary": {
      // ORACLE pane #2: the per-half R/H/E/LOB line the scorer software prints.
      "type": "object",
      "additionalProperties": false,
      "required": ["runs", "hits", "errors", "lob"],
      "properties": {
        "runs":   { "type": "integer", "minimum": 0 },
        "hits":   { "type": "integer", "minimum": 0 },
        "errors": { "type": "integer", "minimum": 0 },
        "lob":    { "type": "integer", "minimum": 0 }
      }
    }
  }
}
```

### 2.4 The box-stat totals question (a minimal-interface judgment call)

The spec says the file "carries box-stat totals." It does — but as an **oracle**, not first-class truth. Rather than inflate `player` with eight batting + twelve pitching columns that duplicate what `events[]` already implies, the box totals live in one optional sibling block whose only job is to be checked:

```jsonc
// optional top-level key, present when the box was parsed:
"box_totals": {
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "batting":  { "type": "object",   // pid -> {ab,r,h,rbi,bb,so,lob}
      "propertyNames": { "$ref": "#/$defs/pid" },
      "additionalProperties": {
        "type": "object", "additionalProperties": false,
        "required": ["ab","r","h","rbi","bb","so","lob"],
        "properties": {
          "ab":{"type":"integer"}, "r":{"type":"integer"}, "h":{"type":"integer"},
          "rbi":{"type":"integer"}, "bb":{"type":"integer"}, "so":{"type":"integer"},
          "lob":{"type":"integer"} } } },
    "pitching": { "type": "object",   // pid -> {ip,h,r,er,bb,so,hr,wp,bf,ab,np}
      "propertyNames": { "$ref": "#/$defs/pid" },
      "additionalProperties": {
        "type": "object", "additionalProperties": false,
        "required": ["ip","h","r","er","bb","so","hr","wp","bf","ab","np"],
        "properties": {
          "ip":{"type":"string"},   // "6.2" — thirds, kept as printed
          "h":{"type":"integer"}, "r":{"type":"integer"}, "er":{"type":"integer"},
          "bb":{"type":"integer"}, "so":{"type":"integer"}, "hr":{"type":"integer"},
          "wp":{"type":"integer"}, "bf":{"type":"integer"}, "ab":{"type":"integer"},
          "np":{"type":"integer"} } } }
  }
}
```

**Why AVG/ERA are dropped:** they are season-to-date, not this-game, so they are neither replay-derivable nor game-scoped — storing them would put non-game state in a game file. Decision (W/L) is dropped for the same reason (season record embedded). What survives is exactly the counting stats the replayer can independently reproduce from `events[]`.

---

## 3. One fully worked example — Top of the 1st, verbatim from the sample

Source: `boxscore_20260709_final.html`, `#pbp-inning-1`, "Top of 1st" (Yuba-Sutter batting, Long Beach's Garrett VanDeventer pitching). Every `verbatim` string is byte-copied from the HTML `<td class="text">` cells (lines 4602–4652).

> **Identity reality shown honestly.** This file was fetched from the *team* site (longbeachcoast.com), which links only its own players. So the Yuba-Sutter batters have **no** Presto id in this source (`presto_id: null`, synthetic pids `syn:away:*`); the Long Beach pitcher has a real one (`4bs3tvwryvtzrvpa`). The spec's remedy — scrape the pioneerleague.com copy to get both sides' ids — would fill those `presto_id`s in without changing any other shape. That is the payoff of the single `pid` indirection: swapping in real ids is a `players{}`-only edit; no event changes.

```json
{
  "schema_version": 1,
  "game_id": "20260709_h94w",
  "season": 2026,
  "date": "2026-07-09",
  "status": "final",

  "teams": {
    "away": { "id": "unknownteamid0000", "name": "Yuba-Sutter Freebirds" },
    "home": { "id": "maotayco79j2g2lx", "name": "Long Beach Coast" }
  },

  "players": {
    "syn:away:1": { "side": "away", "name": "Isaac Nunez",         "last_name": "Nunez",     "presto_id": null },
    "syn:away:2": { "side": "away", "name": "Jordan Donahue",      "last_name": "Donahue",   "presto_id": null },
    "syn:away:3": { "side": "away", "name": "Josh Phillips",       "last_name": "Phillips",  "presto_id": null },
    "syn:away:4": { "side": "away", "name": "Kyle Carlson",        "last_name": "Carlson",   "presto_id": null },
    "syn:away:5": { "side": "away", "name": "Christian Castaneda", "last_name": "Castaneda", "presto_id": null },
    "syn:away:6": { "side": "away", "name": "Andrew Kirchner",     "last_name": "Kirchner",  "presto_id": null },
    "4bs3tvwryvtzrvpa": { "side": "home", "name": "Garrett VanDeventer", "last_name": "VanDeventer", "presto_id": "4bs3tvwryvtzrvpa" }
  },

  "events": [
    {
      "seq": 0, "inning": 1, "half": "top", "type": "single",
      "verbatim": "Isaac Nunez singled to left field (1-1 BS).",
      "outs_after": null,
      "pa": { "batter": "syn:away:1", "pitcher": "4bs3tvwryvtzrvpa", "balls": 1, "strikes": 1, "pitches": "BS", "rbi": 0 },
      "runners": [ { "pid": "syn:away:1", "to": "first", "how": "hit" } ]
    },
    {
      "seq": 1, "inning": 1, "half": "top", "type": "no_play",
      "verbatim": "Isaac Nunez Failed pickoff attempt.",
      "outs_after": null
    },
    {
      "seq": 2, "inning": 1, "half": "top", "type": "single",
      "verbatim": "Jordan Donahue singled to right field (1-2 FBSF); Isaac Nunez advanced to third.",
      "outs_after": null,
      "pa": { "batter": "syn:away:2", "pitcher": "4bs3tvwryvtzrvpa", "balls": 1, "strikes": 2, "pitches": "FBSF", "rbi": 0 },
      "runners": [
        { "pid": "syn:away:2", "to": "first", "how": "hit" },
        { "pid": "syn:away:1", "to": "third", "how": "advance" }
      ]
    },
    {
      "seq": 3, "inning": 1, "half": "top", "type": "no_play",
      "verbatim": "Jordan Donahue Failed pickoff attempt.",
      "outs_after": null
    },
    {
      "seq": 4, "inning": 1, "half": "top", "type": "single",
      "verbatim": "Josh Phillips singled to center field, RBI (0-0); Jordan Donahue advanced to second; Isaac Nunez scored.",
      "outs_after": null,
      "pa": { "batter": "syn:away:3", "pitcher": "4bs3tvwryvtzrvpa", "balls": 0, "strikes": 0, "pitches": null, "rbi": 1 },
      "runners": [
        { "pid": "syn:away:3", "to": "first",  "how": "hit" },
        { "pid": "syn:away:2", "to": "second", "how": "advance" },
        { "pid": "syn:away:1", "to": "home",   "how": "advance", "earned": true }
      ]
    },
    {
      "seq": 5, "inning": 1, "half": "top", "type": "strikeout",
      "verbatim": "Kyle Carlson struck out swinging (3-2 BBSBKS).",
      "outs_after": 1,
      "pa": { "batter": "syn:away:4", "pitcher": "4bs3tvwryvtzrvpa", "balls": 3, "strikes": 2, "pitches": "BBSBKS", "rbi": 0 },
      "runners": [ { "pid": "syn:away:4", "to": "out", "how": "other" } ]
    },
    {
      "seq": 6, "inning": 1, "half": "top", "type": "strikeout",
      "verbatim": "Christian Castaneda struck out swinging (0-2 KSFS); Josh Phillips stole second; Jordan Donahue stole third.",
      "outs_after": 2,
      "pa": { "batter": "syn:away:5", "pitcher": "4bs3tvwryvtzrvpa", "balls": 0, "strikes": 2, "pitches": "KSFS", "rbi": 0 },
      "runners": [
        { "pid": "syn:away:5", "to": "out",   "how": "other" },
        { "pid": "syn:away:3", "to": "second", "how": "steal" },
        { "pid": "syn:away:2", "to": "third",  "how": "steal" }
      ]
    },
    {
      "seq": 7, "inning": 1, "half": "top", "type": "groundout",
      "verbatim": "Andrew Kirchner grounded out to p (1-2 KSB).",
      "outs_after": 3,
      "pa": { "batter": "syn:away:6", "pitcher": "4bs3tvwryvtzrvpa", "balls": 1, "strikes": 2, "pitches": "KSB", "rbi": 0 },
      "runners": [ { "pid": "syn:away:6", "to": "out", "how": "other" } ]
    },
    {
      "seq": 8, "inning": 1, "half": "top", "type": "inning_summary",
      "verbatim": "Inning Summary: 1 Runs, 3 Hits, 0 Errors, 2 LOB",
      "outs_after": null,
      "summary": { "runs": 1, "hits": 3, "errors": 0, "lob": 2 }
    }
  ]

  // teams.away.id, lineups, linescore, box_totals, unparsed, meta elided for length —
  // shapes are exactly as in §2. lineups.away.order[0..2] = the three synthetic pids above;
  // lineups.away.starting_pitcher is Yuba-Sutter's (a home-batting-half concern, not shown).
}
```

---

## 4. How each replayer check reads from these shapes

The replayer folds `events[]` left-to-right over `(base_state, outs, score, pa_counts, lineup_state)`. Each invariant reads from a **named field**, and — critically — each is checked against an **independently sourced** value, never against the same events that produced it:

| Replayer check | Reads from events (the fold) | Checked against (independent oracle) |
|---|---|---|
| **Linescore validation** | sum of `runners[].to=="home"` grouped by `inning`+batting side | `linescore.{away,home}.innings[]` — a *different DOM pane* (§2, line-score table) |
| **Outs == 3 per half** | last event of each (`inning`,`half`) has `outs_after == 3` | walk-off exception: home `half=="bottom"` final inning may end `< 3` **iff** a `runners[].to=="home"` on that event takes home ahead — flagged explicitly, not silently allowed |
| **LOB reconciliation** | runners left on base = (reached) − (scored) − (out) at half end | `inning_summary` event's `summary.lob` — the *scorer-printed* pane, sourced from `.totals` row, independent of the play lines |
| **Per-batter PA counts** | count of PA-outcome events per `pa.batter`, applied to `lineups.*.order` advanced by `substitution` events | `box_totals.batting[pid]`: PA ≈ ab+bb+... reconstruction |
| **Illegal-transition detection** | consecutive `outs_after` must be non-decreasing within a half and never exceed 3; `runners[].to` must be a legal base advance from the runner's prior base | structural (schema `maximum:3`) + fold-time state machine over `runner_move.to` |
| **Identity resolution audit** | `pa.batter`/`runners[].pid` all resolve in `players{}`; `players[pid].last_name` must match the surname token in `verbatim` | `verbatim` string — the ground-truth line itself |

The design point: **`verbatim` and the two oracle panes are what make the check meaningful.** If the parser mis-derived a field, the fold disagrees with a value that came from a *different region of the source*, and CI fails. A schema that stored only derived truth could be self-consistently wrong; this one cannot be.

---

## 5. Self-assessment

**Depth (behavior per unit of interface).** High, by construction. A caller reconstructs the *entire game* — every base state, score, count, PA tally — from **one** array of **one** object shape (`event`) plus **one** map (`players`). The five other top-level keys are inert to the everyday consumer; they surface only for the CI validator. The interface a "what happened" reader must learn is ~2 shapes; the behavior it buys is the full replay. Fewer, deeper structures over optional sprawl: achieved by making `events[]` the sole spine and refusing to restate its consequences (no box truth, no score field, no base snapshots, no per-event lineup copy).

**Locality of change.** Strong. (a) Getting real Presto ids for the opponent (re-scrape from the league site) is a **`players{}`-only** edit — no event touched, because everything references the file-local `pid` and only the `presto_id`/key values change. (b) A new PBP outcome template adds one `type` enum member and, at most, one `runners[].how` value — additive, no shape churn. (c) Provenance changes are quarantined in `meta`, excluded from equality, so re-parsing the same source with a newer parser version is a no-op for semantic diffing.

**Testability.** Strong. Every semantic field is either (i) verbatim-grounded (`verbatim`, `outs_after`, `pitches`, `summary`) and thus re-derivable by hand from the source, or (ii) fold-derived and cross-checked against an independent pane (§4). The `verbatim` backstop means any disputed field has a canonical arbiter in the file itself. Semantic equality is well-defined: drop `meta`, compare. Two independent parsers agreeing on this structure is real evidence they read the game the same way.

**Where it costs.** The single-source-of-truth bet means consumers who want box totals must either read the optional `box_totals` oracle or run the fold themselves — there is no pre-chewed per-player game stat line as first-class data. I judged that acceptable: this is a *data* repo feeding an advanced-stats pipeline that will fold events anyway, and duplicating derivable totals as truth invites drift (the exact failure mode the replayer exists to catch). If a downstream proves it needs cheap totals, `box_totals` is already the place, and it is labeled an oracle so no one mistakes it for truth.

---

## 6. What I deliberately left out, and why

- **Fielder chains** (`"3b to 2b to 1b"`, `unassisted`). Kept only inside `verbatim`, not structured. Justification: no named replayer check reads them, and defensive-attribution stats are out of scope for the foundation issue. Structuring them now is optional sprawl paying for nothing; when a fielding-metrics epic lands, it parses `verbatim` (or adds a `fielders[]` to `pa`) — an additive change.
- **Per-event running score and base-state snapshot.** Derivable by the fold; storing them duplicates truth and invites drift. The replayer *computes* them; that is its job.
- **Per-player box stat lines as first-class `player` fields.** Demoted to the optional `box_totals` oracle (§2.4). Same anti-drift reason.
- **Season-to-date stats (AVG, ERA) and decisions (W/L record).** Not game-scoped, not replay-derivable — they would smuggle season state into a game file. Excluded entirely.
- **Pitch-by-pitch locations / velocities.** Not present in the source; nothing to store.
- **A separate `innings[]` structure.** Redundant: `inning_summary` events already carry the per-half oracle, and `event.inning`/`half` already group the spine. Adding an innings index would be a second way to say what events already say.
- **Roster / cross-season identity.** Out of scope per spec (Presto ids are within-season only). The file records `presto_id` when known and stops there; season-spanning person keys are a separate mapping table's problem, deliberately not this file's.
- **`away`/`home` denormalized names on every event.** Reachable via `inning`+`half`+`teams`; restating per event is sprawl.

The through-line of every omission: **if a field is derivable from the spine or unread by any check, it is not first-class truth.** That is the minimal interface.
