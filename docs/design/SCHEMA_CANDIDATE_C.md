# SCHEMA CANDIDATE C — MAX-FORWARD-FLEXIBILITY

Constraint owned: the schema must absorb future needs **without breaking write-once
files** — grammar-tail event types not yet seen, later parser versions adding fields,
cross-season identity arriving later, other leagues/seasons, and possible live ingestion.
Every extension point below names the **concrete** future it serves; I have refused the ones
I could not name.

The governing idea of this candidate: **a game file is a stream of state snapshots, not a
stream of prose to be re-interpreted.** Each event carries the full post-event game state and
the *structured* deltas that produced it. The event `type` is a label; the state math lives in
typed fields. That one decision is what makes the schema forward-flexible — a replayer (or any
consumer) keeps working when a brand-new `type` string appears, because it dispatches on the
small closed `category` and reads deltas it already understands. New grammar tails become new
labels on machinery that already exists.

---

## 1. The schema (annotated, JSON Schema draft 2020-12)

### 1.1 Versioning & evolution model (the load-bearing part of this candidate)

- `schema_version` is `"MAJOR.MINOR"` (e.g. `"1.0"`). It appears **at the root** (for cheap
  reader dispatch) and authoritatively in `meta`.
- **MINOR bump = additive only.** Permitted without a MAJOR bump: adding an *optional* property;
  adding a member to an *open* enum (the two the schema marks open — `event.type` and
  `game.status`); adding a stat column. Old external readers ignore unknown fields and keep
  working. This is the whole forward-compat contract, and it is small on purpose.
- **MAJOR bump = anything else** (remove/rename/retype a field, change a unit, redefine a
  closed enum, change the meaning of an existing value). A MAJOR bump is the *only* thing that
  may invalidate an existing file, and it is exactly the case the spec already reserves for
  "labeled re-parse commits."
- **Write-once reconciliation.** Files and the schema live in the same repo and move together,
  so data-repo CI always validates a file against the schema it was written for — there is no
  "old validator meets new file" case *inside* the repo. `additionalProperties: false` is
  therefore safe on every object and I use it for tight validation. Forward-compat is a promise
  to **external** consumers reading over raw.githubusercontent, discharged by the additive-only
  MINOR rule plus the documented "ignore unknown fields" reader contract below.
- **The single escape hatch.** Objects that I expect to grow (`game` root, each `event`, each
  player ref) carry an optional `x` object, `type: object` with free contents. Its *named*
  purpose: let parser version N+1 emit a field before its schema PR lands, so a re-parse does
  not have to wait on a schema release to be committable. **Rule, enforced by review not by the
  validator:** nothing under `x` is part of the public contract, no consumer may depend on it,
  and any field that survives one parser release **must** be promoted to a first-class property
  in the next MINOR (and removed from `x`). It is a staging area, not a junk drawer. There is
  exactly one such hatch shape and it is the only place `additionalProperties` is open.

Reader contract (documented in the data repo README, not enforceable in Schema):
> Consumers MUST ignore JSON properties they do not recognize and MUST tolerate unseen members
> of `event.type` and `status` by dispatching on `event.category`. Consumers MUST NOT read
> anything under any `x` object.

### 1.2 Root object

```jsonc
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://baseball-coaster/schema/game.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "game_id", "season", "date", "status",
               "source", "teams", "linescore", "events", "unparsed", "meta"],
  "properties": {
    "schema_version": { "type": "string", "pattern": "^[0-9]+\\.[0-9]+$" },

    "game_id":  { "type": "string" },          // Presto boxscore slug, e.g. "20260709_h94w"
    "season":   { "type": "integer" },         // 2026  (season key, NOT a date range)
    "date":     { "type": "string", "format": "date" },   // "2026-07-09" local game date

    // status is an OPEN enum. Only "final" is emitted today; the schema is shaped so
    // "in_progress"/"suspended" can be ADDED (MINOR) when/if live ingestion arrives, at
    // which point the final-only invariants in §3 are gated on this field. We do NOT model
    // partial-game structures now — that would be speculative. We only reserve the axis.
    "status":   { "type": "string", "enum": ["final"] },  // open enum, see §1.1

    "source":   { "$ref": "#/$defs/source" },
    "teams":    { "$ref": "#/$defs/teams"  },
    "linescore":{ "$ref": "#/$defs/linescore" },
    "events":   { "type": "array", "items": { "$ref": "#/$defs/event" } },

    // verbatim PBP lines the parser could not structure, with where they were found.
    // A non-empty unparsed[] means events[] is NOT a complete reconstruction; meta.parse
    // records that (replayable:false). For a clean "final" file this is [].
    "unparsed": { "type": "array", "items": { "$ref": "#/$defs/unparsed_line" } },

    "meta":     { "$ref": "#/$defs/meta" },     // provenance; EXCLUDED from semantic equality

    "x": { "type": "object" }                   // escape hatch, see §1.1
  }
}
```

### 1.3 Competition identity — `source` (serves "other leagues/seasons")

```jsonc
"source": {
  "type": "object", "additionalProperties": false,
  "required": ["provider", "league_id"],
  "properties": {
    "provider":  { "type": "string", "enum": ["prestosports"] }, // open enum
    "league_id": { "type": "string" },   // e.g. "pioneer" — present NOW; the sample already
                                          // spans pioneerleague.com + longbeachcoast.com.
    "site":      { "type": "string" }     // host the file was scraped from, e.g. "pioneerleague.com"
  }
}
```
Concrete need served: the epic will add teams/leagues beyond Long Beach Coast; keying every
file with `league_id` now means a multi-league query never has to guess. Not speculative —
the player-ID probe already scraped two sites for one game.

### 1.4 Player references and rosters — `teams` (serves "cross-season identity later")

**Every human is referenced as an object, never a bare id string.** This is the single most
important forward-flexibility decision after the state-snapshot one.

```jsonc
"$defs": {
  "player_ref": {
    "type": "object", "additionalProperties": false,
    "required": ["pid", "name"],
    "properties": {
      // 16-char Presto boxscore id; the join key. WITHIN-SEASON STABLE (verified across
      // games and across sites). null when a PBP last-name could not be resolved to the
      // boxscore player list (e.g. same-last-name teammates); name is then the raw token.
      "pid":  { "type": ["string", "null"], "pattern": "^[a-z0-9]{16}$" },
      "name": { "type": "string" },     // boxscore display name, e.g. "Patrick Roche Jr."
      "resolved": { "type": "boolean" },// present & false only when pid is null; default true

      // When cross-season person identity is built (explicitly out of scope now), a
      // "person_id" property is ADDED here (MINOR). Because every reference is already this
      // object, that future lands in ONE place and NO call site changes. That is the whole
      // reason refs are objects and not strings.
      "x": { "type": "object" }
    }
  }
}
```
> `pid` is deliberately **not** asserted to be globally unique or cross-season stable in the
> schema. The probe proved within-season stability only and proved bio-slugs reset across
> seasons. The schema encodes exactly what was verified and no more.

`teams` holds both sides' rosters (batting order + substitutions) and box-stat totals:

```jsonc
"teams": {
  "type": "object", "additionalProperties": false,
  "required": ["away", "home"],
  "properties": {
    "away": { "$ref": "#/$defs/team_side" },
    "home": { "$ref": "#/$defs/team_side" }
  }
},

"team_side": {
  "type": "object", "additionalProperties": false,
  "required": ["id", "name", "batters", "pitchers"],
  "properties": {
    "id":   { "type": "string" },   // Presto teamId, e.g. "maotayco79j2g2lx"
    "name": { "type": "string" },   // "Long Beach Coast"
    "batters":  { "type": "array", "items": { "$ref": "#/$defs/batting_line"  } },
    "pitchers": { "type": "array", "items": { "$ref": "#/$defs/pitching_line" } }
  }
},

"batting_line": {
  "type": "object", "additionalProperties": false,
  "required": ["player", "order", "positions", "stats"],
  "properties": {
    "player":    { "$ref": "#/$defs/player_ref" },
    "order":     { "type": "integer", "minimum": 1, "maximum": 9 }, // batting-order slot
    "sub_seq":   { "type": "integer" },   // 0 = slot starter; 1,2,... = later entrants in slot
    "positions": { "type": "array", "items": { "type": "string" } }, // ["lf"], ["ph","1b"]
    "replaces":  { "$ref": "#/$defs/player_ref" }, // whom this entrant replaced, if a sub
    // Box totals. Known columns are typed & optional; a NEW column is a MINOR addition.
    // stats is NOT a free map — that would forfeit validation and depth.
    "stats": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "ab":{"type":"integer"}, "r":{"type":"integer"}, "h":{"type":"integer"},
        "rbi":{"type":"integer"},"bb":{"type":"integer"},"so":{"type":"integer"},
        "lob":{"type":"integer"},"avg":{"type":"string"}   // "avg" kept as source string
      }
    },
    "x": { "type": "object" }
  }
},

"pitching_line": {
  "type": "object", "additionalProperties": false,
  "required": ["player", "order", "stats"],
  "properties": {
    "player": { "$ref": "#/$defs/player_ref" },
    "order":  { "type": "integer", "minimum": 1 },   // appearance order
    "stats": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "ip":{"type":"string"},"h":{"type":"integer"},"r":{"type":"integer"},
        "er":{"type":"integer"},"bb":{"type":"integer"},"so":{"type":"integer"},
        "ab":{"type":"integer"},"bf":{"type":"integer"}
      }
    },
    "x": { "type": "object" }
  }
}
```

### 1.5 Linescore (serves replay check #1)

```jsonc
"linescore": {
  "type": "object", "additionalProperties": false,
  "required": ["innings", "totals"],
  "properties": {
    // one entry per inning played. Unplayed bottom half (home leads after top 9 → "X")
    // is null, NOT a string, so numeric consumers never trip on "X".
    "innings": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["away", "home"],
        "properties": {
          "away": { "type": ["integer", "null"] },
          "home": { "type": ["integer", "null"] }
        }
      }
    },
    "totals": {
      "type": "object", "additionalProperties": false,
      "required": ["away", "home"],
      "properties": {
        "away": { "$ref": "#/$defs/rhe" },
        "home": { "$ref": "#/$defs/rhe" }
      }
    }
  }
},
"rhe": {
  "type": "object", "additionalProperties": false,
  "required": ["r","h","e"],
  "properties": { "r":{"type":"integer"}, "h":{"type":"integer"}, "e":{"type":"integer"} }
}
```
This is the "independently extracted pane" the replayer validates against: it is parsed from
the linescore table, entirely separately from `events`, so the two can be cross-checked.

### 1.6 Event — the core (serves replay checks #2–#5 and grammar-tail absorption)

```jsonc
"event": {
  "type": "object", "additionalProperties": false,
  "required": ["seq", "inning", "half", "category", "type",
               "narrative", "state_after"],
  "properties": {

    "seq":    { "type": "integer", "minimum": 0 }, // global 0-based order. Explicit (not just
                                                   // array index) so re-parses diff cleanly and
                                                   // events can be sorted independent of storage.
    "inning": { "type": "integer", "minimum": 1 }, // extra innings just keep counting → no cap
    "half":   { "type": "string", "enum": ["top", "bottom"] },

    // CATEGORY is the CLOSED axis every consumer may dispatch on. Small and stable.
    "category": {
      "type": "string",
      "enum": ["plate_appearance", "runner", "substitution", "summary", "administrative"]
    },
    // TYPE is the OPEN axis. Today's ~17 PA types + runner/sub/summary types live here.
    // A never-before-seen tail (triple, caught_stealing, passed_ball, pinch_hit, fielding
    // sub) is a NEW member (MINOR) — and crucially a replayer that has never heard of it
    // STILL replays correctly, because it reads state_after + deltas, not the label. That
    // is how "grammar tails absorbed without breaking files" is actually achieved.
    "type": { "type": "string" },   // e.g. "single","strikeout_swinging","walk","stolen_base",
                                    // "failed_pickoff","wild_pitch","inning_summary","sub_pitcher"

    "batter":  { "$ref": "#/$defs/player_ref" },  // present for plate_appearance
    "pitcher": { "$ref": "#/$defs/player_ref" },  // present whenever a pitcher is on the mound

    // Final count + raw pitch sequence. seq is null on ~13% of PAs (first-pitch balls in
    // play, where StatCrew omits the string) — that is data, not an error. The decoded
    // pitch array is DERIVABLE from the closed alphabet, so it is NOT stored; if a consumer
    // need for it appears it is added under pitches as a MINOR field.
    "pitches": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "balls":   { "type": "integer" },
        "strikes": { "type": "integer" },
        "seq":     { "type": ["string", "null"], "pattern": "^[BFKSH]*$" } // observed alphabet
      }
    },

    "rbi":       { "type": "integer" },  // from the ", RBI" modifier
    "location":  { "type": "string" },   // hit location verbatim: "left field", "cf"
    "fielders":  { "type": "array", "items": { "type": "string" } }, // ["3b","2b","1b"], ["p"], ["1b","unassisted"]

    // Structured modifier flags. Booleans (not free text) so replay/aggregation read them
    // directly. New flags are MINOR additions.
    "flags": {
      "type": "object", "additionalProperties": false,
      "properties": {
        "intentional":  { "type": "boolean" }, // intentional walk
        "sacrifice":    { "type": "boolean" }, // SAC bunt / SF
        "unearned":     { "type": "boolean" }, // run/reach unearned
        "error":        { "type": "boolean" }, // an error was involved
        "double_play":  { "type": "boolean" },
        "game_ending":  { "type": "boolean" }, // the play that ended the game (walk-off, see §3)
        "scoring_play": { "type": "boolean" }  // source rendered the line <strong>
      }
    },

    // The deltas: every base-state change this event caused. One narrative clause may
    // produce two entries (the "advanced to third, scored on an error, unearned" compound),
    // which is exactly why this is a LIST, not one from/to.
    "runners": {
      "type": "array",
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["runner", "from", "to"],
        "properties": {
          "runner": { "$ref": "#/$defs/player_ref" },
          "from":   { "type": ["string","null"], "enum": ["1B","2B","3B", null] }, // null = batter box
          "to":     { "type": "string", "enum": ["1B","2B","3B","H","OUT"] },      // H = scored
          "on":     { "type": "string" }, // reason verbatim tail: "on a wild pitch","stole","error by 2b"
          "earned": { "type": "boolean" } // for to:"H": whether the run was earned
        }
      }
    },

    "narrative": { "type": "string" },  // VERBATIM source line. The ground truth every
                                        // structured field is derived from; kept always.

    // FULL post-event game state. REQUIRED. This is the backbone of replayability: a
    // replayer computes state independently and asserts equality against this snapshot, so
    // every check in §3 is a LOCAL comparison, not a global reconstruction.
    "state_after": {
      "type": "object", "additionalProperties": false,
      "required": ["outs", "score", "bases"],
      "properties": {
        "outs":  { "type": "integer", "minimum": 0, "maximum": 3 },
        "score": {
          "type": "object", "additionalProperties": false,
          "required": ["away","home"],
          "properties": { "away":{"type":"integer"}, "home":{"type":"integer"} }
        },
        "bases": {   // occupant pid (or null) at each base after the event → LOB is free
          "type": "object", "additionalProperties": false,
          "required": ["1B","2B","3B"],
          "properties": {
            "1B": { "type": ["string","null"] },
            "2B": { "type": ["string","null"] },
            "3B": { "type": ["string","null"] }
          }
        }
      }
    },

    "x": { "type": "object" }
  }
}
```

`summary` events (the "Inning Summary" lines) carry `category:"summary"`, `type:"inning_summary"`,
and their counts in `x`-free typed fields:
```jsonc
// a summary event additionally allows (still additionalProperties:false at object level via
// a oneOf branch): "summary": { runs, hits, errors, lob }  — used by replay checks #3.
"summary": {
  "type": "object", "additionalProperties": false,
  "properties": { "runs":{"type":"integer"},"hits":{"type":"integer"},
                  "errors":{"type":"integer"},"lob":{"type":"integer"} }
}
```

### 1.7 unparsed line & meta

```jsonc
"unparsed_line": {
  "type": "object", "additionalProperties": false,
  "required": ["location", "text"],
  "properties": {
    "location": {   // enough to re-find the line in source
      "type": "object", "additionalProperties": false,
      "required": ["inning","half"],
      "properties": {
        "inning": { "type": "integer" },
        "half":   { "type": "string", "enum": ["top","bottom"] },
        "line_index": { "type": "integer" } // 0-based within the half-inning pane
      }
    },
    "text": { "type": "string" }   // the verbatim missed line
  }
},

// PROVENANCE. Excluded from semantic equality: a re-parse by the same parser version must
// reproduce a byte-identical file EXCEPT this block (timestamps, source hash of a re-fetch).
"meta": {
  "type": "object", "additionalProperties": false,
  "required": ["schema_version", "parser", "source", "parse"],
  "properties": {
    "schema_version": { "type": "string", "pattern": "^[0-9]+\\.[0-9]+$" },
    "parser": {
      "type": "object", "additionalProperties": false,
      "required": ["name","version"],
      "properties": { "name":{"type":"string"}, "version":{"type":"string"} }
    },
    "source": {
      "type": "object", "additionalProperties": false,
      "required": ["url","sha256","fetched_at"],
      "properties": {
        "url":       { "type": "string" },  // the boxscore .xml URL scraped
        "sha256":    { "type": "string" },  // hash of the fetched HTML — re-parse provenance
        "fetched_at":{ "type": "string", "format": "date-time" }
      }
    },
    "generated_at": { "type": "string", "format": "date-time" },
    "parse": {   // integrity signals — how honest is this file?
      "type": "object", "additionalProperties": false,
      "required": ["events_count","unparsed_count","replayable"],
      "properties": {
        "events_count":   { "type": "integer" },
        "unparsed_count": { "type": "integer" },
        // false when unparsed_count>0 OR any replay check failed at parse time. A "final"
        // file SHOULD be replayable:true; the schema can REPRESENT the degraded case
        // honestly rather than pretend.
        "replayable":     { "type": "boolean" },
        "warnings":       { "type": "array", "items": { "type": "string" } }
      }
    }
  }
}
```

**Semantic-equality definition (for the write-once / re-parse discipline):** two files are
semantically equal iff they are equal after deleting `meta` (both root and nested) at every
level. Everything else — including `state_after` and `seq` — is content and must match. This
gives re-parse a strong, mechanical correctness net: same parser + same source ⇒ identical
content; a labeled re-parse commit is expected to change content and bumps `parser.version`.

---

## 2. One fully worked example — Top of the 1st, real half-inning

Verbatim from `boxscore_20260709_final.html`, `#pbp-inning-1`, "Top of 1st" (Yuba-Sutter
batting, Long Beach Coast pitching). All eight narrative lines plus the inning summary. Home/
away: away = Yuba-Sutter Freebirds, home = Long Beach Coast. Only the events array shown; root
scaffolding elided for length (`…`).

```json
{
  "schema_version": "1.0",
  "game_id": "20260709_h94w",
  "season": 2026,
  "date": "2026-07-09",
  "status": "final",
  "source": { "provider": "prestosports", "league_id": "pioneer", "site": "pioneerleague.com" },
  "teams": { "away": { "id": "…", "name": "Yuba-Sutter Freebirds", "batters": [ "…" ], "pitchers": [ "…" ] },
             "home": { "id": "…", "name": "Long Beach Coast",      "batters": [ "…" ], "pitchers": [ "…" ] } },
  "linescore": {
    "innings": [ { "away": 1, "home": 1 }, { "away": 1, "home": 1 }, { "away": 1, "home": 0 },
                 { "away": 0, "home": 4 }, { "away": 0, "home": 1 }, { "away": 1, "home": 6 },
                 { "away": 0, "home": 0 }, { "away": 0, "home": 0 }, { "away": 0, "home": null } ],
    "totals": { "away": { "r": 4, "h": 8, "e": 3 }, "home": { "r": 13, "h": 11, "e": 0 } }
  },

  "events": [
    {
      "seq": 0, "inning": 1, "half": "top",
      "category": "plate_appearance", "type": "single",
      "batter":  { "pid": null, "name": "Isaac Nunez" },
      "pitcher": { "pid": null, "name": "starter" },
      "pitches": { "balls": 1, "strikes": 1, "seq": "BS" },
      "location": "left field",
      "runners": [ { "runner": { "pid": null, "name": "Isaac Nunez" }, "from": null, "to": "1B" } ],
      "narrative": "Isaac Nunez singled to left field (1-1 BS).",
      "state_after": { "outs": 0, "score": { "away": 0, "home": 0 },
                       "bases": { "1B": null, "2B": null, "3B": null } }
    },
    {
      "seq": 1, "inning": 1, "half": "top",
      "category": "administrative", "type": "failed_pickoff",
      "runners": [ { "runner": { "pid": null, "name": "Isaac Nunez" }, "from": "1B", "to": "1B" } ],
      "narrative": "Isaac Nunez Failed pickoff attempt.",
      "state_after": { "outs": 0, "score": { "away": 0, "home": 0 },
                       "bases": { "1B": null, "2B": null, "3B": null } }
    },
    {
      "seq": 2, "inning": 1, "half": "top",
      "category": "plate_appearance", "type": "single",
      "batter": { "pid": null, "name": "Jordan Donahue" },
      "pitches": { "balls": 1, "strikes": 2, "seq": "FBSF" },
      "location": "right field",
      "runners": [
        { "runner": { "pid": null, "name": "Jordan Donahue" }, "from": null, "to": "1B" },
        { "runner": { "pid": null, "name": "Isaac Nunez" },    "from": "1B", "to": "3B" }
      ],
      "narrative": "Jordan Donahue singled to right field (1-2 FBSF); Isaac Nunez advanced to third.",
      "state_after": { "outs": 0, "score": { "away": 0, "home": 0 },
                       "bases": { "1B": null, "2B": null, "3B": null } }
    },
    {
      "seq": 3, "inning": 1, "half": "top",
      "category": "administrative", "type": "failed_pickoff",
      "narrative": "Jordan Donahue Failed pickoff attempt.",
      "state_after": { "outs": 0, "score": { "away": 0, "home": 0 },
                       "bases": { "1B": null, "2B": null, "3B": null } }
    },
    {
      "seq": 4, "inning": 1, "half": "top",
      "category": "plate_appearance", "type": "single",
      "batter": { "pid": null, "name": "Josh Phillips" },
      "pitches": { "balls": 0, "strikes": 0, "seq": null },
      "rbi": 1, "location": "center field",
      "flags": { "scoring_play": true },
      "runners": [
        { "runner": { "pid": null, "name": "Josh Phillips" },  "from": null, "to": "1B" },
        { "runner": { "pid": null, "name": "Jordan Donahue" }, "from": "1B", "to": "2B" },
        { "runner": { "pid": null, "name": "Isaac Nunez" },    "from": "3B", "to": "H", "earned": true }
      ],
      "narrative": "Josh Phillips singled to center field, RBI (0-0); Jordan Donahue advanced to second; Isaac Nunez scored.",
      "state_after": { "outs": 0, "score": { "away": 1, "home": 0 },
                       "bases": { "1B": null, "2B": null, "3B": null } }
    },
    {
      "seq": 5, "inning": 1, "half": "top",
      "category": "plate_appearance", "type": "strikeout_swinging",
      "batter": { "pid": null, "name": "Kyle Carlson" },
      "pitches": { "balls": 3, "strikes": 2, "seq": "BBSBKS" },
      "narrative": "Kyle Carlson struck out swinging (3-2 BBSBKS).",
      "state_after": { "outs": 1, "score": { "away": 1, "home": 0 },
                       "bases": { "1B": null, "2B": null, "3B": null } }
    },
    {
      "seq": 6, "inning": 1, "half": "top",
      "category": "plate_appearance", "type": "strikeout_swinging",
      "batter": { "pid": null, "name": "Christian Castaneda" },
      "pitches": { "balls": 0, "strikes": 2, "seq": "KSFS" },
      "runners": [
        { "runner": { "pid": null, "name": "Josh Phillips" },  "from": "1B", "to": "2B", "on": "stole" },
        { "runner": { "pid": null, "name": "Jordan Donahue" }, "from": "2B", "to": "3B", "on": "stole" }
      ],
      "narrative": "Christian Castaneda struck out swinging (0-2 KSFS); Josh Phillips stole second; Jordan Donahue stole third.",
      "state_after": { "outs": 2, "score": { "away": 1, "home": 0 },
                       "bases": { "1B": null, "2B": null, "3B": null } }
    },
    {
      "seq": 7, "inning": 1, "half": "top",
      "category": "plate_appearance", "type": "groundout",
      "batter": { "pid": null, "name": "Andrew Kirchner" },
      "pitches": { "balls": 1, "strikes": 2, "seq": "KSB" },
      "fielders": ["p"],
      "narrative": "Andrew Kirchner grounded out to p (1-2 KSB).",
      "state_after": { "outs": 3, "score": { "away": 1, "home": 0 },
                       "bases": { "1B": null, "2B": null, "3B": null } }
    },
    {
      "seq": 8, "inning": 1, "half": "top",
      "category": "summary", "type": "inning_summary",
      "summary": { "runs": 1, "hits": 3, "errors": 0, "lob": 2 },
      "narrative": "Inning Summary: 1 Runs, 3 Hits, 0 Errors, 2 LOB",
      "state_after": { "outs": 3, "score": { "away": 1, "home": 0 },
                       "bases": { "1B": null, "2B": null, "3B": null } }
    }
  ],

  "unparsed": [],
  "meta": {
    "schema_version": "1.0",
    "parser": { "name": "bc-pbp", "version": "1.0.0" },
    "source": { "url": "https://www.pioneerleague.com/sports/bsb/2026/boxscores/20260709_h94w.xml",
                "sha256": "…", "fetched_at": "2026-07-10T00:00:00Z" },
    "generated_at": "2026-07-10T00:00:05Z",
    "parse": { "events_count": 130, "unparsed_count": 0, "replayable": true, "warnings": [] }
  }
}
```

Notes on fidelity to source, and honest gaps:
- **Base occupancy in `state_after.bases`.** Because Phillips reached (seq 4) then both
  runners stole and the inning ended on the groundout, the *stored* snapshots above show the
  bases occupied at the moment each state is captured. I have shown them empty at the
  half-inning boundaries for brevity of the elision, but the parser MUST populate them with
  the runners' pids (e.g. after seq 6, `"2B": <phillips-pid>`, `"3B": <donahue-pid>`); that is
  precisely what makes LOB=2 fall out of the last in-inning event. The values are shown as
  `null` here only where I did not have pids inlined — see the caveat in the next line.
- **`pid: null` throughout the example.** PBP text carries bare last names only; resolution to
  the 16-char id happens against the *boxscore player list*, which is elided (`…`) in this
  excerpt. In a real emitted file these are resolved 16-char ids (e.g. Eddy Pelc
  `3865oyuz5l2pj51r`), and null appears only for a genuinely unresolvable same-last-name case.
- `pitches.seq: null` on seq 4 is the real StatCrew "first-pitch ball in play" case `(0-0)`.
- seq 4's `<strong>` rendering in source → `flags.scoring_play: true`.

---

## 3. How each replayer check reads from these shapes

Every check is **local** because `state_after` is a stored snapshot the replayer re-derives and
compares. The replayer maintains its own running state, applies each event's `runners[]` +
out/score logic, and asserts equality against `state_after`.

| Replay check | Reads from |
|---|---|
| **Linescore vs independent pane** | Sum `state_after.score` deltas across each `(inning,half)` group → per-inning runs; compare to `linescore.innings[]`. `linescore.totals` cross-checks the final `state_after.score` and box `stats`. The two are parsed from different DOM regions, so agreement is a real check. |
| **Outs == 3 per half-inning** | Group `events` by `(inning,half)`; the max `state_after.outs` in each group must be 3 — **except** the last half-inning of a walk-off, where the closing event has `flags.game_ending:true` and `status` gates the exception (final-only). The replayer allows <3 outs iff `game_ending` is set. |
| **LOB reconciliation** | `state_after.bases` of the last in-inning event (the one reaching `outs:3`, or the walk-off event) lists occupant pids → count non-null = LOB. Compare to the `summary` event's `summary.lob` and to each side's box `stats.lob` sum. Three-way agreement. |
| **Per-batter PA counts** | Count `events` with `category:"plate_appearance"` grouped by `batter.pid`; compare to box line `ab + bb + (hbp) + sac`. Because dispatch is on the closed `category`, this keeps working when a new PA `type` string appears. |
| **Illegal-transition detection** | Walk the `runners[]` from/to against the replayer's own base machine; a `from` base that is not currently occupied by that `runner`, a `to` that double-occupies, or an out/score delta inconsistent with `state_after` is an illegal transition. `state_after.bases` gives the ground truth to diff against at every step, so an illegal transition surfaces at the exact event. |

Because the state math lives in `runners[]` + `state_after` and dispatch is on `category`, a
brand-new `type` (say `"triple"` or `"caught_stealing"`) replays with **zero** replayer changes
— it just carries different `runners[]`. That is the forward-flexibility claim, made concrete
against the actual checks.

---

## 4. Self-assessment

**Depth (behavior per unit of interface).** High where it counts. `state_after` +
`category` + `runners[]` is a small, fixed interface that carries the entire replay contract:
five distinct checks all read those three shapes, and *new* event types add behavior without
adding interface. The player-ref object is one shape reused at ~six sites; cross-season identity
will add one field there and nowhere else. The cost is redundancy: `state_after` stores derived
state on every event. I take that trade deliberately — it converts every replay check from a
global reconstruction into a local comparison, and the semantic-equality rule turns the
redundancy into a *correctness net* (a re-parse that miscomputes state produces a
content-inequality CI failure). Redundancy that is machine-verified against its own derivation
is cheap insurance, not rot.

**Locality of change.** This is the candidate's strongest axis, by construction:
- New event type/tail → one new `type` string; touches the parser's template inventory only.
  No schema MAJOR, no consumer change, no replayer change.
- New parser field → lands in `x`, promoted to a first-class MINOR field in one place.
- Cross-season identity → one `person_id` added to `player_ref`; zero call sites.
- New league/season → data only; `source.league_id` already exists.
- Live ingestion → `status` gains a member; final-only invariants already gated on `status`.
Each future in the brief maps to exactly one edit site. That was the design target.

**Testability.** High. Every game file is self-validating along three independent axes that a
CI job can run with no external data: (1) JSON Schema structural validation; (2) replay from
`events` asserted against stored `state_after` and against `linescore`; (3) semantic-equality
of a re-parse. Faults are localized to a `seq` (illegal transition at event N) or a field
(schema violation), never "somewhere in the game." The closed `category` axis makes the replay
engine itself testable with a tiny fixture set — one event per category exercises every code
path.

**Where it is weakest, honestly.** (a) `state_after` roughly doubles event size on the wire;
for a repo of plain JSON over githubusercontent that is bandwidth, not correctness, but it is
real. (b) The `x` escape hatch is a discipline, not a constraint the validator enforces — if
review lets fields rot in `x`, forward-compat erodes silently; the "promote within one MINOR"
rule needs a human gate. (c) `runners[].on` is a verbatim free-text tail (`"error by 2b"`),
the one place I kept prose inside a structured field rather than fully parsing the fielder/
reason; I judged the fielding-attribution grammar too thin in a single sample to close now, so
I left it as a labeled string to be promoted later rather than invent structure I could not
ground.

---

## 5. What I deliberately left out, and why

- **Decoded pitch arrays.** Derivable from the closed `BFKSH` alphabet + raw `seq`. Storing
  them would be redundant with no replay or query need today. Added under `pitches` as a MINOR
  field if a consumer need appears. (Contrast with `state_after`, which I *did* store despite
  redundancy — because it is load-bearing for replay locality; decoded pitches are not.)
- **Partial/live-game structures.** The `status` axis is reserved (open enum) and the
  final-only invariants are already gated on it, but I built **no** in-progress event shapes,
  no "current batter" pointer, no suspended-game resumption model. That is the speculative
  abstraction the brief warns against — I can name the *axis* the future needs but not the
  *shapes*, so I reserved the axis and stopped.
- **Cross-season person keys / DOB / hometown.** The probe proved these are not derivable from
  a boxscore and reset across seasons. Modeling them now would encode unverified structure. The
  `player_ref` object is the one place they will attach; until then, absent.
- **A free-form `stats` map.** Rejected in favor of typed, optional stat columns. A map would
  buy "any future column for free" but forfeit validation and make every consumer defensive. New
  columns are cheap MINOR additions; the validation is worth more than the saved edits.
- **Team/venue/weather/officials metadata.** Not in the confirmed spec's file contract and not
  needed by any replay check or model in the epic's foundation issues. Additive later via a
  root `game_info` object if a model needs it — no restructure required, so no reason to
  pre-build it.
- **Per-event source line index as the primary key.** I key events on a semantic `seq`, not on
  source DOM position, so a re-parse that reorders how it walks the DOM but produces the same
  game is still semantically equal. Source position is preserved only for *unparsed* lines
  (where re-finding the raw text is the whole point).
