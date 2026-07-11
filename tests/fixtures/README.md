# Test fixtures

## `game_20260709_h94w_top1.json`

Hand-encoded, schema-valid fixture for the sample game **`20260709_h94w`** — Yuba-Sutter
Freebirds @ Long Beach Coast, 2026-07-09 — proving that the **frozen** schema
(`schemas/game.schema.json`) can encode a real game. It validates as-is against
`jsonschema` (Draft 2020-12); see the validation command at the bottom.

> **None of the caveats below relaxes the schema.** The file validates unchanged.
> Every documented item is a matter of *scope* (Top-1st only) or *source honesty*
> (which ids the on-disk copy carries) — not a schema concession.

### 1. This is a PARTIAL game — Top of the 1st only (9 events)

`events[]` holds exactly the 9 events of the Top of the 1st: 8 narrative lines
(`seq 0-7`) plus the inning-summary line (`seq 8`). It is **not** a full game. The
events are encoded from Schema Candidate B §9's worked "Top of the 1st" example
(`docs/design/SCHEMA_CANDIDATE_B.md`), adapted to the committed schema's exact field
names/shapes, and the narratives are **byte-verbatim** from the on-disk HTML
`#pbp-inning-1` top half (`boxscore_20260709_final.html`, lines 4602-4652).

Because only 6 away batters come to the plate in the Top 1st, several root blocks are
scoped to those 6 players (see below). This is a *fixture*, not a published game file.

### 2. Away batters carry synthetic `syn:away:n` pids; VanDeventer carries his real id

The only on-disk copy of this game is the **team-site (longbeachcoast.com) copy**
(`source.site: "longbeachcoast.com"`). That copy links only the **HOME** team's players
to their 16-char Presto ids; the **AWAY** (Yuba-Sutter) batters appear as plain text
with no `players?id`. So:

- The 6 away batters get synthetic file-local keys `syn:away:1`…`syn:away:6`
  (the schema's sanctioned `syn:<home|away>:<n>` fallback for a team-site copy — a
  ratified design decision, see the epic's `DECISION.md`). Their event refs are
  `resolved: true` because the last-name token *does* resolve to a box entry; the pid
  is synthetic only because **no Presto id exists for them in this source**.
- The pitcher **Garrett VanDeventer** carries his **real** 16-char id
  `4bs3tvwryvtzrvpa`, verified present in the on-disk HTML (batter table line 3388,
  pitcher table line 3867). His pitcher refs are `resolved: true`, `name_raw: null`
  (the PBP narrative never names the pitcher — he is resolved from the lineup).

Id map (Candidate B §9's readable placeholders → this fixture):
`ys:nunez`→`syn:away:1`, `ys:donahue`→`syn:away:2`, `ys:phillips`→`syn:away:3`,
`ys:carlson`→`syn:away:4`, `ys:castaneda`→`syn:away:5`, `ys:kirchner`→`syn:away:6`,
`lbc:p1`→`4bs3tvwryvtzrvpa`. Team ids (read from the on-disk sample): away
(Yuba-Sutter) `ndn2a2djbgbd0lh4`, home (Long Beach Coast) `maotayco79j2g2lx`.

### 3. `_derived` is hand-computed

Every `_derived` block is **hand-computed** (reproduced from Candidate B §9), not
CI-stamped. In production, `_derived` is stamped by the canonical replayer that arrives
with issue #19; until then these values are illustrative-but-consistent (they reconcile
against the inning summary: 3 hits, 1 run, 2 LOB at the 3rd out). `meta.parser_version`
is `"fixture-handencoded"` and `meta.derived_replayer_version` is `"hand-computed"` to
make the hand-authored provenance explicit.

### 4. Blocks populated with whole-game context, or a documented partial subset

These blocks are schema-required, so they are populated with **real** on-disk data — but
that data is whole-game or 6-player context, never fabricated:

- **`linescore`** — the REAL **full-game** linescore from the on-disk box: away per-inning
  `[1,1,1,0,0,1,0,0,0]`, home `[1,1,0,4,1,6,0,0,null]` (`null` = the "X" unbatted home
  9th), totals away `{R:4,H:8,E:3}`, home `{R:13,H:11,E:0}`. This is whole-game context,
  not Top-1st-only — the Top 1st alone accounts for only `away[0] == 1` run.
- **`box.batting[ndn2a2djbgbd0lh4]`** — the REAL **whole-game** away batting lines for the
  6 batters who appear (AB/R/H/RBI/BB/SO/LOB/AVG verbatim from the batting table,
  lines 2780-2869). E.g. Nunez shows `AB:5` — a full-game total, not his Top-1st line.
- **`box.pitching[maotayco79j2g2lx]`** — VanDeventer's REAL **whole-game** pitching line
  (`IP:"6.2", H:8, R:4, ER:4, BB:3, SO:6`, lines 3867-3877).
- **`lineups[ndn2a2djbgbd0lh4]`** — a **partial** batting order: slots 1-6 only, for the 6
  away batters who appear in the Top 1st (positions ss/2b/dh/3b/1b/rf from lines
  2783-2858). Slots 7-9 and `home` are omitted because those players are never referenced
  in this Top-1st fixture; adding them would require inventing pids/order not exercised by
  the events. `substitutions: []` (no subs in the Top 1st).
- **`players`** — one entry per player *referenced by the events*: the 6 away batters
  (synthetic pids) + VanDeventer (real id). Not the full rosters.

### 5. `meta` and the sha256

`meta.source_sha256` is the **real** SHA-256 of the on-disk HTML
(`2d2a7f1688c14e547a60f8a854b3b03fbcad78a1f7673a06aba34cec18519f86`), computed locally —
not a placeholder. `meta.source_url` points at the honest team-site host
(`longbeachcoast.com`, the copy actually on disk), consistent with `source.site`.
`fetched_at`/`parsed_at` are provenance timestamps for the hand-encoding pass.
`meta.parse.warnings` records the partial-game / synthetic-pid / hand-computed caveats.

### Validation

```bash
cd C:/PRograms/baseball-coaster-data
py -c "import json,jsonschema; s=json.load(open('schemas/game.schema.json')); jsonschema.Draft202012Validator(s).validate(json.load(open('tests/fixtures/game_20260709_h94w_top1.json'))); print('VALID')"
# -> VALID
```
