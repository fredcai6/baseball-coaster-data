# Fixture-promotion protocol

How one `unparsed[]` line, once its grammar rule lands, becomes a golden or
unit fixture. This doc is mechanically checked to exist by
`tests/test_docs.py`; it is referenced from the README's "Parsing & replay"
section, which is the other half of that mechanical check.

## When this applies

`bc_pipeline.parse.build_events` guarantees every PBP line becomes an event
OR a verbatim `unparsed[]` entry — never dropped, never guessed (see
`parse.py`'s module docstring). A line lands in `unparsed[]` for one of two
reasons:

1. **A genuine taxonomy gap** — no grammar rule (yet) covers this narrative
   shape, or the schema has no honest way to represent it (e.g. the 5 live
   DH-pitching-sub lines in the sample game — a two-way-DH substitution has
   no batting-order slot, and the schema's substitution shape requires one).
2. **A resolution failure** — the shape is covered, but a name didn't
   resolve uniquely (an identity-table problem, not a grammar-table one).

This protocol is about case 1: an `unparsed[]` line whose *narrative shape*
has no rule yet, and the day a rule lands for it.

## The steps

1. **Note the live residue.** `reparse_summary.summarize(game)["unparsed_count"]`
   (and the `unparsed[]` list itself) is the running census of every line
   still pending a rule. Each entry carries its own verbatim `raw` narrative
   and a `reason` string explaining the miss — that `raw` text is exactly
   what step 3 needs.
2. **Land the grammar rule.** Add the new row to `bc_pipeline.grammar`'s
   `PRIMARY_RULES`/`RUNNER_RULES`/`STANDALONE_RULES` (closed taxonomy — see
   that module's docstring: only 17 `outcome.type` values and 12
   `runner.cause` values exist; a new row must map onto one of them, never
   invent a new one). This is a g4-owned change outside this gate's scope —
   this protocol documents what happens AFTER that rule exists.
3. **Confirm it parses.** Re-run `parse.build_events` (or the full
   `parse.parse_game` + `replay.replay_game` pipeline) over the line (or the
   whole game it came from). The line now produces a real event instead of
   an `unparsed[]` entry. `reparse_summary.diff(before, after)` makes this
   visible as a machine-checkable delta: `unparsed_rate_delta` moves
   negative, and `event_type_count_deltas` shows the new event kind's count
   going up.
4. **Promote a fixture.**
   - If the newly-parseable line is part of the LIVE sample game
     (`tests/samples/boxscore_20260709_final.html`), re-run
     `PYTHONIOENCODING=utf-8 PYTHONPATH=pipeline py -m
     bc_pipeline.reparse_summary` (read-only preview of the delta), review
     it, then re-run with `--write` to accept the new golden
     (`tests/fixtures/golden/game_20260709_h94w.json`). The regeneration is
     GATED by that printed delta — never a silent overwrite (see
     `tests/test_golden.py`).
   - If the newly-parseable *shape* is not exercised by the live sample at
     all (a taxonomy-tail case), hand-author a SYNTHETIC unit fixture under
     `tests/fixtures/synthetic_taxonomy_tail/` instead: a minimal
     `identity.PlayerTable` + one `parse.PbpLine` + the captured
     `build_events` output, clearly labeled `"description": "SYNTHETIC. ..."`
     at the top of the file (never placed under `games/**` or treated as a
     real-game fixture).
5. **Wire a regression test.** Add (or extend) a unit test asserting the
   line no longer appears in `unparsed[]` and its event shape matches the
   promoted fixture, so a future grammar refactor that regresses the rule is
   caught immediately.

## Worked exercise (done once, for real)

`tests/fixtures/synthetic_taxonomy_tail/triple_promotion_exercise.json` is
this protocol run once end-to-end against a SYNTHETIC taxonomy-tail shape —
a **triple** (`outcome_type: "triple"`), chosen because it does not occur
anywhere in the live sample (now 122 events / 0 unparsed after the DH
promotion below, all unrelated to this shape), so it is a genuine "shape not
in the sample":

- **(a)** the file's `step_a_hypothetical_pre_rule_unparsed` block shows what
  `build_events` would have emitted to `unparsed[]` if no rule existed yet
  for `"<name> tripled to <loc>"` (illustrative — the rule already exists,
  so this exact miss can't be reproduced without deleting code).
- **(b)** `step_b_rule_already_in_closed_taxonomy` cites the real
  `PRIMARY_RULES` row in `bc_pipeline.grammar` that already covers it (g4's
  closed taxonomy, landed before this gate).
- **(c)** `step_c_and_d_promoted_fixture.build_events_output` is the ACTUAL,
  captured output of running `parse.build_events` on a synthetic one-line
  `PbpLine` ("Smith tripled to left field (1-2 BFK).") against a two-player
  synthetic `identity.PlayerTable` — real code, real output, not hand-typed.
- **(d)** that captured output IS the promoted fixture (the file itself).

Reproduce step (c) with:

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=pipeline py -c "
from bc_pipeline import identity, parse as parse_mod
away = identity.TeamIdentity(team_id='syn:team:away', name='Synthetic Away', players={
    'syn:away:1': identity.PlayerEntry(player_id='syn:away:1', name='Pat Smith', last_name='Smith', team_id='syn:team:away', positions=['cf']),
})
home = identity.TeamIdentity(team_id='syn:team:home', name='Synthetic Home', players={
    'syn:home:1': identity.PlayerEntry(player_id='syn:home:1', name='Jordan Lee', last_name='Lee', team_id='syn:team:home', positions=['p']),
})
player_table = identity.PlayerTable(home=home, away=away)
line = parse_mod.PbpLine(inning=1, half='top', line_index=0, text='Smith tripled to left field (1-2 BFK).', is_strong=False)
print(parse_mod.build_events([line], player_table))
"
```

## The live promotion example — EXERCISED (real, not synthetic)

The 5 `unparsed[]` lines in `game_20260709_h94w` (a DH-team pitching
substitution with no batting-order slot) were this protocol's FIRST REAL run
on genuinely-unparsed live data. Originally (schema 1.0.0) they were step (a) —
genuinely unparsed because `$defs.substitution.slot` required 1-9. The gap was
floated to and ratified by the human (issue #19), the schema evolved additively
(1.0.0 → 1.1.0, `substitution.slot` made nullable — see `docs/design/DECISION.md`
§7), and those 5 lines then followed steps 2-5: they are now real substitution
events (`slot: null`, `kind: "pitching"`), and the golden fixture was
regenerated (gated, per step 4). The reparse-summary delta that gated the
regeneration was `{"event_type_count_deltas": {"substitution": 5},
"unparsed_rate_delta": -0.041, "replay_delta": 0.0}` — the sample went from
117 events / 5 unparsed to 122 events / 0 unparsed.
