# baseball-coaster-data

Canonical, version-controlled store of **baseball game data** for the Long Beach Coast
advanced-stats pipeline. This repo is the single source of truth: parsed game files land here,
derived analysis artifacts are regenerated from them, and everything downstream (run-expectancy
models, Elo, matchup clusters) reads from this repo and nothing else.

New to baseball? A few terms this README uses, one line each:

- **Play-by-play (PBP)** — the ordered narrative of everything that happened in a game, one line
  per play (e.g. "Isaac Nunez singled to left field"). It is the spine every game file is built
  from.
- **Base-out state** — which bases have runners and how many outs there are. There are 24
  combinations (8 base configurations x 3 out counts), and advanced stats bin every play into one
  of them to attach run values.
- **Plate appearance (PA)** — one batter's complete turn at the plate, ending in a hit, walk,
  out, etc.
- **Linescore** — the runs-per-inning grid plus the runs/hits/errors totals for each team.

## Repository layout

```
games/            canonical, write-once game files:  games/<season>/<game_id>.json
artifacts/latest/ derived, mutable analysis outputs (regenerated from games/**)
pipeline/         the Python package that fetches, parses, and replays games (bc_pipeline)
schemas/          the JSON Schema every game file is validated against (added in a later gate)
docs/design/      the schema design record: the three candidates + the DECISION
tests/fixtures/   golden fixtures for the parser/validator (added in a later gate)
scripts/          CI + validation helper scripts (added in a later gate)
.github/workflows/ continuous-integration workflows (added in a later gate)
```

## The caller contract

Three rules govern how data in this repo may change. They are the contract every consumer and
every pipeline run relies on:

1. **`games/**` is write-once.** A final game file changes only in an explicitly labeled
   re-parse commit. It is never silently edited in place. If the parser improves and a game must
   be regenerated, that is a deliberate, labeled commit — not an ambient overwrite.

2. **`artifacts/**` is mutable.** Everything under `artifacts/` is derived from `games/**` and
   may be regenerated freely. Each artifact carries a `meta.generated_at` timestamp so a consumer
   can tell when it was last rebuilt.

3. **Raw scraped HTML is never committed.** The raw boxscore/PBP HTML that games are parsed from
   lives on the local PC, outside git. It is not part of this repo. `.gitignore` excludes `*.html`
   so a stray raw page can never be committed by accident.

## Semantic equality

Two game files describe **the same game** if and only if they are deep-equal after deleting the
`meta` block and every `_derived` block. `meta` is provenance (timestamps, source hash, parser
version) and `_derived` is a regenerable cache — neither is part of the game's identity. This is
the rule the write-once / re-parse discipline is checked against, and it also appears in the
schema's root `$comment`.

## Parsing & replay

The pipeline turns a raw StatCrew boxscore page into a schema-valid `final` game file
in three independently-testable stages, plus a machine-checkable summary of any run:

1. **Parse** (`bc_pipeline.parse.parse_game(html, source_url=..., fetched_at=...)`) reads
   the page's structural DOM and its own closed play-by-play grammar, and folds every
   PBP line forward into the schema's `events[]` spine. Every line becomes an event OR a
   verbatim `unparsed[]` entry — never dropped, never guessed. Parsing is
   **zero-fetch**: it never makes a network call; it only reads the HTML string handed
   to it (fetching that HTML is a separate, earlier gate's job).
2. **Replay** (`bc_pipeline.replay.replay_game(game, html)`) is an INDEPENDENT check —
   it re-derives the linescore/box oracle from the same raw HTML with its own,
   unshared table-reading code, folds the parser's asserted runner primitives forward
   into the `_derived` base-out cache, and runs five checks (linescore, outs-per-half,
   LOB, PA counts, illegal transitions). A failed check flags the game
   (`meta.parse.replayable = False` + a warning); it never silently passes and never
   raises past the caller.
3. **Reparse-summary** (`bc_pipeline.reparse_summary`) turns one parse+replay run into a
   small, stable, JSON-serializable summary (`summarize`: replay pass/fail, unparsed
   rate, event-type counts) and the delta between two runs (`diff`: zero on two
   identical runs, and isolated to exactly what changed otherwise). This is what gates
   golden-fixture regeneration — see below.

**Determinism.** Two game files describe the same game iff they are deep-equal after
removing the root `meta` key and every `_derived` block (`bc_pipeline.serialize.
semantic_equal`/`canonical_dumps` — see "Semantic equality" above). A re-parse of
byte-identical HTML by the same parser version always produces the same
`idempotency_key` (`source sha256 + parser version`).

**`unparsed[]`.** A line the current grammar/schema cannot honestly represent is never
fabricated or dropped — it is preserved verbatim in `unparsed[]` with its location and
the reason it missed. `tests/fixtures/PROMOTION_PROTOCOL.md` documents how an
`unparsed[]` line, once its grammar rule lands, is promoted into a golden or synthetic
unit fixture (exercised once, end to end, in
`tests/fixtures/synthetic_taxonomy_tail/`).

**Golden fixtures.** `tests/fixtures/golden/` holds the full parse+replay output of the
live sample game, with volatile `meta` timestamps normalized so the fixture never
depends on wall-clock time. `PYTHONIOENCODING=utf-8 PYTHONPATH=pipeline py -m
bc_pipeline.reparse_summary` re-parses the sample and prints the reparse-summary delta
against the committed golden — read-only by default; pass `--write` to accept that
delta and regenerate the golden. Regeneration is always gated by this visible delta,
never a silent overwrite.

## License

MIT — see [LICENSE](LICENSE).
