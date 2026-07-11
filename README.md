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

## License

MIT — see [LICENSE](LICENSE).
