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

## Raw archive & fetching

The raw HTML this pipeline scrapes is never committed to this repo (see the caller contract
above). It lives on the local PC instead, under a single configurable root.

- **Archive root (default):** `C:/PRograms/bc-raw-archive` — PC-local, outside this git working
  tree. This is a *default*, not a hard-coded path: pass `--config` (see below) with an
  `archive_root` override to use a different location.
- **Checkpoint file (default):** `C:/PRograms/bc-raw-archive/checkpoint.json` — a JSON map of
  `source-url -> {archived_path, fetched_at, content_hash, status}`. This checkpoint, not the
  archive directory's filenames, is the sole authority on "have I already fetched this URL" — a
  URL is skipped only when its checkpoint entry has `status: "done"`.
- **Archive filename contract:** `<url-slug>__<fetched-at-microseconds>__<content-hash>.html` —
  the source URL (slugified), the fetch timestamp (integer microseconds since the epoch, so two
  fetches of the same URL are always distinguishable), and a truncated sha256 of the body. A name
  collision refuses to overwrite (`FileExistsError`) rather than silently clobbering data.

### Config shape (`PipelineConfig`)

| Field                  | Default                                    | Meaning                                              |
|------------------------|---------------------------------------------|-------------------------------------------------------|
| `min_interval_seconds` | `12.0`                                      | Minimum seconds between the start of any two fetches (must be `>= 10`, per an observed WAF trip). |
| `jitter_seconds`       | `3.0`                                       | Extra random seconds (0..this) added on top of the minimum interval. |
| `seasons`              | `[2026, 2025, 2024]`                        | Season years walked, in this order (2026 first).       |
| `archive_root`         | `C:/PRograms/bc-raw-archive`                | Local filesystem root for archived raw HTML.            |
| `checkpoint_path`      | `C:/PRograms/bc-raw-archive/checkpoint.json`| Local filesystem path to the checkpoint/resume file.    |

Override any subset of these via a small JSON file passed to `--config`; omitted fields keep their
default.

### Running the CLI

From the `pipeline/` directory:

```bash
python -m bc_pipeline.fetch --dry-run                    # walk schedules, print what WOULD be fetched
python -m bc_pipeline.fetch --limit 5                    # fetch and archive up to 5 new boxscore pages
python -m bc_pipeline.fetch --config my-config.json --limit 20
```

- `--limit N` caps the number of URLs *actually fetched* this run. URLs already marked `done` in
  the checkpoint are skipped and never count against the limit — a second run with `--limit 5`
  against a fully-populated checkpoint reports 0 fetched, it does not refuse to run.

  **`--limit` is a continue-crawl bound, not a fetch-count assertion.** It caps how many *new*
  URLs one invocation fetches; it does not stop the crawl at "N total archived so far." Concretely:
  if a season has more not-yet-done final games than `N`, a same-args re-run does **not** report
  "0 fetched" — it *advances the crawl*, fetching the next `N` not-yet-done games, because
  checkpoint-skipped URLs are passed over without stopping the loop. **Per-URL idempotency is still
  guaranteed** (a URL already in the checkpoint is never re-fetched, ever), but "a same-args run
  fetches nothing new" is only literally true once every reachable URL for the configured
  `seasons` is already `done` — i.e. the backlog is exhausted, not merely "at least `N`
  already-archived." A caller that wants "prove nothing changed" semantics should re-run against
  a config/season scope it has already fully exhausted, not assume a small bounded run implies one.
  (Proven at the unit level in `pipeline/tests/test_fetch.py`:
  `test_second_run_against_same_checkpoint_fetches_zero_new` and
  `test_same_bounded_limit_rerun_against_exhausted_backlog_fetches_zero` use a fully-exhaustible
  fixture backlog; a live run against the real, much larger season backlog will keep advancing
  instead, as observed during issue #18's live demo.)
- `--dry-run` walks each season's schedule page (still fetched over the same paced/challenge-aware
  seam, since that's how the FINAL-game boxscore URLs are enumerated) and prints every boxscore URL
  that would be fetched, but never fetches a boxscore page itself.
- On a detected challenge (HTTP 202 / AWS WAF JS-challenge page / empty body), the run stops
  immediately — no internal retry, no further URLs attempted — and exits non-zero. The checkpoint
  reflects only what completed before the challenge; back off at least 60 seconds before
  re-running (the resumed run picks up exactly where the checkpoint left off).

### Why pioneerleague.com specifically

The schedule walker targets `pioneerleague.com` (the league site) rather than an individual team's
site, even though both are PrestoSports-hosted sites with identical schedule-page markup. This is
because pioneerleague.com's boxscore pages carry both teams' real 16-character player IDs, while a
team-site copy of the same boxscore only carries the home team's real player ID (the visiting
team's players are unresolved on a team site). Since the downstream advanced-stats pipeline needs a
real player ID for both teams in every game, the league-site copy is the only canonical fetch
source for this pipeline.

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

**Test fixtures.** `tests/samples/` holds the curated, archived boxscore pages the
zero-fetch tests run against; it is the sole sanctioned location for committed HTML (a
narrow `!tests/samples/*.html` exception to the blanket `*.html` ignore). The
no-raw-HTML caller-contract clause is intent-scoped to the scraped corpus — curated
test fixtures are exempt.

## License

MIT — see [LICENSE](LICENSE).
