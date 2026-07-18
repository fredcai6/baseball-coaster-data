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

## Backfill

`bc_pipeline.backfill` is the until-caught-up driver: fetch (`schedule`/`fetcher`/`archive`) ->
parse (`parse.parse_game`) -> replay (`replay.replay_game`) -> commit, one season at a time, never
overwriting an already-committed `games/<season>/<game_id>.json` (write-once). Run it from the
`pipeline/` directory:

```bash
python -m bc_pipeline.backfill                       # walk every configured season until caught up
python -m bc_pipeline.backfill --limit 20             # cap total NEW fetches this run (bounded slice)
python -m bc_pipeline.backfill --config my-config.json --repo-root ..
```

It stops immediately (exit 1) on a detected challenge/WAF trip after escalating backoff (60s, 10min,
60min); a resumed run picks back up from the checkpoint plus whatever `games/**` files are already
committed. See `bc_pipeline.backfill.BackfillResult`/`GameOutcome`/`SeasonSummary` for the exact
per-game and per-season outcome shape this driver produces.

### Completeness report (`bc_pipeline.completeness`)

`bc_pipeline.completeness` turns one or more `BackfillResult`s into a single honest completeness
report, written to `artifacts/latest/completeness.json` (mutable, regenerable — see the caller
contract above; it carries a `meta.generated_at` timestamp). Run it from the `pipeline/` directory
against one or more serialized backfill-result JSON files:

```bash
python -m bc_pipeline.completeness --input backfill_result.json --output ../artifacts/latest/completeness.json
python -m bc_pipeline.completeness --input season2024.json season2025.json --threshold 0.03
```

**Report shape:**

- `league.*` — league-wide totals across every season in the input: `games_discovered`,
  `games_fetched`, `games_parsed`, `games_replayable`, `games_non_final`, `games_parse_failed`,
  `games_skipped_already_committed`, `failure_rate` (game-level, see below), and `unparsed_rate`
  (line-level, the real UNPARSED metric, see below).
- `by_season["<year>"]` — the same shape as `league`, scoped to one season.
- `enumerated_failures` — one entry per game whose outcome is `parse_failed`, or whose outcome is
  `parsed` with `replayable: false` — `{game_id, season, url, outcome, reason}`. Every such game is
  listed here; none are ever dropped, truncated, or summarized away.
- `non_final_games` — one entry per game that hit `NonFinalPageError` (`{game_id, season, url,
  reason}`) — an expected, non-alarming outcome, kept separate from `enumerated_failures`.
- `threshold.value` / `threshold.exceeded` — the threshold this run was scored against, and whether
  the league-wide LINE-level `unparsed_rate` crossed it.

This report deliberately carries **two distinct rates**, neither one dropping the other:

**`failure_rate` (game-level).** A game counts against this rate if its outcome is `parse_failed`,
OR its outcome is `parsed` but `replayable` is `false`. `non_final` games are excluded from the
numerator (an unfinished game is an expected negative, not a parse failure) but still count in the
denominator (`games_discovered`), since they were genuinely discovered and looked at this run.
`skipped_already_committed` games are likewise excluded from the numerator (they succeeded in a
previous run) but count in the denominator. Concretely:

```
failure_rate = (games_parse_failed + (games_parsed - games_replayable)) / games_discovered
```

This is a valuable, honestly-reported number in its own right — it is reported, just not what the
CLI threshold gates on (see below).

**`unparsed_rate` (line-level — the real UNPARSED metric).** `parse.py` stamps
`meta.parse.events_count` / `meta.parse.unparsed_count` on every successfully parsed game (the
number of PBP narrative lines it turned into structured events, and the number it could not parse
and dropped into `unparsed[]`); `bc_pipeline.backfill.GameOutcome` threads both numbers through as
`events_count` / `unparsed_count` (`None` for any outcome that never went through a parse this run:
`non_final`, `parse_failed`, `skipped_already_committed`). Per game, when both counts are available:

```
line_unparsed_rate = unparsed_count / (events_count + unparsed_count)
```

`league.unparsed_rate` and each `by_season["<year>"].unparsed_rate` are **totals-based**, not an
average of per-game rates — they weight every narrative line equally regardless of which game
produced it, rather than weighting every game equally regardless of size:

```
unparsed_rate = sum(unparsed_count over parsed games) / sum(events_count + unparsed_count over parsed games)
```

A game with no `events_count`/`unparsed_count` is excluded entirely from both the numerator and the
denominator — never treated as a 0%-unparsed game, never fabricated.

**Threshold mechanism.** The CLI exits nonzero when the league-wide LINE-level `unparsed_rate`
exceeds `--threshold` (default **0.02**, i.e. 2%) — `failure_rate` is reported but does not gate the
run. This default is a **provisional placeholder** — the full multi-season backfill corpus this
report is meant to score does not exist yet at the time this gate was built, and a line-level rate
is a much finer-grained quantity than a game-level failure rate, so the placeholder had to be
re-derived rather than reused at the old game-level magnitude. The intended mechanism, once real
data exists, is: take the observed line-level `unparsed_rate` across the actual backfill slice and
add a fixed safety margin (e.g. +1 percentage point), rather than a hand-picked constant.
`--threshold` lets a real run supply that evidence-grounded value without any code change. 0.02 was
chosen deliberately generous (not tight) so a provisional value does not spuriously fail an
otherwise-healthy early run, while still meaning something at line granularity.

## Refresh

`bc_pipeline.refresh` is the ONE command that keeps this repo current: it runs the backfill driver
(above) to pick up every newly-FINAL game, then regenerates the frequency artifact (below) only if
that regeneration actually changed something. It is a thin orchestration layer — it calls
`bc_pipeline.backfill.run_backfill_with_escalation` and `bc_pipeline.frequencies`'s public functions
unchanged; it adds no pick-up/idempotency/batching logic and no aggregation logic of its own. Run it
from the `pipeline/` directory:

```bash
python -m bc_pipeline.refresh                       # backfill + regenerate frequencies if changed
python -m bc_pipeline.refresh --limit 20             # cap total NEW fetches this run (bounded slice)
python -m bc_pipeline.refresh --config my-config.json --repo-root ..
```

Its CLI flags mirror `bc_pipeline.backfill`'s own (`--config`, `--limit`, `--repo-root`, `--push`) —
the two commands are siblings.

**Sequencing:**

1. Run the backfill escalation loop (fetch -> parse -> replay -> commit every discoverable newly-FINAL
   game, one season at a time — see "Backfill" above).
2. If that stopped on a detected challenge/WAF trip, **skip frequency regeneration entirely** and
   exit 1 — `games/**` reflects only a partial refresh at that point, and regenerating the frequency
   artifact over incomplete state would silently mask the stop. A resumed run picks back up exactly
   where the backfill half left off.
3. Otherwise, regenerate the frequency artifact in memory and compare it (with `meta.generated_at`
   normalized on both sides) against whatever is currently committed at
   `artifacts/latest/frequencies.json`. If they compare equal (or nothing is committed yet and there
   is genuinely nothing to aggregate), this is a **NO-OP** — nothing is written, nothing is
   committed. If they differ, the fresh artifact is written and committed with the SAME commit
   mechanism used for game-file commits, under its own distinct commit message
   (`"refresh: regenerate frequency artifacts"`), separate from any game-file batch commit.
4. Print a one-line summary (new games parsed, game-file commit count, frequency-artifact
   NO-OP-or-CHANGED) and exit 0 (or 1 if step 2 fired).

### Artifacts: frequencies (`bc_pipeline.frequencies`)

`bc_pipeline.frequencies` aggregates every `games/**` file's `events[].outcome.type` — the closed
19-type outcome taxonomy at `schemas/game.schema.json`'s `$defs.outcome.properties.type.enum` — into
a season+league **team** and **player** event-frequency artifact, written to
`artifacts/latest/frequencies.json` (mutable, regenerable — see the caller contract above; it carries
a `meta.generated_at` timestamp). It reads `games/**` only and never re-parses, re-derives, or
fabricates an outcome.

**Shape:** top-level `meta` (`generated_at`, `parser_versions`, `games_included.{total,by_season}`,
`coverage`), `league.{batting,pitching}.{teams,players}` (totals across every aggregated game), and
`by_season.<season>.{batting,pitching}.{teams,players}` (per-season breakdown) — the same
`league`/`by_season` nesting `bc_pipeline.completeness`'s own report uses. `batting` is keyed by
`batting_team`/`batter.player_id` (what a team/player did AT THE PLATE); `pitching` is keyed by
`fielding_team`/`pitcher.player_id` (what a team/player ALLOWED). Every count/rate table always
carries all 19 taxonomy keys, even when a type never occurred for that key (0, never sparse, never
silently omitted), with keys emitted alphabetically for determinism.

**Rate definition:**

```
rate = outcome_type_count / total_plate_appearances_for_that_key
```

For a `batting` entry the denominator is the total plate appearances that team/player BATTED in
(this season, or league-wide for the `league` bucket); for a `pitching` entry it is the total plate
appearances that team/player FACED. Both are counted by construction (every `plate_appearance` event
increments exactly one outcome-type count and the same key's `total_plate_appearances`), so
`sum(counts.values()) == total_plate_appearances` always holds.

**Honest-Null coverage:** `meta.coverage` reports the LINE-level unparsed rate across the aggregated
corpus (from each game's `meta.parse.events_count`/`unparsed_count`, stamped by `parse.py` — never
recomputed here), plus an explicit note that outcome-type counts are drawn only from `events[]`: a
source line the parser could not classify (landing in `unparsed[]`) is not represented in any count
here, and may under-count rare event types. Never imputed, never fabricated.

**CLI and the no-commit guard:**

```bash
python -m bc_pipeline.frequencies --input games/ --output artifacts/latest/frequencies.json
python -m bc_pipeline.frequencies --check-no-commit
```

`--check-no-commit` regenerates the artifact in memory and compares it (with `generated_at`
normalized on both sides) against the currently-committed `--output` file **without writing**: exit 0
+ a "NO-OP" message when nothing but the timestamp would change, exit 2 + a "CHANGED" message
otherwise. This CLI flag only reports the comparison — it never decides whether to `git commit`; that
decision (and the actual write) is `bc_pipeline.refresh`'s job (see "Refresh" above), which uses the
same public functions (`load_games`, `build_frequencies`, `normalize_generated_at`) directly rather
than shelling out to this CLI.
