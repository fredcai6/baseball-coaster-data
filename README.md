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
pipeline/         the Python package that fetches, parses, and replays games (bc_pipeline),
                  including bc_pipeline.refresh (the backfill + frequencies orchestrator --
                  see "Refresh" below) and bc_pipeline.frequencies (the season+league
                  event-frequency aggregator -- see "Artifacts: frequencies" below)
                  bc_pipeline.person_map (the within-season person_id builder) and
                  bc_pipeline.team_map (cross-season franchise_id) and
                  bc_pipeline.career_map (cross-season career_id) -- see the
                  "Artifacts: ..." sections below
schemas/          the JSON Schemas game files and artifacts are validated against:
                  game.schema.json (current: 1.9.0) and frequencies.schema.json
docs/design/      the schema design record: the three candidates + the DECISION
tests/fixtures/   golden fixtures for the parser/validator
scripts/          CI + validation helper scripts (including check_artifacts_current.py,
                  which fails CI when a derived artifact is stale w.r.t. games/**)
.github/workflows/ continuous-integration workflows
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

   Because they are derived, **a re-parse leaves every one of them stale by construction**, and a
   schema check cannot see that — a stale artifact is still perfectly well-formed. That is not
   hypothetical: `frequencies.json` sat stale across `reparse(v0.4.0)` (which took replayable games
   from 84 to 999) while `validate_frequencies.py` passed on it the whole time. CI now checks
   freshness directly with `scripts/check_artifacts_current.py`, which regenerates each artifact in
   memory and compares it against what is committed:

   ```bash
   python scripts/check_artifacts_current.py
   ```

   `bc_pipeline.reparse` also prints the regeneration commands after any `--write` run.

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

- **Archive root (default):** `~/bc-raw-archive` — PC-local, outside this git working tree.
  This is a *default*, not a hard-coded path: pass `--config` (see below) with an
  `archive_root` override to use a different location.
- **Checkpoint file (default):** `~/bc-raw-archive/checkpoint.json` — a JSON map of
  `source-url -> {archived_path, fetched_at, content_hash, status}`. This checkpoint, not the
  archive directory's filenames, is the authority on "has *this machine* already fetched this
  URL" — a URL passes that check only when its checkpoint entry has `status: "done"`.
- **The corpus outranks the checkpoint.** `backfill` also skips any URL whose
  `games/<season>/<game_id>.json` already exists, *before* consulting the checkpoint, so a game
  the repo owns is never re-downloaded. The distinction matters because the checkpoint is a
  **machine-local** fact while `games/**` is the **durable** one: a fresh workstation has no
  checkpoint, and a checkpoint-only rule makes it re-download the entire committed corpus (~4.8 h
  at the `>= 10 s` pacing floor for 2024–2026) before it can reach a single new game. This is not
  hypothetical — it cost a real run ~45 minutes of redundant fetching. `bc_pipeline.fetch`'s own
  CLI passes no such predicate, because it archives raw without owning a corpus.

  One consequence worth knowing: a skipped-at-fetch game is never re-parsed, so the committed
  file's content-drift check (its stored idempotency key vs a fresh re-parse of the archived
  HTML) does not run for it. That check only ever ran when the raw HTML was present locally, and
  detecting drift is a deliberate re-parse's job — but it is a behavior change, not a pure
  optimization.
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
| `archive_root`         | `~/bc-raw-archive`                          | Local filesystem root for archived raw HTML (absolute, outside the working tree). |
| `checkpoint_path`      | `~/bc-raw-archive/checkpoint.json`          | Local filesystem path to the checkpoint/resume file.    |

Override any subset of these via a small JSON file passed to `--config`; omitted fields keep their
default.

**Path invariants.** `archive_root` and `checkpoint_path` are normalized on construction
(`~` and `$VAR` expanded, then resolved absolute) and are refused outright if they are
relative, or if they resolve inside this git working tree. Both rules exist because a
Windows-style `C:/...` path is *relative* on a POSIX host: the previous
`C:/PRograms/bc-raw-archive` default silently materialized a raw archive at
`pipeline/C:/PRograms/bc-raw-archive/` inside the tree. `.gitignore`'s `*.html` rule kept
the raw HTML out of git, but the `checkpoint.json` beside it was untracked and unignored —
the contract is "outside the repo", not "gitignored within the repo".

The defaults are home-rooted for the Linux workstation that is the standing single writer
for this repo. Any other host should pass `--config`.

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
   into the `_derived` base-out cache, and runs seven checks. Five read events
   (linescore, outs-per-half, LOB, PA counts, illegal transitions); two do not
   (`content`, `box_linescore`) and therefore run on every game whatever its
   `record_shape`. A failed check flags the game (`meta.parse.replayable = False` + a
   warning); it never silently passes and never raises past the caller.

   `content` is the floor and reads as a tautology until you know why it is there.
   Every other check is written to find a DISAGREEMENT, so every other check passes
   when handed nothing — a page describing no baseball scored as fully validated for
   the life of the corpus. `content` asserts there is something to check, which is the
   one thing that cannot pass vacuously.

**Record shapes.** `record_shape` (schema 1.12.0) is `play_by_play` for all but two
games. `boxscore_only` is a completed final whose play-by-play the league never
published — a real linescore and a real batting box, no events — and it selects the
check set, because the five event oracles have nothing to read on one and running them
anyway would be five vacuous passes. The schema enforces both floors: a `boxscore_only`
record may not carry events, and a `play_by_play` record may not be empty of them.

**Corrections and dispositions.** Two files, both pinned, neither a place to record an
opinion. `corrections/errata.json` rewrites a defective SOURCE line before the grammar
sees it, pinned to that line's sha256, so a one-off scorer error never becomes a reason
to loosen a general rule. `corrections/dispositions.json` records the games that are
DONE without being clean — parsed, committed, and refused by an oracle with no
correction that can reach the defect — pinned to the sha256 of the replay warnings the
entry was authored against. `tests/test_dispositions.py` reconciles that ledger against
a full corpus replay in both directions, so a disposition can go stale but not quietly:
a game failing with no entry is `undisclosed`, an entry whose pinned failure changed is
`stale`, and an entry for a game that now passes is `spent`.

**The score.** `artifacts/latest/completeness.json` carries an `accounting` block —
discovered / replay-validating / disposed / accounted / **unaccounted**. `unaccounted`
is the only one of them that can be a lie, and it is the one to read. Regenerate with
`python scripts/build_completeness.py`, or ask whether it is current with
`--check`; this is the one derived artifact `check_artifacts_current.py` does not
cover, and it went stale silently for most of 2026 before there was a script.
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

**Running the tests.** The suite is split across two directories — repo-root `tests/`
(parse/grammar/identity/replay/serialize) and `pipeline/tests/` (fetch/backfill/
completeness/frequencies/refresh/config) — and both need `bc_pipeline` importable
without an install. Run the whole thing from the repo root with `PYTHONPATH` pointed at
`pipeline/`:

```bash
PYTHONPATH=pipeline python -m pytest tests pipeline/tests -q
```

CI (`.github/workflows/validate.yml`) runs this exact command on every push/PR, in
addition to the schema-validation scripts below.

### Corpus re-parse (`bc_pipeline.reparse`)

`games/**` is write-once, so a committed game file changes only in an explicitly labeled
`reparse(vX.Y.Z): ...` commit. `bc_pipeline.reparse` is how that commit gets made. Run it
from the `pipeline/` directory:

```bash
python -m bc_pipeline.reparse --version 0.4.0                  # dry run, prints the delta
python -m bc_pipeline.reparse --version 0.4.0 --write          # apply
python -m bc_pipeline.reparse --version 0.4.0 --write --commit \
    --message "grammar + identity + schema (issue #40)"        # apply and label
```

It never fetches — it reads archived raw HTML that `bc_pipeline.fetch` or
`bc_pipeline.backfill` already put on disk, located via the checkpoint.

**Coverage gate.** Every committed game must have archived raw HTML. A *partial* re-parse is
refused by default, because it would leave the corpus straddling two parser versions with no
marker saying which file is which. `--allow-partial` opts out deliberately and is reported in
the summary.

**Semantic comparison.** A game counts as changed only if it differs under
`serialize.semantic_equal` (meta and every `_derived` block stripped), so a run that changes
nothing but provenance does not churn the corpus.

**What it prints** is a stable, serializable corpus delta — unparsed lines, clean-parse games,
replayable games, per season and overall — plus, at the top level, any game that stopped
replaying. That last one is the thing a re-parse must never do quietly.

The commit subject it builds is the same shape `scripts/check_write_once.py` recognizes; a test
asserts that cross-module contract directly, so the two cannot drift into a re-parse commit that
its own CI rejects.

### Write-once guard (`scripts/check_write_once.py`)

Caller-contract clause 1 says `games/**` is write-once — a final game file changes only
in an explicitly labeled re-parse commit. `scripts/check_write_once.py` is the check that
makes that clause enforceable instead of merely stated, and CI runs it on every push/PR.

It diffs a commit **range** (`--base`/`--head`, or `BASE_SHA`/`HEAD_SHA` in the
environment) and classifies every `games/**` change in it:

- **Additions pass.** A new game file is the pipeline doing its job.
- **Modifications** are judged under the [semantic equality](#semantic-equality) rule:
  the base and head blobs are compared after deleting the root `meta` block and every
  `_derived` block at any depth. A diff confined to those is provenance/cache churn, not
  a rewrite of the game — it passes with an informational note.
- **Semantic modifications, deletions, and renames** require that *every* non-merge
  commit in the range touching that path is a labeled re-parse commit, meaning a subject
  matching `reparse(vX.Y.Z): ...` (the convention set by `reparse(v0.2.0)` and
  `reparse(v0.3.0)`). Checking every commit rather than just the tip is deliberate: a
  stray hand-edit riding in behind a legitimate re-parse must still be caught.

A base of all zeros — what `github.event.before` carries on a branch's first push — is
reported as a clean SKIP rather than an error; write-once is checked on the next push.
`artifacts/**` is out of scope by construction (clause 2: it is the mutable tier).

```bash
python scripts/check_write_once.py --base <sha> --head <sha>
```

## Analysis: the plate-appearance table (`bc_pipeline.pa_table`)

The corpus's `events[]` spine is the play-by-play, and every analysis consumer
would otherwise re-walk 1,485 files and re-derive the same columns — which is
where a shared bug hides. `bc_pipeline.pa_table` is that walk, done once:
**one row per plate appearance, 125,827 of them, built in about two seconds.**

```bash
python scripts/build_pa_table.py            # -> artifacts/derived/pa_table.csv (38 MB)
python scripts/build_pa_table.py --check    # summarize without writing
python scripts/check_pa_table_vs_box.py     # reconcile against the boxscore
```

The output is a **cache, not an artifact**: regenerable from committed inputs in
seconds, so it lives under `artifacts/derived/` and is gitignored, by the same
argument that keeps `_derived` out of semantic equality. It is not one of the
four artifacts and `check_artifacts_current.py` does not know about it.

**The table encodes two rules so the caller does not have to remember them.**

  - **`player_id` does not join across games** (see person_map, above). The join
    keys are `batter_career` / `pitcher_career` and `batter_person` /
    `pitcher_person`; the raw `*_pid` columns are traceability back to the source
    file and are explicitly not join keys. One row in 125,827 lacks a career id.
  - **`outcome` is an object.** Reading it as a scalar yields nothing and raises
    nothing — which is a real failure mode, not a hypothetical one.

Beyond the flattening it folds three things the spine implies but does not state:
the counting-stat primitives (`is_ab`, `is_hit`, `is_k`, `is_bb`, `is_sac`, …),
matchup context (`tto` — times this batter has faced this pitcher in this game —
plus `pitcher_bf`, `pitcher_is_starter`, `order_slot`), and a coarse absolute
spray field folded from `fielders[0]` or the prose `location`, populated on 89%
of balls in play. Note that **`bats_side` is null on all 41,713 player records**,
so handedness is unavailable and pull/oppo cannot be derived — only direction.

### The boxscore is the oracle

The box is parsed from a different region of the source page than the narrative,
so per-player AB/H/BB/SO folded from `events[]` and the same totals read from the
box are two independent readings of one truth. `tests/test_pa_table.py` reconciles
them across the corpus: **41,122 of 41,122 clean player-games agree on all four
fields.** The 37 residual disagreements all fall in games
`corrections/dispositions.json` already discloses as dropping or misattributing
plate appearances — and `oracle_residual`, the class meaning our check is wrong
rather than the source, shows zero.

**That check earned its keep on its first run.** The fold scored `AB` from the
outcome *type*, and the corpus records only **36 of its 1,895 sacrifices** as
`type: "sacrifice"` — the other 1,859 are ordinary flyouts, groundouts, lineouts,
foul outs, popouts, fielders' choices and reached-on-errors carrying a `SAC` or
`sacrifice fly` **modifier**, spanning eight types. AB was over-counted on 1,749
of 41,122 player-games; `H`, `BB` and `SO` were perfect. Nothing but an oracle
parsed from elsewhere on the page would have found it, and the fold had a green
test suite in every other respect. *Whatever has no oracle turns out to be wrong.*

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
python -m bc_pipeline.backfill --config my-config.json    # --repo-root only if auto-detect can't find it
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
(above) to pick up every newly-FINAL game, then regenerates each derived artifact — the person map
and the frequency artifact — only if that regeneration actually changed something. It is a thin
orchestration layer: it calls `bc_pipeline.backfill.run_backfill_with_escalation` and the public
functions of `bc_pipeline.person_map` and `bc_pipeline.frequencies` unchanged; it adds no
pick-up/idempotency/batching logic and no aggregation or linking logic of its own. Run it from the
`pipeline/` directory:

```bash
python -m bc_pipeline.refresh                       # backfill + regenerate derived artifacts if changed
python -m bc_pipeline.refresh --limit 20             # cap total NEW fetches this run (bounded slice)
python -m bc_pipeline.refresh --config my-config.json     # --repo-root only if auto-detect can't find it
```

Its CLI flags mirror `bc_pipeline.backfill`'s own (`--config`, `--limit`, `--repo-root`, `--push`) —
the two commands are siblings.

**`--repo-root` is auto-detected.** Both CLIs walk up from the current directory looking for a
checkout that has a `games/` directory, so running from `pipeline/` (as above) finds the repo root
without being told. An explicit `--repo-root` is validated the same way and *refused* if it doesn't
look like the data repo, rather than accepted and quietly misused.

This matters because the previous default was `"."`, while the instruction above is to run from
`pipeline/`. That combination resolved the root to `pipeline/`, where no `games/` exists — which
silently disabled the corpus-aware fetch skip *and* the `out_path.exists()` write-once check, and
aimed new game files at `pipeline/games/<season>/`, a second wrong corpus inside the repo. Nothing
errored; it just did the wrong thing. A run that cannot locate the root now exits 2 with a loud
message.

**Sequencing:**

1. Run the backfill escalation loop (fetch -> parse -> replay -> commit every discoverable newly-FINAL
   game, one season at a time — see "Backfill" above).
2. If that stopped on a detected challenge/WAF trip, **skip artifact regeneration entirely** and
   exit 1 — `games/**` reflects only a partial refresh at that point, and regenerating over
   incomplete state would silently mask the stop. That matters most for the person map, which would
   otherwise mint person ids from an incomplete roster picture. A resumed run picks back up exactly
   where the backfill half left off.
3. Otherwise, regenerate each artifact in memory and compare it (with `meta.generated_at` normalized
   on both sides) against whatever is currently committed under `artifacts/latest/`. If they compare
   equal (or nothing is committed yet and there is genuinely nothing to aggregate), this is a
   **NO-OP** — nothing is written, nothing is committed. If they differ, the fresh artifact is
   written and committed with the SAME commit mechanism used for game-file commits, under its own
   distinct commit message, separate from any game-file batch commit. The **person map** goes first
   (`"refresh: regenerate person map"`) because it is the identity layer every other reading of the
   corpus sits on; then the **team map** (`"refresh: regenerate team map"`); then the **career map**
   (`"refresh: regenerate career map"`, which sits on top of both); then the **frequency artifact**
   (`"refresh: regenerate frequency artifacts"`).
4. Report **`person_id` drift**. `person_id` lives in two places: the person-map artifact, which is
   authoritative and was just regenerated, and a materialized copy on every `players[].person_id`,
   which only a labeled `reparse(vX.Y.Z)` commit can refresh because `games/**` is write-once. A
   refresh that picked up new games leaves the two diverged. `run_refresh` does not resolve that (it
   has no license to rewrite game files) — it counts the committed player records whose stored
   `person_id` disagrees with the fresh map and prints the number. `0` means the corpus is in sync;
   anything else is the signal that a re-parse is due, and names the command. `career_id` drift is
   counted and reported separately, because the two move independently — adding a season links new
   careers without changing a single `person_id`.
5. Print a one-line summary (new games parsed, game-file commit count, frequency-artifact
   NO-OP-or-CHANGED) and exit 0 (or 1 if step 2 fired).

### Artifacts: career_map (`bc_pipeline.career_map`)

`person_id` is stable across every *game* of a season and deliberately no further — Presto reissues
every player id each year. `bc_pipeline.career_map` builds the layer above it: **`career_id`**, one
key per person across seasons, written to `artifacts/latest/career_map.json` and populated on
`players[].career_id` (schema 1.9.0).

**Exact display name is necessary but NOT sufficient, and the corpus proves it.** Two different
`Jack Lynch`es both played in 2024 — on the **same date**, for **different franchises**. One person
cannot do that. Three such pairs are proven, recomputed into `meta.evidence` on every build.

**Signals are chosen by measurement, against an exact null** (every consecutive-season person pair
with a *different* name — definitively not the same person):

| signal | fires on same-name pairs | fires on the null | likelihood ratio | used |
|---|---|---|---|---|
| franchise continuity | 51.7% | 7.1% | **7.33** | yes |
| position overlap | 96.1% | 44.7% | 2.15 | no |

Position overlap fires on nearly half of unrelated people, so it is close to a rubber stamp and is
not used — kept in the artifact as a measured negative. Franchise continuity separates the cases, and
independently refuses all three proven-different pairs.

**The rule:** two persons in consecutive seasons link iff they share a display-name spelling, that
name resolves to exactly one person in *each* season, and they share a franchise. Careers are the
connected components. On the current corpus: **173 links, 1,782 careers from 1,955 persons, 160
spanning more than one season.**

**Refused, and enumerated in `unlinked[]`:** `franchise_changed` (98 pairs — a player who changed
clubs stays unlinked; that is exactly the shape of every proven-different pair) and
`ambiguous_within_season` (55 pairs — inherited from `person_map` not merging across teams within a
season). Every person still gets a `career_id`; a singleton career is a complete answer, not a
missing value.

```bash
python -m bc_pipeline.career_map --input games/ --output artifacts/latest/career_map.json
python -m bc_pipeline.career_map --check-no-commit
```

### Artifacts: team_map (`bc_pipeline.team_map`)

**`team_id` cannot be joined on across seasons.** PrestoSports reissues it every year, and the corpus
proves it exhaustively: of the 12 teams appearing in more than one season, **zero** keep their
`team_id`, and no `team_id` is ever reused.

```
Yuba-Sutter Freebirds   2024 yypnc9frxm...   2025 0f7i5wcuhu...   2026 toa4e66upw...
Idaho Falls Chukars     2024 4hgc4se23g...   2025 ik8nryg1d3...   2026 gwwjqo5s6n...
```

`bc_pipeline.team_map` builds the key that survives: **`franchise_id`**, written to
`artifacts/latest/team_map.json` and populated on `teams.{home,away}.franchise_id` (schema 1.8.0).

**The key is the exact team NAME**, with two preconditions checked on every build (not assumed):
within a season name ↔ `team_id` is 1:1, and no team in this corpus has ever renamed. A violation
raises `AmbiguousTeamIdentity` and fails the build rather than degrading quietly.

**Roster continuity is not used, because it was measured and it does not work.** It is the obvious
second signal, so it was scored against the cases where the answer is known (a team in both seasons
under one name): same-name overlap runs only 15–37%, and on **3 of 21** checkable pairs the top
roster match is the *wrong* team — 2025's Colorado Springs Sky Sox best-matches Grand Junction
Jackalopes even though the Sky Sox exist that season under their own name. A signal that
misidentifies cases we can check is not trusted on cases we cannot. That number is recomputed every
run into `meta.not_attempted.roster_signal`, so the refusal stays evidence-backed.

Consequently the 2026 turnover — out: Colorado Springs Sky Sox, Grand Junction Jackalopes, Rocky
Mountain Vibes; in: Modesto Roadsters, RedPocket Mobiles, Long Beach Coast — is **not** resolved into
relocations. Those clubs stay separate franchises and the question stays visible in `continuity`.

Unlike `person_id`, `franchise_id` has **no artifact dependency and no drift**: it is a pure function
of the team name in each file, so `parse` computes it directly. `team_map.json` is a registry and an
evidence record, not an input to parsing.

```bash
python -m bc_pipeline.team_map --input games/ --output artifacts/latest/team_map.json
python -m bc_pipeline.team_map --check-no-commit
```

### Artifacts: person_map (`bc_pipeline.person_map`)

**`player_id` cannot be joined on across games.** It is file-local: a real 16-char Presto id is
stable for a season, but a synthetic `syn:<side>:<n>` is assigned by boxscore ROW ORDER, so the same
value denotes a different person in every file — `syn:away:8` is bound to 99 distinct display names
in 2026 alone. 10.0% of player records carry a synthetic id (29.5% of the 2026 season), so a naive
cross-game join silently mixes people together.

`bc_pipeline.person_map` builds the layer that fixes it: **`person_id`**, stable across every game of
a season, written to `artifacts/latest/person_map.json` (mutable, regenerable) and materialized onto
each `players[].person_id` (schema 1.7.0) at re-parse time. It reads `games/**` only.

**The key is `(season, team_id, name)`.** `team_id` is always a real Presto id, never synthetic, even
on team-site pages where no player row carries an id — which is what makes the grouping tractable.

**Rules** — a real `player_id` is its own `person_id`; only synthetics resolve through their group:

| classification | rule |
|---|---|
| `real_anchor` | group holds exactly one real id → that id is canonical |
| `minted` | group holds no real id → mint `person:<16 hex>` from the key |
| `multi_real_id` | two or more real ids → UNLINKED (which one is not determined) |
| `same_game_conflict` | two of the group's ids occur in one game → UNLINKED |
| `non_person_name` | the "name" is `/`, the `/ for X` source defect → UNLINKED |

Refusals are checked BEFORE the linking rules, so a defect is never merged away by an anchor that
happens to exist. On the current corpus: **4,177 of 4,189 synthetic records (99.7%) linked**, 12
unlinked, each with a reason in the artifact's `unlinked[]`. Never a guess — an unlinked player is a
measured negative.

**Scope is one season and one team**, stated in `meta.not_attempted` as measured numbers rather than
left implicit: 134 `(season, name)` pairs sit on more than one team (a mid-season move is not
separable from two same-named people), and cross-season linkage does not exist at all because
PrestoSports reissues every player id AND every team id each season (issue #41 Gap 1).

**CLI and the no-commit guard:**

```bash
python -m bc_pipeline.person_map --input games/ --output artifacts/latest/person_map.json
python -m bc_pipeline.person_map --check-no-commit
```

`--check-no-commit` behaves exactly like the frequency guard below: regenerate in memory, compare
with `generated_at` normalized on both sides, exit 0 + `NO-OP` or exit 2 + `CHANGED`, writing nothing.

**Ordering matters.** The artifact is derived FROM `games/**` and written back INTO it, so regenerate
it BEFORE a re-parse whenever the corpus has grown:

```bash
python -m bc_pipeline.person_map --input games/ --output artifacts/latest/person_map.json
python -m bc_pipeline.reparse --version X.Y.Z --write
```

That round trip is safe because the map is a function of fields a re-parse does not change (season,
team_id, name, player_id); regenerating it afterwards reports `NO-OP`. A re-parse with no artifact
present is allowed — it leaves every synthetic `person_id` null and says so, and the run summary's
`person_map_loaded` records which happened, so "no map" is never mistaken for "nobody linked".

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
