# Pre-game Monte Carlo game simulator

`spikes/sim/` builds a plate-appearance-level game simulator on top of
SHAPE D (`spikes/pitch/step6_shapes.py`'s winning 9-node binarised tree,
frozen-test deviance 3.94526) and validates it against the frozen TEST
games (`spikes/split.json`), with two pluggable advancement models.

## Files

- `shape_d.py` -- SHAPE D node topology (copied, not imported, from
  `step6_shapes.py`) + its per-node hyperparameters (`step6_result.json`,
  read-only).
- `data.py` -- full-fidelity PA loader. `common.load_pa()` is frozen and
  drops columns this simulator needs (lineups, team ids, base/out state);
  this re-reads the same `pa_table.csv` with those columns kept, and
  re-does `load_pa`'s profile join so the two loaders agree on
  bats/throws.
- `fit_shaped.py` -- refits SHAPE D on **TRAIN games only** at the frozen
  hyperparameters (no search), and derives the generic-reliever bullpen
  effect and the naive advancement model's four fixed probabilities, all
  from TRAIN. Writes `shaped_train.npz`. Runtime: ~2s (a single L-BFGS fit
  per node at fixed hyperparameters, not step6's hyperparameter grid).
- `build_empirical.py` -- builds the empirical advancement transition
  table from TRAIN games, reusing `spikes/value/run_expectancy.py`'s
  game-file walker. Writes `empirical_transitions.npz`.
- `simulator.py` -- the engine: `SimModel` (loads both `.npz` artifacts),
  `build_game_context`, `simulate_from_state` (the live-odds entry point),
  `simulate_game` (pregame convenience wrapper).
- `validate.py` -- runs both advancement models, N=2000 sims/game, on the
  frozen TEST games, and writes `result.json`.
- `run.log` -- concatenated stdout of `fit_shaped.py`, `build_empirical.py`,
  `validate.py`, in that order.

## Model

Each PA's 10-category probability vector comes from SHAPE D exactly as
`spikes/value/player_value.py` composes it: 9 independent binary-node
fits (`step1.ao_prob`, reused unmodified), each node's probability
`ao_prob(alpha + Xs@beta + b[batter] + q[pitcher], psi)`, multiplied along
each category's root-to-leaf path (`shape_d.PATHS_D`). The structural
covariate row (`spikes/fuse/analyze.py`'s `structural()`, reused
unmodified) is: home-park flag, opposite-hand flag, unknown-hand flag,
season dummies (2024 baseline). **No fitted SHAPE D parameters existed
anywhere in the repo** (step6_shapes.py fits and discards them per-shape);
this refit is fresh, on TRAIN only.

### Speed trick

Within one game, a PA's category distribution depends only on (which of 9
lineup slots is batting, which of exactly 2 pitchers -- the real starter
or the one generic reliever -- is pitching). Both are known before any
random draw. So per game we precompute a (9 slots x 2 roles x 10
categories) tensor **once** per side, and the Monte Carlo loop is pure
table lookup + categorical draw -- no per-PA link-function evaluation. This
is what makes N=2000 x 289 test games x 2 advancement models run in about
18 seconds total (see Wall-clock below), not a live-odds problem at all.

### Bullpen policy (stated)

Each side has exactly two pitchers: the real starter (his own fitted
per-node effect, his own throwing hand), who pitches until the number of
batters he has faced in the *simulated* game reaches his **actual**
`pitcher_bf` from the real game (a fixed, pregame-known threshold, not a
stochastic manager decision) -- then a single **generic reliever** for the
rest of the game. The generic reliever's per-node effect is the
PA-weighted mean effect over that team's TRAIN relief appearances
(`pitcher_is_starter == False`); its throwing hand is **not** collapsed to
a single letter but treated as a PA-weighted mixture over L/R/unknown
relievers, so the handedness covariate for the generic reliever is the
correct population expectation rather than an arbitrary mode. No
pinch-hitters, no further mid-game bullpen changes, no in-game lineup
changes beyond the one pitching change.

### Advancement models

**naive** (`simulator.advance_naive`): deterministic rules, vectorized.
- K, F: one out, runners hold.
- G: one out, runners hold, **except** with runner on 1st and <2 outs, a
  fixed double-play probability (TRAIN-estimated **0.5625**, n=3248) makes
  it 2 outs and removes the runner from 1st. This rate is higher than MLB
  intuition suggests because the `G` category here already excludes
  fielder's-choice-only plays (batter safe) -- those are `OTHER` in this
  taxonomy -- so what's left in `G` is disproportionately outs where the
  defense had a real double-play look.
- BB, HBP: force advance only (standard cascading force from the batter
  up; a run scores only with the bases loaded).
- 1B: batter to 1st, runner on 1st always to 2nd, runner on 3rd always
  scores, runner on 2nd scores with a fixed TRAIN-estimated probability
  (**0.6129**, n=3299, conditioned on 2nd occupied and 3rd empty to
  isolate the question) else advances to 3rd.
- 2B: runners +2 bases (1st->3rd, 2nd/3rd score).
- 3B, HR: everyone scores; HR includes the batter.
- **OTHER** (this taxonomy's catch-all for fielder's-choice, reached-on-
  error, sacrifice, and interference -- see `common.OUTCOME_MAP`): the task
  brief asked us to check how this should advance runners rather than
  applying the blanket "one out, runners hold" rule to a bucket that
  includes plays where the batter is safe. We estimated, from TRAIN: (a)
  P(batter reaches safely) = **0.4389** (n=4288, via `outs_recorded==0`)
  -- when true, advancement is applied exactly like a 1B (batter to 1st,
  forces/2nd-runner-scores-prob as above); when false, the batter is out
  and (b) if a runner is on 3rd with <2 outs, that runner scores with a
  second fixed probability, **0.6881** (n=420) -- our stated treatment of
  the sac-fly case this bucket conflates in. All other runners hold in the
  batter-out branch. This is still an approximation (it can't distinguish
  a real sac fly from a fielder's-choice-with-out-elsewhere), stated as
  such.

**empirical** (`simulator.advance_empirical`): a lookup table,
`P(bases_after, outs_after, runs | bases_before, outs_before, category)`,
built from TRAIN. "After" state is the **next** plate-appearance event's
`bases_before`/`outs_before` in the same clean half-inning (or "END" if
none), skipping over any intervening `runner_event`s (steals / wild
pitches / pickoffs) -- so those between-PA events are silently folded into
the observed transition, exactly as the task brief anticipates ("this is
what the `advance` agent is measuring separately; you just need it to
work"). `runs` for a transition is the sum of `runs_on_play` over the PA
event and any runner_events strictly before the next PA (not just the
PA's own runs_on_play), so a run scored on, e.g., a wild pitch right after
a walk is correctly attributed rather than dropped. Cells with fewer than
**30** observations (34 of 240 `bases x outs x category` cells) back off
**wholesale** to that category's outcome-only marginal (pooled over all
bases/outs states), per the task brief's instruction -- this can rarely
produce a physically odd after-state for a thin cell (e.g. a runner
"appearing" from an empty base), not corrected further. 206/240 cells had
enough TRAIN data to stand on their own. Built by walking
`run_expectancy.py`'s clean half-innings only (0 dirty half-innings hit
in the 1173 usable TRAIN games; 20,240 clean half-innings, 99,316 PAs).

## What was NOT done / known limitations

- **Mercy-rule / early-stop mid-inning is not modeled.** The simulator
  replicates a game's real *length in innings* (from the max `inning` in
  that game's `pa_table` rows -- 19 of 289 validated test games are 6 or
  7 innings, not 9), but does not stop early if a large lead develops
  mid-inning the way some short-season leagues' mercy rules would. This
  is a source of the run-total bias for high-scoring blowout games.
- No pinch-hitters, no double-switches, no more than one pitching change
  per side, no intentional-walk strategy, no steals/wild pitches/pickoffs
  as separate mechanisms (the empirical model absorbs their net effect on
  base/out state; the naive model ignores them entirely).
- 5 of 294 non-disposed test games were skipped (not simulated) because
  one side's actual batting order didn't cover all 9 slots in
  `pa_table` (short/incomplete game data) -- excluded and counted, not
  silently dropped.
- SHAPE D's own per-node hyperparameters (`lam_bat`, `lam_pit`, `psi`)
  were **not** re-searched here, per the task brief -- they're taken as
  frozen from `step6_result.json`, refit only on TRAIN instead of
  TRAIN+TEST.
- 12 of the 17 disposed games are inside the TRAIN split and were left in
  the SHAPE D refit and the naive-probability estimation (matching how
  `step1.py`/`step6_shapes.py` themselves treat the split -- they don't
  filter disposed games either); the empirical transition table, the
  win-probability baselines, and all TEST-side validation *do* exclude
  disposed games (4 of 298 test games), since those depend on
  half-inning-chain integrity or ground-truth final scores.

## Validation results (TEST games, N=2000 sims/game)

294 of 298 test games are not disposed; 289 of those have complete
9-batter lineups for both sides and a readable linescore and form the
validation set. 12 of 289 (4.2%) are tied after regulation and are
excluded from win-probability scoring (included in run-total scoring).

### Run totals (578 team-games = 289 games x 2 sides)

| model | bias (pred-actual) | MAE | coverage 50% (nominal .50) | coverage 90% (nominal .90) |
|---|---|---|---|---|
| naive | **-1.362** | 3.759 | 0.540 | 0.898 |
| empirical | **-0.287** | 3.738 | 0.503 | 0.889 |

Both models under-predict mean runs somewhat (more so naive); the
empirical advancement model's mean bias is about 5x smaller. Interval
coverage is close to nominal for both -- the simulated distribution's
*shape* is reasonably well calibrated even where naive's *mean* is off.

### Win probability (277 non-tied games)

| | Brier | log-loss |
|---|---|---|
| **naive sim** | 0.2316 | 0.6555 |
| **empirical sim** | 0.2303 | 0.6526 |
| baseline: constant (TRAIN home-win rate = 0.5245) | 0.2491 | 0.6913 |
| baseline: Pythagorean/log5 (per-team-season TRAIN RS/RA, k=2) | 0.2388 | 0.6698 |

**Both advancement models beat both baselines on win probability.**
Naive beats the constant baseline by 0.0175 Brier / 0.0358 log-loss, and
the Pythagorean baseline by 0.0072 Brier / 0.0143 log-loss. Empirical is
marginally better still (0.0013 Brier / 0.0029 log-loss better than
naive). The gap between the simulator and the baselines is real but
modest -- this is short-season Pioneer League ball between fairly evenly
matched teams (predicted P(home win) never left roughly [0.25, 0.78] in
either model's calibration table), so no method separates the sides by
much.

Calibration (10 bins, `result.json`'s `*.win_prob.sim.calibration`): both
models track the diagonal reasonably through the populated middle bins
(0.3-0.7 predicted, where nearly all the mass sits); the tails are too
data-thin (0-4 games per outer bin) to say much.

### Naive vs. empirical gap

Empirical advancement changes the answer mainly on **mean run-total
calibration** (bias -1.362 -> -0.287, a ~1.1-run/team-game correction)
and only marginally on win probability (Brier -0.0012, log-loss -0.0028,
both favoring empirical) and on MAE (-0.021, favoring empirical). At the
**game-outcome (win/loss) level, advancement modelling barely matters**
here -- both models pick the same favorite in the overwhelming majority
of games, since bullpen/lineup skill differences dominate. At the
**run-total level it matters quite a bit** -- naive's "runners hold on
almost everything" rule systematically leaves runners stranded that the
empirical model's real base-advancement rates (including all the
between-PA events it silently absorbs) correctly cash in.

### Wall-clock

Naive: 10.4s / 289 games / 2000 sims = **36 ms/game** (18 microseconds/sim).
Empirical: 7.8s / 289 games / 2000 sims = **27 ms/game** (14
microseconds/sim). Context-building (lineups, starters, the 9x2x10
probability tensor) for all 289 games took 0.27s total, negligible. This
easily supports a live-odds use case calling `simulate_from_state` at
in-game decision points -- even 2000 sims of a partial game finishes in
tens of milliseconds.
