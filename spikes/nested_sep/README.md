# SPIKE VARIANT A: nested-GLLVM, a SEPARATE latent space per gate

The gate tree (frozen in `spikes/gates.py`) recasts the 10-category PA
outcome as a sequence of conditional choices. This variant gives EACH gate
its own independent `L_bat, F_bat, L_pit, F_pit` -- nothing (not d, not
lambda, not the latent coordinates) is shared across gates. It is the most
flexible, most parameter-hungry of the three sequential-GLLVM variants; its
job is to find out whether that flexibility pays for itself.

## Headline result

| model | test deviance |
|---|---|
| null | 4.01172 |
| ridge/GLMM | 3.95550 |
| NPMR (rank 8/9) | 3.95424 |
| GLLVM (flat, shared d=5) | 3.95563 |
| **nested-GLLVM, separate latents per gate** | **3.94846** |

The flexibility DOES pay off here: separate per-gate latents beat every flat
model, including the best one (NPMR), by 0.00578 -- roughly 4.5x the margin
by which NPMR beat ridge/GLMM. Restart spread per gate is tiny (1e-5 to
5e-5, see `result.json`'s `restart_spread`), so this isn't an artifact of a
lucky initialization; the block-coordinate-descent fits are landing in the
same place across 7 random restarts each.

`n_params` = 13,288 total across the five gates, versus ~9,855 for the flat
GLLVM at its selected d=5 (10 categories x 5 batter + 5 pitcher loadings
plus a much larger shared `L_bat`/`L_pit`). So variant A does spend more
parameters -- about 35% more -- and still generalizes better on the frozen
test split. That is a genuine result in the direction the flexibility
hypothesis predicts, not a coin flip.

**The identity check**: joint test deviance is not a separate computation --
it is the reach-weighted average of the five per-gate deviances
(`joint = sum_g (n_reach_g / n_test) * dev_g`), confirmed by hand to match
`gates.joint_deviance`'s output to full float precision. This is the same
factorization `gates.saturated_check` verifies at the raw-frequency level;
here it's verified again at the fitted-model level.

## Per-gate deviance, selected rank, and selected penalty

| gate | branches | K | test deviance | d* | lambda* | test reach |
|---|---|---|---|---|---|---|
| root | TTO / CONTACT | 2 | 1.24910 | 1 | 40 | 25,382 (100%) |
| tto | K / BB / HBP | 3 | 1.66565 | 2 | 25 | 8,523 (33.6%) |
| contact | OUT / HIT / OTHER | 3 | 1.74475 | 1 | 60 | 16,859 (66.4%) |
| out | F / G | 2 | 1.35257 | 1 | 40 | 9,012 (35.5%) |
| hit | 1B / 2B / 3B / HR | 4 | 1.86841 | 2 | 25 | 6,805 (26.8%) |

## Where does the rank go?

The flat GLLVM needed d=5 shared across all 10 categories; flat NPMR
retained rank 8 of a possible 9. Per gate here, selected rank is much
smaller: **d=1 for root, contact, out; d=2 for tto and hit; nothing needed
d=3** (hit's own CV grid tried d=3 and it was very slightly worse than d=2 --
1.84672 vs 1.84643 validation deviance -- not a rounding artifact but a real
"no further gain," so it was not selected).

Two-branch gates (root, out) were only ever tried at d=1, deliberately: for
a K=2 softmax, `L @ F.T` collapses to a single scalar per player regardless
of how many latent columns you give it (both L and F are free, so d=1
already spans every achievable per-player log-odds effect) -- so testing
d>1 there would only add redundant, unidentified parameters, not
expressiveness. This is stated in `fit.py`'s module docstring and enforced
by `D_GRID_BY_GATE` rather than discovered empirically at the cost of CV
time.

Interesting split: the three-outcome TTO gate (K/BB/HBP) and the four-way
HIT gate (1B/2B/3B/HR) both wanted d=2 -- i.e. more than one axis of player
variation matters for "which of several good/bad outcomes," but not for
"good or bad at all" (root, out) or "in play, safe or converted" (contact,
which barely moved with identity at all -- see the DIPS ladder below). Sum
of selected ranks across gates is 1+2+1+1+2 = 7, a bit more than the flat
model's shared d=5 but split across five much smaller per-gate softmaxes
(K=2..4) rather than one d=5 space shared across 10 categories -- the
gates are NOT asking for the same dimensionality, and the total "rank
budget" is not obviously comparable to the flat number since the objects
being factorized are different shapes (10x5 and 10x5 there; 2x1, 3x2,
3x1, 2x1, 4x2 here).

## The DIPS ladder (free validation)

For each gate: structural-only (home/handedness/season, no player identity)
-> +batter latent only (d_pit=0) -> +pitcher latent only (d_bat=0) -> both,
all at the gate's CV-selected (d*, lambda*) [reusing the selected lambda
for the single-sided ablations rather than re-running CV for each --
see Caveats].

| gate | structural | +batter only | +pitcher only | both | batter gain | pitcher gain |
|---|---|---|---|---|---|---|
| root (TTO vs CONTACT) | 1.27637 | 1.26450 | 1.26097 | 1.24910 | 0.01187 | 0.01540 |
| tto (K/BB/HBP) | 1.71685 | 1.68986 | 1.68984 | 1.66565 | 0.02699 | 0.02701 |
| contact (OUT/HIT/OTHER) | 1.74666 | 1.74545 | 1.74573 | 1.74475 | 0.00121 | 0.00094 |
| out (F vs G) | 1.37790 | 1.36802 | 1.36363 | 1.35257 | 0.00988 | 0.01427 |
| hit (1B/2B/3B/HR) | 1.88242 | 1.86302 | 1.88921 | 1.86841 | 0.01940 | **-0.00679** |

This reproduces McCracken's DIPS result cleanly, gate by gate, with no
prompting toward that answer built into the model:

- **contact (does a ball in play become a hit or an out) is where BOTH
  sides matter least** -- batter and pitcher identity together buy only
  ~0.0019 total off a 1.747 base, versus ~0.03-0.04 combined at the other
  gates. This is exactly the DIPS claim: what happens once the ball is in
  play is mostly not who's at the plate or on the mound.
- **out (fly ball vs ground ball) is pitcher-led** (0.0143 > 0.0099): batted
  ball trajectory tendency is real, measurable pitcher skill (GB/FB rate),
  consistent with the post-DIPS sabermetric refinement (McCracken's
  strongest claim was about *results* on contact, not *trajectory* of
  contact, and pitchers do have persistent GB/FB rates).
- **hit (which extra-base outcome, once it's a hit) is batter-led, and
  pitcher identity actively HURTS here** -- the pitcher-only ablation
  (1.88921) is worse than structural-only (1.88242). Power-type shape is
  close to a pure batter trait in this corpus; giving the pitcher a d=2
  latent to chase it just adds noise the ridge penalty didn't fully damp
  out at the CV-selected lambda. This is the sub-finding the brief asked
  not to paper over: pitcher flexibility does NOT pay off at this specific
  gate, even though it pays off in the joint model overall.
- **root and tto (whether the PA ends without a ball in play, and which of
  K/BB/HBP if so) are the two gates where batter and pitcher contribute
  roughly EQUALLY** (tto: 0.02699 vs 0.02701, essentially tied). Plate
  discipline/approach and stuff/command genuinely trade off here -- neither
  side dominates the three-true-outcomes gate the way one side dominates
  contact-quality or power.

## Fitting

Same recipe as the flat GLLVM sibling (`spikes/gllvm/fit.py`), applied
independently per gate: **block coordinate descent** (solve L given F via
ridge-penalized multinomial logistic regression -- convex in L alone --
then solve F, alpha, beta given L -- convex in that block alone -- and
alternate). We did not re-litigate the "joint L-BFGS collapses L,F to ~0"
failure mode documented there; it's a property of the bilinear objective,
not of any one gate's data, so we started directly from alternating descent.

Zero-dimension latent blocks (used for the DIPS single-sided ablations)
fall out of the same code for free: an `(n, 0) @ (K, 0).T` product is a
correctly-shaped all-zero contribution in numpy, with no special-casing in
`forward()`/`solve_L_given_F()`/`solve_F_given_L()`. Verified in isolation
before relying on it.

**Parallelism**: because nothing is shared across gates, the five gates'
likelihoods do not couple at all under this variant. Each was fit in its
own worker process (`ProcessPoolExecutor`, one process per gate) rather
than fit sequentially, exploiting exactly the structural fact this variant
is testing.

## Compute budget -- what was capped, explicitly

Total wall clock: **60.4 seconds** (see `runtime_sec` in `result.json`) on
this run, against a ~25-minute allowance. Two passes were run: a first
pass with a small grid (2 lambda values, capped rounds/iters) finished in
24s and already beat the flat model; given that much budget headroom, a
second, larger pass (reported here) widened the CV grid roughly 3x and
increased alternation rounds, per-block L-BFGS-B iterations, and restart
count to match or exceed the flat GLLVM's own settings -- and still
finished in under a minute. This spike is NOT compute-bound; if anything it
proves per-gate models are cheap once you stop asking one softmax to cover
10 categories at once. Settings actually used (`budget_notes` in
`result.json`):

- `LAMBDA_GRID = [8, 15, 25, 40, 60, 90]` (double the flat spike's 3-point
  grid), `D_GRID_BY_GATE` per the rank-ceiling argument above (root/out:
  `[1]`; tto/contact: `[1,2]`; hit: `[1,2,3]`).
- CV fits: `ALT_ROUNDS_CV=6, BLOCK_MAXITER_CV=70` (flat spike used 5/60).
- Final fits: `ALT_ROUNDS_FINAL=12, BLOCK_MAXITER_FINAL=180`,
  `N_RESTARTS_FINAL=7` (flat spike used 10/150/5) -- MORE restarts and
  MORE iterations per block than the reference implementation, at 5x less
  wall-clock, because each gate's softmax and row count are both smaller
  than the flat model's.
- Hyperparameter selection: single 80/20 train/validation split BY GAME
  within the training games (never touching test), same method as the flat
  GLLVM sibling, computed ONCE and shared across all five gates (the
  partition of train GAMES is identical across gates; which of a gate's
  rows fall in inner-train vs inner-val follows from which game they're
  in). Not k-fold, to stay directly comparable to the flat sibling's own
  choice, not because of a time constraint here.

### Caveats -- approximations that were made deliberately

- **DIPS ablations reuse the selected (d*, lambda*)** from the "both"
  model rather than re-running CV separately for batter-only and
  pitcher-only. The single-sided optimum lambda could differ slightly;
  this is a diagnostic decomposition, not a claim that batter-only or
  pitcher-only lambda is itself CV-optimal. Flagged in
  `result.json -> budget_notes.dips_ablations_reuse_selected_lambda`.
- **Rank grid for K=2 gates (root, out) was restricted to d=1 by
  construction**, not discovered by trying larger d and finding no gain --
  see "Where does the rank go?" above for the reparameterization argument.
- **lambda tied between batter and pitcher sides** within a gate (as in
  the flat GLLVM sibling), not searched as an independent 2-D grid --
  keeps the CV grid a tractable 1-D sweep per gate.
- No k-fold / bootstrap CI on the final test deviance; a single frozen
  test split is used, same as every other spike in this repo. The 0.00578
  margin over NPMR is well outside the ~1e-5 restart-to-restart
  optimization spread, but that speaks to optimization stability, not to
  sampling uncertainty in the test split itself.

## Files

- `fit.py` -- runnable end to end (`./.venv/bin/python spikes/nested_sep/fit.py`)
- `result.json` -- joint/per-gate deviances, selected d/lambda per gate,
  DIPS ladder, CV grids, n_params, runtime, restart spreads
- `latent.npz` -- per gate: `{gate}_Lbat, {gate}_Fbat, {gate}_Lpit,
  {gate}_Fpit, {gate}_alpha, {gate}_beta, {gate}_bat_ids, {gate}_pit_ids,
  {gate}_branch_names` for gate in {root, tto, contact, out, hit}
- `fit_run.log` -- full run log of the reported (large-grid) pass
