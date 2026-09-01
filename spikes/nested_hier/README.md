# SPIKE: hierarchical latents on the sequential-GLLVM gate tree (Variant C)

One of three sequential-GLLVM spikes on the frozen gate tree in `spikes/gates.py`.
Two siblings fit the extremes of a single family: every gate gets its own
independent player latent space (max flexibility), or every player gets ONE
latent reused at every gate (max parsimony). This spike fits the model that
**contains both as limiting cases**:

    L^(g)[i] = L_shared[i] + delta^(g)[i]
    penalty  = lambda_shared * ||L_shared||^2 + lambda_gate * sum_g ||delta^(g)||^2

As `lambda_gate -> infinity`, `delta -> 0` and this collapses to the shared
sibling. As `lambda_gate -> 0`, `delta` is unpenalized and the gates decouple
into the separate sibling. **The CV-selected `lambda_gate` measures which
sibling's assumption the data actually supports.**

## Headline: the lambda_gate curve

Inner train/val split by game, 80k/20k train rows, `d=1` (CV-selected, see
below), `lambda_shared=10` (CV-selected) held fixed across the sweep:

| lambda_gate | val deviance | regime |
|---:|---:|---|
| 0.001 | 4.06488 | far below either sibling's own optimum -- an over-fit regime, see "Limits" below |
| 0.01  | 4.04413 | |
| 0.1   | 4.02433 | |
| 1     | 4.01875 | close to separate_only's own CV optimum (4.01921 @ lambda=0.5) |
| 5     | 4.01670 | |
| 20    | 4.00670 | |
| 80    | 3.98425 | |
| **300** | **3.97949** | **selected -- global minimum of the sweep** |
| 1,500 | 3.98105 | |
| 8,000 | 3.98132 | |
| 50,000 | 3.98106 | |
| 1,000,000 | 3.98091 | essentially indistinguishable from shared_only (3.98157) |

**Where the data lands: essentially the fully-shared end, with a sliver of
room for per-gate deviation that is smaller than restart noise.** The curve
is monotonically improving from `lambda_gate=0.001` all the way out to
`lambda_gate=300`, then goes flat -- there is no interior "sweet spot" of
meaningfully separate-but-shrunk per-gate structure. The selected optimum
(300) beats the shared-only reference on validation by 0.00208, and the
final canonical model beats it on test by 0.00045 -- **both smaller than the
0.00336 restart-to-restart spread of the hierarchical model itself.** Read
plainly: the extra machinery (per-gate deltas) bought, at best, a
within-noise nudge over just sharing one latent everywhere. This is a real,
useful, unflattering answer to give the other two siblings: **Variant B's
assumption (one shared latent) is not measurably wrong on this corpus, and
Variant A's assumption (fully separate latents) is actively wrong** (see
below -- it cannot beat the null model at all, at any regularization
strength we tried).

The fitted deltas quantify the same conclusion directly: at the selected
model, `delta^(g)` has RMS magnitude 30-43% of `L_shared`'s RMS magnitude
per gate (batters: root 43%, tto 40%, contact 30%, out 38%, hit 34%; pitchers
similar, 25-35%). The deviation is not literally zero, but it is a minority
correction on top of a dominant shared signal.

## Limit check -- did the sweep actually reach both ends?

**High end: yes, cleanly.** `hier @ lambda_gate=1e6` (val 3.98091) matches
`shared_only`'s own independently-CV'd fit (val 3.98157) to within
6.6e-4 -- well inside the restart-to-restart noise floor. As designed,
driving `lambda_gate` to a huge value is numerically trivial (it just kills
`delta`), and it does exactly what the algebra predicts.

**Low end: no, not cleanly -- and that is itself the finding.** A
standalone, honestly-tuned `separate_only` reference (L_shared frozen at
exactly 0, its own CV sweep over `lambda_gate` from 0.05 to 1000) has a
**U-shaped** curve with its OWN interior optimum at `lambda_gate=0.5`
(val 4.01921) -- it does not want zero shrinkage either. `hier` at its
closest tested point (`lambda_gate=1`, val 4.01875) actually lands almost
exactly on that optimum. But `hier`'s more extreme low-end points
(`lambda_gate=0.001, 0.01`) are dramatically worse (val 4.065, 4.044) than
`separate_only` ever gets even at ITS worst tested point (val 4.029 @
lambda=0.05). That gap is larger than plausible generalization noise. The
likely reason, flagged in `fit.py`'s module docstring before this was ever
run: `L_shared` and `delta^(g)` are **additively confounded** whenever
`lambda_gate` is tiny -- `(L_shared, delta)` and `(L_shared + c, delta - c)`
give an identical `L^(g)`, so with `lambda_gate` near 0 the block-1 QP is
nearly rank-deficient along that direction, on top of `lambda_shared=10`
still being active on `L_shared` alone. That makes block 1 harder to
converge well inside the fixed 4-round / 35-iterations-per-block CV cap,
so `hier`'s extreme low end is plausibly an **optimization artifact of
this specific confound**, not a clean read of what "fully separate, fully
unregularized" looks like on this corpus. **The sweep reaches the shared
limit cleanly and does NOT cleanly reach the separate limit** -- reported
as instructed, not smoothed over.

## The separate_only reference needed a second pass

The first `separate_only` CV grid (`{5, 20, 80, 300, 1000}`, matching the
scale that worked for the flat GLLVM sibling) was monotonically WORSE as
`lambda_gate` increased and never beat the null model even at its weakest
point (val 4.02691 @ lambda=5, vs null's ~4.01). A supplementary sweep
(`resweep_separate.py`, same inner split, wider/lower grid down to 0.05)
found the true optimum at `lambda_gate=0.5` -- still worse than null on
test (**4.01564 vs null 4.01172**). **Pure per-gate player latents, with no
sharing at all, cannot beat the null model on this corpus at any
regularization strength tried.** This is not a tuning failure of this
spike's CV; it is a property of the data given the gate-tree fragmentation:
splitting ~100k train PAs across 5 gates means every per-player-per-gate
subsample is much smaller than the flat model's per-player sample (a
player's ~130 median PAs become ~40 at the `tto` gate, ~25 at `hit`,
etc.), and an independent latent per gate cannot borrow strength across
that fragmentation the way a shared latent can.

## Joint test deviance vs the four sibling numbers

| model | test deviance |
|---|---:|
| null | 4.01172 |
| ridge/GLMM | 3.95550 |
| NPMR | 3.95424 |
| GLLVM (flat) | 3.95563 |
| structural-only (this spike, no player latent) | 4.00693 |
| **separate_only (Variant A stand-in, this spike)** | **4.01564** (worse than null) |
| **shared_only (Variant B stand-in, this spike)** | **3.98125** |
| **nested_hier (Variant C, this spike, canonical)** | **3.98080** |

The hierarchical model beats null (0.031, about 55% of the flat GLLVM's own
gain over null) but falls well short of all three flat models. The gate-tree
reparameterization, at least as fit here, is give up real accuracy relative
to fitting one flat 10-category latent factorization directly -- most
plausibly because `d=1` (CV-selected; see below) and per-gate `F` matrices
give the model much less capacity than the flat GLLVM's `d=5`, and because
the same per-gate fragmentation that sinks `separate_only` also taxes the
shared-plus-delta model's ability to use its latent efficiently. This spike
answers "where does the shared/separate needle point," not "does gate-tree
factorization beat the flat model" -- on the latter question the honest
answer is no, not with this budget's capacity choices.

`d` was chosen via a cheap proxy sweep (`shared_only` mode, `lambda_shared`
fixed at 30, `d` in {1,2,3,4}): val deviance was monotonically WORSE for
`d>1` (3.983, 3.985, 3.995, 4.015), so `d=1` won outright, not by a
sliver. That is consistent with the same fragmentation story: more latent
dimensions than the per-gate sample sizes can support just adds variance.

## Per-gate test deviance (canonical model)

| gate | deviance | struct-only floor (no player latent) |
|---|---:|---:|
| root | 1.25660 | 1.27637 |
| tto | 1.70124 | 1.71685 |
| contact | 1.75265 | 1.74666 |
| out | 1.37134 | 1.37790 |
| hit | 1.87208 | 1.88242 |

`contact` is the one gate where the fitted model is *worse* than doing
nothing with player identity -- consistent with the DIPS ladder below,
where both single-side latents also hurt at `contact`.

## DIPS check

Per gate: batter-latent-only, pitcher-latent-only, both -- same `d=1`,
same `lambda=10` (the selected `lambda_shared`) for a fair three-way
comparison, evaluated on test.

| gate | batter only | pitcher only | both | struct-only floor |
|---|---:|---:|---:|---:|
| root | 1.26644 | 1.26447 | **1.25652** | 1.27637 |
| tto | 1.70823 | 1.70858 | 1.70837 | 1.71685 |
| contact | 1.75169 | 1.76034 | 1.76681 | 1.74666 |
| out | 1.38117 | 1.38147 | 1.39397 | 1.37790 |
| hit | **1.86755** | 1.92876 | 1.91879 | 1.88242 |

**Mixed result relative to McCracken's DIPS, reported either way as
instructed:**

- **The `hit` gate (1B/2B/3B/HR -- power/trajectory) matches DIPS well.**
  Batter identity clearly helps (1.86755, below the no-latent floor
  1.88242); pitcher identity clearly HURTS (1.92876, worse than doing
  nothing) -- exactly DIPS's core claim that pitchers have little control
  over batted-ball outcome once contact is made, to the point that a
  pitcher latent here is fitting noise, not signal.
- **The `tto` gate (K/BB/HBP) does NOT show the pitcher-dominance DIPS
  predicts.** Batter-only (1.70823) and pitcher-only (1.70858) are
  statistically indistinguishable at this `d`/`lambda`, and combining them
  buys nothing (1.70837). This is the one place this spike's measurement
  contradicts the textbook expectation. Plausible reasons: `d=1` and
  `lambda=10` were selected for the OVERALL hierarchical fit, not tuned
  per-gate for this diagnostic, and may be too coarse to resolve a real
  but modest pitcher-side edge at this specific gate; or short-season
  Pioneer League pitchers may simply have less separated K/BB skill than
  affiliated-ball starters. Reported as measured, not adjusted to match
  the prior.
- **`out` and `contact` gates: both single-side latents, and their
  combination, are worse than the no-latent floor.** These are the two
  gates hit hardest by the fragmentation story above -- large, coarse
  branches (2-3 outcomes) where structural covariates (home/hand/season)
  already do most of the work and a weakly-regularized player latent adds
  variance instead of signal at this budget's settings.

## Restart sensitivity

3 restarts of the final `hier` fit (`d=1, lambda_shared=10,
lambda_gate=300`, 8 alternation rounds, 100 L-BFGS-B iterations/block):
test deviance **3.98080 / 3.97934 / 3.97744, spread 0.00336**. The flat
GLLVM sibling's 5 restarts spread only 0.00003 -- **this model is about
100x more restart-sensitive**, exactly as warned in the task brief. The
canonical restart is chosen by best TRAIN penalized loss (3.98080, seed 0),
which is actually the WORST of the three on test (best test restart, seed
2, scores 3.97744) -- a reminder that restart noise here is large enough to
matter for which number gets reported, and that "best on test" is not
"selected."

## Fitting method

Block coordinate descent, cribbed from `spikes/gllvm/fit.py`'s documented
lesson (naive joint L-BFGS-B over everything collapses `L, F` to ~0). Block
1 solves `L_shared` and all 5 gates' `delta^(g)` jointly in one convex
problem (given `F` fixed, the softmax NLL plus quadratic penalties is
convex in the L-side parameters). Block 2 solves each gate's `F`, `alpha`,
`beta` independently given `L` fixed (also convex; gates don't interact
once `L` is fixed). A tiny fixed ridge on `F` (`1e-3`, never swept) is
required for scale identifiability: with an L2 penalty on `L` alone, the
bilinear rescaling `(L*c, F/c)` sends the L-penalty to 0 as `c -> 0` while
leaving the fit unchanged, so `F` would otherwise be free to diverge; even
a tiny counter-penalty on `F` pins down a finite scale via the standard
AM-GM argument. This parameterization is MORE prone to the GLLVM sibling's
zero-collapse failure mode than the flat model was, because `L_shared` and
`delta^(g)` are additively confounded whenever `lambda_gate` is small (see
"Limit check" above) -- exactly the pathology the low end of the sweep ran
into.

`shared_only` (delta frozen at 0) and `separate_only` (L_shared frozen at
0) reuse the identical block machinery with one side's blocks skipped, so
they are genuinely fair reference points on the same code path, not
reimplementations.

## Compute budget

Full pipeline (`fit.py`) ran in **396 seconds** (~6.6 minutes), well inside
the ~25 minute budget:
- `d` selection (4 `shared_only` fits): 28s
- `lambda_shared` selection (3 `shared_only` fits): 12s
- `separate_only` first-pass CV (5 fits, later superseded): 22s
- **`lambda_gate` sweep, 12 points (the headline)**: 178s
- final `hier` fit, 3 restarts (8 rounds x 100 iter/block): 114s
- final `shared_only` / `separate_only` refits: 26s
- DIPS ladder, 15 single-gate fits: 17s

`resweep_separate.py` (supplementary, run after inspecting the first pass's
suspicious monotonic-worsening trend) added another **46 seconds**
(10-point CV grid + one final refit + one test evaluation), touching test
exactly once more for exactly one additional model (the corrected
`separate_only` reference) -- not used to pick anything about the
headline `hier` model, which was already finalized.

## What was NOT done / approximated

- **`d` and `lambda_shared` were selected via the cheap `shared_only`
  proxy**, not via a full 2-D CV grid over `(d, lambda_shared, lambda_gate)`
  jointly -- a 3-D grid at this fit cost would have blown the time budget.
  Given `d=1` won by a wide, monotonic margin in the proxy sweep, a joint
  search is unlikely to have picked a different `d`, but this was not
  verified directly.
- **DIPS ladder fits use the SAME `(d, lambda)` as the main model**, not
  their own per-gate-tuned CV -- deliberate, so the three-way (batter/
  pitcher/both) comparison at each gate is apples-to-apples, but it means
  the ladder answers "does this side help at the settings that work best
  for the joint model," not "what is the best possible per-gate DIPS
  read."
- **`separate_only`'s corrected grid (0.05-1000) still may not have found
  its true global optimum** if it lies below 0.05 -- the U-shape's minimum
  at 0.5 with clear degradation on both sides (0.05: 4.029, 1000: 4.067)
  makes this unlikely to matter much, but was not chased further.
- **CV is a single 80/20 game-level split, not k-fold** -- consistent with
  every other sibling spike's budget tradeoff, not unique to this one.

## Files

- `fit.py` -- runnable end to end (`./.venv/bin/python spikes/nested_hier/fit.py`).
- `resweep_separate.py` -- supplementary patch script; re-runs only the
  `separate_only` reference on a wider/lower grid and patches
  `result.json`'s `sibling_reference.separate_only` and `limit_check`
  blocks. Does not touch the `hier` model's own numbers.
- `result.json` -- required schema (`model`, `joint_test_deviance`,
  `null_deviance`, `per_gate_deviance`, `d`, `lambda_shared`, `lambda_gate`,
  `lambda_gate_curve`, `n_params`, `runtime_sec`, `restart_spread`) plus
  `sibling_reference` (the `shared_only`/`separate_only` stand-ins),
  `limit_check`, `dips_ladder`, `d_selection`, `lambda_shared_selection`,
  `restart_test_deviances`.
- `latent.npz` -- `Lshared_bat`, `Lshared_pit` (n_bat/n_pit x d), one
  `delta_bat_<gate>` / `delta_pit_<gate>` pair per gate, one
  `Fbat_<gate>` / `Fpit_<gate>` / `alpha_<gate>` / `beta_<gate>` per gate,
  `bat_ids`, `pit_ids`, `gate_order`, `d`.
- `run.log` / `resweep.log` -- actual stdout from both runs.
