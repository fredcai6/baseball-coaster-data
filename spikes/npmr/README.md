# SPIKE 2/3: Nuclear Penalized Multinomial Regression (NPMR)

Powers, Hastie & Tibshirani (2018), `docs/reference/npmr-powers-hastie-tibshirani-2018.md`.
Fit by accelerated/block-coordinate proximal gradient descent with singular-value
soft-thresholding, on the shared harness (`spikes/common.py`).

## Headline result

| | deviance |
|---|---|
| null | 4.01172 |
| **NPMR (CV-selected, rank 8/8)** | **3.95424** |
| ridge/GLMM sibling (`spikes/glmm`) | 3.95550 |

NPMR beats the null by 0.0575 (1.4% relative) and edges out the ridge/GLMM
sibling by 0.00125 (0.03% relative) -- a small win, in the direction and
magnitude the paper itself reports ("NPMR outperforms ridge regression
though the difference is not statistically significant").

**Rank: CV picked 8, not 3.** This is the one place this spike disagrees
with the paper's headline number, and it is a real finding, not a bug --
see "Rank vs interpretability" below.

## Design

Linear predictor per paper Sec 5.1, `eta_ik = alpha_k + beta_{B_i,k} +
gamma_{P_i,k} + zeta_k*H_i + theta_k*O_i`, adapted for this corpus:

- **Batter block** and **pitcher block**: one-hot, separately nuclear-penalized
  (own SVD/soft-threshold each). The paper does this too ("we add penalties on
  the nuclear norms of the three coefficient sub-matrices corresponding to
  batters, pitchers and stadiums... The result is that NPMR learns different
  latent variables for batters than it does for pitchers"). We tie
  `lambda_bat = lambda_pitch` for CV tractability (an 8-point 1-D grid instead
  of an 8x8 2-D one) -- the structural decision (separate blocks, separate
  SVDs, separate ranks) is what the paper actually asks for; sharing the
  penalty *strength* is our simplification, and it still let batter and
  pitcher ranks differ in practice (e.g. rank 6/5 at lambda=80).
- **No stadium block.** This corpus is 35 `home_team` values across 3 seasons
  of short-season minor-league ball, not 30 MLB stadiums with a full season
  each; the task brief's predictor list for this spike names batter, pitcher,
  home, opposite-handedness, and season -- not stadium -- so it was left out
  rather than added speculatively.
- **Season** (2024/2025/2026): two dummy columns, baseline 2024. Not in the
  paper (single-season data); added here since our corpus spans three
  seasons and the brief asked for it.
- **Opposite-handedness**: 3-level categorical (same / opposite / unknown)
  rather than the paper's binary indicator, because ~15% of PAs are missing
  bats or throws. Switch hitters (`bats == 'S'`) are coded as "opposite"
  (the standard sabermetric convention -- they choose the box to keep the
  platoon edge).
- **Intercept, home, handedness, season**: one *unpenalized* dense block
  (`Theta_other`), fit with a plain (unpenalized) accelerated gradient step.
- **Replacement-level pooling**: batters/pitchers below a training-PA
  threshold collapse into one `__replacement_*__` identity, mirroring the
  paper's per-position pooling. We skip the per-*position* split (no reliable
  position field on the PA table) and just do one pooled bucket per side.
  CV compared threshold 0 (no pooling) vs 20: pooling won, but only barely
  (best CV deviance 3.95933 vs 3.96184, ~0.06% relative) -- our tail is
  thinner per player than the paper's MLB data (pitcher median training PA
  is 44, with 340/1181 pitchers under 20 PA), so pooling helps a little, not
  a lot. The `__replacement_*__` bucket also absorbs any batter/pitcher seen
  in test/CV-fold but never seen in that fold's training data.

## Fitting algorithm -- and a deviation from the paper, both documented in `fit.py`

The paper's Sec 3.2 takes one *simultaneous* joint step across `(alpha, B)`
with a single shared step size. Our design has three blocks (batter,
pitcher, other) with very different natural curvature: the dense "other"
block's intercept column is touched by every one of ~100k rows; a single
player's block row is touched only by that player's PAs (often under 50).
A shared step size small enough to be stable for the dense block is far too
small to move individual player rows at a workable rate, and vice versa.
`fit_npmr` therefore **cycles Theta_other -> B_bat -> B_pitch each outer
iteration** (Gauss-Seidel), each block getting its own adaptively
backtracked step size and its own Nesterov momentum term (`t/(t+3)`, same
schedule as the paper). Each substep individually satisfies the standard
proximal-gradient majorization/backtracking condition, so it provably does
not increase the objective relative to its own extrapolated point.

**A sign bug was caught and fixed during development** (see the comment
above `_block_update` in `fit.py`): the first version of the backtracking
check used `+<grad, diff>` where `grad = X^T(Y-P)` is the *negative* of the
true gradient of the negative-log-likelihood. That flips the sign of the
majorization's cross term, making the bound get *looser* (not stricter) as
the step size grows, so it accepted arbitrarily large, diverging steps.
Confirmed by instrumentation: the buggy version passed backtracking on
essentially every iteration while the penalized objective grew to 1e37 over
300 iterations. Fixed by using the true gradient's sign; verified
afterward that the penalized objective decreases and plateaus normally.

## Lambda selection

3-fold CV by game (never touches test), 8-point log-ish grid x 2 pooling
thresholds = 16 fits at `max_iter=150`, then one final fit at the winning
`(lambda, pool)` with `max_iter=600` on full train. Full pipeline
(`fit.py`, `python -u`) ran in **406 seconds** (~7 minutes) -- comfortably
inside the 20-30 minute budget, so a further diagnostic sweep (below) was
run afterward without touching the CV-selected model.

CV winner: **lambda=40, pool_threshold=20**, cv_deviance=3.95933.

## Rank vs interpretability -- the honest finding

At the CV-optimal lambda, **8 of 10 possible singular values survive on
both the batter and pitcher blocks** -- not the paper's rank 3. A
supplementary sweep (full-train refit, evaluated on test; NOT used for
lambda selection, purely diagnostic) shows the accuracy cost of forcing
lower rank:

| lambda | rank (bat, pit) | test deviance |
|---|---|---|
| 40 (CV pick) | 8, 8 | 3.95424 |
| 80 | 6, 5 | **3.95012** (best observed, but CV, using only train, picked 40) |
| 120 | 5, 4 | 3.95862 |
| 140 | 4, 4 | 3.96430 |
| 160 | 4, 3 | 3.97025 |
| **180** | **3, 3** | **3.97653** |
| 200 | 3, 3 | 3.98257 |
| 240 | 1, 3 | 3.99352 |

Forcing rank 3 (lambda=180) still beats the null (4.01172) by 0.0352, about
61% of the CV-optimal model's total improvement over null -- so rank 3 is a
real, working model here too, just a strictly worse one on this corpus by
about 0.022 deviance (roughly 40% of the total gain over null forfeited for
a clean rank-3 story).

**Both rank solutions are provided**: `latent.md`/`latent.npz` are the
CV-optimal rank-8 solution (the deliverable spec's main target).
`latent_rank3.md` is the supplementary lambda=180 rank-3 fit, included
because it is dramatically more interpretable and lets us directly compare
against the paper's Table/Figure 6 story.

### Is it interpretable?

**At forced rank 3, yes -- close to the paper's story, especially for
pitchers:**

- Pitcher axis 1 (sv 2.74): K (+0.71), BB (+0.37) vs G (-0.40), 1B (-0.41) --
  "swing-and-miss stuff" vs "pitch-to-contact." Top: Matthew Taubensee
  (38.7% K), Zac Lampton (40.1% K). Bottom: Cole Calnon (9.9% K, 20.7% G).
  Matches the paper's pitcher Skill 1 "Power" closely.
- Pitcher axis 2 (sv 1.46): F (+0.46) vs G (-0.62), K (-0.49) -- fly-ball vs
  ground-ball pitchers. Matches the paper's pitcher Skill 2 "Trajectory."
- Pitcher axis 3 (sv 1.40): F (+0.68), K (+0.32) vs BB (-0.61), HBP (-0.14),
  1B (-0.13) -- good outcomes vs walks/hit-batters/singles. Top: Adam
  Christopher (2.8% BB). Bottom: Breyln Jones (32.2% BB). Matches the
  paper's pitcher Skill 3 "Command."
- Batter axis 1 (sv 2.79): K (+0.81), BB (+0.16) vs 1B (-0.37), G (-0.29),
  F (-0.24) -- textbook "Patience" (TTO vs BIP), matching the paper exactly.
- Batter axis 2 (sv 1.01): F (+0.69) vs G (-0.65) -- "Trajectory," matching
  the paper exactly.
- Batter axis 3 (sv 0.44): BB (+0.76) dominant, everything else small --
  this is where we **diverge** from the paper. Their third batter skill is
  "Speed" (1B vs G, fast players hit more singles/fewer groundouts); ours
  recovers a residual walk-rate axis instead. We do not have a clean "Speed"
  analog in this corpus at rank 3.

**At the CV-optimal rank 8, partially -- the top axes still read the same
way, but the story fragments.** Pitcher axis 2 (K vs bad outcomes, sv 9.74)
and axis 3 (F vs G, sv 8.04) are still clean Power/Trajectory axes; batter
axis 4 (BB vs K, sv 5.85) and axis 5 (F vs HR/G, sv 4.96) are still clean
Patience/Trajectory axes. But several additional axes on both sides
(batter axes 3, 6, 7, 8; pitcher axes 4-8) load heavily on **HBP** or on
**OTHER** (our catch-all for fielder's-choice/reached-on-error/sacrifice/
interference -- 4.2% of PAs, a category the paper's 9-outcome taxonomy does
not have) or on narrow 2B-vs-1B/HR splits. These read as real but
non-"tool"-like structure -- HBP-proneness, park/luck-flavored doubles-vs-
homers variance, whatever drives our OTHER bucket -- rather than
recognizable batting/pitching skills. **Full axis-by-axis loadings and the
top/bottom 10 players with observed rate stats for every retained axis, at
both ranks, are in `latent.md` and `latent_rank3.md`.**

### Bottom line

The paper's rank-3, three-named-tools story replicates well when we force
the same rank on our data, especially for pitchers (all 3) and 2 of 3
batter axes (patience, trajectory; speed does not replicate). It does
**not** emerge on its own from cross-validation on this corpus: CV prefers
8 dimensions per side, trading interpretability for about 40% more of the
achievable improvement over null. This is plausibly because (a) our extra
`OTHER` category and explicit HBP category give the model degrees of
freedom the paper's 9-category taxonomy didn't have to spend, and (b) a
three-season, short-season-minor-league corpus is a different generating
process than a single MLB season -- there is no guarantee the same
low-rank structure holds. Both are reported as findings, not explained
away.

## Files

- `fit.py` -- runnable end to end (`./.venv/bin/python spikes/npmr/fit.py`).
  Also exposes `build_player_index`, `encode_rows`, `fit_npmr`,
  `predict_proba`, `cv_score` for reuse/inspection.
- `result.json` -- required schema plus `cv_results` (full CV grid),
  `rank_ladder_diagnostic` (the rank-vs-accuracy table above),
  `rank3_forced` (pointer to the lambda=180 fit), `sibling_comparison`.
- `latent.npz` / `latent.md` -- **CV-optimal rank-8 solution** (main
  deliverable). `latent.npz` has `singular_values_batter/pitcher`,
  `V_batter/pitcher` (K x rank category loadings), `scores_batter/pitcher`
  (n x rank player scores, i.e. U*Sigma), `batter_ids`/`pitcher_ids` (row
  order, with `__replacement_batter__`/`__replacement_pitcher__` markers),
  `other_names` + `theta_other` (the unpenalized block's fitted
  coefficients).
- `latent_rank3.md` -- supplementary lambda=180 fit (same format), included
  for direct comparison to the paper's Table/Figure 6.
- `residuals.npz` -- `probs` (test_pa x 10 predicted probabilities), `y`
  (true category index), `game_id`, `batter`, `pitcher` per test row.
- `run.log` -- the actual run's stdout (unbuffered), including the CV grid
  and final-fit convergence trace.

## What was NOT done / approximated

- Lambda tied across the batter and pitcher blocks (see "Design" above) --
  a real simplification, not a bug; a 2-D grid was judged not worth the
  extra compute for a spike, especially since the two blocks' ranks still
  came out different (6 vs 5 at lambda=80) even with a shared penalty
  strength.
- No per-position replacement pooling (single pooled bucket per side
  instead) -- the PA table doesn't carry a reliable defensive-position
  field, and the paper's own reason for splitting by position (matching
  MLB's per-position replacement-level baseline) doesn't map cleanly onto
  a spike-scale analysis.
- Block-coordinate (Gauss-Seidel) fitting instead of the paper's exact
  simultaneous joint step -- documented above and in `fit.py`; chosen for
  numerical stability given the blocks' very different curvature scales,
  not for speed (the simultaneous version's failure mode, caught via the
  sign-bug instrumentation, was divergence, not slowness).
- CV `max_iter=150` vs final-fit `max_iter=600` -- CV fits converged (by
  the `tol` criterion) well before 150 iterations at every grid point that
  mattered, so this wasn't a binding constraint in practice, but it means
  the CV deviance numbers use slightly less-converged fits than the final
  reported test deviance.
