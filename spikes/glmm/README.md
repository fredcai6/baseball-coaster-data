# SPIKE 1: ridge-penalized multinomial logistic (additive GLMM baseline)

## What this is

A 10-category multinomial logistic regression of PA outcome on batter
identity, pitcher identity, season, home field, opposite-handedness, and
times-through-the-order, with an **L2 (ridge) penalty on the batter and
pitcher coefficient blocks only**. On the log-odds scale, an L2 penalty is
exactly a Gaussian prior on those effects — the penalized-likelihood form of
a mixed model with batter and pitcher random intercepts-by-category, crossed.
This is the linear predictor from Powers/Hastie/Tibshirani (2018) Sec 5.1,
minus their stadium term (the harness's rows carry `home_team` as an id but
no separate park/stadium effect was fit — see "what I'd do differently"):

```
eta_ik = alpha_k + beta_{batter_i,k} + gamma_{pitcher_i,k}
         + zeta_k * home_i + theta_k * opposite_hand_i + season_k
```

This model has **no batter-by-pitcher interaction term** — it is the honest
additive floor the NPMR and GLLVM sibling spikes need to beat.

## Implementation choice

Hand-rolled `scipy.optimize.minimize(method="L-BFGS-B")` with an analytic
gradient, not `sklearn.linear_model.LogisticRegression`. Reasons:

1. sklearn's multinomial solver applies one scalar `C` to *every*
   non-intercept coefficient. There's no clean way to tell it "penalize only
   the batter/pitcher blocks, leave season/home/handedness alone" short of
   rescaling columns, which the task brief itself flags as awkward.
2. **Reference-category parameterization.** Rather than fitting a free
   coefficient per category for all 10 outcomes (softmax's classic
   shift-invariance: adding the same constant to every category's logit for
   every row changes nothing), the model fixes the most frequent category
   (`F`, flyout, 24,259 training PA) at logit 0 and fits 9 free coefficient
   columns. This means the intercept and all "lightly penalized" terms
   (season, home, handedness, TTO) are left **fully unpenalized** — no
   near-singular directions, and no need to invent a penalty weight for them.
   Ridge is applied only to the two blocks that are genuinely
   high-dimensional and need it: batter (758 x 9) and pitcher (1181 x 9).

Design matrix: `scipy.sparse.csr_matrix`, ~1,948 columns for the full model
(intercept, 2 season dummies, home, opp-hand, unknown-hand, 3 TTO dummies,
758 batter dummies, 1181 pitcher dummies), each row having ~6-8 nonzeros.
Sparse matmuls make each L-BFGS iteration cheap even though the parameter
count (17,532 for the full model) looks large.

## Handedness

`bats`/`throws` missing on ~15% of PA (matches harness's own reported
14.6%/15.4% split-by-side figures). Handled as **an explicit unknown
indicator column**, not row-dropping or imputation: `opposite_hand` is held
at its reference value (0) when either side is unknown, and a separate
`unknown_handedness` dummy absorbs whatever average effect that group has.
Switch hitters (`bats == "S"`) are always coded as opposite-hand, per spec.

## Unseen players

Batter/pitcher vocabularies are built from the **training set only**. 14
batters and 39 pitchers appear in the test set but never in training; PAs
involving them get zero contribution from the player block (equivalent to
"average player," a soft version of the paper's "replacement level" pooling,
without a formal PA-count threshold — see "what I'd do differently").

## Lambda selection

5-fold `GroupKFold`-by-game cross-validation **within the training set**
(never touching the test games), grid `[1, 3, 10, 30, 100, 300]`, a single
shared lambda applied to both the batter and pitcher blocks, evaluated on
the full model (batter + pitcher + home + handedness + TTO):

| lambda | mean CV deviance |
|---|---|
| 1 | 4.00736 |
| 3 | 3.97208 |
| **10** | **3.95663** (chosen) |
| 30 | 3.96298 |
| 100 | 3.98011 |
| 300 | 3.99240 |

Chosen `lambda = 10`. The curve is fairly flat between 3 and 30 (all within
~0.007 deviance of each other) — the model is not highly sensitive to the
exact penalty within that range. **This lambda was tuned once on the full
model and reused, unchanged, for every step of the ablation ladder below**,
rather than re-tuned per submodel. That's a budget simplification, not a
correctness requirement — a batter-only model might technically prefer a
slightly different lambda than the batter block does inside the full model.
Given the flatness of the curve this is unlikely to change the qualitative
picture, but it means the individual ladder numbers for "+batter" and
"+pitcher" alone are very slightly pessimistic relative to their own
best-tuned lambda.

CV was also done with fold-local (not global) batter/pitcher vocabularies —
each of the 5 folds re-derives its own vocab from its 4/5 training slice, so
a handful of low-PA players are "unseen" in a given fold's validation split
even though they are in the overall training set. This is a conservative
approximation (slightly understates how well a model with the full training
vocab will do) and reused for simplicity across the grid.

## Deviance ladder (all on the frozen test split, 25,382 PA)

| step | deviance | delta vs null | % of full model's total reduction | runtime (s) | params |
|---|---|---|---|---|---|
| null (league frequencies) | 4.01172 | -- | -- | -- | 10 |
| intercept + season | 4.00900 | 0.00272 | 4.8% | 0.6 | 27 |
| + batter only | 3.98089 | 0.03083 | 54.8% | 2.1 | 6,849 |
| + pitcher only | 3.98539 | 0.02633 | 46.8% | 12.5 | 10,656 |
| + both (batter + pitcher) | 3.95786 | 0.05386 | 95.8% | 15.1 | 17,478 |
| + home + handedness | 3.95666 | 0.05506 | 97.9% | 14.0 | 17,505 |
| + TTO (full model) | **3.95550** | **0.05622** | **100%** | 24.7 | 17,532 |

**The number that matters most for the project:** of the total deviance
reduction this additive model finds (null 4.01172 -> full 3.95550, a 1.40%
relative reduction), **batter + pitcher identity alone accounts for 95.8% of
it**. Season, home field, handedness, and times-through-the-order — every
contextual main effect this model has room for — together add the remaining
4.2%. And since this whole model is additive by construction, the entire
0.05622 reduction (100%) is "main effects": there is no interaction term
here for the interaction spikes to be compared against except by omission.
**The batter+pitcher-only ladder step (3.95786) is the single most useful
number to hand the NPMR/GLLVM spikes**: it is what a matchup model has to
beat using something other than "sum of two independent player identities."

Total pipeline runtime: **759 seconds (~12.7 minutes)**, comfortably inside
the 20-30 minute budget, no subsampling needed (full 100,445-PA training
set, full 25,382-PA test set, on all 24 cores via BLAS's sparse-matmul
threading — L-BFGS-B itself is single-threaded but each function/gradient
eval is cheap thanks to sparsity).

## Sanity check

Correlated the fitted, ridge-shrunk K (strikeout) coefficient for each
player against their raw observed K% in the training data, restricted to
players with >= 20 training PA:

- batter K-coefficient vs observed K%: **r = 0.885** (n = 610 batters)
- pitcher K-coefficient vs observed K%: **r = 0.908** (n = 841 pitchers)

Both strongly positive, as expected — the model recovers something close to
raw rates for players with enough data, shrunk toward the mean by the ridge
penalty for players with less. This is the expected signature of a sane
ridge fit, not a broken one.

## Is any of this too good to be true?

No — if anything the opposite. A 1.40% relative deviance reduction over the
null is modest, consistent with PA outcomes being dominated by within-PA
noise that no amount of player-identity information resolves. This is the
same magnitude Powers et al. report for ridge on comparable data. If this
spike had come back with, say, a 30% reduction, that would be the signal to
go hunting for a leak (e.g., some form of season/game leakage across the
train/test game split, or an id join that lets a player's *test-set*
identity marker leak into their *training-set* rows). A game-level split
rules out the most obvious form of that (shared park/day/pitcher-quality
context within a game) but does not rule out, for instance, `career:` ids
that don't actually track a single person consistently across seasons — I
did not independently verify the career-id join logic beyond what
`common.py` already computes, and neither should the other two spikes, since
they all consume the same harness.

Two smaller things worth someone's attention but not alarming:
- The `+pitcher only` step is very slightly *worse* than `+batter only`
  (3.98539 vs 3.98089) despite having more parameters and more player
  coverage (1181 vs 758). This is plausible — batters likely have more
  stable, higher-signal category tendencies per PA than pitchers do given
  the categories used here — but it's the kind of thing worth cross-checking
  against the NPMR/GLLVM results, since if their pitcher-side signal is much
  stronger, that would say something about this fit's pitcher-block penalty
  or feature encoding rather than about pitchers.
- The individual `+batter` and `+pitcher` reductions (0.03083 + 0.02633 =
  0.05716) nearly equal the joint `+both` reduction (0.05386) — batter and
  pitcher main effects are close to additive/non-redundant, which is
  reassuring for an *additive* model (it would be more surprising if they
  overlapped a lot) but also means this spike cannot speak to whether a
  batter-pitcher *interaction* would find anything beyond what's already
  here — that's exactly the sibling spikes' job.

## What I'd do differently with more budget

- **Separate lambda for batter vs pitcher blocks.** I used one shared
  lambda for both; a 2D grid (or an alternating-block coordinate search)
  would let the pitcher block (more players, more parameters, seemingly
  weaker per-player signal) regularize more aggressively than the batter
  block without forcing them to match.
- **Replacement-level pooling** for below-threshold batters/pitchers (as in
  the paper), rather than the current zero-effect-for-unseen-test-players
  fallback. This mainly matters for rookies/late-callups who accumulate a
  handful of test PA with no training history.
- **Per-submodel lambda re-tuning** for the ablation ladder steps, since the
  ladder is currently a slight approximation (see "Lambda selection" above).
- **Global (not fold-local) batter/pitcher vocab** in the CV loop, so the CV
  score reflects the actual deployed vocabulary rather than a slightly
  smaller one.
- **A park/stadium term** distinct from the home/away indicator — the
  paper's `delta_S` — since `home_team` here doubles as a park id that the
  current model doesn't use as such.

## Addendum: follow-up diagnostics (batter-vs-pitcher asymmetry, handedness)

Three quick, cheap follow-up checks (not part of `fit.py`, run standalone,
results not persisted to `result.json`) to stress-test two findings above.

**Is "+batter beats +pitcher" a ridge-lambda artifact?** No. An independent,
non-GLM check — a Dirichlet-shrunk empirical-rate oracle,
`p_hat(player) = (counts + alpha*league_freq) / (n + alpha)`, no
optimization at all — ranks batter above pitcher at every shrinkage strength
tested (alpha = 5, 10, 20, 50, 100, 200; e.g. at alpha=200: batter-oracle
deviance 3.98186 vs pitcher-oracle 3.98931). Batters average more PA each
than pitchers in training (mean 132.5 / median 84 vs mean 85.1 / median 44),
so pitcher estimates are individually noisier, but the ranking survives
across the whole shrinkage grid, not just near the model's chosen lambda.
This looks like a real property of this corpus/taxonomy, not a fitting
artifact — though it does not distinguish genuine batter/pitcher skill
heterogeneity from role effects (starter/reliever platoon usage) this model
doesn't separate out.

**Is the small handedness effect being soaked up by player identity?**
Mostly no. Fitting home+handedness completely alone (no batter/pitcher
terms at all) yields test deviance 4.00692 vs an intercept+season baseline
of 4.00900 — a 0.00209 gain, barely bigger than the 0.00120 gain measured
after batter+pitcher are already in the model. The fitted `opp_hand`
log-odds coefficients are essentially unchanged (if anything slightly
larger) whether or not player fixed effects are present. So the shrinking
delta down the ladder looks like ordinary diminishing marginal returns as
deviance approaches its achievable floor, not confounding between
handedness and player identity.

**Is deviance under-counting a large true platoon effect?** No — the raw
platoon split in this corpus, measured directly with no model at all, is
K% 19.49% (same-hand) vs 17.86% (opposite-hand), and a BB+HBP+H "on-base
proxy" of 39.67% vs 41.33%. In baseball's own units that's about a 16-point
platoon gap on the on-base proxy and a 1.6-percentage-point gap on K% —
squarely in line with standard published platoon-split magnitudes (widely
cited as roughly 15-20 points of wOBA/OBP), not evidence of something
unusually large hiding in this data. A shift of that size, spread mostly
across the four categories (F, G, K, 1B) that already carry most of the
probability mass, produces exactly the small deviance delta observed — this
looks like deviance behaving correctly for an effect of this size, not a
blind spot. It would be a real blind spot specifically for *power* platoon
effects (HR/2B/3B), which are rare categories whose large *relative* shifts
contribute little to aggregate 10-category deviance; a category-specific
log-likelihood breakdown would be the right tool if that's the effect of
interest.

## Deliverables in this directory

- `fit.py` — runnable end to end (`./.venv/bin/python spikes/glmm/fit.py`), ~13 min.
- `result.json` — deviance ladder, chosen lambda, CV curve, sanity-check correlations, params/runtime.
- `effects.npz` — `batter_coef` (758 x 9), `pitcher_coef` (1181 x 9): the fitted, ridge-penalized coefficients for the 9 non-reference categories (reference category `F`'s coefficient is fixed at 0 and not stored). **This is the additive part downstream interaction analysis needs to residualize against.**
- `ids.json` — row order for `effects.npz` (`batter_ids`, `pitcher_ids`), and the 9-category column order (`nonref_category_order`) with the reference category noted.
- `residuals.npz` — per-test-PA predicted probability vectors (`probs`, 25,382 x 10, matching `common.CATEGORIES` order) plus `y`, `game_id`, `batter`, `pitcher` for later residual analysis without a refit.
- `run.log` — full stdout from the run that produced the above.

No files outside `spikes/glmm/` were touched; nothing was committed.
