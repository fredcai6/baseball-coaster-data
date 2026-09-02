# Venue main effect + venue x stance deviation on shape D

Extends shape D (frozen-test deviance 3.94526, `spikes/pitch/step6_result.json`) with two
ridge-shrunk additive terms per node: a venue main effect (16 parks + a fixed-zero "unknown"
bucket, penalty `lam_ven`) and a venue x stance deviation (16 x 2 = 32 cells + fixed-zero
unknown, penalty `lam_vs`, nested on top of both the venue main effect and the existing
`hand_opposite` platoon column in `structural()`). `lam_bat`, `lam_pit`, `psi` are inherited
from shape D unchanged; only `(lam_ven, lam_vs)` are selected, per node, on the same inner
80/20 train split (seed 90210) shape D itself used.

Code: `venue_model.py` (the extended `nll_grad`/`fit`/`node_dev`, layered on `step1.ao_prob` --
`step1.py` itself is untouched), `venue_common.py` (venue/stance loading + indexing),
`fit_venue.py` (staged selection + final refit + frozen-test scoring), `leaderboard.py`
(item 5), `check_grad.py` (gradient verification). Raw per-node fits (including the full 772 +
1220 + 16 + 32 coefficient vectors) are in `result.json`; the leaderboard comparison is in
`leaderboard_result.json`; console output from both is in `run.log`.

## Gradient check

`check_grad.py`, 400-row subsample, 593-parameter theta, random nonzero lambdas. scipy
`check_grad` (forward differences) reported error 1.21e-5 at epsilon=1e-7 (relative to
||analytic grad|| = 40.25: **3.0e-7 relative error**). A per-component central-difference cross
check (epsilon=1e-6) matched the analytic gradient to 6.8e-8 max absolute component error. The
forward-difference error scales ~linearly with epsilon as expected for truncation error, not a
sign of a wrong gradient (confirmed by sweeping epsilon 1e-5 -> 1e-7 in the script's output).
Gradient trusted.

## 1. Frozen-test deviance

**3.93971** (single run) / **3.93976** (after widening `LAM_GRID`'s low end, see below) vs
**3.94526** for shape D alone: an improvement of **-0.0055**, comfortably beating the 3.94526
target and every other number in this spike series (NULL 4.01172, flat ridge 3.95550, NPMR
3.95424, nested_sep 3.94846).

Per-node delta (venue-augmented minus shape D):

| node | shape D dev | venue dev | delta |
|---|---|---|---|
| root | 1.24917 | 1.24874 | -0.00043 |
| tto_K | 0.44309 | 0.44246 | -0.00063 |
| tto_BB | 0.11595 | 0.11613 | **+0.00018** |
| con_HR | 0.26435 | 0.26132 | **-0.00303** |
| con_OTH | 0.30370 | 0.30364 | -0.00007 |
| con_OUT | 0.79000 | 0.78915 | -0.00086 |
| out_F | 0.48108 | 0.48078 | -0.00030 |
| hit_1B | 0.26272 | 0.26260 | -0.00012 |
| hit_2B | 0.03519 | 0.03494 | -0.00025 |
| **total** | **3.94526** | **3.93976** | **-0.00550** |

**Expectation check: mostly held.** `con_HR` is the single largest contributor by a wide margin
(55% of the total gain, -0.00303 of -0.00550) -- exactly as predicted, and consistent with the
design doc's §2a finding that home runs are the biggest, most reliable park effect. `hit_2B` and
`out_F` both improved as predicted, though modestly (-0.00025, -0.00030). `tto_BB` (walks) is
flat to slightly **worse** (+0.00018) -- consistent with the design doc's §2c finding that walks
have no real park effect and a walk-park adjustment fits noise; the tiny positive delta here is
that noise-fitting cost, not a real signal. The one part of the expectation that did NOT fully
hold: `root`, `tto_K` and `con_OUT` also picked up small but non-trivial gains (-0.00043,
-0.00063, -0.00086 respectively -- together over 30% of the total) that weren't called out in the
pre-registration. These three are also the three nodes whose `lam_ven` selection hit the grid's
low edge (see below), so at least part of this is the model exploiting a weakly-regularized,
flat-surface degree of freedom rather than a strong, well-identified park signal the way `con_HR`
is.

## 2. Per-node selected lambdas

| node | lam_ven | lam_vs | edge? |
|---|---|---|---|
| root | 1 | 3 | interior |
| tto_K | 0.1 | 1000 | **lam_ven at low edge** |
| tto_BB | 30 | 100 | interior |
| con_HR | 10 | 100 | interior |
| con_OTH | 100 | 10 | interior |
| con_OUT | 0.1 | 300 | **lam_ven at low edge** |
| out_F | 300 | 300 | interior |
| hit_1B | 300 | 1000 | interior |
| hit_2B | 0.1 | 1000 | **lam_ven at low edge** |

`lam_vs` never hit an edge at any node (selected values range 3-1000, comfortably inside the
grid `[0.1 .. 3000]`). `lam_ven` hit the **low** edge at three nodes -- `tto_K`, `con_OUT`,
`hit_2B` -- meaning the plateau tie-break (smallest-norm-within-`LAM_TOL`=1e-4 of the best
validation deviance) kept choosing less and less regularisation as the grid widened. The task's
suggested grid was `[1..3000]`; the initial run selected `lam_ven=1` (then the low edge) at
`root`, `tto_K`, `con_OUT` and `hit_2B`, so the grid was widened down to `[0.1, 0.3, 1 .. 3000]`
(mirroring step6_shapes.py's own defect-1 fix for exactly this failure mode) and rerun for all 9
nodes. `root` resolved to an interior point (`lam_ven=1`) once the edge moved; the other three
did not.

**Diagnostic (not part of the official grid, a direct probe): is this a real edge or a flat
surface?** Validation deviance for `lam_ven` in `{0.01, 0.03, 0.1, 0.3, 1, 3, 10}` at fixed
selected `lam_vs`, for all four nodes that ever touched an edge:

- `root`: 1.238309 -> 1.238162 across the whole range (5th significant figure).
- `tto_K`: 1.336511 -> 1.336571, U-shaped, minimum near 0.3 but flat to the 4th figure.
- `con_OUT`: 1.334757 -> 1.334907 (4th figure).
- `hit_2B`: 0.598249 -> 0.602325 (a 0.004 move over three orders of magnitude -- the largest of
  the four, still small).

This is the same genuinely-flat/weakly-identified likelihood surface step6_shapes.py documented
for `hit_2B`'s `lam_pit` (its docstring's "defect 1" discussion): no finite grid produces a
stable interior argmin on a surface this flat, and further widening would just chase the new
edge. **Self-disabled, not a failure**: at these three nodes (`tto_K` = K vs BB/HBP, `con_OUT` =
fly/ground vs everything else, `hit_2B` = double vs triple), the venue main effect is not
resolving a real, well-identified park signal -- the data cannot tell the optimizer how much to
shrink it, and the improvement it buys (see table above: -0.00063, -0.00086, -0.00025) is
correspondingly small and not to be over-trusted as a real park effect at those specific gates,
even though the frozen-test number does go down. `con_HR`, by contrast, landed cleanly interior
at `lam_ven=10` with the largest gain of any node -- that is the well-identified case.

## 3. Venue main effects, ranked

### con_HR (K/BB/HBP-vs-contact... no: contact-vs-HR gate; positive = more home runs)

`lam_ven=10`, `psi=10` (a very high fitted `psi`, i.e. a link that's steep/threshold-like for
this rare-event node -- inherited from shape D unchanged).

| rank | park | franchise | v |
|---|---|---|---|
| 1 | Raimondi Park | **Oakland Ballers** | **+0.748** |
| 2 | Suplizio Field | **Grand Junction Jackalopes** | **+0.738** |
| 3 | 4Rivers Equipment Park | Colorado Springs Sky Sox (2024 only) | +0.371 |
| 4 | Ogren Allegiance Park | Missoula PaddleHeads | +0.295 |
| 5 | Lindquist Field | Ogden Raptors | +0.268 |
| 6 | Bryant Field | Yuba-Sutter Freebirds | +0.171 |
| 7 | Melaleuca Field | Idaho Falls Chukars | +0.166 |
| 8 | Memorial Stadium | Boise Hawks | +0.045 |
| 9 | Blocktickets Park | Colorado Springs / Rocky Mountain (2025, shared) | -0.086 |
| 10 | Glacier Bank Park | Glacier Range Riders | -0.125 |
| 11 | Voyager Stadium | Great Falls Voyagers | -0.188 |
| 12 | Dehler Park | Billings Mustangs | -0.197 |
| 13 | Dobbins Stadium | Yuba-Sutter Freebirds (2024 only) | -0.439 |
| 14 | UCHealth Park | **Rocky Mountain Vibes (2024 only)** | **-0.484** |
| 15 | Blair Field | Long Beach Coast (2026 only) | -0.546 |
| 16 | Modern Woodmen Field | Modesto Roadsters (2026 only) | -0.565 |

**Lines up well with the design doc.** Oakland/Raimondi (#1) and Grand Junction/Suplizio (#2) are
the two strongest HR-inflating parks by a clear margin over #3, exactly matching the doc's
naming of Oakland as "the highest HR/100 PA" park and Grand Junction/Suplizio as the highest raw
run-value park. Rocky Mountain/UCHealth sits at #14 of 16 (third-lowest), matching the doc's
"Rocky Mountain Vibes ... lowest HR/100 PA, near-highest BABIP" characterization -- it is not
dead last, but it is decisively on the suppression side. The two lowest, Blair Field (Long Beach
Coast) and Modern Woodmen Field (Modesto Roadsters), are both single-season-only parks (2026);
their venue coefficient is partially confounded with the 2026 season effect per the pre-flagged
caveat, so their extreme ranking should be read with that discount in mind rather than as a pure
park property.

### root (K/BB/HBP-vs-contact gate; positive = MORE strikeout/walk/HBP, i.e. fewer balls in play)

`lam_ven=1` (interior after grid widening), `psi=1`. Every park is negative here (against a
league-average `alpha`), meaning the fitted per-park deviations all pull toward *more* contact
relative to the grand mean, with the season/structural intercept absorbing the rest -- read the
spread, not the sign.

Ranked highest (closest to average / least contact-favoring) to lowest (most contact-favoring):

| rank | park | franchise | v |
|---|---|---|---|
| 1 | Bryant Field | Yuba-Sutter Freebirds | -0.046 |
| 2 | Suplizio Field | Grand Junction Jackalopes | -0.094 |
| 3 | Dehler Park | Billings Mustangs | -0.101 |
| 4 | Raimondi Park | Oakland Ballers | -0.110 |
| ... | ... | ... | ... |
| 13 | Glacier Bank Park | Glacier Range Riders | -0.376 |
| 14 | Lindquist Field | Ogden Raptors | -0.419 |
| 15 | UCHealth Park | Rocky Mountain Vibes (2024) | -0.429 |
| 16 | Melaleuca Field | Idaho Falls Chukars | -0.475 |

This gate blends K + BB + HBP together, so it is not a clean read against the design doc's pure
K% table, but the strongest single point of contact is real: **Melaleuca Field / Idaho Falls
Chukars is the most contact-favoring park in the corpus at root** (v=-0.475, the extreme), and the
design doc independently measured Idaho Falls Chukars as having the **lowest strikeout rate**
(15.4%, the extreme low) of any park in §2a. Glacier Bank Park / Glacier Range Riders, the doc's
**highest** strikeout-rate park (22.7%), is 4th-most-negative here (rank 13 of 16) rather than
least-negative --
i.e. it doesn't come out as the *most* three-true-outcome park at root, likely because root also
carries BB and HBP (which the doc's §2c found have weak/no park stability), diluting the pure-K
signal. Given `root`'s `lam_ven` sits on a flat surface (§2), this ranking should be read as
directionally informative at the extremes (Idaho Falls) rather than reliable rank-by-rank.

## 4. Venue x stance deviations at con_HR

`lam_vs=100`, shrinking fairly hard against thin cells (per the ground rules, park x stance cells
range ~950-6,800 PA with 16-320 HR each). The largest surviving deviations after that shrinkage:

| park x stance | w |
|---|---|
| Raimondi Park \| L | **+0.082** |
| Bryant Field \| L | +0.054 |
| Voyager Stadium \| L | -0.055 |
| UCHealth Park \| L | -0.052 |
| Glacier Bank Park \| R | -0.042 |
| Modern Woodmen Field \| R | -0.032 |
| Blocktickets Park \| L | -0.037 |

Everything else is under 0.03 in magnitude. Read together with the con_HR main effects: Raimondi
Park (Oakland) is already the strongest HR-boosting park overall (+0.748) and shows an
*additional* boost specifically for left-handed stances (+0.082 on top) -- a real, if modest,
L/R asymmetry surviving shrinkage, consistent with a physical HR-favoring park with an
asymmetric fence. Voyager Stadium and UCHealth Park both show the opposite pattern: left-handed
stances are pulled *further down* than their (already negative or near-zero) main effect would
suggest. None of these are large relative to the main effects (the biggest, Raimondi|L at 0.082,
is about 11% of Raimondi's own +0.748 main effect) -- the venue x stance term is real but a
second-order correction on top of the main park effect, not a comparable-sized force, which is
the expected shape for a *nested* deviation term by construction.

## 5. Leaderboard movement (`leaderboard.py`, `leaderboard_result.json`)

Done within budget. Sanity check passed first: refitting shape D with no venue terms on ALL data
(train+test, matching `player_value.py`'s STEPS 2-3 convention exactly, applying the STORED
step-0 calibration rather than refitting it) reproduced `value_result.json`'s stored top-10
batter-season order exactly (`Adam Fogel 2024/2025`, `Kelly Dugan 2024`, ... all 10 names and
order match). The venue-augmented leaderboard was then built the same way, with venue held at
the **population mix**: since the venue and venue x stance terms are linear in `eta` (categorical
selectors, no interaction with the AO link), their population contribution is the PA-weighted
mean of `v[venue]` (and `w[venue,stance]`) over ALL PA rows, added once to each node's constant
term -- the same "average the linear covariate, not a probability" convention `populations.py`'s
Task 3 already established for `X_fixed`, extended to these two new terms rather than reinvented.

- **Rank correlation** (Spearman) between the original (no-venue) and venue-adjusted RAA/350
  across all 356 qualified batter-seasons: **rho = 0.937** (p = 3e-164). High, but well short of
  1 -- this is real reordering, not noise.
- **sd of the per-player-season change** in RAA/350: **3.58 runs**. About half the
  pre-registered 6.65 (the raw, uncontrolled within-player park-exposure sd) -- expected, since
  this number is the *model's regularized, nested* venue adjustment (shrunk per node, summed
  across 9 gates, further passed through the AO link and the existing calibration), not the raw
  park index itself; a shrunk model-based adjustment coming in at roughly half the raw estimate's
  spread is a reasonable, not alarming, discount.
- **Top movers up**: dominated by 2026-season players (Osiris Johnson +9.25, Cuba Bess +8.88,
  Emilio Corona +8.74, Jacob Jablonski +8.68, Justin Boyd +7.60, Jacob Lojewski +7.22, ...).
- **Top movers down**: also concentrated in specific players across multiple seasons -- Tyler
  Wyatt moves down in **all three** of his seasons (2024 -8.23, 2025 -8.53, 2026 -9.01), the
  single most consistent mover in the corpus; also Justin Trimble -8.18, Mason Minzey -8.07,
  Evan Scavotto -7.38, Wesley Mitchell -7.32, Sam Canton -7.23 (2026).
- **Caveat, stated rather than omitted**: 2026 players dominate both tails. Three of the venues
  (Modern Woodmen, Blair, and to a lesser extent the Blocktickets/UCHealth transition) are
  season-confounded per the pre-flagged caveat (6 of 16 parks are single-season), so some of this
  movement for 2026-specific players may be absorbing residual season drift the venue term picked
  up rather than a pure park effect, on top of the season fixed effect already in `structural()`.
  Tyler Wyatt's move is the strongest evidence *against* pure season-confound explaining
  everything, since it is consistent across three different seasons (and presumably three
  different park mixes) for the same player.

## What was not done / limits

- The joint (lam_bat, lam_pit) x (lam_ven, lam_vs) grid was NOT re-searched -- per the task spec,
  shape D's `lam_bat`/`lam_pit`/`psi` are inherited fixed. A joint re-search might find a
  different optimum but was explicitly out of scope.
- No re-derivation of the design doc's raw within-player park indices -- those are taken as given
  ("facts already measured") and compared against qualitatively, not recomputed.
- The three edge-hitting nodes' `lam_ven` selection is reported honestly as sitting on a flat
  surface rather than forced to an artificial interior point; no further grid widening was done
  once the flatness was confirmed by direct probe (widening again would just chase the new edge,
  per step6_shapes.py's own documented experience with this failure mode).
- Item 5 assumed "population mix" means holding the linear venue covariate at its population mean
  (the same convention `populations.py` Task 3 already uses for the other structural covariates),
  not a full average-over-venues probability marginalization (which would reintroduce the
  Jensen's-gap issue `populations.py` Task 2 measured and deliberately avoided for the other
  terms). This is a modeling choice, stated here rather than assumed silently.
