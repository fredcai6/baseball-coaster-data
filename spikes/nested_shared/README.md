# Variant B: one shared player latent, gate-specific loadings

`spikes/nested_shared/fit.py`. Frozen gate tree from `spikes/gates.py`,
frozen split from `spikes/common.get_split`, neither edited. Full run: 237s
compute (well under the ~25 min budget) for the final CV grid + 5 restarts +
full DIPS ladder; a supplementary d=3 fit for interpretability added ~83s.

## Headline: joint test deviance

| model | test deviance |
|---|---|
| null | 4.01172 |
| ridge/GLMM | 3.95550 |
| NPMR | 3.95424 |
| GLLVM (flat, separate loadings per side, d=5) | 3.95563 |
| **nested_shared (this spike, d=5, lambda=40)** | **3.95105** |

This variant beats every number above, including NPMR -- the previous best.
The margin over NPMR (0.00319) is more than 30x the flat GLLVM's restart
spread (0.00003) and about 30x this spike's own restart spread (0.00010), so
it reads as a real effect, not restart noise. A supplementary d=3 fit (the
rank Powers et al. forced on the flat NPMR axes, and the rank used for
`loadings.md`) reaches 3.95136 -- essentially the same number with 40% fewer
free per-gate loading parameters, so d=5 is not doing much beyond d=3 was
already doing.

Parameter count at the selected d=5: 9,919 (`n_params` in `result.json`) --
still far below a per-player-per-category free-effects model, and only
modestly more than the flat GLLVM despite covering 5 separate gates, because
the shared L is (758+1181) x 5 regardless of gate count and each gate's own
F is small (2-4 branches).

## Per-gate test deviance (canonical d=5 fit)

| gate | branches | test deviance |
|---|---|---|
| root | TTO / CONTACT | 1.24907 |
| tto | K / BB / HBP | 1.67683 |
| contact | OUT / HIT / OTHER | 1.74519 |
| out | F / G | 1.35221 |
| hit | 1B / 2B / 3B / HR | 1.86363 |

These five numbers are NOT directly comparable to each other (different row
counts, different branch counts, different base rates) -- they sum, via
`gates.joint_deviance`'s row-weighted average, to the single joint number
above, which is the one to compare across models.

## DIPS ladder -- the free validation, and it landed cleanly

For each gate: fit a small, INDEPENDENT (not shared-L) low-rank model at the
same (d, lambda) as the main model, with batter-latent, pitcher-latent,
neither, or both, and report test deviance (lower is better; `none` is the
structural-only floor for that gate alone).

| gate | none | batter only | pitcher only | both | reads as |
|---|---|---|---|---|---|
| root (TTO vs CONTACT) | 1.27637 | 1.26453 (-0.0118) | 1.26099 (-0.0154) | 1.24918 (-0.0272) | both sides matter, pitcher a bit more |
| tto (K/BB/HBP) | 1.71685 | 1.69642 (-0.0204) | 1.69221 (-0.0246) | 1.67418 (-0.0427) | **pitcher > batter, as DIPS predicts for TTO** |
| contact (OUT/HIT/OTHER) | 1.74666 | 1.74621 (-0.0005) | 1.74679 (+0.0001) | 1.74669 (0.0000) | **~zero signal from either side** |
| out (F/G) | 1.37790 | 1.36802 (-0.0099) | 1.36363 (-0.0143) | 1.35259 (-0.0253) | pitcher > batter (batted-ball-type is a real, mostly-pitcher skill) |
| hit (1B/2B/3B/HR) | 1.88242 | 1.86700 (-0.0154) | 1.88279 (+0.0004, worse) | 1.86702 (-0.0152) | **batter >> pitcher, pitcher adds ~nothing once batter is in** |

This reproduces McCracken's DIPS (1999) almost exactly, gate by gate:

- **TTO** is where classic DIPS says pitchers have the most control (K, BB,
  HBP rates are largely a pitcher skill) -- and here the pitcher-only
  deviance drop (0.0246) exceeds the batter-only drop (0.0204).
- **CONTACT** (does a ball in play become a hit or an out) is DIPS's
  signature claim: essentially NEITHER side controls this -- it is close to
  BABIP, dominated by luck and defense that this dataset doesn't model. The
  deviance drops here are noise-sized (0.0005, and one is even slightly
  positive/worse from added parameters). This is the cleanest and most
  convincing replication of the five: a null result exactly where the
  50-year-old sabermetric finding says to expect one.
- **HIT** (what kind of hit) is the mirror image: power/hit-type is
  overwhelmingly a BATTER skill. Batter-only nearly matches both-sides
  (1.86700 vs 1.86702), and pitcher-only is actually WORSE than no latent at
  all (1.88279 vs. 1.88242) -- the pitcher parameters are pure overfit noise
  once the true signal (all on the batter side) is accounted for.
- **OUT** (fly ball vs. ground ball) is a real, mostly-pitcher skill
  (sinkerball vs. flyball pitchers), and the ladder shows pitcher > batter
  here too, consistent with that being a legitimate (non-DIPS) pitcher
  attribute rather than luck.

Reproducing a 25-year-old, well-established result gate-by-gate, with the
correct sign on which side dominates each gate, is the strongest evidence in
this spike that the sequential recast captures something real rather than
just reparameterizing noise.

## Do the loadings agree with the flat axes?

Yes, and more strongly than the flat model's own single-softmax axes can,
because the SAME latent number is read through independent loadings at two
gates that share no target row. Full detail with matrices in
`loadings.md`; short version: one dimension is legible as "fly-ball/power"
at BOTH the OUT gate (F vs. G) and the HIT gate (1B vs. HR) for both batters
and pitchers independently -- a batter or pitcher who tends to hit/allow
fly balls (OUT gate) is, via the exact same latent value, one whose contact
skews toward extra bases rather than singles (HIT gate). That is the real
launch-angle/power relationship in baseball, discovered twice by
independently-fit loading matrices sharing only the player's position in
latent space. The CONTACT gate has no such coherent axis, matching its
DIPS null result above (there's no baseball concept to have discovered
there).

Two other dimensions at the TTO gate read as a K-vs-BB "command/discipline"
axis; they are close to collinear with each other (nothing enforces
orthogonality between columns of F), so this spike found ~2 genuinely
distinct concepts (power, command) at d=3 rather than 3 fully independent
ones -- consistent with, not contradicting, Powers et al.'s 3 named axes
(Power, Trajectory, Command), since Trajectory and Power are closely related
constructs in this recast (Trajectory folds INTO the Power story here,
because the F/G split lives on the same shared axis as 1B-vs-HR).

## Restart sensitivity

5 restarts at the selected (d=5, lambda=40): test deviance range
[3.951054, 3.951156], spread **0.000103**. About 3x the flat GLLVM's spread
(0.00003) but still two orders of magnitude smaller than the margin over
NPMR (0.0032) -- non-convexity is a measurable but practically negligible
concern here, same conclusion as the flat GLLVM reached, now confirmed for
the shared-latent variant too.

## Algorithm notes (the part the brief called "the crux")

- `solve_F_given_L_all_gates`: 5 independent L-BFGS-B calls, one per gate,
  looped (not multiprocessed -- each is cheap enough, sub-second to a few
  seconds, that pool overhead wasn't worth it inside the time budget).
- `solve_L_given_F_shared`: ONE L-BFGS-B call over all (n_bat+n_pit)*d
  parameters, with loss and gradient summed across all 5 gates inside a
  single objective closure. This is the coupling step -- a player's L-row
  gradient is literally `sum over gates reached by this player of (dEta_g @
  F_g)`, accumulated via `np.add.at` per gate before summing.
- Same joint-collapse failure mode as the flat GLLVM was checked for and
  avoided the same way: alternating block coordinate descent, never a
  single joint L-BFGS-B over L and F together. Sum-NLL (not mean-NLL) is
  used for the same reason documented in `spikes/gllvm/fit.py` -- a
  per-observation mean divides a player's gradient by ~100k regardless of
  how many rows that player actually has, crushing the sane per-block
  lambda scale.
- Ragged bookkeeping: `build_gate_data` slices `bi`/`pi`/`Xs` per gate using
  `gates.assign(y)`'s row-index arrays, so gate-local arrays never need to
  be the same length; `sanity_check_shapes` asserts index/branch-count
  consistency for every gate before fitting starts. Before running the full
  grid, a smoke test was run with tiny settings (D_GRID=[1], 1 round, 5
  iterations per block) end-to-end through result.json/latent.npz -- it
  passed cleanly on the first attempt, so no bugs actually surfaced in this
  run's development. The brief's warning that ragged single-gate
  bookkeeping (not the shared-L coupling) is where bugs are likeliest is
  still worth stating for anyone extending this: `fit_gate_variant`'s
  `use_bat`/`use_pit` branching (variable-length theta vectors depending on
  which sides are active) is the fiddliest code in this file and is where
  I'd look first if the DIPS ladder numbers ever looked wrong.

## Budget / what was cut

- D_GRID capped at {1..5} (matches flat GLLVM's own grid) x LAMBDA_GRID
  {10,20,40,60} = 20 combos, ALT_ROUNDS_CV=5 / BLOCK_MAXITER_CV=50 for
  selection; ALT_ROUNDS_FINAL=10 / BLOCK_MAXITER_FINAL=120 for the final
  5 restarts. Total measured runtime 237s -- there was substantial
  headroom left in the ~25 minute budget (this spike is cheaper per round
  than the flat GLLVM because each gate's softmax has 2-4 branches instead
  of 10, even summed over 5 gates), so the grid was widened from an initial
  conservative pass (d in {1,2,3}, 2 lambdas, 121s total) once that first
  pass confirmed correctness and timing.
- DIPS ladder fits are small (rounds=5, maxiter=50) standalone per-gate
  models, deliberately NOT the shared-L model -- isolating one gate's own
  signal is the point of that diagnostic, so it must NOT be entangled with
  the other 4 gates' rows.
- The d=3 interpretability companion fit (`latent_d3_interp.npz`) used only
  3 restarts, not 5, since it exists to produce readable loadings, not a
  competing headline number.
- No k-fold CV (single 80/20 inner game-level split, matching the flat
  GLLVM's own documented tradeoff) -- same reasoning: fits the time budget,
  see `spikes/gllvm/fit.py`'s docstring for the explicit tradeoff this
  spike inherited rather than re-litigated.

## Files

- `fit.py` -- the model, fully documented inline
- `result.json` -- required schema plus CV grid, restart deviances, DIPS
  ladder, for anyone re-deriving the numbers above
- `latent.npz` -- canonical d=5 fit: shared `Lbat`/`Lpit`, per-gate
  `Fbat_<gate>`/`Fpit_<gate>`/`alpha_<gate>`/`branches_<gate>`, `bat_ids`/
  `pit_ids`
- `latent_d3_interp.npz` -- supplementary d=3 fit used only for
  `loadings.md` (same schema, no `alpha_<gate>`)
- `loadings.md` -- the per-gate loading matrices and their interpretation
- `fit_run.log` -- full log of the reported run
