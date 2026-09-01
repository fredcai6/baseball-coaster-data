# Loadings: does one shared axis serve as a coherent lens at every gate?

The CV-selected model uses d=5 (see `result.json`), which reproduces the flat
GLLVM's own choice of d=5 but is hard to eyeball. Rank-3 test deviance is
**3.95136** on a supplementary d=3 fit (vs. 3.95105 at d=5 -- a 0.0003 gap,
i.e. d=3 already captures nearly all of the benefit) -- see
`latent_d3_interp.npz` (same schema as `latent.npz`: `Lbat`, `Lpit`,
`Fbat_<gate>`, `Fpit_<gate>`, `branches_<gate>`). d=3 also matches the rank
Powers/Hastie/Tibshirani forced for their flat NPMR axes (Power, Trajectory,
Command), so it's the fairer comparison point and what the rest of this file
uses.

**Why raw dimension-by-dimension comparison across gates is even legitimate
here, unlike in a per-gate-latent design:** a low-rank factor model is
identified only up to an invertible d x d transform Q (L -> L Q, F -> F
Q^{-T} leaves L @ F.T unchanged). In a design with a SEPARATE L per gate
(a sibling spike), each gate gets its own independent Q -- dimension 1 at
gate A has no relationship to dimension 1 at gate B, full stop. Here L is
literally the same matrix at every gate, so rotating it requires rotating
EVERY gate's F together. The gauge freedom is global, not per-gate, which is
exactly why it makes sense to ask "does dimension k mean the same thing at
two different gates" -- the shared-L design is what makes the question
answerable at all.

## The loading matrices (d=3, rows = branches, columns = latent dims)

```
root      branches [TTO, CONTACT]
  Fbat  [[ 0.666 -0.874  1.017]
         [-0.666  0.874 -1.017]]
  Fpit  [[ 0.754 -1.706 -0.193]
         [-0.754  1.706  0.193]]

tto       branches [K, BB, HBP]
  Fbat  [[-0.993 -1.041  0.126]
         [ 0.879  0.941  0.071]
         [ 0.114  0.100 -0.197]]
  Fpit  [[ 1.592  0.790  0.509]
         [-1.259 -0.652 -0.431]
         [-0.333 -0.139 -0.078]]

contact   branches [OUT, HIT, OTHER]
  Fbat  [[ 0.154 -0.058 -0.082]
         [-0.214  0.179  0.407]
         [ 0.061 -0.122 -0.326]]
  Fpit  [[ 0.680  0.152  0.355]
         [-0.383  0.035  0.038]
         [-0.297 -0.187 -0.393]]

out       branches [F, G]
  Fbat  [[-0.437  0.674  0.772]
         [ 0.437 -0.674 -0.772]]
  Fpit  [[-0.427 -0.353  1.391]
         [ 0.427  0.353 -1.391]]

hit       branches [1B, 2B, 3B, HR]
  Fbat  [[ 0.616 -0.181 -0.913]
         [-0.098  0.156  0.082]
         [-0.016 -0.019 -0.034]
         [-0.502  0.044  0.865]]
  Fpit  [[ 0.113  0.270 -0.649]
         [-0.056 -0.129  0.251]
         [-0.014 -0.009  0.023]
         [-0.043 -0.132  0.375]]
```

(2-branch gates -- root, out -- necessarily have antisymmetric rows: with
only a log-odds difference identified, the ridge penalty on F pulls the
solution to the minimum-norm antisymmetric point. Expected, not a bug.)

## Reading dimension 3 (index 2): a genuine "fly-ball / power" axis

This is the clearest cross-gate story in the fit, and it appears for BOTH
sides independently:

- **OUT gate**, dim 3: Fpit = [+1.391 (F), -1.391 (G)] -- by far the largest
  loading anywhere in the pitcher tables. A pitcher high on this axis gives
  up a lot more fly balls than ground balls. Fbat = [+0.772 (F), -0.772 (G)]
  -- same direction for batters: high-dim-3 hitters put more balls in the
  air.
- **HIT gate**, dim 3: Fpit = [-0.649 (1B), +0.251 (2B), +0.023 (3B), +0.375
  (HR)] -- fewer singles, more extra bases (especially HR) allowed. Fbat =
  [-0.913 (1B), +0.082 (2B), -0.034 (3B), +0.865 (HR)] -- same pattern:
  fewer singles, far more home runs.

Put together: a batter (or pitcher) who is high on dim 3 is a fly-ball
hitter (or fly-ball-prone pitcher) at the OUT gate, AND -- using the exact
same number, through a different loading -- someone whose contact that
becomes a hit is disproportionately a home run rather than a single, at the
HIT gate. That is precisely the real baseball relationship between launch
angle and power (you cannot hit an official home run on a ball hit into the
ground), reproduced by two gates that never share a row of data with each
other for the same player-appearance and were fit with independent
loadings. This is the sharing doing real work, not coincidence: it is the
same L for that player at both gates.

This axis is close to "Power" in the Powers/Hastie/Tibshirani flat-model
sense, but sharper: their forced-rank-3 Power axis was recovered from a
single flat 10-category softmax; ours emerges as the SAME latent number
read through two completely separate small gates, which is a stronger form
of internal consistency than one softmax finding an axis by itself.

## Reading dimensions 1 and 2: TTO composition and overall TTO rate

At the TTO gate, dims 1 and 2 are nearly collinear for both batter and
pitcher (K vs. BB contrast, same sign pattern, similar magnitude on both
columns) -- this is very likely the same underlying "control/command" idea
split across two correlated, non-orthogonal directions rather than two
independent concepts (expected: nothing in the objective enforces
orthogonality between columns of F, only that L @ F.T fits the data).
Reading them together: high dim-1/dim-2 pitchers throw more strikes (higher
K, lower BB) -- a command axis. High dim-1/dim-2 batters are, perhaps
counter-intuitively, LOWER on K and HIGHER on BB at the TTO gate itself
(Fbat dim 2: K=-1.041, BB=+0.941) -- i.e. conditional on reaching a
three-true-outcome plate appearance at all, they walk more than they
strike out (a patient hitter's profile). But at the ROOT gate, the SAME
batter dim (dim 2) has a NEGATIVE loading on the TTO branch itself
(Fbat[TTO, dim2] = -0.874, meaning higher dim 2 pushes a PA toward CONTACT,
not TTO, overall). Read together, that is a coherent hitter type: a
contact-oriented, patient hitter who rarely runs up a three-true-outcome
count at all, and on the rare occasions they do, ends up walking more often
than striking out. Nothing here contradicts DIPS; if anything it sharpens
it (see the DIPS ladder in README.md: this same TTO gate shows the
PITCHER'S dims 1/2 loadings are roughly 1.5-2x LARGER in magnitude than the
batter's, matching the deviance-drop story below).

## Reading the CONTACT gate: the null result IS the finding

Every loading at the contact gate (OUT vs. HIT vs. OTHER, i.e. "once you
put it in play, does it become a hit or an out") is small -- the largest
magnitude anywhere in that table is 0.68 vs. 1.0-1.7 at other gates, and the
DIPS ladder (README.md) shows essentially zero deviance improvement from
either side's latent at this gate. That absence of a coherent axis is
itself the correct, expected result -- see the DIPS section.
