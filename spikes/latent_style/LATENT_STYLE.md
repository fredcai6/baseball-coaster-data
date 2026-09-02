# Latent style: is the 3-D player latent a meaningful summary?

Question: latent2's arm 1 (shared rank-3 batter/pitcher latent inside shape D's 9-node binarised tree, Aranda-Ordaz link, d=3 lam_L=lam_M=40, psi inherited from step6 shape D) compresses 9 free per-node player effects into 3 numbers at a cost of +0.00583 test deviance (3.95109 vs shape D's 3.94526) -- i.e. it retains 91% of the 0.0665 total player signal. Is that 3-D space a MEANINGFUL summary, or an arbitrary rotation of a quality-plus-noise ellipsoid? Two tests below decide it; both use the FIXED hyperparameters already selected in spikes/latent2/result.json -- no new hyperparameter search was run here.

Canonical full-train fit: d=3 lam_L=lam_M=40, n_bat=772 n_pit=1220 n_train_pa=100445, 3 restarts (canonical seed 1, train-loss spread 0.0000 -- restart spread at these hyperparameters was already shown negligible in latent2/result.json (8.35e-7), so 3 restarts rather than 5 were used here for all three fits (full/halfA/halfB) to keep each stage inside the foreground time budget).

Halves split: seed recorded in halves_split.json, stratified by season, disjoint games. Half A: 592 games / 50035 PA. Half B: 593 games / 50410 PA.

## Test 1 -- split-half reliability (the decisive one)

Alignment: orthogonal (rotation/reflection only, no scaling) -- Pearson r is scale-invariant so the optimal rotation from SVD(B^T A) is identical whether or not a scale factor is also fit; only rotation resolves axis-mixing/reflection, which is what matters for per-axis correlation.

**Batters** -- 405 / 772 players with >= 30 PA/BF in BOTH halves.

Procrustes-aligned latent-coordinate split-half r (per axis):

| axis | split-half r | Spearman-Brown r (full-sample) |
|---|---|---|
| 0 | 0.616 | 0.762 |
| 1 | 0.477 | 0.646 |
| 2 | 0.227 | 0.370 |

Alignment-free per-node predicted-effect (L @ f.T) split-half r:

| node | split-half r | Spearman-Brown r (full-sample) |
|---|---|---|
| root | 0.633 | 0.775 |
| tto_K | 0.472 | 0.641 |
| tto_BB | 0.182 | 0.308 |
| con_HR | 0.554 | 0.713 |
| con_OTH | 0.539 | 0.700 |
| con_OUT | 0.108 | 0.195 |
| out_F | 0.509 | 0.674 |
| hit_1B | 0.500 | 0.667 |
| hit_2B | -0.324 | -0.957 |

**Pitchers** -- 394 / 1220 players with >= 30 PA/BF in BOTH halves.

Procrustes-aligned latent-coordinate split-half r (per axis):

| axis | split-half r | Spearman-Brown r (full-sample) |
|---|---|---|
| 0 | 0.397 | 0.568 |
| 1 | 0.420 | 0.592 |
| 2 | 0.606 | 0.755 |

Alignment-free per-node predicted-effect (L @ f.T) split-half r:

| node | split-half r | Spearman-Brown r (full-sample) |
|---|---|---|
| root | 0.617 | 0.763 |
| tto_K | 0.348 | 0.516 |
| tto_BB | -0.001 | -0.002 |
| con_HR | 0.396 | 0.567 |
| con_OTH | 0.376 | 0.546 |
| con_OUT | 0.319 | 0.483 |
| out_F | 0.415 | 0.587 |
| hit_1B | 0.432 | 0.604 |
| hit_2B | -0.457 | -1.683 |

**Reading the axes (batters):** axis 0 (r=0.616 / SB 0.762) and axis 1 (r=0.477 / SB 0.646) replicate reasonably; axis 2 (r=0.227 / SB 0.370) is borderline-to-weak. **Pitchers:** axis 2 (r=0.606 / SB 0.755) is the strongest, axes 0-1 (r~0.40-0.42, SB ~0.57-0.59) are moderate. The alignment-free per-node check tells the same story at the node level: most nodes land r=0.35-0.63 (SB 0.5-0.78), except hit_2B, which is NEGATIVE for both sides (batters -0.324, pitchers -0.457) -- consistent with step6_shapes.py's own finding that hit_2B's validation surface is flat/degenerate (pitcher identity carries ~no signal for 2B once conditioned on reaching {2B,3B,HR}); the Spearman-Brown formula is not meaningful for negative r and is reported as-is for completeness, not as evidence of anti-correlation.

## Test 2 -- external validity

Method: nested CV: outer 5-fold KFold(shuffle, seed=4242) for R^2, RidgeCV(alphas=logspace(-2,5,30)) inner generalized-CV for alpha, features standardized. Categorical targets one-hot + multi-output R^2 (uniform average). Qualified: >= 100 training PA/BF.

Caveats stated, not hidden: `gb_rate` (derived from `bb_type`) is PARTIALLY in-target -- out_F/con_OUT directly model F vs G, so a high R2 there is expected and is not evidence of external validity. `pitches_per_pa`, `swing_miss_rate`, `called_strike_rate`, `first_pitch_strike_rate` are 2026-only (pitch_seq is null in 2024/2025), so their qualified-n is much smaller (~136-140) than the other variables (~255-355).

**Batters**

| variable | n qualified | latent3 R2 | free9 R2 | pca3 R2 |
|---|---|---|---|---|
| pull_score | 310 | +0.003 | +0.046 | +0.050 |
| gb_rate | 346 | +0.725 | +0.931 | +0.930 |
| pitches_per_pa | 140 | +0.282 | +0.325 | +0.036 |
| swing_miss_rate | 140 | +0.411 | +0.382 | +0.147 |
| called_strike_rate | 140 | +0.026 | +0.040 | +0.003 |
| first_pitch_strike_rate | 140 | +0.098 | +0.052 | -0.008 |
| height_in | 262 | +0.039 | +0.128 | +0.088 |
| weight_lb | 305 | +0.061 | +0.152 | +0.145 |
| position | 314 | -0.010 | +0.009 | -0.011 |
| bats | 324 | +0.024 | +0.015 | +0.002 |

**Pitchers**

| variable | n qualified | latent3 R2 | free9 R2 | pca3 R2 |
|---|---|---|---|---|
| gb_rate | 355 | +0.890 | +0.897 | -0.017 |
| pitches_per_pa | 136 | +0.240 | +0.204 | +0.273 |
| swing_miss_rate | 136 | +0.459 | +0.488 | +0.503 |
| called_strike_rate | 136 | -0.033 | -0.024 | -0.029 |
| first_pitch_strike_rate | 136 | +0.137 | +0.128 | +0.180 |
| height_in | 255 | +0.011 | -0.013 | -0.024 |
| weight_lb | 299 | +0.010 | -0.004 | -0.009 |
| throws | 336 | -0.007 | -0.020 | -0.013 |

## Presentation: canonical axes

Canonical rotation: SVD of L_bat @ F_bat.T (batters) and M_pit @ G_pit.T (pitchers) separately, axes ordered by singular value. Sign is arbitrary (SVD sign ambiguity) -- read +/- as 'one side' / 'the other side', not as an absolute direction, per spikes/npmr/latent.md's convention. Category rates are computed on the SAME training games the latent was fit on.

### Batter axis 0  (singular value 10.24)

Run-value correlation: -0.022 -> **STYLE (not quality)**

Node loadings, most negative to most positive:

| node | loading |
|---|---|
| hit_1B | -0.238 |
| con_OTH | -0.148 |
| hit_2B | +0.021 |
| con_OUT | +0.043 |
| tto_BB | +0.108 |
| tto_K | +0.390 |
| root | +0.417 |
| out_F | +0.478 |
| con_HR | +0.594 |

Top 10 (most positive on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Easton Amundson | +1.179 | 90 | 32.2 | 12.2 | 0.0 | 28.9 | 8.9 | 5.6 | 3.3 | 0.0 | 6.7 | 2.2 |
| Kyle Schmack | +1.129 | 365 | 21.4 | 12.3 | 5.5 | 24.9 | 6.3 | 15.1 | 6.0 | 0.3 | 5.8 | 2.5 |
| Tyler Shelnut | +1.033 | 273 | 27.5 | 15.8 | 2.2 | 24.9 | 8.1 | 10.6 | 4.0 | 0.4 | 3.3 | 3.3 |
| Jake Millan | +1.033 | 319 | 32.0 | 12.5 | 1.6 | 19.1 | 13.2 | 10.7 | 3.8 | 0.6 | 4.7 | 1.9 |
| Antonio Barranca | +0.999 | 39 | 43.6 | 12.8 | 0.0 | 23.1 | 2.6 | 5.1 | 7.7 | 0.0 | 5.1 | 0.0 |
| Roman Kuntz | +0.887 | 137 | 27.7 | 8.0 | 1.5 | 24.8 | 9.5 | 12.4 | 6.6 | 0.7 | 6.6 | 2.2 |
| Gabriel Vasquez | +0.867 | 159 | 28.9 | 10.1 | 0.0 | 21.4 | 12.6 | 10.1 | 6.3 | 0.6 | 5.0 | 5.0 |
| Steven Rivas | +0.858 | 306 | 21.6 | 5.9 | 0.7 | 20.6 | 15.4 | 16.7 | 6.5 | 1.0 | 7.5 | 4.2 |
| Jake Hjelle | +0.850 | 283 | 20.8 | 6.4 | 1.4 | 24.7 | 10.6 | 17.0 | 8.8 | 0.4 | 5.7 | 4.2 |
| Sebastian Greico | +0.845 | 222 | 21.6 | 8.6 | 2.7 | 19.8 | 11.7 | 16.2 | 6.8 | 0.9 | 9.0 | 2.7 |

Bottom 10 (most negative on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kenneth Oyama | -1.377 | 315 | 7.3 | 10.2 | 2.5 | 19.4 | 23.5 | 23.8 | 4.1 | 0.3 | 0.0 | 8.9 |
| Josh Duarte | -1.346 | 379 | 7.9 | 13.5 | 2.4 | 18.7 | 25.6 | 23.0 | 3.2 | 0.0 | 0.8 | 5.0 |
| Edwin Pagani | -1.232 | 116 | 7.8 | 9.5 | 0.0 | 14.7 | 31.0 | 19.8 | 5.2 | 0.0 | 0.9 | 11.2 |
| John Mabry | -1.167 | 129 | 10.9 | 4.7 | 0.8 | 17.1 | 31.0 | 28.7 | 2.3 | 0.0 | 0.0 | 4.7 |
| Drew Stengren | -1.121 | 156 | 7.7 | 13.5 | 1.3 | 14.7 | 25.6 | 24.4 | 5.8 | 0.0 | 1.9 | 5.1 |
| Anthony Mata | -1.119 | 253 | 13.0 | 8.3 | 1.2 | 12.6 | 25.3 | 24.5 | 5.5 | 0.8 | 1.2 | 7.5 |
| Kyle Ashworth | -1.069 | 423 | 13.0 | 20.1 | 1.9 | 13.7 | 23.2 | 19.6 | 3.8 | 0.2 | 0.7 | 3.8 |
| Tristin Garcia | -1.068 | 225 | 10.7 | 11.1 | 3.1 | 14.7 | 25.3 | 24.4 | 3.6 | 0.9 | 1.3 | 4.9 |
| Jacob Gutierrez | -1.058 | 222 | 4.5 | 4.5 | 2.3 | 24.3 | 20.7 | 27.0 | 8.1 | 0.5 | 2.3 | 5.9 |
| Euro Diaz | -1.039 | 214 | 16.8 | 10.3 | 1.4 | 8.9 | 28.0 | 20.6 | 4.7 | 0.0 | 1.9 | 7.5 |

### Batter axis 1  (singular value 8.21)

Run-value correlation: +0.531 -> **QUALITY-aligned**

Node loadings, most negative to most positive:

| node | loading |
|---|---|
| tto_K | -0.614 |
| root | -0.480 |
| hit_1B | -0.152 |
| con_OTH | -0.119 |
| con_OUT | -0.053 |
| hit_2B | +0.031 |
| tto_BB | +0.057 |
| con_HR | +0.193 |
| out_F | +0.558 |

Top 10 (most positive on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kai Moody | +0.978 | 264 | 12.5 | 15.2 | 0.4 | 27.7 | 11.4 | 18.9 | 6.4 | 1.1 | 2.7 | 3.8 |
| Jack Cone | +0.894 | 318 | 12.3 | 11.3 | 2.2 | 28.6 | 13.8 | 19.5 | 6.6 | 0.3 | 2.2 | 3.1 |
| Patrick Chung | +0.860 | 281 | 10.0 | 13.9 | 1.8 | 20.6 | 16.0 | 19.6 | 7.8 | 1.1 | 3.2 | 6.0 |
| Quintt Landis | +0.843 | 250 | 11.6 | 17.2 | 3.2 | 26.0 | 12.8 | 20.4 | 3.6 | 0.8 | 1.6 | 2.8 |
| Xavier Casserilla | +0.813 | 319 | 12.5 | 11.6 | 4.4 | 21.3 | 11.6 | 16.9 | 7.8 | 0.3 | 8.2 | 5.3 |
| Nin Burns II | +0.793 | 88 | 8.0 | 22.7 | 1.1 | 34.1 | 10.2 | 15.9 | 3.4 | 0.0 | 1.1 | 3.4 |
| Cuba Bess | +0.787 | 296 | 15.2 | 15.5 | 3.0 | 24.7 | 12.5 | 13.9 | 3.0 | 0.3 | 8.4 | 3.4 |
| Eddy Pelc | +0.770 | 353 | 9.9 | 27.8 | 0.6 | 21.8 | 12.2 | 19.8 | 4.2 | 0.3 | 1.7 | 1.7 |
| Collin Runge | +0.747 | 218 | 11.0 | 13.3 | 1.8 | 22.5 | 17.4 | 20.2 | 5.5 | 0.5 | 3.7 | 4.1 |
| Sam Linscott | +0.740 | 287 | 8.4 | 7.3 | 2.1 | 23.0 | 17.4 | 24.0 | 7.3 | 0.7 | 2.8 | 7.0 |

Bottom 10 (most negative on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Eli Young | -0.960 | 43 | 44.2 | 4.7 | 4.7 | 7.0 | 30.2 | 9.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| Royce Clayton, Jr. | -0.812 | 61 | 47.5 | 4.9 | 0.0 | 13.1 | 8.2 | 11.5 | 3.3 | 1.6 | 0.0 | 9.8 |
| Dreylin Holmes | -0.795 | 52 | 50.0 | 15.4 | 0.0 | 9.6 | 7.7 | 11.5 | 1.9 | 0.0 | 0.0 | 3.8 |
| Demarckus Smiley | -0.713 | 106 | 20.8 | 15.1 | 0.0 | 6.6 | 25.5 | 25.5 | 4.7 | 0.0 | 0.0 | 1.9 |
| Paul Winland Jr. | -0.675 | 52 | 28.8 | 1.9 | 7.7 | 5.8 | 23.1 | 13.5 | 0.0 | 0.0 | 7.7 | 11.5 |
| Gavin Tonkel | -0.673 | 142 | 35.9 | 11.3 | 2.8 | 14.1 | 15.5 | 9.2 | 2.1 | 0.7 | 2.1 | 6.3 |
| Caden Matlon | -0.670 | 57 | 36.8 | 12.3 | 0.0 | 12.3 | 19.3 | 7.0 | 8.8 | 0.0 | 1.8 | 1.8 |
| Chris Brady | -0.646 | 43 | 39.5 | 7.0 | 4.7 | 11.6 | 18.6 | 11.6 | 0.0 | 0.0 | 2.3 | 4.7 |
| Lucas Terilli | -0.623 | 120 | 20.8 | 22.5 | 1.7 | 9.2 | 20.8 | 15.0 | 0.8 | 0.8 | 1.7 | 6.7 |
| Harold Torres | -0.619 | 25 | 36.0 | 20.0 | 4.0 | 4.0 | 28.0 | 8.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Batter axis 2  (singular value 6.46)

Run-value correlation: +0.288 -> **weak/QUALITY-leaning**

Node loadings, most negative to most positive:

| node | loading |
|---|---|
| tto_K | -0.656 |
| con_OUT | -0.096 |
| out_F | -0.092 |
| con_OTH | -0.047 |
| con_HR | -0.005 |
| hit_2B | +0.014 |
| tto_BB | +0.016 |
| hit_1B | +0.046 |
| root | +0.740 |

Top 10 (most positive on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Eddy Pelc | +0.857 | 353 | 9.9 | 27.8 | 0.6 | 21.8 | 12.2 | 19.8 | 4.2 | 0.3 | 1.7 | 1.7 |
| Austin Folds | +0.856 | 142 | 9.9 | 23.2 | 7.7 | 9.2 | 23.2 | 16.2 | 3.5 | 0.7 | 2.1 | 4.2 |
| Armando Albert | +0.686 | 108 | 20.4 | 18.5 | 7.4 | 9.3 | 20.4 | 16.7 | 3.7 | 0.0 | 0.0 | 3.7 |
| Kyle Ashworth | +0.642 | 423 | 13.0 | 20.1 | 1.9 | 13.7 | 23.2 | 19.6 | 3.8 | 0.2 | 0.7 | 3.8 |
| Ty Yukumoto | +0.603 | 234 | 5.6 | 20.1 | 3.4 | 21.8 | 20.1 | 20.5 | 5.1 | 0.0 | 0.4 | 3.0 |
| Evan Berkey | +0.577 | 51 | 11.8 | 15.7 | 13.7 | 23.5 | 11.8 | 11.8 | 2.0 | 0.0 | 7.8 | 2.0 |
| Connor Denning | +0.572 | 233 | 13.3 | 21.5 | 5.2 | 24.0 | 15.5 | 9.9 | 3.9 | 0.0 | 3.0 | 3.9 |
| Isaac Lovings | +0.549 | 40 | 30.0 | 22.5 | 7.5 | 17.5 | 2.5 | 15.0 | 2.5 | 0.0 | 0.0 | 2.5 |
| Carson Tucker | +0.546 | 46 | 17.4 | 26.1 | 4.3 | 19.6 | 4.3 | 23.9 | 4.3 | 0.0 | 0.0 | 0.0 |
| Michael Koszewski | +0.542 | 308 | 7.5 | 20.8 | 1.3 | 23.4 | 14.9 | 19.5 | 4.2 | 1.0 | 1.3 | 6.2 |

Bottom 10 (most negative on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Omar Veloz | -0.689 | 119 | 21.8 | 1.7 | 0.8 | 16.8 | 28.6 | 16.8 | 5.9 | 0.8 | 0.0 | 6.7 |
| Jacob Kline | -0.677 | 303 | 20.5 | 6.6 | 0.3 | 27.7 | 13.5 | 17.2 | 3.6 | 0.7 | 2.6 | 7.3 |
| Emilio Corona | -0.671 | 174 | 20.7 | 3.4 | 0.0 | 24.7 | 11.5 | 20.7 | 6.9 | 2.9 | 4.6 | 4.6 |
| Alonzo Zuniga | -0.666 | 49 | 12.2 | 0.0 | 0.0 | 16.3 | 36.7 | 22.4 | 2.0 | 0.0 | 0.0 | 10.2 |
| Steven Rivas | -0.652 | 306 | 21.6 | 5.9 | 0.7 | 20.6 | 15.4 | 16.7 | 6.5 | 1.0 | 7.5 | 4.2 |
| Tyner Hughes | -0.618 | 168 | 17.3 | 3.0 | 3.0 | 24.4 | 22.0 | 16.7 | 7.1 | 0.6 | 2.4 | 3.6 |
| Omar Veloz | -0.617 | 222 | 21.2 | 2.3 | 3.6 | 22.5 | 20.3 | 20.3 | 3.6 | 0.0 | 2.7 | 3.6 |
| Cuba Bess | -0.590 | 298 | 15.8 | 7.0 | 1.7 | 30.2 | 14.8 | 15.8 | 5.4 | 0.0 | 6.7 | 2.7 |
| Cameron Bowen | -0.569 | 654 | 17.1 | 6.9 | 1.4 | 21.3 | 17.9 | 18.7 | 6.0 | 0.8 | 3.5 | 6.6 |
| Gabe Wurtz | -0.567 | 201 | 22.9 | 5.5 | 2.0 | 25.4 | 15.9 | 13.9 | 3.5 | 0.0 | 6.5 | 4.5 |

### Pitcher axis 0  (singular value 13.51)

Run-value correlation: +0.548 -> **QUALITY-aligned**

Node loadings, most negative to most positive:

| node | loading |
|---|---|
| tto_K | -0.831 |
| con_OUT | -0.147 |
| hit_1B | -0.068 |
| tto_BB | -0.015 |
| hit_2B | -0.001 |
| con_HR | +0.070 |
| out_F | +0.158 |
| con_OTH | +0.233 |
| root | +0.446 |

Top 10 (most positive on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Branden Blankenship | +1.236 | 17 | 0.0 | 58.8 | 5.9 | 0.0 | 11.8 | 11.8 | 0.0 | 0.0 | 5.9 | 5.9 |
| Starling Perez | +1.170 | 50 | 6.0 | 30.0 | 4.0 | 18.0 | 8.0 | 16.0 | 8.0 | 0.0 | 6.0 | 4.0 |
| Alex Verdugo | +1.153 | 39 | 15.4 | 35.9 | 7.7 | 5.1 | 12.8 | 5.1 | 5.1 | 0.0 | 2.6 | 10.3 |
| Danny Fox | +1.072 | 191 | 11.5 | 20.4 | 8.4 | 17.3 | 9.4 | 18.8 | 5.2 | 0.0 | 3.1 | 5.8 |
| Gabe Emmett | +1.070 | 13 | 7.7 | 61.5 | 7.7 | 0.0 | 7.7 | 7.7 | 0.0 | 0.0 | 0.0 | 7.7 |
| Breyln Jones | +1.050 | 146 | 8.9 | 32.2 | 0.0 | 17.8 | 13.7 | 17.8 | 4.8 | 0.0 | 2.1 | 2.7 |
| Andrew Vail | +1.048 | 33 | 12.1 | 33.3 | 9.1 | 9.1 | 12.1 | 12.1 | 3.0 | 0.0 | 3.0 | 6.1 |
| Devin Norton | +1.020 | 98 | 3.1 | 22.4 | 3.1 | 19.4 | 11.2 | 19.4 | 6.1 | 0.0 | 4.1 | 11.2 |
| Andrew Rust | +1.005 | 19 | 10.5 | 42.1 | 21.1 | 5.3 | 21.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Stephen Greenlees | +0.992 | 16 | 0.0 | 37.5 | 12.5 | 12.5 | 6.2 | 18.8 | 0.0 | 6.2 | 0.0 | 6.2 |

Bottom 10 (most negative on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Noah Millikan | -1.259 | 263 | 26.6 | 3.4 | 2.7 | 21.3 | 22.1 | 14.1 | 4.6 | 0.4 | 1.9 | 3.0 |
| David Thomas | -1.223 | 84 | 21.4 | 4.8 | 0.0 | 20.2 | 26.2 | 19.0 | 1.2 | 0.0 | 4.8 | 2.4 |
| Adam Christopher | -1.213 | 360 | 15.0 | 2.8 | 1.7 | 26.4 | 17.2 | 18.9 | 8.3 | 0.8 | 5.3 | 3.6 |
| Cole Calnon | -1.161 | 324 | 9.9 | 3.4 | 1.2 | 28.1 | 20.7 | 21.0 | 5.2 | 0.3 | 5.2 | 4.9 |
| Connor Langrell | -1.140 | 128 | 27.3 | 3.1 | 2.3 | 14.1 | 28.9 | 16.4 | 2.3 | 0.8 | 3.1 | 1.6 |
| Aidan Elfering | -1.116 | 354 | 18.1 | 5.6 | 1.1 | 24.9 | 19.2 | 20.6 | 3.4 | 0.3 | 4.2 | 2.5 |
| Grant Taylor | -1.083 | 343 | 25.1 | 7.0 | 0.0 | 22.7 | 18.7 | 17.2 | 3.5 | 0.0 | 3.5 | 2.3 |
| Nico Saltaformaggio | -1.029 | 868 | 19.4 | 6.7 | 1.2 | 11.9 | 26.3 | 21.3 | 5.0 | 0.6 | 2.9 | 5.0 |
| Cole Calnon | -1.013 | 55 | 10.9 | 1.8 | 0.0 | 38.2 | 20.0 | 12.7 | 3.6 | 0.0 | 7.3 | 5.5 |
| Ren Abe-Arias | -0.999 | 310 | 13.5 | 7.4 | 0.3 | 25.2 | 20.0 | 17.4 | 6.5 | 0.6 | 5.8 | 3.2 |

### Pitcher axis 1  (singular value 11.63)

Run-value correlation: -0.511 -> **QUALITY-aligned**

Node loadings, most negative to most positive:

| node | loading |
|---|---|
| out_F | -0.106 |
| con_OTH | -0.072 |
| con_HR | -0.023 |
| hit_1B | -0.016 |
| hit_2B | +0.005 |
| tto_BB | +0.034 |
| con_OUT | +0.178 |
| tto_K | +0.403 |
| root | +0.887 |

Top 10 (most positive on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Rane Pfeifer | +1.483 | 60 | 51.7 | 11.7 | 5.0 | 13.3 | 6.7 | 6.7 | 3.3 | 0.0 | 1.7 | 0.0 |
| Connor Butler | +1.171 | 79 | 44.3 | 21.5 | 0.0 | 12.7 | 6.3 | 8.9 | 2.5 | 0.0 | 2.5 | 1.3 |
| Reese Miller | +1.148 | 150 | 40.7 | 7.3 | 2.0 | 19.3 | 14.0 | 10.7 | 2.7 | 0.0 | 0.7 | 2.7 |
| Julio Rosario | +1.134 | 38 | 42.1 | 26.3 | 2.6 | 7.9 | 7.9 | 0.0 | 7.9 | 0.0 | 5.3 | 0.0 |
| Matthew Taubensee | +1.114 | 225 | 38.7 | 16.4 | 0.9 | 17.3 | 8.4 | 8.0 | 5.8 | 0.0 | 1.3 | 3.1 |
| Zac Lampton | +1.069 | 147 | 40.1 | 12.9 | 2.7 | 19.0 | 8.8 | 8.8 | 4.1 | 0.7 | 1.4 | 1.4 |
| Jack Maruskin | +1.060 | 84 | 38.1 | 16.7 | 2.4 | 10.7 | 11.9 | 9.5 | 4.8 | 1.2 | 3.6 | 1.2 |
| Jonathan Ramallo | +1.050 | 38 | 34.2 | 26.3 | 5.3 | 7.9 | 15.8 | 5.3 | 0.0 | 0.0 | 2.6 | 2.6 |
| Trey Valka | +0.980 | 26 | 38.5 | 19.2 | 19.2 | 3.8 | 7.7 | 11.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| Jimmy Loper | +0.919 | 16 | 56.2 | 12.5 | 0.0 | 0.0 | 25.0 | 0.0 | 6.2 | 0.0 | 0.0 | 0.0 |

Bottom 10 (most negative on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Craig Corliss | -1.154 | 39 | 0.0 | 0.0 | 2.6 | 35.9 | 12.8 | 28.2 | 10.3 | 0.0 | 7.7 | 2.6 |
| Connor Barton | -1.131 | 66 | 3.0 | 7.6 | 0.0 | 33.3 | 12.1 | 30.3 | 6.1 | 1.5 | 1.5 | 4.5 |
| Noah Blythe | -0.984 | 41 | 0.0 | 7.3 | 7.3 | 36.6 | 7.3 | 29.3 | 4.9 | 2.4 | 0.0 | 4.9 |
| Benjamin Rosengard | -0.933 | 24 | 0.0 | 0.0 | 4.2 | 45.8 | 4.2 | 29.2 | 4.2 | 0.0 | 8.3 | 4.2 |
| C.J. Backer Jr. | -0.844 | 48 | 4.2 | 10.4 | 0.0 | 29.2 | 16.7 | 29.2 | 8.3 | 0.0 | 0.0 | 2.1 |
| Austin Mora | -0.835 | 107 | 5.6 | 8.4 | 0.9 | 23.4 | 24.3 | 22.4 | 6.5 | 0.9 | 1.9 | 5.6 |
| Jacob Bradshaw | -0.807 | 45 | 2.2 | 6.7 | 6.7 | 28.9 | 17.8 | 26.7 | 8.9 | 0.0 | 0.0 | 2.2 |
| Aidan Risse | -0.806 | 63 | 4.8 | 7.9 | 3.2 | 17.5 | 20.6 | 27.0 | 11.1 | 0.0 | 1.6 | 6.3 |
| Tim Cunningham | -0.802 | 15 | 0.0 | 6.7 | 0.0 | 13.3 | 13.3 | 46.7 | 13.3 | 0.0 | 0.0 | 6.7 |
| Garrett VanDeventer | -0.795 | 34 | 2.9 | 14.7 | 2.9 | 8.8 | 14.7 | 32.4 | 5.9 | 5.9 | 2.9 | 8.8 |

### Pitcher axis 2  (singular value 10.12)

Run-value correlation: +0.003 -> **STYLE (not quality)**

Node loadings, most negative to most positive:

| node | loading |
|---|---|
| out_F | -0.892 |
| con_HR | -0.243 |
| tto_K | -0.134 |
| con_OUT | -0.121 |
| root | -0.006 |
| hit_2B | -0.006 |
| tto_BB | +0.007 |
| con_OTH | +0.210 |
| hit_1B | +0.262 |

Top 10 (most positive on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Braydon Nelson | +1.250 | 134 | 24.6 | 11.2 | 0.7 | 3.7 | 32.8 | 14.9 | 5.2 | 0.0 | 3.0 | 3.7 |
| Quinn Waterhouse | +1.018 | 159 | 22.6 | 15.1 | 3.1 | 7.5 | 24.5 | 15.7 | 1.9 | 0.6 | 2.5 | 6.3 |
| Nico Saltaformaggio | +0.991 | 868 | 19.4 | 6.7 | 1.2 | 11.9 | 26.3 | 21.3 | 5.0 | 0.6 | 2.9 | 5.0 |
| Billy Rozakis | +0.910 | 114 | 12.3 | 15.8 | 0.9 | 10.5 | 22.8 | 21.9 | 5.3 | 0.0 | 0.0 | 10.5 |
| Cale Mathison | +0.904 | 74 | 18.9 | 18.9 | 8.1 | 5.4 | 24.3 | 12.2 | 5.4 | 1.4 | 1.4 | 4.1 |
| Jacob Hasty | +0.876 | 263 | 28.1 | 19.8 | 2.7 | 9.1 | 19.8 | 12.9 | 3.0 | 0.8 | 0.8 | 3.0 |
| Mason Bryant | +0.873 | 145 | 28.3 | 11.0 | 1.4 | 9.0 | 24.1 | 16.6 | 2.1 | 0.0 | 2.8 | 4.8 |
| Dutch Landis | +0.849 | 148 | 25.0 | 10.1 | 4.1 | 8.1 | 29.1 | 14.2 | 3.4 | 0.0 | 2.7 | 3.4 |
| Mason Bryant | +0.798 | 44 | 15.9 | 13.6 | 2.3 | 4.5 | 31.8 | 18.2 | 2.3 | 2.3 | 2.3 | 6.8 |
| Evan Massie | +0.794 | 589 | 19.2 | 11.0 | 3.6 | 12.7 | 23.6 | 20.4 | 3.1 | 0.7 | 1.9 | 3.9 |

Bottom 10 (most negative on this axis):

| player | score | PA | K | BB | HBP | F | G | 1B | 2B | 3B | HR | OTHER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Greg Blackman | -1.050 | 175 | 24.6 | 7.4 | 2.9 | 30.9 | 9.7 | 13.1 | 2.9 | 0.0 | 5.1 | 3.4 |
| Steven Washilewski | -0.998 | 205 | 11.2 | 13.2 | 1.0 | 31.7 | 10.7 | 14.1 | 6.3 | 0.5 | 6.8 | 4.4 |
| Luke Wechsler | -0.793 | 83 | 12.0 | 13.3 | 4.8 | 31.3 | 7.2 | 21.7 | 6.0 | 0.0 | 2.4 | 1.2 |
| Carson Angeroth | -0.766 | 50 | 20.0 | 8.0 | 2.0 | 30.0 | 8.0 | 8.0 | 12.0 | 2.0 | 6.0 | 4.0 |
| Zach DeVito | -0.752 | 93 | 26.9 | 5.4 | 3.2 | 29.0 | 12.9 | 9.7 | 6.5 | 0.0 | 5.4 | 1.1 |
| Dawson Day | -0.750 | 302 | 32.1 | 11.6 | 2.3 | 22.8 | 10.9 | 10.6 | 5.6 | 0.7 | 2.0 | 1.3 |
| Jayden Drake | -0.739 | 197 | 16.2 | 9.1 | 2.0 | 33.0 | 15.7 | 13.7 | 5.1 | 0.0 | 3.6 | 1.5 |
| Jason Pineda | -0.734 | 327 | 17.1 | 11.0 | 5.2 | 26.6 | 13.5 | 12.8 | 6.1 | 0.9 | 4.3 | 2.4 |
| Jacob Hughes | -0.729 | 247 | 24.3 | 10.5 | 1.6 | 25.5 | 12.6 | 11.7 | 5.7 | 0.8 | 3.6 | 3.6 |
| Aydan Alger | -0.724 | 287 | 22.3 | 9.4 | 2.1 | 23.7 | 10.5 | 18.1 | 4.5 | 0.0 | 7.0 | 2.4 |
