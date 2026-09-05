# Design Note — environmental controls for player-impact estimation

**Status:** survey + recommendation. No pipeline code is proposed here yet.
**Motivation:** a separate effort estimates individual player impact on game outcomes. This note
inventories the confounding vectors we can actually control for *from this corpus*, ranks them by
confounding-reduction per unit of effort, and records what is data-blocked.

Every number below was measured on the corpus as of this commit (1,485 games, 1,483 with
play-by-play, seasons 2024-2026, 125,827 plate appearances). The measurement scripts are
throwaway; the numbers are reproducible from `games/**` alone.

---

## 0. The one-paragraph summary

Park is the dominant environmental confound in this league and it is **much larger than in MLB** --
a 37% spread in run value per plate appearance between the extreme parks, which survives controlling
for roster. It is also **not one number per park**: the park that inflates home runs most is the one
that suppresses balls-in-play hits most. On top of that sit a strong league-wide run inflation across
the three seasons and a schedule so unbalanced that opponent quality is a first-order confound rather
than a rounding error. Most published MLB prior art is usable to us as *method* but not as *values*,
and the single most-recommended shortcut in the literature -- join a pre-computed park-factor table --
does not exist for this league. We derive our own.

---

## 1. What the corpus does and does not carry

The schema (`schemas/game.schema.json`, currently 1.12.0) was designed backwards from the model
layer, and it carries the base-out spine that run-expectancy work needs. What it does **not** carry
is any physical-environment field at all.

Present and usable:

| Field | Path | Enables |
|---|---|---|
| Park identity | `teams.home.franchise_id` | park factors |
| Season | `season` | league-year fixed effects |
| Date | `date` | days rest, travel, month/seasonal drift |
| Base-out state | `events[]._derived.base_out_state`, `bases_before`, `outs_before` | RE24, context-neutral vs. context-dependent splits |
| Batter / pitcher identity | `events[].batter.player_id`, `events[].pitcher.player_id` | opponent-quality control, matchup terms |
| Cross-season identity | `players[].career_id`, `person_id` | multi-season player effects |
| Batting order slot | `lineups[].batting_order[].slot` | lineup-context control |
| Outcome taxonomy | `events[].outcome.type` (18 types) | component park factors |

Absent:

- **No `venue` field.** Park identity is inferred from the home franchise. This is sound only if
  every home game is played in that franchise's park. **We cannot detect a neutral-site game**, and
  would silently misattribute one. See §6.
- **No weather, temperature, wind, humidity, or roof state.** Every atmospheric vector in the
  literature is unavailable. Altitude is not in the data either, though it is a static per-franchise
  constant we could supply.
- **No umpire, attendance, or first-pitch time.** No day/night split is possible.
- **`players[].bats_side` exists in the schema but is `null` in all 41,713 player-game records.**
  Handedness and platoon control are therefore data-blocked today. This matters more than it looks:
  handedness-split park factors are the single most-recommended refinement in the modern literature.

`franchise_id` is a stable *identity* key: no franchise changed name across the three seasons. It is
**not** established as a stable *park* key. The corpus has no venue field, so a relocation is
invisible to it by construction -- a club that moved parks mid-corpus would look identical to one
that did not. External research on one club (`franchise:4e7dd733f635c973`, Yuba-Sutter Freebirds)
turned up a possible 2024 -> 2025 relocation that the corpus cannot confirm or deny; the run
environment is essentially unchanged across that boundary (11.9 R/G both years), while the club's
real environment shift is 2025 -> 2026. Unresolved. See `reference/venues.json`, which assigns a park
per *season* rather than per franchise precisely so this cannot be assumed away.

---

## 2. Measured: park is the dominant confound

The naive scoring spread is 11.93 to 21.66 runs per game between parks. That could be roster rather
than park, so the estimate below is **within-player**: each batter's linear-weight run value per PA
in a given park, against that same batter's rate across every *other* park (minimum 15 PA in-park and
50 PA elsewhere). This is the Baseball Savant estimator, and it avoids the roster-composition bias
that makes naive home/road ratios circular.

| Park | index | | Park | index |
|---|---|---|---|---|
| Grand Junction Jackalopes | 1.141 | | Billings Mustangs | 0.963 |
| Idaho Falls Chukars | 1.117 | | Oakland Ballers | 0.957 |
| Ogden Raptors | 1.063 | | Boise Hawks | 0.952 |
| Colorado Springs Sky Sox | 1.040 | | Yuba-Sutter Freebirds | 0.939 |
| Missoula PaddleHeads | 1.035 | | Modesto Roadsters | 0.892 |
| Rocky Mountain Vibes | 1.018 | | Glacier Range Riders | 0.858 |
| Great Falls Voyagers | 1.017 | | Long Beach Coast | 0.834 |

**These are reliable, not noise.** Splitting the corpus into two independent halves by game and
recomputing gives a split-half correlation of **r = 0.824** across the 14 parks, a Spearman-Brown
full-sample reliability of **0.904**. Pooled over three seasons, observed park deviations need only
about 10% shrinkage toward 1.00 on average -- far less than the heavy regression MLB single-season
component factors require. Shrinkage should still be applied **per park by sample size**, not
uniformly: Modesto (45 games) and Long Beach (44 games) have visibly wider half-to-half disagreement
than Missoula (136) or Boise (129).

### 2a. It must be component-wise

A single multiplier would mis-adjust in both directions, because the components diverge sharply:

| Park | HR/100 PA | BABIP | K% |
|---|---|---|---|
| Oakland Ballers | **5.17** (highest) | **.317** (near-lowest) | 21.8 |
| Rocky Mountain Vibes | **1.80** (lowest) | **.391** (highest) | 18.0 |
| Idaho Falls Chukars | 3.73 | .393 | **15.4** (lowest) |
| Glacier Range Riders | 2.55 | .342 | **22.7** (highest) |

Oakland is a home-run park that suppresses hits on balls in play; Rocky Mountain is the exact
inverse. Strikeout rate also swings 15.4% to 22.7% and tracks the run environment. A scalar park
factor applied to a slugger and a contact hitter in the same park would corrupt both. Component
factors (at minimum HR / BABIP / K / BB) are required here, notwithstanding that FanGraphs uses the
aggregate factor for MLB -- our component spread is much wider than theirs, so the usual argument
that components add variance faster than precision does not hold at this effect size.

### 2b. How the effect actually manifests, outcome by outcome

The park index in §2 is a run-value summary. Underneath it, the park is moving specific plate-appearance
outcomes, and not uniformly. Every outcome in the taxonomy was indexed within-batter, then re-indexed
**within-pitcher** as a control (see below), and checked for split-half reliability.

| Outcome | % of PA | index spread | reliability | batter-ctl vs pitcher-ctl |
|---|---|---|---|---|
| home_run | 3.32 | 1.169 | 0.94 | 0.97 |
| foul_out | 0.88 | 1.647 | 0.95 | 0.98 |
| popout | 3.00 | 1.221 | 0.92 | 0.98 |
| lineout | 2.60 | 1.160 | 0.90 | 0.96 |
| reached_on_error | 1.71 | 0.836 | 0.77 | 0.95 |
| strikeout_looking | 4.84 | 0.490 | 0.78 | 0.83 |
| double | 5.12 | 0.485 | 0.73 | 0.96 |
| strikeout_swinging | 13.80 | 0.470 | 0.82 | 0.94 |
| groundout | 15.35 | 0.360 | 0.81 | 0.87 |
| flyout | 12.78 | 0.323 | 0.84 | 0.97 |
| single | 17.40 | 0.311 | 0.85 | 0.90 |
| walk | 11.95 | 0.277 | 0.51 | 0.95 |

Read `spread` with care: it is max-minus-min of a *ratio*, so a rare outcome shows a wider spread for the
same absolute effect. The frequency column is there to keep that honest. Three outcomes are omitted as
unusable: bare `strikeout` (0.09% of PA, reliability 0.06), `fielders_choice` (reliability -0.23), and
`sacrifice` -- the last has a spectacular 22x spread on roughly 40 events corpus-wide, which is a warning
about small denominators, not a park effect.

**It is the park, not the home roster.** The obvious rival explanation is that a park's "effect" is really
its home team's batters and pitchers leaking through, since they play half their games there. The test that
separates these: compute each index twice, once holding the **batter** fixed and once holding the
**pitcher** fixed. Roster leakage would make the two disagree. They agree at **r = 0.83 to 0.98 on every
usable outcome** (0.94 for strikeout_swinging, 0.97 for home runs). These are properties of the venue.

This also corrected an earlier reading. Splitting the strikeout index by whether the batting side was home
or away gave only r = 0.55 between the two, which looked like roster contamination. The two-control test
says it is not; the home/away split is simply the noisier instrument.

### 2c. Stability across and within a season

A park factor is only useful if it holds still. Correlating each park's index across seasons, and across
thirds of a season (May-June / July / August):

| Outcome | across seasons | within season | verdict |
|---|---|---|---|
| home_run | 0.60 | 0.86 | stable both ways |
| foul_out | 0.72 | 0.71 | stable both ways |
| groundout | 0.64 | 0.65 | stable both ways |
| double | 0.75 | 0.55 | stable across years, noisy in-season |
| lineout | 0.71 | 0.56 | stable across years, noisy in-season |
| strikeout_looking | 0.70 | 0.20 | stable across years, noisy in-season |
| single | 0.59 | 0.62 | drifts across years |
| strikeout_swinging | 0.42 | 0.63 | drifts across years |
| walk | 0.02 | 0.06 | unstable -- no park effect |

Only home runs, foul outs and groundouts are dependable on both axes. **Walks are the clean negative
result**: smallest spread, weakest reliability, and no stability on either axis. That is a real finding
rather than a gap -- a walk is the outcome least mediated by the physical environment, so a walk-rate park
adjustment would be fitting noise. It also matches the published MLB pattern.

The two strikeout flavours split in an interesting way: *looking* is stable across years but not within
one, *swinging* is the reverse. Worth noting, not yet worth explaining -- with 14 parks per correlation
these coefficients carry wide error bars, and the split should be re-checked on more seasons before anyone
builds on it.

The practical consequence: pooled three-season factors are the right unit. Only a few components earn a
per-season factor, and walks should not get a park factor at all.

---

## 3. Measured: three structural confounds specific to this league

**3a. League-wide run inflation.** Runs per game go 15.09 -> 16.40 -> 18.11 across 2024-2026;
context-neutral run value per PA goes 0.3632 -> 0.3762 -> 0.3885, about +7% over three seasons. A
season fixed effect is mandatory regardless of cause. Within-player year-over-year change runs
higher than the league mean (+2.2% for 2024->2025, +9.2% for 2025->2026), which *hints* at a real
environment shift beyond roster turnover -- but n is only 26-28 qualifying players and the comparison
is contaminated by survivorship and aging, so **we should not attribute a cause** from this. The
literature's equivalents (annual wOBA-weight re-centering, the yearly FIP constant) absorb exactly
this kind of drift without explaining it, and that is the right posture for us.

**3b. Severe schedule imbalance.** In 2026 a team plays one opponent as few as 3 times and another
as many as 27. MLB park-factor methodology assumes a roughly balanced slate so that home and road
opponent quality cancel; that assumption fails badly here. Opponent quality is a first-order term,
not a refinement.

**3c. A team with no home park.** **RedPocket Mobiles played 83 games in 2026 and zero at home.**
This breaks any home/road-ratio park factor outright -- the denominator does not exist -- and it means
every RedPocket player's raw line is a mixture over the league's road parks with no home anchor.
The within-player estimator in §2 handles this correctly and the classic ratio estimator does not.
This alone is a sufficient reason to reject the naive method.

---

## 4. Prior art, and what transfers

Surveyed: park-factor lineage (naive ratio, multi-year regressed, Statcast handedness/batted-ball
splits, component factors, geometry- and weather-aware ML), atmospheric effects, ball/era effects,
rule-regime changes, opponent and usage context, and the modeling frameworks that net them out.

What transfers to us:

- **Konaka (2021), "Park factor estimation improvement using pairwise comparison method"**
  ([arXiv:2109.09287](https://arxiv.org/abs/2109.09287)) -- models each plate appearance as a
  contest between {batting team, pitching team, ballpark} in a logistic regression, estimating park
  and team-quality effects *simultaneously* rather than via a home/road ratio. This is the closest
  match to our situation: it is precisely the fix for §3b and §3c.
- **DRC+ (Baseball Prospectus)** -- the most rigorous published "separate player from context"
  framework. Its structure is directly portable: random effects on batter, pitcher, and
  park x handedness; fixed effects for temperature, time-through-order, and pitcher groundball rate;
  fit as a multinomial over event outcomes. We can implement the batter / pitcher / park random
  effects today; the handedness interaction and the temperature term we cannot.
  ([Entirely Beyond WOWY](https://www.baseballprospectus.com/news/article/48293/entirely-beyond-wowy-a-breakdown-of-drc/))
- **RE24 / linear weights / WPA-LI** -- the context-neutral vs. context-dependent distinction. Our
  `_derived.base_out_state` was designed for exactly this and needs no new data.
- **Statcast within-player park estimator** -- already adopted in §2, with its interpretive caveat:
  an index of 1.14 means *players who batted both there and elsewhere* produced 14% more, not that
  the park produces 14% more in the abstract.
- **Multi-year regression toward the mean** -- adopted in §2, but our measured reliability says we
  need much less of it than MLB single-season factors do.

What does **not** transfer:

- Pre-computed park-factor tables (Baseball Savant, FanGraphs Guts!) are MLB-only. No lookup shortcut
  exists for the Pioneer League; §2 is our table.
- Every atmospheric method (Callahan et al.'s ~2%/degC home-run effect, Nathan's drag/Magnus
  trajectory model, wind-vector and roof-state controls) is blocked -- we have no weather fields and
  no game-time timestamps to join external weather against. Worth noting the irony: this is a
  high-altitude league where those effects are likely *larger* than in MLB, and our park factors are
  silently absorbing them as a static per-park constant. That is an acceptable v1 conflation as long
  as we do not later claim the park term is pure geometry.
- Shift-alignment, umpire-zone, and catcher-framing controls all require pitch-level location data we
  do not have.
- MLB rule-regime flags (pitch clock, shift ban, universal DH, ghost runner) are irrelevant to this
  league's data; we would need this league's own rule history, which is not in the corpus.

---

## 5. Ranked vectors

**Tier 1 -- do these; high confound reduction, data already present.**

1. **Component park factors keyed on `franchise_id`**, estimated within-player, shrunk per park by
   sample size. Largest single confound (37% spread), measured reliability 0.90.
2. **Season fixed effect.** One categorical column; absorbs the +7% three-season run inflation and
   any unexplained regime shift. Essentially free.
3. **Opposing pitcher quality.** Elevated from its usual "nice refinement" rank because of §3b --
   with a 3-to-27 game imbalance, who you faced is a first-order term. A rolling contemporaneous
   quality proxy captures most of it without the full mixed model.
4. **Base-out / leverage context (RE24).** Table stakes, and the schema already carries it. This is
   what separates context-neutral value from context-dependent value; every framework surveyed is
   built on it.

**Tier 2 -- worth doing, moderate payoff.**

5. **Home/away.** Small measured effect on home runs (3.41 vs 3.22 per 100 PA for away vs. home
   batters), but free to include and it absorbs any residual home advantage the park term misses.
6. **Days rest and travel.** Derivable from `date` sequences per team; the PNAS jet-lag literature
   documents real effects, and this league's geography (sea-level California to 6,000-foot Colorado)
   makes travel more extreme than MLB's. Direction of travel needs a static city table.
7. **Batting-order slot.** Present in `lineups`; controls for plate-appearance quality and
   opportunity rather than skill.

**Tier 3 -- data-blocked, listed so we know what we are missing.**

8. Platoon / handedness -- blocked by `bats_side` being universally null. **Highest-value unblock.**
9. Temperature, wind, humidity, altitude -- no fields. Altitude is the cheapest to add (a static
   per-franchise constant) and would let us test whether park effects are altitude-driven.
10. Umpire, catcher framing, defensive positioning -- no pitch-location data.
11. Extra-innings context -- **there are no extra innings in this league.** A tied game is settled by
    a home-run derby, which is not baseball played under the run-expectancy model and must never be
    binned as a 10th inning. See §6.

---

## 6. Data-quality findings surfaced by this survey

Three items found while measuring, each arguably its own issue:

1. **No `venue` field.** We cannot distinguish a home game played in the home park from one played
   at a neutral site, and would misattribute the latter. Recommend adding `venue` (or at minimum a
   `neutral_site` boolean) to the schema as an additive MINOR bump.
2. **`bats_side` is null in 100% of records** (41,713 of 41,713) despite being in the schema.
   Populating it unlocks the handedness-split park factors that the modern literature treats as the
   default refinement.
3. **The 10th inning is a home-run derby, and it is not parsed.** This league settles a tie with a
   home-run derby rather than extra innings (repo owner, 2026-08-31), and the parser does not yet
   handle it. What lands in the data instead: **60 games carry a 10th linescore column, and all 60
   are tied after regulation.** The 10th column is `0` for both clubs in every one of them, and the
   linescore innings always sum to the recorded totals -- so the derby result is nowhere in the run
   data. A subset of those games also carries stray inning-10 `substitution` and `inning_summary`
   events (one reads `0 Runs, 0 Hits, 0 Errors, 1 LOB`, which is the derby being mangled into an
   inning shape), but **zero plate appearances exist past the 9th anywhere in the corpus.**

   Two consequences, and the second is the one that bites:

   - *For rate work, the data is clean.* Derby outcomes never enter the run totals or the PA stream,
     so per-PA and per-game run environments are uncontaminated. The park factors in §2 are safe.
   - *For any win-based player-impact model, 82 games have no recoverable winner.* Across the corpus
     **82 games end with equal runs** (26 in 2024, 27 in 2025, 29 in 2026 -- 5.5% of all games), and
     the derby that actually decided them is not represented. A model keyed on wins will silently
     treat these as ties or drop them. They should be explicitly excluded, or the derby captured at
     parse time, before anyone fits on game outcomes.

   Note the derby is also a *player* event -- someone hit those home runs -- so parsing it would add
   real signal, not just tidy the record. But it is a fundamentally different skill sample than a
   plate appearance and must not be pooled into ordinary batting lines.

A venue reference table now exists at `reference/venues.json` (validated by
`scripts/build_venues.py`), keyed franchise -> season -> park, carrying city, coordinates, elevation
and an approximate field orientation. It does **not** close finding 1: it records which park a club
called home, not where a given game was played, so it still cannot see a neutral-site game. Each
season row flags via `park_season_link` whether its park was researched for that specific year or
carried forward unverified.

Also noted, not defects: 42 doubleheader dates, and roughly 120 games shortened to 7 innings. These
affect per-game denominators but not per-PA rates, so they are harmless to rate-based adjustment and
should simply not be pooled into any per-game measure.

### 2d. Two different objects, not two venue properties

Splitting each venue's effect by which side was batting, with the home-vs-visiting *difference*
tested directly (full game-clustered covariance between the two sides, since they share games), gives
20 real gaps out of 112 park x outcome tests after Benjamini-Hochberg at 10% FDR -- against about 6
expected by chance. Where they fall inverts the park-effect ranking:

| Outcome | park spread ÷ CI | real home-away gaps | home advantage persists across seasons? |
|---|---|---|---|
| home_run | 4.8x | **0 / 14** | -- |
| strikeout | 4.8x | 3 / 14 | no (r 0.38, 0.13, -0.45) |
| popout | 4.0x | 5 / 14 | **yes (r 0.43, 0.73, 0.25)** |
| groundout | 2.3x | 3 / 14 | weak |
| walk | 1.7x | **6 / 14** | **no (r 0.08, 0.21, 0.14)** |

Home runs have the largest venue effect in the corpus and **not one park favours a dugout** -- exactly
how a physical property of a building should behave. Walks are the mirror image: no venue effect worth
modelling, the most home-away gaps, and those gaps **do not replicate across seasons**. Per-season
estimates are noisy but not that noisy: with a true between-club sd near 0.18 and a per-season standard
error near 0.09, a stable effect would still correlate around 0.8 across years. Observing 0.1 means the
walk home-advantage is a team-season transient, not a standing property of a club or its park.

**These are therefore two different kinds of object, and only one of them is a venue property.**

The exception is `popout`, the one outcome whose home-away split *does* persist year over year while
its roster turns over. A park-attached, home-favouring, season-stable effect on the single outcome most
exposed to scorer discretion (popout vs. flyout is a judgement call, and a park's scoring crew is the
same people each year) is far better explained as observer bias than as baseball. It is the strongest
reason to treat the `popout` column as a data-quality finding rather than a park factor, and it would
survive every statistical control in this note.

---

## 7. Recommended shape

Not sequential ratio adjustments. Park, opponent, and season are **correlated with each other** here
-- the schedule imbalance means a team's park mix and its opponent mix are not independent, and the
2026 roster churn (three franchises out, three in, one of them park-less) ties season to both. Chained
ratio corrections would double-count the overlap.

The right v1 is a **mixed-effects model over plate-appearance outcomes**: random intercepts for
batter, pitcher, and venue; fixed effect for season; outcome as the multinomial event taxonomy already
in `events[].outcome.type`. Prefer one multinomial over eight independent binomials -- the outcomes are
mutually exclusive categories of a single plate appearance, and separate binomials do not constrain
them to sum to one. That is DRC+'s structure minus the terms our data cannot support.

**Split "home" in two: batting last is structural, playing at your own park is not.** A neutral-site
game separates them, and they behave differently there -- the designated home club still bats last,
but gets none of the familiarity. Bundling them into one `home` flag asserts a venue effect at a
neutral site, which is wrong.

The structural half is real and measurable here. Home clubs take **3.17% fewer plate appearances**
(61,901 vs 63,926, a 2,025-PA deficit) because **42.6% of games never bat the bottom of the 9th** --
the home side was ahead. That is an *opportunity* effect, not a rate effect: it leaves per-PA rates
untouched but systematically shortchanges the home club on any counting stat. Any impact metric
expressed as a total rather than a rate inherits it, and it has nothing to do with the ballpark.

The familiarity half is what should carry a venue condition. So:

- `bats_last` -- a global fixed effect, true wherever the game is played, including a neutral site.
- `at_own_venue` -- **not** `is_home_team`. A team deviation, structurally zero for a neutral-site
  game and for a club with no home park.

In this corpus the two are **perfectly confounded**, because we cannot detect a neutral-site game
(§6, finding 1) and so `designated home == at own venue` by assumption. That is the third distinct
thing blocked on the missing `venue` field, and the reason to write the term as `at_own_venue` now:
when venue arrives, neutral-site games get the right treatment without a restructure. Worth noting
the walk advantage is not concentrated in late innings (+0.049 early vs +0.069 from the 7th on), so
it is not a last-licks leverage artifact -- it sits in the half that should carry the venue condition.

**Home advantage is a team term, not a venue term, and the two cannot both be fitted.** Because each
franchise maps 1:1 to its park within a season, a `venue x side` effect and a `team x home` effect are
the same column of the design matrix wearing different labels; fitting both leaves the model
unidentified. Choose `team-season x home`, for four reasons: §2d shows the split is a team-season
transient rather than a venue property; a club plays road games at many parks, so its home effect is
identified against the venue main effect, which `venue x side` is not; a relocating franchise then
keeps its home effect while the park keeps its venue effect (the reason `reference/venues.json` is
keyed per season); and a travelling club with no home games simply shrinks to the prior instead of
leaving a hole. Add a global home fixed effect too -- small but real for several outcomes (walks
+0.049 log-odds, 95% CI [+0.024, +0.074]) -- while remembering the between-club sd is about 4x that,
so the league constant is the least useful part of it.

**Venue x handedness is a genuine venue property** and is the interaction to add once `bats_side` is
populated -- asymmetric-geometry physics, unrelated to the home-away split above.

**Nest every one of these as a deviation, never as a free-standing effect.** Written as
`(1 | venue) + (1 | venue:bat_side)`, the interaction has its own variance component and is shrunk
toward zero, so it only picks up what the venue main effect leaves behind; the two are not competing
on equal footing, and the pair cannot thrash. The main effect is estimated from every PA at the venue,
the deviation only from the handedness contrast within it. The same discipline applies to the home
terms (`bats_last` fixed, `at_own_venue` as a team deviation) and it buys a useful safety property:
where a split carries no signal its variance component collapses to roughly zero and the term
self-disables, rather than absorbing noise the main effect had already explained. That is the
mechanism that lets us include a term like `venue:bat_side` before knowing whether these parks are
asymmetric at all -- if they are not, the model says so by shrinking it away.

Konaka's pairwise-comparison formulation is the fallback if the full mixed model proves too heavy:
it delivers the simultaneous park-and-opponent estimation that §3b and §3c demand, at logistic-
regression cost.
