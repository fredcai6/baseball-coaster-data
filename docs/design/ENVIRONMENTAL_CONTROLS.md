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

---

## 7. Recommended shape

Not sequential ratio adjustments. Park, opponent, and season are **correlated with each other** here
-- the schedule imbalance means a team's park mix and its opponent mix are not independent, and the
2026 roster churn (three franchises out, three in, one of them park-less) ties season to both. Chained
ratio corrections would double-count the overlap.

The right v1 is a **mixed-effects model over plate-appearance outcomes**: random intercepts for
batter, pitcher, and park; fixed effect for season; outcome as the multinomial event taxonomy already
in `events[].outcome.type`. That is DRC+'s structure minus the terms our data cannot support, and it
degrades gracefully -- when `bats_side` is eventually populated, the park x handedness interaction
drops straight in without restructuring.

Konaka's pairwise-comparison formulation is the fallback if the full mixed model proves too heavy:
it delivers the simultaneous park-and-opponent estimation that §3b and §3c demand, at logistic-
regression cost.
