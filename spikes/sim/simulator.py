"""Pre-game / in-game Monte Carlo game simulator.

Loads the artifacts fit_shaped.py and build_empirical.py produced
(spikes/sim/shaped_train.npz, spikes/sim/empirical_transitions.npz) and
simulates a full game (or the remainder of one, from an arbitrary state) many
times, with a pluggable advancement model ("naive" or "empirical").

Design used to make N=2000 sims/game fast: within a single game, the SHAPE D
category-probability distribution for a plate appearance depends only on
(which of the 9 lineup slots is batting, which of exactly two pitchers --
that team's actual starter, or the single "generic reliever" -- is on the
mound). Both are known before any random draw is made. So per game we
precompute an (9 slots x 2 roles x 10 categories) probability tensor ONCE for
each side, and the hot Monte Carlo loop is pure table lookups + categorical
draws -- no per-PA link-function evaluation, no Python-level loop over
players. This is what makes 2000 sims x 298 test games x 2 advancement models
tractable in the time budget.

Bullpen policy (STATED, per task brief): each side has exactly two pitchers
in the simulation -- the real starter (his own fitted per-node effect and
throwing hand), who pitches until the number of batters he has faced in the
SIMULATED game reaches his ACTUAL pitcher_bf from the real game; then a
single "generic reliever" for the rest of the game, whose per-node effect is
the PA-weighted mean effect over that team's actual TRAIN relief appearances,
and whose throwing hand is treated as a mixture (weighted by the same
PA-weighted mix) rather than a single hand -- see fit_shaped.py. No
pinch-hitters, no mid-game lineup changes beyond the one pitching change.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPIKES = HERE.parent
sys.path.insert(0, str(SPIKES))
sys.path.insert(0, str(SPIKES / "pitch"))
sys.path.insert(0, str(SPIKES / "fuse"))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import step1 as S1  # noqa: E402  (ao_prob, reused unmodified)
from analyze import structural  # noqa: E402
from shape_d import PATHS_D, N_NODES  # noqa: E402

CI = common.CAT_INDEX
CATS = common.CATEGORIES
K, BB, HBP, F, G, ONEB, TWOB, THREEB, HR, OTHER = (CI[c] for c in CATS)

END_BASES = 8  # sentinel used by the empirical table for "half ends here"


# ============================================================================
# model / artifact loading
# ============================================================================
class SimModel:
    def __init__(self, sim_dir=HERE):
        z = np.load(sim_dir / "shaped_train.npz", allow_pickle=True)
        self.node_names = list(z["node_names"])
        self.season_list = [int(s) for s in z["season_list"]]
        self.season_idx = {s: i for i, s in enumerate(self.season_list)}
        self.bats = list(z["bats"])
        self.pits = list(z["pits"])
        self.BI = {b: i for i, b in enumerate(self.bats)}
        self.PI = {p: i for i, p in enumerate(self.pits)}
        self.n_bat, self.n_pit = len(self.bats), len(self.pits)
        self.node_alpha = z["node_alpha"]
        self.node_beta = z["node_beta"]
        self.node_b = z["node_b"]     # (N_NODES, n_bat+1)  last col = OOV = 0
        self.node_q = z["node_q"]     # (N_NODES, n_pit+1)
        self.node_psi = z["node_psi"]
        self.teams = list(z["teams"])
        self.TI = {t: i for i, t in enumerate(self.teams)}
        self.team_reliever_q = z["team_reliever_q"]         # (n_teams, N_NODES)
        self.team_reliever_hand = z["team_reliever_hand"]   # (n_teams, 3) L,R,unknown
        self.league_avg_reliever_q = self.team_reliever_q.mean(axis=0)
        self.league_avg_reliever_hand = self.team_reliever_hand.mean(axis=0)
        self.dp_prob = float(z["dp_prob"])
        self.other_reach_prob = float(z["other_reach_prob"])
        self.other_sac_prob = float(z["other_sac_prob"])
        self.single_2nd_scores_prob = float(z["single_2nd_scores_prob"])

        e = np.load(sim_dir / "empirical_transitions.npz", allow_pickle=True)
        self.emp_bases_after = e["bases_after"]
        self.emp_outs_after = e["outs_after"]
        self.emp_runs = e["runs"].astype(np.int64)
        self.emp_cumprob = e["cumprob"]
        self.emp_n_obs = e["n_obs"]
        self.emp_thin = e["thin"]
        self.emp_max_bins = int(e["max_bins"])


def node_category_probs(model, Xs_row, b_vec, q_vec):
    eta = model.node_alpha + Xs_row @ model.node_beta.T + b_vec + q_vec
    p, _, _, _ = S1.ao_prob(eta, model.node_psi)
    cat_p = np.ones(10)
    for ci in range(10):
        for ni, is_pos in PATHS_D[ci]:
            cat_p[ci] *= p[ni] if is_pos else (1.0 - p[ni])
    return cat_p


def compute_cat_table(model, batter_ids, batter_bats, opp_starter_id, opp_starter_throws,
                       opp_team_id, season, batting_at_home_park, season_idx):
    """(9 lineup slots x 2 pitching roles x 10 categories). Role 0 = opposing
    starter (real individual). Role 1 = generic reliever: q is the team's
    PA-weighted mean reliever effect; the throwing-hand covariate is a
    PA-weighted MIXTURE over L/R/unknown-throws relievers (not a single
    assumed hand) -- both computed once, offline, in fit_shaped.py."""
    n_bat, n_pit = model.n_bat, model.n_pit
    starter_pidx = model.PI.get(opp_starter_id, n_pit)
    team_idx = model.TI.get(opp_team_id)
    if team_idx is None:
        reliever_q = model.league_avg_reliever_q
        reliever_hand = model.league_avg_reliever_hand
    else:
        reliever_q = model.team_reliever_q[team_idx]
        reliever_hand = model.team_reliever_hand[team_idx]

    table = np.zeros((9, 2, 10))
    for slot in range(9):
        bid, bats = batter_ids[slot], batter_bats[slot]
        bidx = model.BI.get(bid, n_bat)
        b_vec = model.node_b[:, bidx]

        row0 = [{"batting_at_home_park": batting_at_home_park, "bats": bats,
                 "throws": opp_starter_throws, "season": season}]
        Xs0 = structural(row0, season_idx)[0]
        table[slot, 0] = node_category_probs(model, Xs0, b_vec, model.node_q[:, starter_pidx])

        p1 = np.zeros(10)
        for throws_letter, w in zip(("L", "R", None), reliever_hand):
            if w <= 0:
                continue
            rowr = [{"batting_at_home_park": batting_at_home_park, "bats": bats,
                     "throws": throws_letter, "season": season}]
            Xsr = structural(rowr, season_idx)[0]
            p1 += w * node_category_probs(model, Xsr, b_vec, reliever_q)
        table[slot, 1] = p1
    return table


# ============================================================================
# game context
# ============================================================================
@dataclass
class GameContext:
    game_id: str
    season: int
    home_team: str
    away_team: str
    n_innings: int
    home_starter_id: str
    home_starter_max_bf: int
    away_starter_id: str
    away_starter_max_bf: int
    cat_table_away: np.ndarray  # (9,2,10) -- away batters vs home pitching
    cat_table_home: np.ndarray  # (9,2,10) -- home batters vs away pitching


def build_game_context(model, game_rows):
    if not game_rows:
        return None
    season = game_rows[0]["season"]
    home_team = game_rows[0]["home_team"]
    away_rows = [r for r in game_rows if not r["batting_is_home"]]
    home_rows = [r for r in game_rows if r["batting_is_home"]]
    if not away_rows or not home_rows:
        return None
    away_team = away_rows[0]["batting_team"]
    n_innings = max(r["inning"] for r in game_rows)

    def first_by_slot(rows_):
        seen = {}
        for r in sorted(rows_, key=lambda x: x["seq"]):
            s = r["order_slot"]
            if s is None or s in seen:
                continue
            seen[s] = r
        if set(seen) != set(range(1, 10)):
            return None
        return [seen[i] for i in range(1, 10)]

    away_lineup = first_by_slot(away_rows)
    home_lineup = first_by_slot(home_rows)
    if away_lineup is None or home_lineup is None:
        return None

    def find_starter(rows_):
        cand = [r for r in rows_ if r["is_starter"]]
        if not cand:
            return None
        pid = cand[0]["pitcher"]
        own = [r for r in cand if r["pitcher"] == pid]
        max_bf = max(r["pitcher_bf"] for r in own)
        throws = next((r["throws"] for r in own if r["throws"]), None)
        return pid, max_bf, throws

    home_starter = find_starter(away_rows)   # home team pitches while away bats
    away_starter = find_starter(home_rows)
    if home_starter is None or away_starter is None:
        return None
    home_starter_id, home_starter_max_bf, home_starter_throws = home_starter
    away_starter_id, away_starter_max_bf, away_starter_throws = away_starter

    away_park = away_rows[0]["batting_at_home_park"]
    home_park = home_rows[0]["batting_at_home_park"]

    away_batters = [r["batter"] for r in away_lineup]
    away_bats = [r["bats"] for r in away_lineup]
    home_batters = [r["batter"] for r in home_lineup]
    home_bats = [r["bats"] for r in home_lineup]

    cat_table_away = compute_cat_table(model, away_batters, away_bats, home_starter_id,
                                        home_starter_throws, home_team, season, away_park,
                                        model.season_idx)
    cat_table_home = compute_cat_table(model, home_batters, home_bats, away_starter_id,
                                        away_starter_throws, away_team, season, home_park,
                                        model.season_idx)

    return GameContext(
        game_id=game_rows[0]["game_id"], season=season, home_team=home_team,
        away_team=away_team, n_innings=n_innings,
        home_starter_id=home_starter_id, home_starter_max_bf=home_starter_max_bf,
        away_starter_id=away_starter_id, away_starter_max_bf=away_starter_max_bf,
        cat_table_away=cat_table_away, cat_table_home=cat_table_home,
    )


# ============================================================================
# advancement models
# ============================================================================
def _decode(codes):
    return (codes & 1).astype(bool), (codes & 2).astype(bool), (codes & 4).astype(bool)


def _encode(b0, b1, b2):
    return (b0.astype(np.uint8) | (b1.astype(np.uint8) << 1) | (b2.astype(np.uint8) << 2))


def advance_naive(cat, bases_code, outs, rng, model):
    """Deterministic-rule advancement. See SIM.md for the full rule table.
    Operates on 1-D arrays (the currently-active subset of sims)."""
    n = len(cat)
    occ1, occ2, occ3 = _decode(bases_code)
    new1, new2, new3 = occ1.copy(), occ2.copy(), occ3.copy()
    new_outs = outs.copy()
    runs = np.zeros(n, dtype=np.int64)

    # K, F: one out, runners hold.
    m = (cat == K) | (cat == F)
    new_outs[m] += 1

    # G: one out, runners hold, EXCEPT a fixed double-play prob when a runner
    # is on 1st and <2 outs (batter + lead runner both out, runner removed).
    m_g = (cat == G)
    new_outs[m_g] += 1
    dp_elig = m_g & occ1 & (outs < 2)
    dp_hit = dp_elig & (rng.random(n) < model.dp_prob)
    new_outs[dp_hit] += 1
    new1[dp_hit] = False

    # BB, HBP: force advance only (cascading force from the batter up).
    m_bb = (cat == BB) | (cat == HBP)
    if m_bb.any():
        loaded = occ1 & occ2 & occ3
        new1[m_bb] = True
        new2[m_bb] = (occ2 | occ1)[m_bb]
        new3[m_bb] = (occ3 | (occ1 & occ2))[m_bb]
        runs[m_bb & loaded] += 1

    # 1B: batter to 1st, runner on 1st always to 2nd, runner on 3rd always
    # scores, runner on 2nd scores with a fixed TRAIN-estimated probability
    # (else advances to 3rd).
    m_1b = (cat == ONEB)
    if m_1b.any():
        take2 = rng.random(n) < model.single_2nd_scores_prob
        _apply_single_like(m_1b, occ1, occ2, occ3, new1, new2, new3, runs, take2)

    # 2B: batter to 2nd, runner on 1st to 3rd, runners on 2nd/3rd score.
    m_2b = (cat == TWOB)
    if m_2b.any():
        new1[m_2b] = False
        new2[m_2b] = True
        new3[m_2b] = occ1[m_2b]
        runs[m_2b] += (occ2.astype(np.int64) + occ3.astype(np.int64))[m_2b]

    # 3B: everyone scores, batter to 3rd.
    m_3b = (cat == THREEB)
    if m_3b.any():
        runs[m_3b] += (occ1.astype(np.int64) + occ2.astype(np.int64) + occ3.astype(np.int64))[m_3b]
        new1[m_3b] = False
        new2[m_3b] = False
        new3[m_3b] = True

    # HR: everyone scores including the batter, bases empty.
    m_hr = (cat == HR)
    if m_hr.any():
        runs[m_hr] += (1 + occ1.astype(np.int64) + occ2.astype(np.int64)
                       + occ3.astype(np.int64))[m_hr]
        new1[m_hr] = False
        new2[m_hr] = False
        new3[m_hr] = False

    # OTHER (error / fielder's choice / sac / interference, conflated): with
    # a fixed TRAIN-estimated prob the batter reaches like a single (this is
    # our stated treatment of reached_on_error / fielders_choice, which
    # OUTCOME_MAP also buckets here); otherwise the batter is out and, if a
    # runner is on 3rd with <2 outs, that runner scores with a second fixed
    # TRAIN-estimated prob (our stated treatment of the sac-fly case OTHER
    # also conflates in). All other runners hold in the "batter out" branch.
    m_oth = (cat == OTHER)
    if m_oth.any():
        reach = rng.random(n) < model.other_reach_prob
        m_reach = m_oth & reach
        if m_reach.any():
            take2 = rng.random(n) < model.single_2nd_scores_prob
            _apply_single_like(m_reach, occ1, occ2, occ3, new1, new2, new3, runs, take2)
        m_out = m_oth & ~reach
        if m_out.any():
            new_outs[m_out] += 1
            sac_elig = m_out & occ3 & (outs < 2)
            sac_hit = sac_elig & (rng.random(n) < model.other_sac_prob)
            runs[sac_hit] += 1
            new3[sac_hit] = False

    return _encode(new1, new2, new3), new_outs, runs


def _apply_single_like(mask, occ1, occ2, occ3, new1, new2, new3, runs, take2_draw):
    """Shared 1B-shaped advancement (used for real 1B and for OTHER's
    'batter reaches' branch): batter to 1st, 1st-runner always to 2nd,
    3rd-runner always scores, 2nd-runner scores w.p. take2_draw else to 3rd."""
    idx = mask
    runs[idx] += occ3[idx].astype(np.int64)
    scored2 = idx & occ2 & take2_draw
    runs[scored2] += 1
    new1[idx] = True
    new2[idx] = occ1[idx]
    new3[idx] = (occ2 & ~take2_draw)[idx]


def advance_empirical(cat, bases_code, outs, rng, model):
    """Table lookup: P(bases_after, outs_after, runs | bases_before,
    outs_before, category), built by build_empirical.py from TRAIN games."""
    n = len(cat)
    key = bases_code.astype(np.int64) * 30 + outs.astype(np.int64) * 10 + cat.astype(np.int64)
    cum = model.emp_cumprob[key]              # (n, max_bins)
    u = rng.random(n)
    bin_idx = (u[:, None] < cum).argmax(axis=1)
    rows = np.arange(n)
    new_bases = model.emp_bases_after[key[rows], bin_idx]
    new_outs = model.emp_outs_after[key[rows], bin_idx]
    runs = model.emp_runs[key[rows], bin_idx]
    # END sentinel already carries outs_after == 3, so no separate handling needed.
    return new_bases.astype(np.uint8), new_outs.astype(np.int64), runs.astype(np.int64)


# ============================================================================
# Monte Carlo engine
# ============================================================================
def simulate_from_state(model, ctx, *, inning, half, outs, bases, away_score, home_score,
                         away_lineup_pos, home_lineup_pos, away_pitcher_bf, home_pitcher_bf,
                         advancement, n_sims, rng, n_innings=None):
    """Simulate a game to completion from an arbitrary state. `half` is
    'top' or 'bottom'. `bases` is a (b1,b2,b3) bool tuple. `*_lineup_pos` is
    the 0-based index of the NEXT batter due up for that side. `*_pitcher_bf`
    is the number of batters that side's CURRENT pitcher has already faced
    in the (simulated) game so far. Returns a dict of length-n_sims arrays:
    away_runs, home_runs, home_win (bool, undefined/False on a tie),
    tie (bool)."""
    N = n_sims
    n_innings = n_innings or ctx.n_innings
    adv = advance_naive if advancement == "naive" else advance_empirical

    away_runs = np.full(N, away_score, dtype=np.int64)
    home_runs = np.full(N, home_score, dtype=np.int64)
    alp = np.full(N, away_lineup_pos, dtype=np.int64)
    hlp = np.full(N, home_lineup_pos, dtype=np.int64)
    bf_home = np.full(N, home_pitcher_bf, dtype=np.int64)  # faced by HOME pitcher (vs away batters)
    bf_away = np.full(N, away_pitcher_bf, dtype=np.int64)  # faced by AWAY pitcher (vs home batters)

    start_half_idx = 0 if half == "top" else 1
    seq = []
    for inn in range(inning, n_innings + 1):
        for h in (0, 1):
            if inn == inning and h < start_half_idx:
                continue
            seq.append((inn, h))

    init_bases_code = np.uint8((1 if bases[0] else 0) | (2 if bases[1] else 0) | (4 if bases[2] else 0))
    init_outs = np.int64(outs)

    for step_i, (inn, h) in enumerate(seq):
        is_last_half = (inn == n_innings and h == 1)
        if h == 0:
            lineup_pos, pitcher_bf, starter_max_bf = alp, bf_home, ctx.home_starter_max_bf
            cat_table, runs_acc, other_runs_acc = ctx.cat_table_away, away_runs, home_runs
        else:
            lineup_pos, pitcher_bf, starter_max_bf = hlp, bf_away, ctx.away_starter_max_bf
            cat_table, runs_acc, other_runs_acc = ctx.cat_table_home, home_runs, away_runs

        play_mask = np.ones(N, dtype=bool)
        if is_last_half:
            # home team already leading after the top of the last inning:
            # bottom is not played at all.
            play_mask = ~(home_runs > away_runs)

        if step_i == 0:
            bases_arr = np.where(play_mask, init_bases_code, np.uint8(0)).astype(np.uint8)
            outs_arr = np.where(play_mask, init_outs, np.int64(0)).astype(np.int64)
        else:
            bases_arr = np.zeros(N, dtype=np.uint8)
            outs_arr = np.zeros(N, dtype=np.int64)

        still = play_mask.copy()
        while still.any():
            idx = np.where(still)[0]
            slot = (lineup_pos[idx] % 9).astype(np.int64)
            role = (pitcher_bf[idx] >= starter_max_bf).astype(np.int64)
            probs = cat_table[slot, role]           # (n_active, 10)
            cum = probs.cumsum(axis=1)
            cum[:, -1] = 1.0
            u = rng.random(len(idx))
            cat = (u[:, None] < cum).argmax(axis=1)

            new_bases, new_outs, runs = adv(cat, bases_arr[idx], outs_arr[idx], rng, model)

            runs_acc[idx] += runs
            pitcher_bf[idx] += 1
            lineup_pos[idx] += 1
            bases_arr[idx] = new_bases
            outs_arr[idx] = new_outs

            ended = new_outs >= 3
            if is_last_half:
                ended = ended | (home_runs[idx] > away_runs[idx])
            still[idx[ended]] = False

    tie = away_runs == home_runs
    home_win = (home_runs > away_runs) & ~tie
    return dict(away_runs=away_runs, home_runs=home_runs, home_win=home_win, tie=tie)


def simulate_game(model, ctx, advancement, n_sims, rng):
    """Pregame entry point -- a call to simulate_from_state from the true
    initial state of the game."""
    return simulate_from_state(
        model, ctx, inning=1, half="top", outs=0, bases=(False, False, False),
        away_score=0, home_score=0, away_lineup_pos=0, home_lineup_pos=0,
        away_pitcher_bf=0, home_pitcher_bf=0, advancement=advancement,
        n_sims=n_sims, rng=rng)
