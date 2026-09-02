"""naive_v0 / naive_v1 deterministic-ish base-advancement rule engines.

naive_v0: no strategy baked in at all.
  K, F, G, OTHER  -> one out, runners hold, 0 runs.
  BB, HBP         -> force advance only (standard force-advance algebra).
  1B              -> every existing runner +1 base; on1->2b, on2->3b,
                     on3->scores; batter to 1st.
  2B              -> every existing runner +2 bases; on1->3b, on2/on3 score;
                     batter to 2nd.
  3B              -> everyone on base scores; batter to 3rd.
  HR              -> everyone including batter scores.

naive_v1 = naive_v0 plus three fixed-probability refinements, each
estimated from TRAIN only (see fit_params below):
  (1) on a 1B, a runner who started on 2nd scores instead of stopping at
      3rd with probability p_score_from_2nd_on_1b.
  (2) a G with a runner on 1st and <2 outs is a double play (batter AND the
      lead forced runner both out; 2 outs charged, at most one runner left
      on base at the state the naive_v0 rule would otherwise have produced)
      with probability p_dp_on_groundout.
  (3) an OTHER PA flagged is_sac (raw outcome_type in
      {sacrifice, reached_on_error, fielders_choice} with a "sac" modifier)
      advances runners by one base (like a 1B's runner movement, minus the
      batter reaching) instead of holding. This is deterministic given the
      is_sac flag -- not a fitted probability -- see ADVANCE.md for the
      one game-log cross-check that motivated the rule (bases_before
      [1,1,0] -> bases_after [0,1,1] on a real sacrifice, i.e. each runner
      +1, batter out).

All advancement functions return (new_bases: tuple[bool,bool,bool],
outs_added: int, runs: int). Callers are responsible for capping outs
at 3 and stopping the half there.
"""
from __future__ import annotations

import random


def force_advance(bases):
    on1, on2, on3 = bases
    forced2 = on1
    forced3 = on1 and on2
    forced_home = on1 and on2 and on3
    n1 = True
    n2 = True if forced2 else on2
    n3 = True if forced3 else on3
    runs = 1 if forced_home else 0
    return (n1, n2, n3), runs


def advance_one_each(bases, batter_reaches_first):
    """Every existing runner moves up exactly one base; on3 scores.
    batter_reaches_first=True for a single, False for a sac-advance."""
    on1, on2, on3 = bases
    runs = 1 if on3 else 0
    n1 = bool(batter_reaches_first)
    n2 = on1
    n3 = on2
    return (n1, n2, n3), runs


def advance_two_each(bases):
    on1, on2, on3 = bases
    runs = (1 if on2 else 0) + (1 if on3 else 0)
    n1 = False
    n2 = True  # batter
    n3 = on1
    return (n1, n2, n3), runs


def advance_three_each(bases):
    on1, on2, on3 = bases
    runs = (1 if on1 else 0) + (1 if on2 else 0) + (1 if on3 else 0)
    return (False, False, True), runs  # batter on 3rd


def advance_hr(bases):
    on1, on2, on3 = bases
    runs = (1 if on1 else 0) + (1 if on2 else 0) + (1 if on3 else 0) + 1
    return (False, False, False), runs


def apply_v0(bases, outs_before, cat, is_sac=False, rng=None):
    """Deterministic naive_v0. rng accepted for a uniform call signature
    with apply_v1 but unused."""
    if cat in ("K", "F", "G", "OTHER"):
        return bases, 1, 0
    if cat in ("BB", "HBP"):
        nb, runs = force_advance(bases)
        return nb, 0, runs
    if cat == "1B":
        nb, runs = advance_one_each(bases, True)
        return nb, 0, runs
    if cat == "2B":
        nb, runs = advance_two_each(bases)
        return nb, 0, runs
    if cat == "3B":
        nb, runs = advance_three_each(bases)
        return nb, 0, runs
    if cat == "HR":
        nb, runs = advance_hr(bases)
        return nb, 0, runs
    raise ValueError(cat)


def apply_v1(bases, outs_before, cat, is_sac, params, rng):
    """Stochastic naive_v1 -- draws from rng (a random.Random) for the two
    fitted-probability refinements. Caller should average over many draws
    (or many replays) to get an expectation."""
    p2 = params["p_score_from_2nd_on_1b"]
    pdp = params["p_dp_on_groundout"]

    if cat == "OTHER" and is_sac:
        nb, runs = advance_one_each(bases, False)
        return nb, 1, runs  # batter still out

    if cat == "G":
        on1 = bases[0]
        if on1 and outs_before < 2 and rng.random() < pdp:
            # double play: batter + lead forced runner both out; the forced
            # runner (from 1st) is removed, other runners hold.
            on1b, on2, on3 = bases
            return (False, on2, on3), 2, 0
        return bases, 1, 0

    if cat == "1B":
        on1, on2, on3 = bases
        if on2 and rng.random() < p2:
            # runner from 2nd scores instead of stopping at 3rd
            runs = (1 if on3 else 0) + 1
            return (True, on1, False), 0, runs
        nb, runs = advance_one_each(bases, True)
        return nb, 0, runs

    return apply_v0(bases, outs_before, cat, is_sac, rng)


def fit_params(train_records):
    """Estimate the two fixed probabilities from TRAIN PA records (as built
    by records.py). Uses the actual game-log ground truth (bases_after /
    outs_recorded), not the naive rule's own predictions."""
    # (1) P(runner from 2nd scores | 1B, runner was on 2nd before), isolated
    # to the clean before-state 010 (only runner on 2nd) so we're not
    # confounding with simultaneous movement of other runners.
    n2, scored2 = 0, 0
    for r in train_records:
        if r["cat"] != "1B":
            continue
        b = tuple(r["bases_before"])
        if b != (False, True, False):
            continue
        n2 += 1
        # runner originally on 2nd scores iff new_on3 is False (a runner
        # from 2nd who merely advanced would occupy 3rd; the only other
        # runner in this before-state is the batter, who cannot reach 3rd
        # on a single) -- equivalently runs_pa >= 1.
        if not r["bases_after"][2]:
            scored2 += 1
    p2 = scored2 / n2 if n2 else 0.0

    # (2) P(double play | G, runner on 1st, outs_before < 2)
    ndp_elig, ndp_yes = 0, 0
    for r in train_records:
        if r["cat"] != "G":
            continue
        if not r["bases_before"][0]:
            continue
        if r["outs_before"] >= 2:
            continue
        ndp_elig += 1
        if r["outs_recorded"] >= 2:
            ndp_yes += 1
    pdp = ndp_yes / ndp_elig if ndp_elig else 0.0

    return {
        "p_score_from_2nd_on_1b": p2, "p_score_from_2nd_on_1b_n": n2,
        "p_dp_on_groundout": pdp, "p_dp_on_groundout_n": ndp_elig,
    }
