"""Shared model machinery: empirical transition tables, exact
distribution-propagation (no Monte Carlo needed -- naive_v1's two fitted
probabilities and the empirical table's realized-outcome frequencies are
both just categorical distributions over a small set of branches, so exact
expectation propagation over the 24(+1 terminal)-state space is both cheap
and noise-free), and the RE24 solver used by task 3.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import naive

BASES_ORDER = ["000", "100", "010", "001", "110", "101", "011", "111"]
ALL_BASES = [tuple(c == "1" for c in b) for b in BASES_ORDER]
ALL_STATES = [(b, o) for b in ALL_BASES for o in (0, 1, 2)]
CATS = ["K", "BB", "HBP", "F", "G", "1B", "2B", "3B", "HR", "OTHER"]


def bases_str(b):
    return "".join("1" if x else "0" for x in b)


def state_key(bases, outs):
    return f"{bases_str(bases)}|{outs}"


# --------------------------------------------------------------------
# naive_v0 / naive_v1 as exact branch distributions
# --------------------------------------------------------------------

def v0_dist(bases, outs_before, cat, is_sac=False):
    # naive.apply_v0 returns OUTS ADDED (0/1/2), not absolute outs -- the
    # distribution-propagation state key needs the absolute count (same
    # convention as the empirical table's state_before_next_outs, which
    # comes straight from game-log _derived, always absolute).
    nb, oa, runs = naive.apply_v0(bases, outs_before, cat, is_sac)
    return [(nb, min(3, outs_before + oa), runs, 1.0)]


def v1_dist(bases, outs_before, cat, is_sac, params):
    p2 = params["p_score_from_2nd_on_1b"]
    pdp = params["p_dp_on_groundout"]

    if cat == "OTHER" and is_sac:
        nb, runs = naive.advance_one_each(bases, False)
        return [(nb, min(3, outs_before + 1), runs, 1.0)]

    if cat == "G":
        on1, on2, on3 = bases
        if on1 and outs_before < 2:
            dp_next = (False, on2, on3)
            return [(dp_next, min(3, outs_before + 2), 0, pdp),
                    (bases, min(3, outs_before + 1), 0, 1.0 - pdp)]
        return [(bases, min(3, outs_before + 1), 0, 1.0)]

    if cat == "1B":
        on1, on2, on3 = bases
        if on2:
            score_next = (True, on1, False)
            runs_score = (1 if on3 else 0) + 1
            hold_next, hold_runs = naive.advance_one_each(bases, True)
            return [(score_next, outs_before, runs_score, p2), (hold_next, outs_before, hold_runs, 1.0 - p2)]
        nb, runs = naive.advance_one_each(bases, True)
        return [(nb, outs_before, runs, 1.0)]

    return v0_dist(bases, outs_before, cat, is_sac)


def v1_dist_marginal_sac(bases, outs_before, cat, p_sac_other, params):
    """Same as v1_dist but for synthetic/simulated categories where we don't
    have a real per-PA is_sac flag -- marginalize OTHER over TRAIN's
    P(is_sac | OTHER). Used only for RE24 forward simulation (task 3), never
    for replay (task 1), which always has the real flag."""
    if cat == "OTHER":
        d_sac = v1_dist(bases, outs_before, "OTHER", True, params)
        d_non = v1_dist(bases, outs_before, "OTHER", False, params)
        out = [(nb, no, r, p * p_sac_other) for nb, no, r, p in d_sac]
        out += [(nb, no, r, p * (1.0 - p_sac_other)) for nb, no, r, p in d_non]
        return out
    return v1_dist(bases, outs_before, cat, False, params)


# --------------------------------------------------------------------
# Empirical table: built from a list of PA records (records.py schema).
# `mode="full"` uses state_before_next_{bases,outs} + runs_pa+between_runs
# (this PA's outcome rolled forward through any intervening runner_events --
# what a replay/RE24 model needs). `mode="within"` uses this PA's own
# bases_after/outs_after/runs_pa only (no between-PA movement) -- used for
# the task-2 decomposition.
# --------------------------------------------------------------------

def build_empirical_table(records, mode="full"):
    raw = defaultdict(Counter)
    for r in records:
        key = (tuple(r["bases_before"]), r["outs_before"], r["cat"])
        if mode == "full":
            nb = tuple(r["state_before_next_bases"])
            no = r["state_before_next_outs"]
            runs = r["runs_pa"] + r["between_runs"]
        else:
            nb = tuple(r["bases_after"])
            no = r["outs_after"]
            runs = r["runs_pa"]
        no_capped = 3 if no >= 3 else no
        raw[key][(nb, no_capped, runs)] += 1

    table = {}
    for key, counter in raw.items():
        n = sum(counter.values())
        branches = [(nb, no, runs, c / n) for (nb, no, runs), c in counter.items()]
        mean_runs = sum(runs * c for (_, _, runs), c in counter.items()) / n
        table[key] = {"n": n, "branches": branches, "mean_runs": mean_runs}
    return table


def empirical_dist(table, bases, outs_before, cat, min_n=1, fallback_table=None):
    """Look up (bases,outs_before,cat) in `table`. If missing or below
    min_n, fall back to `fallback_table` (a cat-only marginal table, keyed
    just by cat) built by build_marginal_table. If that's also missing,
    fall back to naive_v0 (a play with truth never observed at all)."""
    key = (bases, outs_before, cat)
    cell = table.get(key)
    if cell is not None and cell["n"] >= min_n:
        return cell["branches"]
    if fallback_table is not None:
        fb = fallback_table.get(cat)
        if fb is not None and fb["n"] > 0:
            return fb["branches"]
    return v0_dist(bases, outs_before, cat, False)


def build_marginal_table(records, mode="full"):
    """Cat-only fallback table (ignoring bases_before/outs_before) for cells
    the per-state empirical table never observed in TRAIN."""
    raw = defaultdict(Counter)
    for r in records:
        cat = r["cat"]
        if mode == "full":
            nb = tuple(r["state_before_next_bases"])
            no = r["state_before_next_outs"]
            runs = r["runs_pa"] + r["between_runs"]
        else:
            nb = tuple(r["bases_after"])
            no = r["outs_after"]
            runs = r["runs_pa"]
        no_capped = 3 if no >= 3 else no
        raw[cat][(nb, no_capped, runs)] += 1
    table = {}
    for cat, counter in raw.items():
        n = sum(counter.values())
        branches = [(nb, no, runs, c / n) for (nb, no, runs), c in counter.items()]
        table[cat] = {"n": n, "branches": branches}
    return table


# --------------------------------------------------------------------
# RE24 solver: exact value iteration over the 24-state absorbing chain
# (outs never decrease within a half, and every category has positive
# probability of an out, so this converges geometrically).
# --------------------------------------------------------------------

def solve_re24(dist_fn, cat_probs, n_iter=2000, tol=1e-13):
    RE = {s: 0.0 for s in ALL_STATES}
    for it in range(n_iter):
        new_RE = {}
        max_delta = 0.0
        for (bases, outs) in ALL_STATES:
            val = 0.0
            for cat, pcat in cat_probs.items():
                if pcat <= 0:
                    continue
                for nb, no, runs, bp in dist_fn(bases, outs, cat):
                    p = pcat * bp
                    val += p * runs
                    if no < 3:
                        val += p * RE[(nb, no)]
            new_RE[(bases, outs)] = val
            max_delta = max(max_delta, abs(val - RE[(bases, outs)]))
        RE = new_RE
        if max_delta < tol:
            break
    return RE, it + 1
