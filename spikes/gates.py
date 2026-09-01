"""Shared gate definitions for the sequential (continuation-ratio) spikes.

DO NOT EDIT -- this is what makes the nested variants comparable to each other
AND to the flat models already fit (GLMM 3.95550, NPMR 3.95424, GLLVM 3.95563).

A plate appearance is recast as a sequence of conditional choices:

    PA
    |-- TTO        41,651 (33.1%)  -> K / BB / HBP
    `-- Contact    84,176 (66.9%)
        |-- Out    45,651          -> F / G        [trajectory]
        |-- Hit    33,195          -> 1B/2B/3B/HR  [power]
        `-- OTHER   5,330

Why this tree and not the textbook DIPS one: DIPS needs "given a fly ball, hit
or out?", but this corpus labels batted-ball trajectory only on OUTS. A single
may have been a liner or a chopper and the source never says. So the
hit/out split has to come before the trajectory split, not after.

The payoff is concentration. Home runs are 3.3% of all plate appearances but
12.6% of hits, so a power interaction tested at the HIT gate gets ~4x the
signal density it has in a flat 10-category deviance dominated by F/G/K/1B.

THE KEY IDENTITY, and the unit test at the bottom of this file: the
continuation-ratio likelihood factorizes EXACTLY, so

    log P(outcome) = sum over gates of log P(branch | reached gate)

and a saturated nested fit reproduces the flat multinomial exactly. That means
`joint_deviance` here is directly comparable to the flat spikes' numbers --
the nesting is a reparameterization until low-rank structure is imposed on it,
which is the entire point of trying it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

CATS = common.CATEGORIES
CI = common.CAT_INDEX

# gate name -> (list of branches, each branch a list of leaf categories)
# A branch that is a single leaf ends the sequence; a branch naming a gate
# continues it.
GATES = {
    "root":    [("TTO", ["K", "BB", "HBP"]),
                ("CONTACT", ["F", "G", "1B", "2B", "3B", "HR", "OTHER"])],
    "tto":     [("K", ["K"]), ("BB", ["BB"]), ("HBP", ["HBP"])],
    "contact": [("OUT", ["F", "G"]),
                ("HIT", ["1B", "2B", "3B", "HR"]),
                ("OTHER", ["OTHER"])],
    "out":     [("F", ["F"]), ("G", ["G"])],
    "hit":     [("1B", ["1B"]), ("2B", ["2B"]), ("3B", ["3B"]), ("HR", ["HR"])],
}

# which gate a row proceeds to after taking a branch (None = sequence ends)
NEXT = {("root", "TTO"): "tto", ("root", "CONTACT"): "contact",
        ("contact", "OUT"): "out", ("contact", "HIT"): "hit"}

GATE_ORDER = ["root", "tto", "contact", "out", "hit"]


def _leaf_to_branch(gate):
    m = {}
    for bi, (bname, leaves) in enumerate(GATES[gate]):
        for lf in leaves:
            m[CI[lf]] = bi
    return m


LEAF_TO_BRANCH = {g: _leaf_to_branch(g) for g in GATE_ORDER}
N_BRANCH = {g: len(GATES[g]) for g in GATE_ORDER}


def assign(y):
    """For each gate, which rows reach it and which branch they take.

    Returns {gate: (row_index_array, branch_index_array)}.
    """
    y = np.asarray(y)
    out = {}
    reach = {"root": np.arange(len(y))}
    for g in GATE_ORDER:
        if g not in reach:
            continue
        idx = reach[g]
        m = LEAF_TO_BRANCH[g]
        br = np.array([m[v] for v in y[idx]])
        out[g] = (idx, br)
        for bi, (bname, _) in enumerate(GATES[g]):
            nxt = NEXT.get((g, bname))
            if nxt is not None:
                reach[nxt] = idx[br == bi]
    return out


def joint_deviance(gate_logp, y):
    """Total deviance from per-gate conditional log-probabilities.

    `gate_logp[gate]` is an (n_reaching_gate, n_branches) array of log-probs,
    row-aligned with `assign(y)[gate][0]`. Comparable to the flat models.
    """
    a = assign(y)
    total = np.zeros(len(y))
    for g, (idx, br) in a.items():
        total[idx] += gate_logp[g][np.arange(len(idx)), br]
    return -2.0 * total.mean()


def saturated_check(y):
    """Unit test: a saturated nested fit == the flat multinomial. Must be ~0."""
    a = assign(y)
    logp = {}
    for g, (idx, br) in a.items():
        cnt = np.bincount(br, minlength=N_BRANCH[g]).astype(float)
        p = cnt / cnt.sum()
        logp[g] = np.log(np.maximum(p, 1e-300))[None, :].repeat(len(idx), axis=0)
    nested = joint_deviance(logp, y)
    cnt = np.bincount(y, minlength=len(CATS)).astype(float)
    flat_p = cnt / cnt.sum()
    flat = -2.0 * np.mean(np.log(np.maximum(flat_p[y], 1e-300)))
    return nested, flat, abs(nested - flat)


if __name__ == "__main__":
    rows = common.load_pa()
    y = np.array([r["y"] for r in rows])
    a = assign(y)
    print("gate                rows   branches")
    for g in GATE_ORDER:
        print(f"  {g:12s} {len(a[g][0]):8d}   {N_BRANCH[g]}  "
              f"{[b for b,_ in GATES[g]]}")
    nested, flat, diff = saturated_check(y)
    print()
    print(f"saturated nested deviance = {nested:.6f}")
    print(f"flat multinomial deviance = {flat:.6f}")
    print(f"|difference|              = {diff:.3e}   "
          f"{'OK - factorization exact' if diff < 1e-9 else 'BROKEN'}")
