"""Shared loading/indexing helpers for the venue spike.

Does not modify spikes/common.py or spikes/fuse/analyze.py. Wraps
common.load_pa() to attach two extra fields per row, joined by row order
(common.load_pa() reads artifacts/derived/pa_table.csv with a plain
csv.DictReader and no reordering, so a second synchronized read of the same
file lines up 1:1 -- the same trick populations.py and player_value.py rely
on for batter_name/pitcher_name lookups):

  r["venue"]   -- venue_canonical, or None for the 156 PAs from the two
                  Long Beach games that have no canonical venue.
  r["stance"]  -- the batter's stance side vs THIS pitcher's throwing hand,
                  per the ground-rules formula:
                      stance = bats if bats in {L,R}
                               else 'L' if throws == 'R'
                               else 'R' if throws == 'L'
                               else None
                  This applies uniformly to switch hitters (bats == 'S')
                  AND to batters with unknown recorded hand -- both cases
                  fall through to the throws-based inference, exactly as
                  specified. Only a fully-unknown pairing (throws also
                  unknown) yields stance = None.

Also builds the venue / (venue, stance) coefficient index maps used by
venue_model.py's index-with-sentinel trick: known venues/combos get a real
index into the fitted coefficient vector; unknown ones get the sentinel
index one past the end, which nll_grad_venue treats as a fixed zero (never
touched by the optimizer) -- the same "append a zero row for missing
roster members" pattern spikes/fuse/analyze.py uses for gate rosters.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

STANCES = ("L", "R")


def _stance(bats, throws):
    if bats in ("L", "R"):
        return bats
    if throws == "R":
        return "L"
    if throws == "L":
        return "R"
    return None


def load_rows():
    """common.load_pa() rows, augmented with venue + stance, in the same order."""
    rows = common.load_pa()
    venues = []
    with open(common.PA_TABLE) as fh:
        for r in csv.DictReader(fh):
            venues.append(r["venue_canonical"] or None)
    assert len(venues) == len(rows), (len(venues), len(rows))
    for r, ven in zip(rows, venues):
        r["venue"] = ven
        r["stance"] = _stance(r["bats"], r["throws"])
    return rows


def build_venue_indices(rows):
    """Sorted venue list + (venue, stance) list, and lookup dicts.

    n_ven = len(VENUES) (16); n_vs = len(VS_KEYS) (32, 16 parks x 2 stances).
    Rows with venue None / stance None map to the sentinel index n_ven / n_vs
    respectively when looked up via `venue_index` / `venue_stance_index`.
    """
    venues = sorted({r["venue"] for r in rows if r["venue"]})
    VI = {v: i for i, v in enumerate(venues)}
    vs_keys = [(v, s) for v in venues for s in STANCES]
    VSI = {k: i for i, k in enumerate(vs_keys)}
    return venues, VI, vs_keys, VSI


def venue_index(r, VI):
    return VI.get(r["venue"], len(VI))


def venue_stance_index(r, VSI, n_ven_dummy=None):
    key = (r["venue"], r["stance"]) if r["venue"] and r["stance"] else None
    return VSI.get(key, len(VSI))
