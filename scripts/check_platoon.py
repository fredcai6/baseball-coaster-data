#!/usr/bin/env python3
"""Does the platoon split exist in this league at all?

The first question worth asking of newly-recovered handedness, and the one that
decides whether the matchup-clustering project has a known-truth target.

In real baseball the batter-handedness x pitcher-handedness interaction is the
one pitcher-vs-batter interaction that survives scrutiny -- roughly 20-30 points
of wOBA and 2-3 points of K rate between same- and opposite-handed matchups.
If this league reproduces it, then a latent-cluster method that recovers
handedness blind (it is recorded nowhere in the play-by-play) is validated
against ground truth. If the league does NOT reproduce it, that is the more
important finding: it means the simulation has no handedness effect to find,
and any "style interaction" a clustering turns up is measuring our own noise.

Switch hitters bat opposite the pitcher by definition, so they are reported
separately rather than folded into either arm.
"""

import argparse
import collections
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from bc_pipeline import pa_table  # noqa: E402

PROFILES = ROOT / "profiles" / "player_profiles.csv"


def load_profiles():
    """Index handedness by (season, person_id), NOT by player_id.

    `person_id` is stable across every game of a season; `player_id` is not --
    10% of player records carry a synthetic id, and keying on it drops those
    rows even when the very same person is profiled under a real id in another
    game. Keying on person_id lifts matchup coverage from 76% of plate
    appearances to 93%.
    """
    bats, throws = {}, {}
    with open(PROFILES) as fh:
        for r in csv.DictReader(fh):
            if not r["person_id"]:
                continue
            key = (int(r["season"]), r["person_id"])
            if r["bats"]:
                bats[key] = r["bats"]
            if r["throws"]:
                throws[key] = r["throws"]
    return bats, throws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=str(ROOT / "games"))
    args = ap.parse_args()

    bats, throws = load_profiles()
    cells = collections.defaultdict(collections.Counter)
    covered = missing = 0

    for r in pa_table.build(args.games):
        b = bats.get((r["season"], r["batter_person"]))
        p = throws.get((r["season"], r["pitcher_person"]))
        if not (b and p):
            missing += 1
            continue
        covered += 1
        c = cells[(b, p)]
        c["PA"] += 1
        c["K"] += bool(r["is_k"])
        c["BB"] += bool(r["is_bb"])
        c["HR"] += bool(r["is_hr"])
        c["H"] += bool(r["is_hit"])
        c["TB"] += r["bases"]
        c["AB"] += bool(r["is_ab"])
        c["OB"] += bool(r["is_hit"] or r["is_bb"] or r["is_hbp"])

    print(f"PA with handedness on both sides: {covered}  (missing one or both: {missing}, "
          f"{100*covered/(covered+missing):.1f}% covered)")
    print()
    hdr = f"{'bats':>5} {'throws':>7} {'PA':>7} {'K%':>7} {'BB%':>7} {'HR%':>7} {'AVG':>7} {'OBP':>7} {'SLG':>7}"
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(cells):
        c = cells[key]
        n, ab = c["PA"], max(1, c["AB"])
        print(f"{key[0]:>5} {key[1]:>7} {n:>7} {c['K']/n:>7.3f} {c['BB']/n:>7.3f} "
              f"{c['HR']/n:>7.4f} {c['H']/ab:>7.3f} {c['OB']/n:>7.3f} {c['TB']/ab:>7.3f}")

    # Same vs opposite arm, switch hitters excluded (they always bat opposite).
    def agg(pairs):
        t = collections.Counter()
        for k in pairs:
            t.update(cells[k])
        return t

    same = agg([("L", "L"), ("R", "R")])
    opp = agg([("L", "R"), ("R", "L")])
    print()
    print("Platoon contrast (switch hitters excluded):")
    print(f"{'':>10} {'PA':>7} {'K%':>7} {'BB%':>7} {'HR%':>7} {'OBP':>7} {'SLG':>7}")
    for label, c in (("same arm", same), ("opposite", opp)):
        n, ab = max(1, c["PA"]), max(1, c["AB"])
        print(f"{label:>10} {c['PA']:>7} {c['K']/n:>7.3f} {c['BB']/n:>7.3f} "
              f"{c['HR']/n:>7.4f} {c['OB']/n:>7.3f} {c['TB']/ab:>7.3f}")

    ns, no = max(1, same["PA"]), max(1, opp["PA"])
    d_k = opp["K"]/no - same["K"]/ns
    d_ob = opp["OB"]/no - same["OB"]/ns
    print()
    print(f"  delta K%  (opposite - same): {d_k*100:+.2f} pts")
    print(f"  delta OBP (opposite - same): {d_ob*1000:+.1f} pts")
    print("  MLB reference: opposite-arm batters strike out ~2-3 pts LESS and get on base ~20-30 pts MORE.")


if __name__ == "__main__":
    main()
