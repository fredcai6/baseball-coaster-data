#!/usr/bin/env python3
"""Build the plate-appearance table from the committed corpus.

    python scripts/build_pa_table.py                 # writes artifacts/derived/pa_table.csv
    python scripts/build_pa_table.py --check         # build in memory, print the summary only

The output is a cache, not an asserted artifact: it is regenerable in seconds
from `games/**`, so it lives under `artifacts/derived/` and is not committed,
by the same argument that keeps `_derived` out of semantic equality.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from bc_pipeline import pa_table  # noqa: E402


def summarize(rows):
    by_season = collections.defaultdict(collections.Counter)
    for r in rows:
        c = by_season[r["season"]]
        c["pa"] += 1
        for k in ("is_k", "is_bb", "is_hit", "is_hr", "is_ab", "is_bip"):
            c[k] += bool(r[k])
        if r["bb_type"]:
            c["bb_" + r["bb_type"]] += 1
        if r["spray"]:
            c["sprayed"] += 1
    return by_season


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=str(ROOT / "games"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "derived" / "pa_table.csv"))
    ap.add_argument("--check", action="store_true", help="do not write; summarize only")
    args = ap.parse_args()

    rows = pa_table.build(args.games)
    print(f"plate appearances: {len(rows)}")

    careers_p = {r["pitcher_career"] for r in rows if r["pitcher_career"]}
    careers_b = {r["batter_career"] for r in rows if r["batter_career"]}
    pairs = {(r["pitcher_career"], r["batter_career"]) for r in rows
             if r["pitcher_career"] and r["batter_career"]}
    print(f"pitcher careers: {len(careers_p)}  batter careers: {len(careers_b)}  pairs: {len(pairs)}")

    missing = sum(1 for r in rows if not (r["pitcher_career"] and r["batter_career"]))
    print(f"rows without both career ids: {missing}")

    print()
    print(f"{'season':>7} {'PA':>7} {'K%':>6} {'BB%':>6} {'HR%':>6} {'BIP%':>6} {'spray%':>7}")
    for season, c in sorted(summarize(rows).items()):
        n = c["pa"]
        print(f"{season:>7} {n:>7} {c['is_k']/n:>6.3f} {c['is_bb']/n:>6.3f} "
              f"{c['is_hr']/n:>6.4f} {c['is_bip']/n:>6.3f} {c['sprayed']/max(1,c['is_bip']):>7.3f}")

    if not args.check:
        out = pa_table.write_csv(rows, args.out)
        print()
        print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
