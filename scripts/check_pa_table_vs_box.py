#!/usr/bin/env python3
"""Reconcile the plate-appearance table against the boxscore.

The boxscore is parsed from a different region of the source page than the
play-by-play narrative, so it is a genuine independent oracle for the table's
counting-stat fold: if `classify()` mis-partitions the outcome taxonomy, the
per-player AB/H/BB/SO totals stop matching and this says so.

Games the corpus already knows are incomplete are reported in their own bucket
rather than allowed to move the headline number -- a disposed game's missing
plate appearances are a disclosed source defect, not a fold bug.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from bc_pipeline import pa_table  # noqa: E402

FIELDS = ("AB", "H", "BB", "SO")


def disposed_game_ids():
    path = ROOT / "corrections" / "dispositions.json"
    if not path.exists():
        return set()
    doc = json.load(open(path))
    return {d["game_id"] for d in doc.get("dispositions", [])}


def reconcile(games_dir, disposed):
    agree = collections.Counter()
    disagree = collections.Counter()
    per_field_bad = collections.Counter()
    examples = collections.defaultdict(list)
    skipped_shape = 0
    rows_disposed = 0

    for path in pa_table.iter_game_files(games_dir):
        game = json.load(open(path))
        gid = game["game_id"]
        if game.get("record_shape") == "boxscore_only":
            skipped_shape += 1
            continue

        folded = collections.defaultdict(collections.Counter)
        for r in pa_table.rows_for_game(game):
            c = folded[r["batter_pid"]]
            c["AB"] += bool(r["is_ab"])
            c["H"] += bool(r["is_hit"])
            c["BB"] += bool(r["is_bb"])
            c["SO"] += bool(r["is_k"])

        bucket = "disposed" if gid in disposed else "clean"
        for team_rows in (game.get("box") or {}).get("batting", {}).values():
            for brow in team_rows:
                pid = brow.get("player_id")
                if pid is None:
                    continue
                got = folded.get(pid, collections.Counter())
                bad = [f for f in FIELDS if got[f] != brow.get(f, 0)]
                if bad:
                    disagree[bucket] += 1
                    if bucket == "clean":
                        for f in bad:
                            per_field_bad[f] += 1
                        if len(examples[tuple(bad)]) < 3:
                            examples[tuple(bad)].append(
                                (gid, pid, {f: (got[f], brow.get(f, 0)) for f in bad})
                            )
                else:
                    agree[bucket] += 1
        if bucket == "disposed":
            rows_disposed += 1

    return agree, disagree, per_field_bad, examples, skipped_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=str(ROOT / "games"))
    args = ap.parse_args()

    disposed = disposed_game_ids()
    agree, disagree, per_field, examples, skipped = reconcile(args.games, disposed)

    tot_clean = agree["clean"] + disagree["clean"]
    tot_disp = agree["disposed"] + disagree["disposed"]
    print(f"boxscore-only games skipped (no play-by-play to fold): {skipped}")
    print()
    print(f"{'bucket':>10} {'rows':>8} {'agree':>8} {'disagree':>9} {'rate':>8}")
    print(f"{'clean':>10} {tot_clean:>8} {agree['clean']:>8} {disagree['clean']:>9} "
          f"{agree['clean']/max(1,tot_clean):>8.4f}")
    print(f"{'disposed':>10} {tot_disp:>8} {agree['disposed']:>8} {disagree['disposed']:>9} "
          f"{agree['disposed']/max(1,tot_disp):>8.4f}")
    if per_field:
        print()
        print("clean-bucket disagreements by field:", dict(per_field))
        print()
        for fields, ex in sorted(examples.items(), key=lambda kv: -len(kv[1]))[:6]:
            print(f"  {'+'.join(fields)}:")
            for gid, pid, delta in ex:
                shown = {f: f"got {g} box {b}" for f, (g, b) in delta.items()}
                print(f"    {gid} {pid} {shown}")


if __name__ == "__main__":
    main()
