"""Split the TRAINING games into two disjoint halves, stratified by season,
seeded. Writes spikes/latent_style/halves_split.json. Fast (seconds) --
run once before run_halfA.py / run_halfB.py.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common_style as CS

SEED = 130917  # arbitrary, recorded
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "halves_split.json")


def main():
    base = CS.base_universe()
    rows, train_g = base["rows"], base["train_g"]
    by_season = {}
    for r in rows:
        if r["game_id"] in train_g:
            by_season.setdefault(r["season"], set()).add(r["game_id"])
    rng = random.Random(SEED)
    halfA, halfB = set(), set()
    for season, games in sorted(by_season.items()):
        g = sorted(games)
        rng.shuffle(g)
        cut = len(g) // 2
        halfA.update(g[:cut])
        halfB.update(g[cut:])
    assert halfA.isdisjoint(halfB)
    assert halfA | halfB == train_g
    d = dict(seed=SEED, halfA_games=sorted(halfA), halfB_games=sorted(halfB))
    with open(OUT, "w") as fh:
        json.dump(d, fh)
    CS.log(f"halves split: seed={SEED} halfA={len(halfA)} games halfB={len(halfB)} games "
           f"(train total {len(train_g)})")
    CS.log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
