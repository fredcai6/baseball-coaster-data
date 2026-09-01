"""Shared harness for the three modelling spikes. DO NOT EDIT -- it is the
only reason the three results are comparable.

Every spike predicts the SAME target on the SAME split and reports the SAME
metric, so "which method is better" is a real question rather than three
notebooks that each graded their own homework.

Target: a 10-category outcome, a faithful superset of the 9 categories used by
Powers/Hastie/Tibshirani (2018) with an explicit residual bucket rather than a
silent one.

Split: by GAME, not by plate appearance. PAs within a game share a pitcher, a
park and a day, so a PA-level split leaks badly. Stratified by season and
frozen to `spikes/split.json` so all three spikes see identical folds.

Metric: multinomial deviance (2 x negative log-likelihood per PA), the same
loss Powers et al. report. Lower is better. Always report against the NULL
model (league-wide category frequencies) -- a method that cannot beat the null
has found nothing.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PA_TABLE = ROOT / "artifacts" / "derived" / "pa_table.csv"
PROFILES = ROOT / "profiles" / "player_profiles.csv"
SPLIT = Path(__file__).resolve().parent / "split.json"

CATEGORIES = ["K", "BB", "HBP", "F", "G", "1B", "2B", "3B", "HR", "OTHER"]
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

OUTCOME_MAP = {
    "strikeout_swinging": "K", "strikeout_looking": "K", "strikeout": "K",
    "walk": "BB", "intentional_walk": "BB",
    "hit_by_pitch": "HBP",
    "flyout": "F", "popout": "F", "infield_fly": "F", "foul_out": "F", "lineout": "F",
    "groundout": "G", "grounded_into_double_play": "G",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "fielders_choice": "OTHER", "reached_on_error": "OTHER", "sacrifice": "OTHER",
    "reached_on_interference": "OTHER", "batter_interference": "OTHER",
}


def _b(v):
    return v == "1"


def load_pa(with_handedness=True):
    """Every plate appearance as a dict, with `y` = category index.

    Adds `bats` / `throws` (L/R/S, or None) joined through person_id -- NOT
    player_id, which is file-local for the ~10% synthetic ids and would drop
    them even when the same person is profiled elsewhere in the season.
    """
    bats, throws = {}, {}
    if with_handedness and PROFILES.exists():
        with open(PROFILES) as fh:
            for r in csv.DictReader(fh):
                if not r["person_id"]:
                    continue
                key = (int(r["season"]), r["person_id"])
                if r["bats"]:
                    bats[key] = r["bats"]
                if r["throws"]:
                    throws[key] = r["throws"]

    rows = []
    with open(PA_TABLE) as fh:
        for r in csv.DictReader(fh):
            cat = OUTCOME_MAP.get(r["outcome_type"])
            if cat is None:
                raise ValueError(f"unmapped outcome_type {r['outcome_type']!r}")
            season = int(r["season"])
            rows.append({
                "game_id": r["game_id"], "season": season,
                "batter": r["batter_career"] or r["batter_pid"],
                "pitcher": r["pitcher_career"] or r["pitcher_pid"],
                "batter_person": r["batter_person"], "pitcher_person": r["pitcher_person"],
                "y": CAT_INDEX[cat], "cat": cat,
                "tto": int(r["tto"]), "pitcher_bf": int(r["pitcher_bf"]),
                "is_starter": _b(r["pitcher_is_starter"]),
                "batting_is_home": _b(r["batting_is_home"]),
                "home_team": r["home_team"], "inning": int(r["inning"]),
                "bats": bats.get((season, r["batter_person"])),
                "throws": throws.get((season, r["pitcher_person"])),
                # 2026 only; None for 2024/2025 -- see pa_table.pitch_detail
                "n_pitches": int(r["n_pitches"]) if r["n_pitches"] else None,
                "count_balls": int(r["count_balls"]) if r["count_balls"] else None,
                "count_strikes": int(r["count_strikes"]) if r["count_strikes"] else None,
                "pitch_seq": r["pitch_seq"] or None,
            })
    return rows


def get_split(rows, test_frac=0.2, seed=20260830):
    """Frozen train/test game ids, stratified by season. Written once, reused."""
    if SPLIT.exists():
        d = json.loads(SPLIT.read_text())
        return set(d["train_games"]), set(d["test_games"])
    by_season = {}
    for r in rows:
        by_season.setdefault(r["season"], set()).add(r["game_id"])
    rng = random.Random(seed)
    train, test = set(), set()
    for season, games in sorted(by_season.items()):
        g = sorted(games)
        rng.shuffle(g)
        cut = int(len(g) * (1 - test_frac))
        train.update(g[:cut])
        test.update(g[cut:])
    SPLIT.write_text(json.dumps(
        {"seed": seed, "test_frac": test_frac,
         "train_games": sorted(train), "test_games": sorted(test)}, indent=1))
    return train, test


def deviance(probs, ys):
    """2 x negative log-likelihood per observation. Lower is better.

    `probs` is a sequence of length-10 probability vectors aligned with `ys`.
    """
    eps = 1e-12
    total = 0.0
    for p, y in zip(probs, ys):
        total += -math.log(max(eps, p[y]))
    return 2.0 * total / max(1, len(ys))


def null_model(train_rows):
    """League-wide category frequencies -- the floor every method must beat."""
    c = Counter(r["y"] for r in train_rows)
    n = sum(c.values())
    return [c[i] / n for i in range(len(CATEGORIES))]


def report(name, test_rows, probs, extra=None):
    """Print a comparable result block AND return it as a dict."""
    ys = [r["y"] for r in test_rows]
    dev = deviance(probs, ys)
    out = {"model": name, "test_pa": len(ys), "deviance": dev}
    if extra:
        out.update(extra)
    print(f"\n=== {name} ===")
    print(f"  test PA        : {len(ys)}")
    print(f"  deviance       : {dev:.5f}")
    if extra:
        for k, v in extra.items():
            print(f"  {k:15s}: {v}")
    return out


if __name__ == "__main__":
    rows = load_pa()
    train_g, test_g = get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    print(f"total PA {len(rows)}  train {len(tr)}  test {len(te)}")
    print(f"games: train {len(train_g)}  test {len(test_g)}")
    print("category counts:", Counter(r["cat"] for r in rows).most_common())
    hb = sum(1 for r in rows if r["bats"] and r["throws"])
    print(f"PA with handedness both sides: {hb} ({100*hb/len(rows):.1f}%)")
    p = null_model(tr)
    print(f"NULL deviance on test: {deviance([p]*len(te), [r['y'] for r in te]):.5f}")
