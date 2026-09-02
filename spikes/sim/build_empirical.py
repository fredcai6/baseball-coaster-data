"""Build the empirical advancement transition table on TRAIN games only.

P(bases_after, outs_after, runs | bases_before, outs_before, outcome_category),
estimated from TRAIN plate appearances.

Reuses spikes/value/run_expectancy.py's game-file walker (iter_games,
half_innings, check_clean, load_disposed_ids) instead of re-parsing games/**,
per the task brief. half_innings() groups BOTH plate_appearance and
runner_event kinds by (inning, half); check_clean() verifies the state chain
has no gaps.

"After" state, per the task brief: for a plate-appearance event at index i in
a clean half-inning, look forward to the NEXT plate_appearance event at index
j (skipping over any runner_events -- steals / wild pitches / pickoffs -- in
between). after = (bases_before[j], outs_before[j]), or END if no such event
exists (half ends here, whether via 3 outs or a walk-off). This deliberately
folds whatever happened between the two PAs (including runner_events) into
the observed transition -- exactly the between-PA noise the task brief says
this table is allowed to silently absorb.

`runs` for that transition = sum of runs_on_play over events i..j-1 inclusive
(this PA's own runs_on_play, PLUS any runner_event's runs_on_play before the
next PA) -- so a run scored on a wild pitch immediately after a walk is
correctly attributed to the walk's transition instead of silently dropped.
This is slightly more inclusive than "just this PA's own runs_on_play"; it's
the natural reading of "next-PA-before plus runs_on_play" once you notice
run-scoring runner_events exist between PAs. Stated here and in SIM.md.

Thin cells (< MIN_CELL_N observations for a given (bases_before, outs_before,
category) key) back off WHOLESALE to that category's outcome-only marginal
(pooled over all bases_before/outs_before for that category, TRAIN-wide) --
literally as the task brief specifies. This can occasionally produce a
physically inconsistent bases_after (e.g. a runner "on" a base with no
predecessor) for the rare thin cells; not corrected further, per the brief's
instruction to use the simple backoff. Stated in SIM.md.
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPIKES = HERE.parent
sys.path.insert(0, str(SPIKES))
sys.path.insert(0, str(SPIKES / "value"))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import run_expectancy as RE_MOD  # noqa: E402  (reuse the walker only)
from data import bases_code  # noqa: E402

OUT_NPZ = HERE / "empirical_transitions.npz"
MIN_CELL_N = 30
END_BASES = 8   # sentinel for "half ends here"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def main():
    import json
    split = json.loads((SPIKES / "split.json").read_text())
    train_g = set(split["train_games"])
    disposed = RE_MOD.load_disposed_ids()
    log(f"train games={len(train_g)} disposed(total)={len(disposed)}")

    # key = bases_code(before)*30 + outs_before*10 + cat_idx  -> Counter[(after_bases,after_outs,runs)]
    cell_counts = defaultdict(lambda: defaultdict(int))
    cat_marginal = defaultdict(lambda: defaultdict(int))  # cat_idx -> Counter[...]

    n_games_used = 0
    n_halves_clean = n_halves_dirty = 0
    n_pa_seen = 0
    n_end_of_half = 0

    for season in RE_MOD.SEASONS:
        for path, game in RE_MOD.iter_games(season):
            game_id = game.get("game_id") or path
            if game_id in disposed or game_id not in train_g:
                continue
            n_games_used += 1
            groups = RE_MOD.half_innings(game["events"])
            for _, evs in groups.items():
                if not RE_MOD.check_clean(evs):
                    n_halves_dirty += 1
                    continue
                n_halves_clean += 1
                n = len(evs)
                for i, e in enumerate(evs):
                    if e["kind"] != "plate_appearance":
                        continue
                    n_pa_seen += 1
                    d = e["_derived"]
                    cat_idx = common.CAT_INDEX[common.OUTCOME_MAP[e["outcome"]["type"]]]
                    bc = bases_code(tuple(d["bases_before"]))
                    ob = d["outs_before"]
                    key = bc * 30 + ob * 10 + cat_idx

                    j = None
                    for k in range(i + 1, n):
                        if evs[k]["kind"] == "plate_appearance":
                            j = k
                            break
                    if j is None:
                        n_end_of_half += 1
                        runs = sum(evs[k]["_derived"]["runs_on_play"] for k in range(i, n))
                        outcome = (END_BASES, 3, runs)
                    else:
                        dj = evs[j]["_derived"]
                        after_bc = bases_code(tuple(dj["bases_before"]))
                        after_outs = dj["outs_before"]
                        runs = sum(evs[k]["_derived"]["runs_on_play"] for k in range(i, j))
                        outcome = (after_bc, after_outs, runs)

                    cell_counts[key][outcome] += 1
                    cat_marginal[cat_idx][outcome] += 1

    log(f"games used={n_games_used}  halves clean={n_halves_clean} dirty={n_halves_dirty}  "
        f"PAs seen={n_pa_seen}  end-of-half transitions={n_end_of_half}")

    N_KEYS = 8 * 3 * 10
    n_thin = 0
    n_present = 0
    final_dists = {}   # key -> list[(bases,outs,runs,count)]
    for key in range(N_KEYS):
        cat_idx = key % 10
        counts = cell_counts.get(key)
        total = sum(counts.values()) if counts else 0
        if counts and total >= MIN_CELL_N:
            n_present += 1
            final_dists[key] = [(b, o, r, c) for (b, o, r), c in counts.items()]
        else:
            n_thin += 1
            mc = cat_marginal[cat_idx]
            final_dists[key] = [(b, o, r, c) for (b, o, r), c in mc.items()]
    log(f"cells with >= {MIN_CELL_N} obs: {n_present}/{N_KEYS}  "
        f"backed off to category marginal: {n_thin}/{N_KEYS}")

    max_bins = max(len(v) for v in final_dists.values())
    log(f"max distinct outcomes for any key/marginal: {max_bins}")

    BASES_AFTER = np.full((N_KEYS, max_bins), END_BASES, dtype=np.uint8)
    OUTS_AFTER = np.full((N_KEYS, max_bins), 3, dtype=np.uint8)
    RUNS = np.zeros((N_KEYS, max_bins), dtype=np.uint8)
    CUMPROB = np.ones((N_KEYS, max_bins), dtype=np.float64)
    N_OBS = np.zeros(N_KEYS, dtype=np.int64)
    THIN = np.zeros(N_KEYS, dtype=bool)

    for key in range(N_KEYS):
        dist = sorted(final_dists[key])  # deterministic order
        total = sum(c for *_, c in dist)
        N_OBS[key] = total
        THIN[key] = key not in cell_counts or sum(cell_counts.get(key, {}).values()) < MIN_CELL_N
        cum = 0.0
        for m, (b, o, r, c) in enumerate(dist):
            cum += c / total
            BASES_AFTER[key, m] = b
            OUTS_AFTER[key, m] = o
            RUNS[key, m] = min(r, 255)
            CUMPROB[key, m] = cum
        # pad remainder (if any) by repeating the last entry, cumprob already 1.0
        for m in range(len(dist), max_bins):
            BASES_AFTER[key, m] = BASES_AFTER[key, len(dist) - 1]
            OUTS_AFTER[key, m] = OUTS_AFTER[key, len(dist) - 1]
            RUNS[key, m] = RUNS[key, len(dist) - 1]
            CUMPROB[key, m] = 1.0
        # float rounding safety: force the last real column to exactly 1.0
        CUMPROB[key, max_bins - 1] = 1.0

    np.savez(
        OUT_NPZ,
        bases_after=BASES_AFTER, outs_after=OUTS_AFTER, runs=RUNS, cumprob=CUMPROB,
        n_obs=N_OBS, thin=THIN, min_cell_n=MIN_CELL_N, end_bases_sentinel=END_BASES,
        max_bins=max_bins,
    )
    log(f"wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
