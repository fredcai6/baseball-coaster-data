"""Does n_pitches qualify as a covariate at all? No -- and this measures it.

An exposure offset assumes the exposure is fixed BEFORE the outcome is drawn.
Pitch count is not: a strikeout cannot happen in under three pitches and a
walk cannot happen in under four, so the count is a DESCENDANT of the outcome,
not a cause of it. Conditioning on it is conditioning on a collider.

This script refuses to argue the point and measures it instead: fit a model
whose ONLY feature is the pitch count, and see how much of the outcome it
"predicts". Anything well above the null is leakage, because a pitch count
carries no information about the matchup -- only about what already happened.
"""
import sys, os, math, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common

rows = [r for r in common.load_pa(with_handedness=False) if r.get("n_pitches")]
print(f"2026 PAs with a pitch count: {len(rows)}")

train_g, test_g = common.get_split(rows)
train = [r for r in rows if r["game_id"] in train_g]
test = [r for r in rows if r["game_id"] in test_g]
print(f"train {len(train)}  test {len(test)}")

K = len(common.CATEGORIES)
idx = {c: i for i, c in enumerate(common.CATEGORIES)}


def fit_by(keyfn, train, smooth=5.0):
    """Smoothed empirical category distribution within each key bucket."""
    base = [0.0] * K
    tab = collections.defaultdict(lambda: [0.0] * K)
    for r in train:
        y = idx[r["cat"]]
        base[y] += 1
        tab[keyfn(r)][y] += 1
    tot = sum(base)
    prior = [c / tot for c in base]
    out = {}
    for k, cnt in tab.items():
        n = sum(cnt)
        out[k] = [(cnt[i] + smooth * prior[i]) / (n + smooth) for i in range(K)]
    return prior, out


def score(prior, tab, keyfn, test):
    tot = 0.0
    for r in test:
        p = tab.get(keyfn(r), prior)
        tot += -2.0 * math.log(max(p[idx[r["cat"]]], 1e-12))
    return tot / len(test)


null_p, _ = fit_by(lambda r: 0, train)
null_dev = score(null_p, {}, lambda r: 0, test)
print(f"\nNULL (2026 subset)          deviance = {null_dev:.5f}")

for name, fn in [
    ("n_pitches alone",          lambda r: min(int(r["n_pitches"]), 12)),
    ("balls/strikes count alone", lambda r: (r["count_balls"], r["count_strikes"])),
]:
    prior, tab = fit_by(fn, train)
    d = score(prior, tab, fn, test)
    print(f"{name:27} deviance = {d:.5f}   ({d - null_dev:+.5f} vs null)")

print("""
For reference, the entire additive player model -- every batter and pitcher
latent, three seasons, 100k PAs -- buys about -0.056 against its null.
""")
