"""Presentation: fix a canonical rotation for the full fit by SVD of the
batter effect matrix L_bat @ F_bat.T (and separately M_pit @ G_pit.T for
pitchers), axes ordered by singular value. For each axis: node loadings,
top-10/bottom-10 players with raw category rates (style of
spikes/npmr/latent.md), and a run-value correlation to call each axis
QUALITY vs STYLE. Writes into result.json["presentation"]; LATENT_STYLE.md
is assembled by a separate small script from result.json (kept apart so the
prose file can be regenerated without re-deriving numbers).
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_style as CS

log = CS.log
import common as COMMON  # spikes/common.py CATEGORIES

CATEGORIES = COMMON.CATEGORIES

# Rough linear run-value weights (not tuned; monotonic direction is what
# matters for calling an axis "quality-aligned"). K and outs are negative,
# extra-base power scales up, OTHER is treated as mildly negative (mostly
# ROE/FC, not a batter accomplishment).
RUN_VALUE = {"K": -0.30, "BB": 0.69, "HBP": 0.72, "F": -0.25, "G": -0.25,
             "1B": 0.89, "2B": 1.27, "3B": 1.62, "HR": 2.10, "OTHER": -0.10}


def category_rates(rows, train_g, id_field, ids_of_interest):
    """Per-player (id in ids_of_interest) rate over each of the 10
    categories, PA count -- computed on TRAINING games only, same PA the
    latent was fit on."""
    idx = {pid: i for i, pid in enumerate(ids_of_interest)}
    counts = np.zeros((len(ids_of_interest), len(CATEGORIES)))
    for r in rows:
        if r["game_id"] not in train_g:
            continue
        i = idx.get(r[id_field])
        if i is None:
            continue
        counts[i, r["y"]] += 1
    n = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(n[:, None] > 0, counts / np.maximum(n, 1)[:, None], np.nan)
    return rates, n


def canonical_svd(effect_matrix):
    """effect_matrix: (n_players, NNODE). Returns (scores (n,3), node_loadings
    (3, NNODE), singular_values (3,))."""
    U, S, Vt = np.linalg.svd(effect_matrix, full_matrices=False)
    d = min(3, len(S))
    scores = U[:, :d] * S[:d]
    return scores, Vt[:d], S[:d]


def axis_table(node_names, loadings_row):
    order = np.argsort(loadings_row)
    return [(node_names[i], float(loadings_row[i])) for i in order]


def top_bottom(names, scores, rates, n_pa, k=10):
    order = np.argsort(scores)
    bottom = order[:k]
    top = order[::-1][:k]

    def rows_for(idxs):
        out = []
        for i in idxs:
            out.append(dict(player=names[i], score=float(scores[i]), pa=int(n_pa[i]),
                             rates={c: (float(rates[i, ci]) if rates[i, ci] == rates[i, ci] else None)
                                    for ci, c in enumerate(CATEGORIES)}))
        return out
    return rows_for(top), rows_for(bottom)


def run_value_score(rates):
    w = np.array([RUN_VALUE[c] for c in CATEGORIES])
    return np.nansum(rates * w[None, :], axis=1)


def analyze_side(label, rows, train_g, ids, id_field, effect_matrix, n_pa, node_names,
                  name_lookup):
    scores, loadings, svals = canonical_svd(effect_matrix)
    rates, npa_check = category_rates(rows, train_g, id_field, ids)
    rv = run_value_score(rates)

    axes_out = []
    for a in range(scores.shape[1]):
        s = scores[:, a]
        ok = ~np.isnan(rv) & (n_pa >= 20)
        rv_corr = CS.pearson(s[ok], rv[ok]) if ok.sum() > 5 else float("nan")
        names = [name_lookup.get(pid, pid) for pid in ids]
        top, bottom = top_bottom(names, s, rates, n_pa)
        axes_out.append(dict(
            axis=a, singular_value=float(svals[a]),
            node_loadings=axis_table(list(node_names), loadings[a]),
            run_value_corr=rv_corr,
            interpretation="QUALITY-aligned" if abs(rv_corr) >= 0.5 else
                            ("weak/QUALITY-leaning" if abs(rv_corr) >= 0.25 else "STYLE (not quality)"),
            top10=top, bottom10=bottom))
        log(f"  [{label}] axis {a} (sv={svals[a]:.2f}): run-value corr={rv_corr:+.3f} "
            f"-> {axes_out[-1]['interpretation']}")
    return axes_out


def main():
    log("=" * 70)
    log("PRESENTATION: canonical rotation, axis loadings, top/bottom players")
    full = np.load(os.path.join(HERE, "full.npz"), allow_pickle=True)
    base = CS.base_universe()
    rows, train_g = base["rows"], base["train_g"]

    # human-readable names: pa_table has batter_name/pitcher_name but load_pa
    # doesn't carry them -- pull a career_id -> most-common-name map directly.
    import csv
    name_of_bat, name_of_pit = {}, {}
    with open(os.path.join(os.path.dirname(os.path.dirname(HERE)), "artifacts", "derived",
                            "pa_table.csv")) as fh:
        for r in csv.DictReader(fh):
            bkey = r["batter_career"] or r["batter_pid"]
            pkey = r["pitcher_career"] or r["pitcher_pid"]
            name_of_bat.setdefault(bkey, r["batter_name"])
            name_of_pit.setdefault(pkey, r["pitcher_name"])

    node_names = list(full["node_names"])
    L, f = full["L"], full["f"]
    M, g = full["M"], full["g"]
    bat_ids, pit_ids = full["bat_ids"], full["pit_ids"]
    bat_pa, pit_pa = full["bat_pa"], full["pit_pa"]

    bat_effect = L @ f.T
    pit_effect = M @ g.T

    out = CS.load_result()
    out["presentation"] = {}
    out["presentation"]["run_value_weights"] = RUN_VALUE
    out["presentation"]["batters"] = analyze_side(
        "batters", rows, train_g, list(bat_ids), "batter", bat_effect, bat_pa,
        node_names, name_of_bat)
    out["presentation"]["pitchers"] = analyze_side(
        "pitchers", rows, train_g, list(pit_ids), "pitcher", pit_effect, pit_pa,
        node_names, name_of_pit)
    CS.save_result(out)


if __name__ == "__main__":
    main()
