"""STEP 6: widen the lambda grid (defect 1) and test four tree shapes (defect 2).

Does NOT modify step1.py. Reuses step1's ao_prob / fit / node_dev / PSI_GRID by
import, and re-implements only the staged lambda/psi selection loop (copied
from step1.main(), because that loop is not factored into a function there)
with a widened LAM_GRID and explicit grid-edge detection.

Usage:
    python step6_shapes.py --shape A
    python step6_shapes.py --shape A,B,C,D
    python step6_shapes.py --shape all

Results accumulate (merge) into step6_result.json, keyed by shape name, so
each shape can be run as its own foreground command.
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "fuse"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import common
from analyze import structural
import step1  # reuse ao_prob, fit, node_dev, PSI_GRID, LAM_STRUCT unmodified

CI = common.CAT_INDEX
S = lambda *cs: frozenset(CI[c] for c in cs)
ALL = S(*common.CATEGORIES)

# ---- defect 1: widened grid. Every selected lambda must land INTERIOR. ----
# First widened to [1..10000] (per the task spec); a live run (shape A) then
# selected lam_pit=10000 at hit_2B (top edge). Widening further to [0.1..1e5]
# just pushed the edge to 100000 -- a direct probe (see report) showed
# hit_2B's val-deviance vs lam_pit is FLAT to 5-6 significant figures from
# ~3000 up through 1e7: this is a genuinely degenerate/asymptotic likelihood
# (pitcher identity carries ~no signal for 2B once conditioned on reaching
# {2B,3B,HR}), not a grid that's still "too small". No finite grid produces a
# stable interior argmin on a flat surface -- it will always chase
# floating-point noise to whatever the top happens to be. So two fixes:
#   1. a modest widened grid with headroom on both sides of the ORIGINAL
#      [3..300], not the extreme [0.1..1e5] that was only needed to chase
#      the flat tail, and
#   2. a plateau tie-break (LAM_TOL below): among (lam_bat, lam_pit) pairs
#      whose validation deviance is within LAM_TOL of the observed best,
#      choose the one with the smallest ||(lam_bat, lam_pit)||, i.e. the
#      least amount of regularisation that is statistically indistinguishable
#      from the best. This is what lets a flat tail resolve to an interior,
#      reproducible point instead of an artifact of wherever the grid ends.
LAM_GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]
LAM_TOL = 1e-4  # plateau tolerance in per-row validation deviance units
PSI_GRID = step1.PSI_GRID

RESULT_PATH = os.path.join(os.path.dirname(__file__), "step6_result.json")
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- shapes --

SHAPE_A = [
    ("root",       ALL,                                S("K", "BB", "HBP")),
    ("tto_K",      S("K", "BB", "HBP"),                S("K")),
    ("tto_BB",     S("BB", "HBP"),                     S("BB")),
    ("con_OTH",    ALL - S("K", "BB", "HBP"),          S("OTHER")),
    ("con_OUT",    ALL - S("K", "BB", "HBP", "OTHER"), S("F", "G")),
    ("out_F",      S("F", "G"),                        S("F")),
    ("hit_1B",     S("1B", "2B", "3B", "HR"),          S("1B")),
    ("hit_2B",     S("2B", "3B", "HR"),                S("2B")),
    ("hit_3B",     S("3B", "HR"),                      S("3B")),
]

SHAPE_B = [
    ("root",       ALL,                                S("K", "BB", "HBP")),
    ("tto_K",      S("K", "BB", "HBP"),                S("K")),
    ("tto_BB",     S("BB", "HBP"),                     S("BB")),
    ("con_OTH",    ALL - S("K", "BB", "HBP"),          S("OTHER")),
    ("con_OUT",    ALL - S("K", "BB", "HBP", "OTHER"), S("F", "G")),
    ("out_F",      S("F", "G"),                        S("F")),
    ("hit_XB",     S("1B", "2B", "3B", "HR"),          S("3B", "HR")),
    ("hit_1v2",    S("1B", "2B"),                      S("1B")),
    ("hit_3vHR",   S("3B", "HR"),                      S("HR")),
]

SHAPE_C = [
    ("root",       ALL,                                     S("K", "BB", "HBP")),
    ("tto_K",      S("K", "BB", "HBP"),                     S("K")),
    ("tto_BB",     S("BB", "HBP"),                          S("BB")),
    ("con_OUT",    ALL - S("K", "BB", "HBP"),               S("F", "G")),
    ("con_OTH",    ALL - S("K", "BB", "HBP", "F", "G"),     S("OTHER")),
    ("out_F",      S("F", "G"),                             S("F")),
    ("hit_XB",     S("1B", "2B", "3B", "HR"),               S("3B", "HR")),
    ("hit_1v2",    S("1B", "2B"),                           S("1B")),
    ("hit_3vHR",   S("3B", "HR"),                           S("HR")),
]

SHAPE_D = [
    ("root",       ALL,                                          S("K", "BB", "HBP")),
    ("tto_K",      S("K", "BB", "HBP"),                          S("K")),
    ("tto_BB",     S("BB", "HBP"),                               S("BB")),
    ("con_HR",     ALL - S("K", "BB", "HBP"),                    S("HR")),
    ("con_OTH",    ALL - S("K", "BB", "HBP", "HR"),              S("OTHER")),
    ("con_OUT",    ALL - S("K", "BB", "HBP", "HR", "OTHER"),     S("F", "G")),
    ("out_F",      S("F", "G"),                                  S("F")),
    ("hit_1B",     S("1B", "2B", "3B"),                          S("1B")),
    ("hit_2B",     S("2B", "3B"),                                S("2B")),
]

SHAPES = {"A": SHAPE_A, "B": SHAPE_B, "C": SHAPE_C, "D": SHAPE_D}


# ------------------------------------------------------- correctness check --

def structural_check(name, nodes):
    """(a) every category reaches >=1 node, (b) all 10 paths are distinct.

    NO early exit: a category's path includes every node whose `reach`
    contains it, walked in list order, regardless of which branch was taken
    at an earlier node. Membership in `reach` alone defines the path.
    """
    assert len(nodes) == 9, f"{name}: expected 9 nodes, got {len(nodes)}"
    cat_idx = {c: CI[c] for c in common.CATEGORIES}
    paths = {}
    depths = {}
    for cat, cidx in cat_idx.items():
        path = []
        for node_name, reach, pos in nodes:
            if cidx in reach:
                path.append((node_name, cidx in pos))
        assert len(path) >= 1, f"{name}: category {cat} reaches no node"
        paths[cat] = tuple(path)
        depths[cat] = len(path)
    sig_to_cats = {}
    for cat, sig in paths.items():
        sig_to_cats.setdefault(sig, []).append(cat)
    dupes = {sig: cs for sig, cs in sig_to_cats.items() if len(cs) > 1}
    assert not dupes, f"{name}: non-distinct paths: {dupes}"
    log(f"shape {name}: structural check OK -- 10 distinct paths, "
        f"depths {depths}")
    return depths


def sample_sum_check(name, nodes, node_fit, sample_rows, season_idx, BI, PI,
                      tol=1e-10):
    """(c) reconstructed category probabilities sum to 1 on a row sample."""
    Xs = structural(sample_rows, season_idx)
    bi = np.fromiter((BI[r["batter"]] for r in sample_rows), int, len(sample_rows))
    pj = np.fromiter((PI[r["pitcher"]] for r in sample_rows), int, len(sample_rows))
    node_p = {}
    for node_name, reach, pos in nodes:
        th, psi = node_fit[node_name]
        ps = Xs.shape[1]
        n_bat, n_pit = len(BI), len(PI)
        eta = (th[0] + Xs @ th[1:1 + ps] + th[1 + ps:1 + ps + n_bat][bi]
               + th[1 + ps + n_bat:][pj])
        p, _, _, _ = step1.ao_prob(eta, psi)
        node_p[node_name] = p
    n = len(sample_rows)
    cat_probs = np.zeros((n, len(common.CATEGORIES)))
    for cat in common.CATEGORIES:
        cidx = CI[cat]
        prob = np.ones(n)
        for node_name, reach, pos in nodes:
            if cidx in reach:
                p = node_p[node_name]
                prob = prob * (p if cidx in pos else (1.0 - p))
        cat_probs[:, cidx] = prob
    sums = cat_probs.sum(axis=1)
    max_err = float(np.max(np.abs(sums - 1.0)))
    assert max_err < tol, f"{name}: category probs sum to 1 +/- {max_err}, exceeds {tol}"
    log(f"shape {name}: sum-to-1 check OK on {n} sampled rows, max |sum-1| = {max_err:.3e}")
    return max_err


def select_plateau(cands, tol=LAM_TOL):
    """Among (dev, lb, lp) candidates within `tol` of the best dev, return the
    one with the smallest (lb, lp) magnitude -- least regularisation that is
    statistically indistinguishable from the observed best. Falls back to the
    strict argmin when the surface is NOT flat (tol excludes everything else)."""
    best_dev = min(c[0] for c in cands)
    within = [c for c in cands if c[0] <= best_dev + tol]
    return min(within, key=lambda c: (c[1] ** 2 + c[2] ** 2, c[0]))


# --------------------------------------------------------------- fitting --

def fit_shape(name, nodes, tr, te, ifit, BI, PI, n_bat, n_pit, season_idx):
    def pack(rs_, reach, pos):
        sub = [r for r in rs_ if r["y"] in reach]
        Xs = structural(sub, season_idx)
        bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
        pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
        yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
        return Xs, bi, pj, yv

    n_test = len(te)
    tot_ao = 0.0
    node_results = []
    node_fit = {}  # node_name -> (theta_on_TR, psi_hat)  for the sum-to-1 check
    edge_hits = []

    for node_name, reach, pos in nodes:
        F = pack([r for r in tr if r["game_id"] in ifit], reach, pos)
        V = pack([r for r in tr if r["game_id"] not in ifit], reach, pos)
        TR = pack(tr, reach, pos)
        TE = pack(te, reach, pos)
        nv = max(1, len(V[3]))

        def val(psi, lb, lp, warm=None):
            th = step1.fit(*F, n_bat, n_pit, psi, lb, lp, warm)
            return step1.node_dev(th, *V, n_bat, n_pit, psi) / nv, th

        # stage 1: per-side lambda at logit (psi=1), plateau-tie-broken
        cands1 = []
        for lb in LAM_GRID:
            warm = None
            for lp in LAM_GRID:
                d, warm = val(1.0, lb, lp, warm)
                cands1.append((d, lb, lp))
        best_ls = select_plateau(cands1)
        _, lb1, lp1 = best_ls

        # stage 2: psi at stage-1 lambdas
        best_psi = None
        warm = None
        for psi in PSI_GRID:
            d, warm = val(psi, lb1, lp1, warm)
            if best_psi is None or d < best_psi[0]:
                best_psi = (d, psi)
        psi_hat = best_psi[1]

        # stage 3: re-select lambda at psi_hat, plateau-tie-broken
        cands3 = []
        for lb in LAM_GRID:
            warm = None
            for lp in LAM_GRID:
                d, warm = val(psi_hat, lb, lp, warm)
                cands3.append((d, lb, lp))
        best_ao = select_plateau(cands3)
        _, lb2, lp2 = best_ao

        lo, hi = LAM_GRID[0], LAM_GRID[-1]
        edge = []
        if lb2 in (lo, hi):
            edge.append(f"lam_bat={lb2:g}")
        if lp2 in (lo, hi):
            edge.append(f"lam_pit={lp2:g}")
        if edge:
            edge_hits.append((node_name, edge))

        theta_final = step1.fit(*TR, n_bat, n_pit, psi_hat, lb2, lp2)
        d_ao = step1.node_dev(theta_final, *TE, n_bat, n_pit, psi_hat)
        tot_ao += d_ao
        node_fit[node_name] = (theta_final, psi_hat)

        node_results.append(dict(node=node_name, n=int(len(TR[3])),
                                  rate=float(TR[3].mean()),
                                  lam_bat=lb2, lam_pit=lp2, psi=psi_hat,
                                  dev_ao=d_ao / n_test))
        log(f"[{name}] {node_name:10} n={len(TR[3]):>6} rate={TR[3].mean():.4f}  "
            f"lam_bat={lb2:<7g} lam_pit={lp2:<7g} psi={psi_hat:<5g} "
            f"dev={d_ao/n_test:.5f}" + ("  <<< EDGE " + ",".join(edge) if edge else ""))

    total_dev = tot_ao / n_test
    log(f"shape {name}: TOTAL frozen-test deviance = {total_dev:.5f}")
    if edge_hits:
        log(f"shape {name}: GRID EDGE HITS -- {edge_hits}")
    else:
        log(f"shape {name}: no lambda selection at grid edge "
            f"(grid = {LAM_GRID[0]:g}..{LAM_GRID[-1]:g})")
    return total_dev, node_results, node_fit, edge_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="all",
                     help="comma list of shapes (A,B,C,D) or 'all'")
    ap.add_argument("--sample-n", type=int, default=200,
                     help="rows sampled for the sum-to-1 check")
    args = ap.parse_args()
    want = list(SHAPES.keys()) if args.shape == "all" else args.shape.split(",")

    # structural checks run for ALL four shapes regardless of --shape, since
    # they're cheap and don't depend on data/fitting.
    depths_all = {}
    for name, nodes in SHAPES.items():
        depths_all[name] = structural_check(name, nodes)

    rows = common.load_pa()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}

    g = sorted(train_g)
    rs = np.random.RandomState(90210)
    rs.shuffle(g)
    ifit = set(g[: int(0.8 * len(g))])

    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)
    log(f"train {len(tr)} test {len(te)}  batters {n_bat} pitchers {n_pit}  "
        f"LAM_GRID={LAM_GRID}")

    rs_sample = np.random.RandomState(42)
    sample_rows = [te[i] for i in rs_sample.choice(len(te), size=min(args.sample_n, len(te)),
                                                     replace=False)]

    out = {}
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as fh:
            out = json.load(fh)
    out["lam_grid"] = LAM_GRID
    out["depths"] = depths_all

    for name in want:
        nodes = SHAPES[name]
        log(f"=== fitting shape {name} ===")
        total_dev, node_results, node_fit, edge_hits = fit_shape(
            name, nodes, tr, te, ifit, BI, PI, n_bat, n_pit, season_idx)
        max_err = sample_sum_check(name, nodes, node_fit, sample_rows, season_idx, BI, PI)
        out[name] = dict(
            total_deviance=total_dev,
            nodes=node_results,
            depths=depths_all[name],
            edge_hits=edge_hits,
            sum_to_1_max_err=max_err,
        )
        with open(RESULT_PATH, "w") as fh:
            json.dump(out, fh, indent=1)
        log(f"wrote {RESULT_PATH} (shape {name} done)")

    log("")
    log("reference (frozen test): NULL 4.01172 | flat ridge 3.95550 | NPMR 3.95424")
    log("  nested_sep 3.94846 | shape A (old grid) 3.94729 | best known 3.94600")
    for name in SHAPES:
        if name in out and "total_deviance" in out[name]:
            log(f"  shape {name} (widened grid): {out[name]['total_deviance']:.5f}")


if __name__ == "__main__":
    main()
