"""Venue main effect + venue x stance deviation, layered on shape D (the
frozen 3.94526 winner from spikes/pitch/step6_result.json).

Per node: lam_bat, lam_pit, psi are INHERITED from step6_result.json shape D
-- fixed, no re-search (per the task spec). Only (lam_ven, lam_vs) are
selected, on the SAME inner validation split step1.py/step6_shapes.py use
(80/20 of train games, seed 90210), via a grid search with the same
grid-edge detection + plateau tie-break step6_shapes.py introduced (defect 1
in that file's docstring: a flat validation surface chases the grid edge
forever unless you tie-break toward the smallest norm).

Usage:
    python fit_venue.py --node root
    python fit_venue.py --node root,tto_K,tto_BB
    python fit_venue.py --node all

Results accumulate into result.json keyed by node name, so a slow node can
be run as its own foreground command within the 10-minute cap (mirrors
step6_shapes.py's --shape splitting).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "fuse"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pitch"))

import common  # noqa: E402
from analyze import structural  # noqa: E402
import step1  # noqa: E402  (NODES, the 9-node shape D... no: step1.NODES is shape A/original; use local SHAPE_D)
import venue_common as VC  # noqa: E402
import venue_model as VM  # noqa: E402

CI = common.CAT_INDEX
S = lambda *cs: frozenset(CI[c] for c in cs)  # noqa: E731
ALL = S(*common.CATEGORIES)

# SHAPE D, copied verbatim from spikes/pitch/step6_shapes.py / value/player_value.py
# (not imported -- step6_shapes.py has no importable SHAPES without its argparse
# main(), and value/player_value.py's copy is the second precedent for copying
# this literal, not re-deriving it).
SHAPE_D = [
    ("root",       ALL,                                          S("K", "BB", "HBP")),
    ("tto_K",      S("K", "BB", "HBP"),                          S("K")),
    ("tto_BB",     S("BB", "HBP"),                                S("BB")),
    ("con_HR",     ALL - S("K", "BB", "HBP"),                    S("HR")),
    ("con_OTH",    ALL - S("K", "BB", "HBP", "HR"),               S("OTHER")),
    ("con_OUT",    ALL - S("K", "BB", "HBP", "HR", "OTHER"),      S("F", "G")),
    ("out_F",      S("F", "G"),                                  S("F")),
    ("hit_1B",     S("1B", "2B", "3B"),                           S("1B")),
    ("hit_2B",     S("2B", "3B"),                                 S("2B")),
]
NODE_NAMES = [n for n, _, _ in SHAPE_D]

STEP6_RESULT = os.path.join(os.path.dirname(HERE), "pitch", "step6_result.json")
RESULT_PATH = os.path.join(HERE, "result.json")

LAM_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]
# Widened below the task's suggested [1..3000] after a first pass selected
# lam_ven=1 (the then-lower edge) at root, tto_K, con_OUT and hit_2B -- the
# same grid-edge-chasing failure mode step6_shapes.py's docstring documents
# for lambda selection generally. 0.1/0.3 added with headroom on the low
# side; see result.json/VENUE.md for which of those four re-resolved to an
# interior point vs. genuinely wanted the least regularisation on the grid.
LAM_TOL = 1e-4

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def load_shape_d_hp():
    d = json.loads(open(STEP6_RESULT).read())
    hp = {}
    for nd in d["D"]["nodes"]:
        hp[nd["node"]] = dict(lam_bat=nd["lam_bat"], lam_pit=nd["lam_pit"], psi=nd["psi"])
    return hp, d["D"]["total_deviance"]


def select_plateau(cands, tol=LAM_TOL):
    """cands: list of (dev, lam_ven, lam_vs). Same tie-break as step6_shapes:
    among candidates within tol of the best validation deviance, take the
    smallest (lam_ven, lam_vs) norm -- least regularisation indistinguishable
    from the observed best."""
    best_dev = min(c[0] for c in cands)
    within = [c for c in cands if c[0] <= best_dev + tol]
    return min(within, key=lambda c: (c[1] ** 2 + c[2] ** 2, c[0]))


def pack(rs_, reach, pos, season_idx, BI, PI, VI, VSI):
    sub = [r for r in rs_ if r["y"] in reach]
    Xs = structural(sub, season_idx)
    bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
    pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
    vi = np.fromiter((VC.venue_index(r, VI) for r in sub), int, len(sub))
    vsi = np.fromiter((VC.venue_stance_index(r, VSI) for r in sub), int, len(sub))
    yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
    return Xs, bi, pj, vi, vsi, yv


def fit_one_node(node_name, reach, pos, hp, tr, te, ifit, season_idx, BI, PI, VI, VSI,
                  n_bat, n_pit, n_ven, n_vs, n_test_full):
    h = hp[node_name]
    lam_b, lam_p, psi = h["lam_bat"], h["lam_pit"], h["psi"]

    F = pack([r for r in tr if r["game_id"] in ifit], reach, pos, season_idx, BI, PI, VI, VSI)
    V = pack([r for r in tr if r["game_id"] not in ifit], reach, pos, season_idx, BI, PI, VI, VSI)
    TR = pack(tr, reach, pos, season_idx, BI, PI, VI, VSI)
    TE = pack(te, reach, pos, season_idx, BI, PI, VI, VSI)
    nv = max(1, len(V[5]))
    # per step6_shapes.py convention: dev_ao is this NODE's total held-out
    # deviance divided by the FULL frozen-test PA count, not this node's own
    # subset count -- so per-node values sum to the whole-model deviance.
    n_test = max(1, n_test_full)

    def val(lv, lvs, warm=None):
        th = VM.fit(*F, n_bat, n_pit, n_ven, n_vs, psi, lam_b, lam_p, lv, lvs, warm)
        return VM.node_dev(th, *V, n_bat, n_pit, n_ven, n_vs, psi) / nv, th

    cands = []
    for lv in LAM_GRID:
        warm = None
        for lvs in LAM_GRID:
            d, warm = val(lv, lvs, warm)
            cands.append((d, lv, lvs))
    best = select_plateau(cands)
    _, lv2, lvs2 = best

    lo, hi = LAM_GRID[0], LAM_GRID[-1]
    edge = []
    if lv2 in (lo, hi):
        edge.append(f"lam_ven={lv2:g}")
    if lvs2 in (lo, hi):
        edge.append(f"lam_vs={lvs2:g}")

    theta_final = VM.fit(*TR, n_bat, n_pit, n_ven, n_vs, psi, lam_b, lam_p, lv2, lvs2)
    d_final = VM.node_dev(theta_final, *TE, n_bat, n_pit, n_ven, n_vs, psi)
    dev_per_pa = d_final / n_test

    ps = TR[0].shape[1]
    _, _, _, _, v, w = VM.unpack(theta_final, ps, n_bat, n_pit, n_ven, n_vs)

    log(f"{node_name:9} n={len(TR[5]):>6} rate={TR[5].mean():.4f}  "
        f"lam_ven={lv2:<7g} lam_vs={lvs2:<7g} (inherited lam_bat={lam_b:g} "
        f"lam_pit={lam_p:g} psi={psi:g})  dev={dev_per_pa:.5f}" +
        ("  <<< EDGE " + ",".join(edge) if edge else ""))

    return dict(
        node=node_name, n=int(len(TR[5])), rate=float(TR[5].mean()),
        lam_bat=lam_b, lam_pit=lam_p, psi=psi,
        lam_ven=lv2, lam_vs=lvs2, edge_hits=edge,
        dev=dev_per_pa, n_test=n_test,
        v=v.tolist(), w=w.tolist(),
        theta=theta_final.tolist(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default="all", help="comma list of node names or 'all'")
    args = ap.parse_args()
    want = NODE_NAMES if args.node == "all" else args.node.split(",")
    for w in want:
        assert w in NODE_NAMES, f"unknown node {w!r}, choices {NODE_NAMES}"

    hp, ref_dev = load_shape_d_hp()
    log(f"inherited shape D hyperparameters from step6_result.json (total_deviance={ref_dev:.5f})")

    rows = VC.load_rows()
    train_g, test_g = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    season_idx = {s: i for i, s in enumerate(sorted({r["season"] for r in rows}))}

    g = sorted(train_g)
    rs = np.random.RandomState(90210)  # SAME inner split as step1.py/step6_shapes.py
    rs.shuffle(g)
    ifit = set(g[: int(0.8 * len(g))])

    bats = sorted({r["batter"] for r in rows})
    pits = sorted({r["pitcher"] for r in rows})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)

    venues, VI, vs_keys, VSI = VC.build_venue_indices(rows)
    n_ven, n_vs = len(venues), len(vs_keys)
    log(f"train {len(tr)} test {len(te)}  batters {n_bat} pitchers {n_pit}  "
        f"venues {n_ven} venue x stance {n_vs}  LAM_GRID={LAM_GRID}")

    out = {}
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH) as fh:
            out = json.load(fh)
    out["lam_grid"] = LAM_GRID
    out["lam_tol"] = LAM_TOL
    out["venues"] = venues
    out["vs_keys"] = [f"{v}|{s}" for v, s in vs_keys]
    out["shape_d_reference_deviance"] = ref_dev
    out.setdefault("nodes", {})

    for node_name, reach, pos in SHAPE_D:
        if node_name not in want:
            continue
        log(f"=== fitting node {node_name} ===")
        res = fit_one_node(node_name, reach, pos, hp, tr, te, ifit, season_idx,
                            BI, PI, VI, VSI, n_bat, n_pit, n_ven, n_vs, len(te))
        out["nodes"][node_name] = res
        with open(RESULT_PATH, "w") as fh:
            json.dump(out, fh, indent=1)
        log(f"wrote {RESULT_PATH} (node {node_name} done)")

    if all(n in out["nodes"] for n in NODE_NAMES):
        # each node's "dev" is already (that node's total held-out NLL*2) / full
        # frozen-test PA count, so the whole-model deviance is a plain sum.
        total_dev = sum(out["nodes"][n]["dev"] for n in NODE_NAMES)
        out["total_deviance"] = total_dev
        with open(RESULT_PATH, "w") as fh:
            json.dump(out, fh, indent=1)
        log(f"ALL 9 NODES DONE. total_deviance = {total_dev:.5f} "
            f"(shape D reference {ref_dev:.5f}, delta {total_dev - ref_dev:+.5f})")

    log("done.")


if __name__ == "__main__":
    main()
