"""Shared SHAPE D definition, reused by fit_shaped.py and simulator.py.

Copied verbatim from spikes/pitch/step6_shapes.py's SHAPE_D (also duplicated,
identically, in spikes/value/player_value.py). Not imported from either --
step6_shapes.py has no __main__-guard-safe importable SHAPES without also
pulling in its argparse-driven main(), and player_value.py is a large
unrelated script. This is the "some duplication accepted for parallelism"
case the task brief calls out. No fitted parameters live here -- only the
node topology and the per-node hyperparameters loaded from
spikes/pitch/step6_result.json (read-only).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

CATS = common.CATEGORIES
CI = common.CAT_INDEX
N_CAT = len(CATS)

S = lambda *cs: frozenset(CI[c] for c in cs)  # noqa: E731
ALL = S(*CATS)

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
N_NODES = len(SHAPE_D)
NODE_NAMES = [n[0] for n in SHAPE_D]

STEP6_RESULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pitch", "step6_result.json")


def build_paths(nodes):
    paths = {ci: [] for ci in range(N_CAT)}
    for ni, (name, reach, pos) in enumerate(nodes):
        for ci in range(N_CAT):
            if ci in reach:
                paths[ci].append((ni, ci in pos))
    return paths


PATHS_D = build_paths(SHAPE_D)


def load_shape_d_hp():
    """Per-node (lam_bat, lam_pit, psi) selected in step6_shapes.py -- shape D,
    frozen-test deviance 3.94526 (see step6_result.json['D']['total_deviance']).
    Read-only reference; nothing here is refit."""
    d = json.loads(open(STEP6_RESULT).read())
    hp = {}
    for nd in d["D"]["nodes"]:
        hp[nd["node"]] = dict(lam_bat=nd["lam_bat"], lam_pit=nd["lam_pit"], psi=nd["psi"])
    return hp, d["D"]["total_deviance"]
