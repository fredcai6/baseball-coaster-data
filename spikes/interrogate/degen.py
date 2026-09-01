"""Is the 'interaction' style structure, or identity memorisation via a null axis?

The tree interrogation raised a specific alarm. The top cross-side pairs are
B4xP5, B1xP5, B5xP4, B2xP5 -- P5 appears in three of the top four. But the
GLLVM latent singular values are:

    batter  [5.493, 3.333, 2.790, 1.558, 0.922]
    pitcher [6.751, 5.224, 4.122, 1.525, 0.008]

Pitcher axis 5 has singular value 0.008. It carries essentially NO fitted
effect -- it is a numerically degenerate direction. Its coordinates are tiny,
but trees are scale-free, so U5 still works as a per-pitcher FINGERPRINT.
Crossed with a batter axis, a depth-4 tree can carve out "this subset of
pitchers against that subset of batters", which is identity memorisation
wearing the costume of style interaction.

Feature importance was also suspiciously flat (0.132 down to 0.066, no dominant
axis), which is what identity-fingerprinting looks like and not what a real
low-dimensional style effect looks like.

Test: rerun the identical additive-vs-joint comparison restricted to the
well-conditioned axes only. If the interaction survives on the top 3 (or top 4)
axes per side, it is style. If it collapses, it lived in the degenerate
channels and the finding is an artifact.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "boost"))
sys.path.insert(0, str(HERE.parent / "fuse"))
import common  # noqa: E402
import fit as BF  # noqa: E402
import analyze as FZ  # noqa: E402
from control import boost_additive  # noqa: E402

SEP = HERE.parent / "nested_sep"
GLLVM = HERE.parent / "gllvm"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def main():
    rows = common.load_pa()
    train_g, _ = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    season_idx = {s: i for i, s in enumerate([2024, 2025, 2026])}
    y = np.array([r["y"] for r in tr])

    z = np.load(GLLVM / "latent.npz", allow_pickle=True)
    sv = {}
    axes = {}
    for side in ("bat", "pit"):
        B = z[f"L{side}"] @ z[f"F{side}"].T
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        axes[side] = (U * S)[:, :5]
        sv[side] = S[:5]
    log(f"batter sv  {np.round(sv['bat'],3).tolist()}")
    log(f"pitcher sv {np.round(sv['pit'],3).tolist()}")

    bat_ids, pit_ids = list(z["bat_ids"]), list(z["pit_ids"])
    _, Ab, Ap, _, _ = BF.additive_offset_and_coords(
        tr, z, {b: i for i, b in enumerate(bat_ids)},
        {p: i for i, p in enumerate(pit_ids)}, season_idx,
        len(bat_ids), len(pit_ids), axes)

    zA = np.load(SEP / "latent.npz", allow_pickle=True)
    PA = FZ.nested_category_probs(tr, zA, False, season_idx)
    off = np.log(np.maximum(PA, 1e-300))

    g = sorted(train_g)
    rs = np.random.RandomState(11)
    rs.shuffle(g)
    itr = set(g[: int(len(g) * 0.8)])
    m = np.array([r["game_id"] in itr for r in tr])

    out = {}
    for r in (5, 4, 3, 2, 1):
        B_, P_ = Ab[:, :r], Ap[:, :r]
        X = np.hstack([B_, P_])
        base = BF.deviance(off[~m], y[~m])
        add = boost_additive(B_[m], P_[m], off[m], y[m], B_[~m], P_[~m], off[~m], y[~m])
        joint, rnd = BF.boost(X[m], off[m], y[m], X[~m], off[~m], y[~m])
        gap = joint - add
        out[r] = dict(base=float(base), additive=float(add), joint=float(joint),
                      gap=float(gap))
        log(f"top-{r} axes per side: add={add:.5f} joint={joint:.5f}  GAP={gap:+.5f}")

    log("")
    log("full 5-axis gap was -0.01410. If the gap tracks the well-conditioned")
    log("axes it is style; if it needs axis 5 it was identity memorisation.")
    (HERE / "degen_result.json").write_text(json.dumps(
        dict(singular_values={k: v.tolist() for k, v in sv.items()}, by_rank=out,
             runtime_sec=time.time() - T0), indent=1) + "\n")
    log("wrote degen_result.json")


if __name__ == "__main__":
    main()
