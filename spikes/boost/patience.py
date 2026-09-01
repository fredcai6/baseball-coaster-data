"""Is the additive control undertrained? Give it far more rope and re-measure.

The parametric bootstrap showed the procedure does not manufacture a gap out
of nothing (simulated gaps: mean -0.00002, sd 0.00002, vs a real gap of
-0.00748). But it does not settle the ORIGINAL worry, because outcomes there
were simulated from the shrunken additive model itself -- so that null had no
residual main-effect signal for the additive arm to chase, and the arm
converged immediately at round 0-6.

The real data DOES have residual main effects (ridge shrinkage left them), and
the additive arm stopped on early-stopping patience after ~250 rounds. If it
quit on a plateau rather than at a true optimum, the leftover -0.00748 is the
control being undertrained, not interaction.

Direct test: rerun the additive-by-construction arm with 5x the patience and
many more passes. If the gap closes, it was undertraining. If it holds, the
interaction survives its most likely mundane explanation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402
import fit as F  # noqa: E402

F.PATIENCE = 100                      # was 20
GLLVM = HERE.parent / "gllvm"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def main():
    import control  # imported AFTER PATIENCE is raised so it picks up the value
    control.PATIENCE = 100
    rows = common.load_pa()
    train_g, _ = common.get_split(rows)
    tr = [r for r in rows if r["game_id"] in train_g]
    z = np.load(GLLVM / "latent.npz", allow_pickle=True)
    bat_ids, pit_ids = list(z["bat_ids"]), list(z["pit_ids"])
    bat_idx = {b: i for i, b in enumerate(bat_ids)}
    pit_idx = {p: i for i, p in enumerate(pit_ids)}
    season_idx = {s: i for i, s in enumerate([2024, 2025, 2026])}
    axes = {}
    for side in ("bat", "pit"):
        B = z[f"L{side}"] @ z[f"F{side}"].T
        U, S, Vt = np.linalg.svd(B, full_matrices=False)
        axes[side] = (U * S)[:, :5]
    off, Ab, Ap, y, _ = F.additive_offset_and_coords(
        tr, z, bat_idx, pit_idx, season_idx, len(bat_ids), len(pit_ids), axes)
    g = sorted(train_g)
    rs = np.random.RandomState(11)
    rs.shuffle(g)
    itr_g = set(g[: int(len(g) * 0.8)])
    m = np.array([r["game_id"] in itr_g for r in tr])

    base = F.deviance(off[~m], y[~m])
    log(f"offset-only val = {base:.5f}")
    add = control.boost_additive(Ab[m], Ap[m], off[m], y[m], Ab[~m], Ap[~m],
                                 off[~m], y[~m], passes=20, rounds_per=60)
    log(f"additive arm, patience=100, up to 20 passes: val = {add:.5f} "
        f"(delta vs offset {add-base:+.5f})")
    log(f"previous additive arm (patience=20): 3.88202")
    log(f"joint arm (unchanged):               3.87454")
    log(f"NEW interaction gap = {3.87454 - add:+.5f}   (was -0.00748)")

    (HERE / "patience_result.json").write_text(json.dumps(dict(
        additive_patience100=float(add), additive_patience20=3.88202,
        joint=3.87454, new_gap=float(3.87454 - add), old_gap=-0.00748,
        runtime_sec=time.time() - T0), indent=1) + "\n")


if __name__ == "__main__":
    main()
