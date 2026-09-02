"""Fit arm1 at the SAME fixed hyperparameters (d=3, lam_L=lam_M=40, inherited
psi) on one half of the training games (A or B), from halves_split.json.
Writes halfA.npz or halfB.npz. Run as two separate foreground invocations:

    python run_half.py A
    python run_half.py B
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_style as CS

log = CS.log


def main():
    which = sys.argv[1].upper()
    assert which in ("A", "B")
    split = json.load(open(os.path.join(HERE, "halves_split.json")))
    games = set(split[f"half{which}_games"])

    log("=" * 70)
    log(f"STAGE: half{which}.npz -- canonical arm1 fit on half {which} "
        f"({len(games)} games, seed {split['seed']})")

    base = CS.base_universe()
    rows = base["rows"]
    BI, PI, season_idx = base["BI"], base["PI"], base["season_idx"]
    n_bat, n_pit = base["n_bat"], base["n_pit"]
    psi0, lam_bat0, lam_pit0 = base["psi0"], base["lam_bat0"], base["lam_pit0"]

    sub = [r for r in rows if r["game_id"] in games]
    log(f"half {which} PA = {len(sub)}")
    D_sub = CS.build_node_data(sub, BI, PI, season_idx)
    ps = D_sub[0][0].shape[1]

    fit = CS.fit_canonical_arm1(D_sub, psi0, lam_bat0, lam_pit0, n_bat, n_pit, ps,
                                 tag=f"half{which}")

    bat_pa, pit_pa = CS.pa_counts(sub, BI, PI)
    bats = base["bats"]; pits = base["pits"]

    out_path = os.path.join(HERE, f"half{which}.npz")
    np.savez(out_path,
             L=fit["L"], f=fit["f"], M=fit["M"], g=fit["g"],
             alpha=fit["alpha"], beta=fit["beta"],
             B_free=fit["B_free"], Q_free=fit["Q_free"],
             bat_ids=np.array(bats, dtype=object), pit_ids=np.array(pits, dtype=object),
             bat_pa=bat_pa, pit_pa=pit_pa,
             node_names=np.array(CS.NODE_NAMES, dtype=object),
             d=fit["d"], lam=fit["lam"], n_pa=len(sub),
             canonical_seed=fit["canonical_seed"], train_loss_spread=fit["train_loss_spread"])
    log(f"wrote {out_path}")

    out = CS.load_result()
    out.setdefault("half_fits", {})[which] = dict(
        n_pa=len(sub), n_games=len(games), seed=split["seed"],
        canonical_seed=fit["canonical_seed"], train_loss_spread=fit["train_loss_spread"],
        n_restarts=CS.N_RESTARTS_STYLE)
    CS.save_result(out)


if __name__ == "__main__":
    main()
