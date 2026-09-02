"""Stage: fit arm1 at FIXED hyperparameters (d=3, lam_L=lam_M=40, inherited
psi) on ALL training games. This is the canonical fit for presentation and
for Test 2 (external validity). Also saves the free-9 per-node effect
matrices (B_free, Q_free) from the SAME node_free_fits call, which Test 2(b)
needs -- refit once here, reused later rather than refit again.

Writes spikes/latent_style/full.npz.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_style as CS

log = CS.log


def main():
    log("=" * 70)
    log("STAGE: full.npz -- canonical arm1 fit on ALL training games")
    base = CS.base_universe()
    rows, train_g = base["rows"], base["train_g"]
    BI, PI, season_idx = base["BI"], base["PI"], base["season_idx"]
    n_bat, n_pit = base["n_bat"], base["n_pit"]
    psi0, lam_bat0, lam_pit0 = base["psi0"], base["lam_bat0"], base["lam_pit0"]

    tr = [r for r in rows if r["game_id"] in train_g]
    log(f"train PA = {len(tr)}  batters={n_bat} pitchers={n_pit}")
    D_tr = CS.build_node_data(tr, BI, PI, season_idx)
    ps = D_tr[0][0].shape[1]

    fit = CS.fit_canonical_arm1(D_tr, psi0, lam_bat0, lam_pit0, n_bat, n_pit, ps,
                                 tag="full")

    bat_pa, pit_pa = CS.pa_counts(tr, BI, PI)
    bats = base["bats"]; pits = base["pits"]

    out_path = os.path.join(HERE, "full.npz")
    np.savez(out_path,
             L=fit["L"], f=fit["f"], M=fit["M"], g=fit["g"],
             alpha=fit["alpha"], beta=fit["beta"],
             B_free=fit["B_free"], Q_free=fit["Q_free"],
             alpha_free=fit["alpha_free"], beta_free=fit["beta_free"],
             bat_ids=np.array(bats, dtype=object), pit_ids=np.array(pits, dtype=object),
             bat_pa=bat_pa, pit_pa=pit_pa,
             node_names=np.array(CS.NODE_NAMES, dtype=object),
             d=fit["d"], lam=fit["lam"],
             canonical_seed=fit["canonical_seed"], train_loss_spread=fit["train_loss_spread"])
    log(f"wrote {out_path}")

    out = CS.load_result()
    out["full_fit"] = dict(d=fit["d"], lam=fit["lam"], n_bat=n_bat, n_pit=n_pit, ps=ps,
                            n_train_pa=len(tr), canonical_seed=fit["canonical_seed"],
                            train_loss_spread=fit["train_loss_spread"],
                            n_restarts=CS.N_RESTARTS_STYLE)
    CS.save_result(out)


if __name__ == "__main__":
    main()
