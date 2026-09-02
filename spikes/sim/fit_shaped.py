"""Refit SHAPE D on TRAIN games only, at the per-node hyperparameters selected
by spikes/pitch/step6_shapes.py (no search here -- those are frozen). Also
derives everything else the simulator needs that isn't already sitting in a
frozen result file:

  - the generic-reliever bullpen policy: PA-weighted mean per-node q effect,
    and PA-weighted throwing-hand mix, over each team's TRAIN relief
    appearances (pitcher_is_starter == False for that PA).
  - the naive advancement model's four fixed probabilities, estimated
    directly from TRAIN PA rows (outs_recorded / bases_before / runs_on_play
    are already columns in pa_table.csv -- no event-level re-derivation
    needed for these).

Everything is written to spikes/sim/shaped_train.npz. Node fitting reuses
step1.fit / step1.ao_prob unmodified (per the task brief); it is NOT
rewritten here. Takes a few minutes -- run in the foreground.
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPIKES = HERE.parent
sys.path.insert(0, str(SPIKES))
sys.path.insert(0, str(SPIKES / "pitch"))
sys.path.insert(0, str(SPIKES / "fuse"))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import step1 as S1  # noqa: E402  (reuse ao_prob / fit / node_dev unmodified)
from analyze import structural  # noqa: E402
from shape_d import SHAPE_D, N_NODES, NODE_NAMES, load_shape_d_hp  # noqa: E402
from data import load_pa_full  # noqa: E402

OUT_NPZ = HERE / "shaped_train.npz"
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def main():
    log("loading full PA rows...")
    rows = load_pa_full()
    train_g, test_g = common.get_split(rows)
    season_list = sorted({r["season"] for r in rows})
    season_idx = {s: i for i, s in enumerate(season_list)}
    tr = [r for r in rows if r["game_id"] in train_g]
    te = [r for r in rows if r["game_id"] in test_g]
    log(f"rows={len(rows)} train_games={len(train_g)} test_games={len(test_g)} "
        f"train_pa={len(tr)} test_pa={len(te)} seasons={season_list}")

    bats = sorted({r["batter"] for r in tr})
    pits = sorted({r["pitcher"] for r in tr})
    BI = {b: i for i, b in enumerate(bats)}
    PI = {p: i for i, p in enumerate(pits)}
    n_bat, n_pit = len(bats), len(pits)
    log(f"TRAIN vocab: batters={n_bat} pitchers={n_pit}")

    hp, ref_dev = load_shape_d_hp()
    log(f"loaded shape D hyperparameters from step6_result.json "
        f"(reference frozen-test deviance {ref_dev:.5f}, fit on train+test split there; "
        f"we refit params fresh on TRAIN only, same hyperparameters, no search)")

    p_dim = None
    node_alpha = np.zeros(N_NODES)
    node_beta = None
    node_b = np.zeros((N_NODES, n_bat + 1))   # last col = OOV (unseen batter) = 0
    node_q = np.zeros((N_NODES, n_pit + 1))   # last col = OOV (unseen pitcher) = 0
    node_psi = np.zeros(N_NODES)

    for ni, (name, reach, pos) in enumerate(SHAPE_D):
        sub = [r for r in tr if r["y"] in reach]
        Xs = structural(sub, season_idx)
        if p_dim is None:
            p_dim = Xs.shape[1]
            node_beta = np.zeros((N_NODES, p_dim))
        bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
        pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
        yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
        h = hp[name]
        th = S1.fit(Xs, bi, pj, yv, n_bat, n_pit, h["psi"], h["lam_bat"], h["lam_pit"])
        d = S1.node_dev(th, Xs, bi, pj, yv, n_bat, n_pit, h["psi"]) / max(1, len(yv))
        node_alpha[ni] = th[0]
        node_beta[ni] = th[1:1 + p_dim]
        node_b[ni, :n_bat] = th[1 + p_dim:1 + p_dim + n_bat]
        node_q[ni, :n_pit] = th[1 + p_dim + n_bat:]
        node_psi[ni] = h["psi"]
        log(f"  {name:10} n={len(sub):>6} rate={yv.mean():.4f} "
            f"lam_bat={h['lam_bat']:<7g} lam_pit={h['lam_pit']:<7g} psi={h['psi']:<5g} "
            f"train_dev(in-sample)={d:.5f}")

    log("node fitting done")

    # ---- generic reliever: PA-weighted mean q + PA-weighted hand mix, per
    # fielding_team, over TRAIN relief appearances (pitcher_is_starter=False
    # on that PA). PA count of the appearance IS the weight -- summing per-PA
    # values is exactly a PA-weighted mean over pitchers. ----
    teams = sorted({r["fielding_team"] for r in tr})
    TI = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    team_q_sum = np.zeros((n_teams, N_NODES))
    team_q_n = np.zeros(n_teams)
    team_hand_count = np.zeros((n_teams, 3))  # L, R, unknown
    HAND_COL = {"L": 0, "R": 1}
    for r in tr:
        if r["is_starter"]:
            continue
        ti = TI[r["fielding_team"]]
        pidx = PI[r["pitcher"]]
        team_q_sum[ti] += node_q[:, pidx]
        team_q_n[ti] += 1
        col = HAND_COL.get(r["throws"], 2)
        team_hand_count[ti, col] += 1
    team_reliever_q = team_q_sum / np.maximum(team_q_n, 1)[:, None]
    team_reliever_hand = team_hand_count / np.maximum(team_hand_count.sum(axis=1, keepdims=True), 1)
    log(f"generic reliever built for {n_teams} teams, relief-PA counts "
        f"min={team_q_n.min():.0f} max={team_q_n.max():.0f} mean={team_q_n.mean():.1f}")

    # ---- naive advancement model: four fixed probabilities from TRAIN -----
    # (1) double play: G, runner on 1st, <2 outs -> P(outs_recorded==2)
    dp_num = dp_den = 0
    # (2) OTHER: P(batter reaches safely) = P(outs_recorded==0)
    oth_reach_num = oth_reach_den = 0
    # (3) OTHER, batter out, runner on 3rd & <2 outs -> P(that runner scores)
    #     (a sac-fly-shaped exception, since OTHER lumps sac/FC/error/interference)
    oth_sac_num = oth_sac_den = 0
    # (4) 1B, runner on 2nd only (not also on 3rd, to isolate the question) ->
    #     P(that runner scores)
    sgl2_num = sgl2_den = 0
    for r in tr:
        cat, bb, ob, outsrec, runs = r["cat"], r["bases_before"], r["outs_before"], r["outs_recorded"], r["runs_on_play"]
        if cat == "G" and bb[0] and ob < 2:
            dp_den += 1
            dp_num += 1 if outsrec == 2 else 0
        if cat == "OTHER":
            oth_reach_den += 1
            oth_reach_num += 1 if outsrec == 0 else 0
            if outsrec >= 1 and bb[2] and ob < 2:
                oth_sac_den += 1
                oth_sac_num += 1 if runs >= 1 else 0
        if cat == "1B" and bb[1] and not bb[2]:
            sgl2_den += 1
            sgl2_num += 1 if runs >= 1 else 0

    dp_prob = dp_num / dp_den if dp_den else 0.0
    other_reach_prob = oth_reach_num / oth_reach_den if oth_reach_den else 0.0
    other_sac_prob = oth_sac_num / oth_sac_den if oth_sac_den else 0.0
    single_2nd_scores_prob = sgl2_num / sgl2_den if sgl2_den else 0.0
    log(f"naive params (TRAIN): dp_prob={dp_prob:.4f} (n={dp_den})  "
        f"other_reach_prob={other_reach_prob:.4f} (n={oth_reach_den})  "
        f"other_sac_prob={other_sac_prob:.4f} (n={oth_sac_den})  "
        f"single_2nd_scores_prob={single_2nd_scores_prob:.4f} (n={sgl2_den})")

    np.savez(
        OUT_NPZ,
        node_names=np.array(NODE_NAMES),
        season_list=np.array(season_list),
        bats=np.array(bats), pits=np.array(pits),
        node_alpha=node_alpha, node_beta=node_beta,
        node_b=node_b, node_q=node_q, node_psi=node_psi,
        teams=np.array(teams),
        team_reliever_q=team_reliever_q, team_reliever_hand=team_reliever_hand,
        team_relief_pa_n=team_q_n,
        dp_prob=dp_prob, other_reach_prob=other_reach_prob,
        other_sac_prob=other_sac_prob, single_2nd_scores_prob=single_2nd_scores_prob,
        ref_dev=ref_dev,
    )
    log(f"wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
