"""STEP 7: audit step6's shape-selection result for test-set overfitting.

Does NOT modify step1.py or step6_shapes.py. Reuses SHAPES / LAM_GRID /
PSI_GRID / select_plateau from step6_shapes by import, and step1's ao_prob /
fit / node_dev, so every number here is computed with the identical model
and the identical frozen split -- only the SCORING SPLIT and the psi grid
change.

Three tasks:
  1. Score all four shapes (using the hyperparameters step6 already selected)
     on the INNER VALIDATION split, not the frozen test. If a different shape
     wins there than on frozen test, the frozen-test ranking was partly
     noise-chasing.
  2. Paired per-row test deviance for A-vs-D and C-vs-D: mean diff, sd, se,
     t, p, and both a by-row and a by-game (cluster) bootstrap.
  3. Widen PSI_GRID upward and re-select shape D's con_HR node (which
     selected psi=10, the top of the original grid) with the SAME plateau
     tie-break already used for lambda, applied to psi.

Usage:
    python step7_validate.py
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "fuse"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats
import common
from analyze import structural
import step1
import step6_shapes as S6

CI = common.CAT_INDEX
S = lambda *cs: frozenset(CI[c] for c in cs)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT6_PATH = os.path.join(HERE, "step6_result.json")
RESULT_PATH = os.path.join(HERE, "step7_result.json")
T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------------ setup --

def setup():
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
    return rows, tr, te, ifit, BI, PI, n_bat, n_pit, season_idx


def pack(rows_sub, reach, pos, BI, PI, season_idx):
    sub = [r for r in rows_sub if r["y"] in reach]
    Xs = structural(sub, season_idx)
    bi = np.fromiter((BI[r["batter"]] for r in sub), int, len(sub))
    pj = np.fromiter((PI[r["pitcher"]] for r in sub), int, len(sub))
    yv = np.fromiter((1.0 if r["y"] in pos else 0.0 for r in sub), float, len(sub))
    return Xs, bi, pj, yv


# -------------------------------------------------- TASK 1: validation win --

def validation_deviance(shape_name, tr, ifit, BI, PI, n_bat, n_pit, season_idx,
                         node_hp):
    """Fit each node on F (inner-fit rows) at step6's SELECTED hyperparameters
    and score on V (inner-val rows, held out of hyperparameter selection but
    never touching the frozen test). Mirrors exactly the F/V split step6 used
    to pick lam_bat/lam_pit/psi in the first place."""
    nodes = S6.SHAPES[shape_name]
    fit_rows = [r for r in tr if r["game_id"] in ifit]
    val_rows = [r for r in tr if r["game_id"] not in ifit]
    n_val_total = len(val_rows)

    tot = 0.0
    per_node = []
    for node_name, reach, pos in nodes:
        lb, lp, psi = node_hp[shape_name][node_name]
        F = pack(fit_rows, reach, pos, BI, PI, season_idx)
        V = pack(val_rows, reach, pos, BI, PI, season_idx)
        th = step1.fit(*F, n_bat, n_pit, psi, lb, lp)
        d_val = step1.node_dev(th, *V, n_bat, n_pit, psi)
        tot += d_val
        per_node.append(dict(node=node_name, lam_bat=lb, lam_pit=lp, psi=psi,
                              n_val=int(len(V[3])), dev_val=d_val / n_val_total))
    total_val = tot / n_val_total
    return total_val, per_node


# --------------------------------------------- TASK 2: per-row test deviance --

def refit_row_dev(shape_name, tr, te, BI, PI, n_bat, n_pit, season_idx, node_hp):
    """Refit theta_final on the FULL train set at step6's selected
    hyperparameters (identical computation to step6_shapes.fit_shape's
    theta_final call), then reconstruct the per-row category probability on
    every frozen-test row and return the per-row deviance array (len = n_test,
    aligned to `te` order) plus the summary total (sanity check against the
    recorded step6 total_deviance)."""
    nodes = S6.SHAPES[shape_name]
    node_fit = {}
    for node_name, reach, pos in nodes:
        lb, lp, psi = node_hp[shape_name][node_name]
        TR = pack(tr, reach, pos, BI, PI, season_idx)
        theta = step1.fit(*TR, n_bat, n_pit, psi, lb, lp)
        node_fit[node_name] = (theta, psi)

    Xs = structural(te, season_idx)
    bi = np.fromiter((BI[r["batter"]] for r in te), int, len(te))
    pj = np.fromiter((PI[r["pitcher"]] for r in te), int, len(te))
    node_p = {}
    for node_name, reach, pos in nodes:
        th, psi = node_fit[node_name]
        ps = Xs.shape[1]
        eta = (th[0] + Xs @ th[1:1 + ps] + th[1 + ps:1 + ps + n_bat][bi]
               + th[1 + ps + n_bat:][pj])
        p, _, _, _ = step1.ao_prob(eta, psi)
        node_p[node_name] = p

    n = len(te)
    cat_probs = np.zeros((n, len(common.CATEGORIES)))
    for cat in common.CATEGORIES:
        cidx = CI[cat]
        prob = np.ones(n)
        for node_name, reach, pos in nodes:
            if cidx in reach:
                p = node_p[node_name]
                prob = prob * (p if cidx in pos else (1.0 - p))
        cat_probs[:, cidx] = prob

    y = np.fromiter((r["y"] for r in te), int, n)
    eps = 1e-300
    row_dev = -2.0 * np.log(np.maximum(cat_probs[np.arange(n), y], eps))
    return row_dev, float(row_dev.sum() / n)


def paired_stats(diff):
    n = len(diff)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(n)
    t = mean / se
    p = float(stats.t.sf(abs(t), df=n - 1) * 2)
    return dict(n=n, mean=mean, sd=sd, se=se, t=float(t), p=p)


def bootstrap_by_row(diff, n_boot=10000, seed=1):
    rng = np.random.RandomState(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means[b] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    frac_d_wins = float((means > 0).mean())  # diff = devA - devD; >0 means D beats A
    return dict(lo=float(lo), hi=float(hi), frac_d_wins=frac_d_wins)


def bootstrap_by_game(diff, game_ids, n_boot=10000, seed=1):
    games = sorted(set(game_ids))
    gidx = {g: i for i, g in enumerate(games)}
    ng = len(games)
    g_sum = np.zeros(ng)
    g_cnt = np.zeros(ng)
    gi = np.fromiter((gidx[g] for g in game_ids), int, len(game_ids))
    np.add.at(g_sum, gi, diff)
    np.add.at(g_cnt, gi, 1.0)

    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.randint(0, ng, size=ng)
        means[b] = g_sum[samp].sum() / g_cnt[samp].sum()
    lo, hi = np.percentile(means, [2.5, 97.5])
    frac_d_wins = float((means > 0).mean())
    return dict(lo=float(lo), hi=float(hi), frac_d_wins=frac_d_wins, n_games=ng)


# -------------------------------------------------- TASK 3: psi grid edge --

def select_plateau_min(cands, tol):
    """Among (dev, key) candidates within `tol` of the best dev, return the
    one with the SMALLEST key -- least extreme choice statistically
    indistinguishable from the observed best. Same idea as step6's
    select_plateau, specialised to a single scalar (psi) instead of a
    (lam_bat, lam_pit) pair."""
    best_dev = min(c[0] for c in cands)
    within = [c for c in cands if c[0] <= best_dev + tol]
    return min(within, key=lambda c: c[1])


def refit_con_HR_widened(tr, ifit, BI, PI, n_bat, n_pit, season_idx):
    nodes = S6.SHAPES["D"]
    node_name, reach, pos = [nd for nd in nodes if nd[0] == "con_HR"][0]

    fit_rows = [r for r in tr if r["game_id"] in ifit]
    val_rows = [r for r in tr if r["game_id"] not in ifit]
    F = pack(fit_rows, reach, pos, BI, PI, season_idx)
    V = pack(val_rows, reach, pos, BI, PI, season_idx)
    nv = max(1, len(V[3]))

    def val(psi, lb, lp, warm=None):
        th = step1.fit(*F, n_bat, n_pit, psi, lb, lp, warm)
        return step1.node_dev(th, *V, n_bat, n_pit, psi) / nv, th

    LAM_GRID = S6.LAM_GRID
    TOL = S6.LAM_TOL

    # stage 1: lambda at psi=1 (does not depend on the psi grid at all)
    cands1 = []
    for lb in LAM_GRID:
        warm = None
        for lp in LAM_GRID:
            d, warm = val(1.0, lb, lp, warm)
            cands1.append((d, lb, lp))
    _, lb1, lp1 = S6.select_plateau(cands1, TOL)

    # stage 2: psi over the WIDENED grid, at (lb1, lp1), with a probe of the
    # full surface so we can report exactly where it flattens (or, if it
    # doesn't flatten but instead turns back up, where the true interior
    # minimum sits).
    PSI_WIDE = sorted(set(S6.PSI_GRID) | {11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                           17.0, 18.0, 19.0, 20.0, 25.0, 30.0,
                                           50.0, 100.0, 200.0, 500.0})
    psi_surface = []
    warm = None
    for psi in PSI_WIDE:
        d, warm = val(psi, lb1, lp1, warm)
        psi_surface.append((d, psi))
    psi_hat_argmin = min(psi_surface)[1]
    psi_hat = select_plateau_min(psi_surface, TOL)[1]

    # where does the surface flatten? first psi (in ascending order) after
    # which every subsequent candidate is within TOL of the eventual best.
    best_dev = min(d for d, _ in psi_surface)
    ordered = sorted(psi_surface, key=lambda c: c[1])
    flatten_at = None
    for i, (d, psi) in enumerate(ordered):
        if all(dd <= best_dev + TOL for dd, _ in ordered[i:]):
            flatten_at = psi
            break

    # stage 3: reselect lambda at the new psi_hat
    cands3 = []
    for lb in LAM_GRID:
        warm = None
        for lp in LAM_GRID:
            d, warm = val(psi_hat, lb, lp, warm)
            cands3.append((d, lb, lp))
    _, lb2, lp2 = S6.select_plateau(cands3, TOL)

    return dict(
        psi_surface=[(float(d), float(p)) for d, p in psi_surface],
        psi_hat_argmin=float(psi_hat_argmin),
        psi_hat_plateau=float(psi_hat),
        plateau_tol=TOL,
        flatten_at=float(flatten_at) if flatten_at is not None else None,
        lam_bat=lb2, lam_pit=lp2,
    ), (reach, pos)


def shape_D_test_deviance_with_new_con_HR(tr, te, BI, PI, n_bat, n_pit,
                                           season_idx, lb2, lp2, psi_hat,
                                           reach, pos, other_node_dev):
    TR = pack(tr, reach, pos, BI, PI, season_idx)
    TE = pack(te, reach, pos, BI, PI, season_idx)
    theta_final = step1.fit(*TR, n_bat, n_pit, psi_hat, lb2, lp2)
    d_ao = step1.node_dev(theta_final, *TE, n_bat, n_pit, psi_hat)
    n_test = len(te)
    new_con_HR_dev = d_ao / n_test
    new_total = other_node_dev + new_con_HR_dev
    return new_con_HR_dev, new_total


# ------------------------------------------------------------------- main --

def main():
    with open(RESULT6_PATH) as fh:
        r6 = json.load(fh)

    node_hp = {}
    for shape in ("A", "B", "C", "D"):
        node_hp[shape] = {}
        for nd in r6[shape]["nodes"]:
            node_hp[shape][nd["node"]] = (nd["lam_bat"], nd["lam_pit"], nd["psi"])

    rows, tr, te, ifit, BI, PI, n_bat, n_pit, season_idx = setup()
    n_test = len(te)
    log(f"train {len(tr)} test {n_test}  batters {n_bat} pitchers {n_pit}")

    out = {}

    # ---- TASK 1 ----
    log("=== TASK 1: validation-split ranking ===")
    val_results = {}
    for shape in ("A", "B", "C", "D"):
        total_val, per_node = validation_deviance(
            shape, tr, ifit, BI, PI, n_bat, n_pit, season_idx, node_hp)
        val_results[shape] = dict(total_val_deviance=total_val, nodes=per_node)
        log(f"  shape {shape}: validation deviance = {total_val:.5f}  "
            f"(frozen test was {r6[shape]['total_deviance']:.5f})")
    val_rank = sorted(val_results, key=lambda s: val_results[s]["total_val_deviance"])
    test_rank = sorted(("A", "B", "C", "D"), key=lambda s: r6[s]["total_deviance"])
    log(f"  validation ranking (best first): {val_rank}")
    log(f"  frozen-test ranking (best first): {test_rank}")
    out["task1_validation"] = dict(results=val_results, val_rank=val_rank,
                                    test_rank=test_rank,
                                    winner_agrees=(val_rank[0] == test_rank[0]))

    # ---- TASK 2 ----
    log("=== TASK 2: paired significance of the A/C-vs-D margin ===")
    row_dev = {}
    recon_total = {}
    for shape in ("A", "C", "D"):
        rd, tot = refit_row_dev(shape, tr, te, BI, PI, n_bat, n_pit, season_idx, node_hp)
        row_dev[shape] = rd
        recon_total[shape] = tot
        log(f"  shape {shape}: reconstructed per-row total = {tot:.5f}  "
            f"(recorded {r6[shape]['total_deviance']:.5f}, "
            f"diff {tot - r6[shape]['total_deviance']:+.6f})")

    game_ids = [r["game_id"] for r in te]

    task2 = {}
    for a_name in ("A", "C"):
        diff = row_dev[a_name] - row_dev["D"]  # >0 means D has lower (better) deviance
        stat = paired_stats(diff)
        boot_row = bootstrap_by_row(diff, n_boot=10000, seed=1)
        boot_game = bootstrap_by_game(diff, game_ids, n_boot=10000, seed=1)
        log(f"  {a_name} vs D: mean={stat['mean']:.5f} sd={stat['sd']:.4f} "
            f"se={stat['se']:.6f} t={stat['t']:.3f} p={stat['p']:.3g}")
        log(f"    bootstrap by row : 95% CI [{boot_row['lo']:.5f}, {boot_row['hi']:.5f}]  "
            f"P(D beats {a_name})={boot_row['frac_d_wins']:.3f}")
        log(f"    bootstrap by game: 95% CI [{boot_game['lo']:.5f}, {boot_game['hi']:.5f}]  "
            f"P(D beats {a_name})={boot_game['frac_d_wins']:.3f}  (n_games={boot_game['n_games']})")
        task2[f"{a_name}_vs_D"] = dict(stat=stat, bootstrap_by_row=boot_row,
                                        bootstrap_by_game=boot_game)
    out["task2_paired"] = dict(reconstructed_total=recon_total, comparisons=task2)

    # ---- TASK 3 ----
    log("=== TASK 3: widen PSI_GRID and re-select shape D's con_HR ===")
    sel, (reach, pos) = refit_con_HR_widened(tr, ifit, BI, PI, n_bat, n_pit, season_idx)
    log(f"  psi surface (val dev per row) at widened grid:")
    for d, psi in sorted(sel["psi_surface"], key=lambda c: c[1]):
        log(f"    psi={psi:<7g} val_dev={d:.6f}")
    log(f"  argmin psi = {sel['psi_hat_argmin']:g}   "
        f"plateau-selected psi = {sel['psi_hat_plateau']:g}  (tol={sel['plateau_tol']:g})")
    if sel["flatten_at"] is None:
        log("  surface does NOT flatten above the old grid top -- it turns back "
            "UP past the argmin, so the argmin is a genuine interior minimum, "
            "not a truncated asymptote")
    else:
        log(f"  surface flattens (within tol of eventual best) at psi >= {sel['flatten_at']:g}")
    log(f"  re-selected lam_bat={sel['lam_bat']:g} lam_pit={sel['lam_pit']:g}")

    other_node_dev = sum(nd["dev_ao"] for nd in r6["D"]["nodes"] if nd["node"] != "con_HR")
    old_con_HR_dev = [nd["dev_ao"] for nd in r6["D"]["nodes"] if nd["node"] == "con_HR"][0]
    new_con_HR_dev, new_total_D = shape_D_test_deviance_with_new_con_HR(
        tr, te, BI, PI, n_bat, n_pit, season_idx,
        sel["lam_bat"], sel["lam_pit"], sel["psi_hat_plateau"], reach, pos, other_node_dev)
    log(f"  con_HR frozen-test dev/row: old(psi=10) {old_con_HR_dev:.5f} -> "
        f"new(psi={sel['psi_hat_plateau']:g}) {new_con_HR_dev:.5f}")
    log(f"  shape D total frozen-test deviance: old {r6['D']['total_deviance']:.5f} -> "
        f"new {new_total_D:.5f}")
    lands_interior = sel["psi_hat_plateau"] < max(S6.PSI_GRID) or sel["psi_hat_plateau"] != max(
        p for _, p in sel["psi_surface"])
    out["task3_psi_edge"] = dict(
        psi_surface=sel["psi_surface"], psi_hat_argmin=sel["psi_hat_argmin"],
        psi_hat_plateau=sel["psi_hat_plateau"], plateau_tol=sel["plateau_tol"],
        flatten_at=sel["flatten_at"], lam_bat=sel["lam_bat"], lam_pit=sel["lam_pit"],
        old_con_HR_dev=old_con_HR_dev, new_con_HR_dev=new_con_HR_dev,
        old_shape_D_total=r6["D"]["total_deviance"], new_shape_D_total=new_total_D,
        lands_interior_of_widened_grid=(sel["psi_hat_plateau"] != max(p for _, p in sel["psi_surface"])),
    )

    out["runtime_sec"] = time.time() - T0
    with open(RESULT_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    log(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
