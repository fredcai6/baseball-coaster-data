"""Driver for the three latent2 arms. See latent_fit.py's module docstring
for the model definitions and the discipline this follows (inner-val-only
selection, plateau tie-break, grid-edge detection, null guard, restarts,
gradient check).

Usage:
    python run_all.py --arm grad          # gradient checks only (seconds)
    python run_all.py --arm 1             # arm 1 (shared latent, per-side lambda)
    python run_all.py --arm 2             # arm 2 (+ per-node loading multipliers, needs arm1 result)
    python run_all.py --arm 3             # arm 3 (hybrid: free effects + low rank)
    python run_all.py --arm all           # all of the above, in order

Results accumulate into result.json (merge, like step6_shapes.py).
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import latent_fit as L

T0 = time.time()


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def load_result():
    if os.path.exists(L.RESULT_PATH):
        with open(L.RESULT_PATH) as fh:
            return json.load(fh)
    return {}


def save_result(out):
    with open(L.RESULT_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    log(f"wrote {L.RESULT_PATH}")


# ------------------------------------------------------------- utilities --

def pick_plateau_1d(cands, tol=L.LAM_TOL, center=1.0):
    """cands: list of (val, x, extra...). Pick the x closest to `center` in
    log-space among those within tol of the best val -- same spirit as
    step6_shapes.select_plateau, specialised to a single scalar knob."""
    best_val = min(c[0] for c in cands)
    within = [c for c in cands if c[0] <= best_val + tol]
    return min(within, key=lambda c: (abs(np.log(c[1] / center)), c[0]))


def restart_final(fit_fn, score_fn, x0_base, jitter_fn, D_tr, D_te, psi0,
                   d, n_bat, n_pit, extra_args, n_restarts=L.N_RESTARTS, maxiter=900):
    """Multiple random restarts of the FINAL (full-train) fit at the selected
    hyperparameters. jitter_fn(x0_base, seed) -> perturbed init. Returns
    (canonical_test_dev, spread, restarts_log)."""
    n_test = len(D_te[0][3]) if False else None  # unused; n_test computed by caller
    log_rows = []
    for seed in range(n_restarts):
        t0 = time.time()
        x0 = jitter_fn(x0_base, seed)
        th, train_loss = fit_fn(x0, D_tr, psi0, d, n_bat, n_pit, *extra_args, maxiter=maxiter)
        test_raw = score_fn(th, D_te, psi0, d, n_bat, n_pit)
        log_rows.append(dict(seed=seed, train_penalized_loss=float(train_loss),
                              test_raw=float(test_raw), theta=th, fit_sec=time.time() - t0))
        log(f"    restart {seed}: train_loss={train_loss:.2f} ({time.time()-t0:.1f}s)")
    canonical = min(log_rows, key=lambda r: r["train_penalized_loss"])
    return canonical, log_rows


# ------------------------------------------------------------------ ARM 1 --

def run_arm1(d0):
    log("=" * 70)
    log("ARM 1: shared latent, per-side lambda")
    D_fit, D_val, D_tr, D_te = d0["D_fit"], d0["D_val"], d0["D_tr"], d0["D_te"]
    psi0, n_bat, n_pit, ps = d0["psi0"], d0["n_bat"], d0["n_pit"], d0["ps"]
    n_val, n_test = d0["n_val"], d0["n_test"]
    c1 = np.ones(L.NNODE)

    alpha0, beta0, B, Q = L.node_free_fits(D_fit, psi0, d0["lam_bat0"], d0["lam_pit0"],
                                            n_bat, n_pit, ps)

    # ---- Stage A: diagonal (lam_L = lam_M) sweep over d, to pick d* -------
    diag = []
    x0_by_d = {}
    for d in L.D_GRID:
        Lm, f, M, g = L.svd_init(B, Q, d)
        x0 = L.pack_shared(alpha0, beta0, f, g, Lm, M)
        x0_by_d[d] = x0
        warm = x0
        for lam in L.LAM_GRID:
            t0 = time.time()
            th, _ = L.fit_shared(warm, D_fit, psi0, d, n_bat, n_pit, lam, lam, c1, c1)
            val = L.score_shared(th, D_val, psi0, d, n_bat, n_pit) / n_val
            warm = th
            diag.append(dict(d=d, lam=lam, val=val, theta=th))
            log(f"  [arm1 diag] d={d} lam={lam:<7g} val={val:.5f} ({time.time()-t0:.1f}s)")
    best_diag = min(diag, key=lambda r: r["val"])
    d_star = best_diag["d"]
    log(f"  arm1 stage A: best diagonal d*={d_star} lam={best_diag['lam']:g} val={best_diag['val']:.5f}")

    # ---- Stage B: full per-side grid at d* ---------------------------------
    side = [dict(lam_L=r["lam"], lam_M=r["lam"], val=r["val"], theta=r["theta"])
            for r in diag if r["d"] == d_star]
    for lb in L.LAM_GRID:
        warm = x0_by_d[d_star]
        for lp in L.LAM_GRID:
            if lb == lp:
                continue  # already have the diagonal from stage A
            t0 = time.time()
            th, _ = L.fit_shared(warm, D_fit, psi0, d_star, n_bat, n_pit, lb, lp, c1, c1)
            val = L.score_shared(th, D_val, psi0, d_star, n_bat, n_pit) / n_val
            warm = th
            side.append(dict(lam_L=lb, lam_M=lp, val=val, theta=th))
            log(f"  [arm1 side] lam_L={lb:<7g} lam_M={lp:<7g} val={val:.5f} ({time.time()-t0:.1f}s)")

    cands = [(r["val"], r["lam_L"], r["lam_M"]) for r in side]
    best_val, lam_L_star, lam_M_star = L.select_plateau(cands)
    best_row = next(r for r in side if r["lam_L"] == lam_L_star and r["lam_M"] == lam_M_star
                     and abs(r["val"] - best_val) < 1e-12)
    lo, hi = L.LAM_GRID[0], L.LAM_GRID[-1]
    edge = []
    if lam_L_star in (lo, hi):
        edge.append(f"lam_L={lam_L_star:g}")
    if lam_M_star in (lo, hi):
        edge.append(f"lam_M={lam_M_star:g}")
    log(f"  arm1 SELECTED: d*={d_star} lam_L*={lam_L_star:g} lam_M*={lam_M_star:g} "
        f"inner_val={best_val:.5f}" + (f"  <<< GRID EDGE {edge}" if edge else "  (interior)"))

    guard_pass = bool(best_val < L.NULL_DEV)
    log(f"  NULL GUARD: inner-val {best_val:.5f} vs null {L.NULL_DEV}  {'PASS' if guard_pass else 'FAIL'}")
    if not guard_pass:
        return dict(status="guard_failed", inner_val=best_val, d_star=d_star,
                    lam_L=lam_L_star, lam_M=lam_M_star, grid_edge=edge)

    # ---- psi re-selection verification (arm 1 only; see module docstring) --
    ps_dim = ps
    alphaF, betaF, fF, gF, LF, MF = L.unpack_shared(best_row["theta"], d_star, n_bat, n_pit, ps_dim)
    psi_check = []
    inherited_tot, reselect_tot = 0.0, 0.0
    for n, (name, reach, pos) in enumerate(L.NODES):
        Fn = D_fit[n]; Vn = D_val[n]
        best_psi, best_raw, inh_raw = L.resweep_node_psi(
            n, Fn, Vn, LF, MF, d_star, ps_dim, lam_L_star, lam_M_star, 1.0, 1.0, psi0[n])
        inherited_tot += inh_raw
        reselect_tot += best_raw
        psi_check.append(dict(node=name, inherited_psi=float(psi0[n]), reselected_psi=float(best_psi),
                               changed=bool(best_psi != psi0[n])))
        log(f"  [psi check] {name:9} inherited={psi0[n]:g} reselected={best_psi:g} "
            f"{'CHANGED' if best_psi != psi0[n] else 'same'}")
    inherited_tot /= n_val
    reselect_tot /= n_val
    log(f"  psi check TOTAL inner-val: inherited={inherited_tot:.5f} reselected={reselect_tot:.5f} "
        f"delta={reselect_tot-inherited_tot:+.6f}")

    # ---- final fit + restarts on full TRAIN --------------------------------
    alphaT, betaT, BT, QT = L.node_free_fits(D_tr, psi0, d0["lam_bat0"], d0["lam_pit0"], n_bat, n_pit, ps)
    LmT, fT, MT, gT = L.svd_init(BT, QT, d_star)
    x0_final = L.pack_shared(alphaT, betaT, fT, gT, LmT, MT)

    def jitter(x0_base, seed):
        rng = np.random.RandomState(3000 + seed)
        if seed == 0:
            return x0_base.copy()  # restart 0 = clean SVD init, no jitter
        alpha, beta, f, g, Lm, M = L.unpack_shared(x0_base, d_star, n_bat, n_pit, ps_dim)
        scale_f = 0.3 * (np.std(f) + 1e-6)
        scale_g = 0.3 * (np.std(g) + 1e-6)
        scale_L = 0.3 * (np.std(Lm) + 1e-6)
        scale_M = 0.3 * (np.std(M) + 1e-6)
        f2 = f + rng.normal(scale=scale_f, size=f.shape)
        g2 = g + rng.normal(scale=scale_g, size=g.shape)
        L2 = Lm + rng.normal(scale=scale_L, size=Lm.shape)
        M2 = M + rng.normal(scale=scale_M, size=M.shape)
        return L.pack_shared(alpha, beta, f2, g2, L2, M2)

    canonical, restarts_log = restart_final(
        L.fit_shared, L.score_shared, x0_final, jitter, D_tr, D_te, psi0, d_star, n_bat, n_pit,
        extra_args=(lam_L_star, lam_M_star, c1, c1))
    test_devs = [r["test_raw"] / n_test for r in restarts_log]
    spread = max(test_devs) - min(test_devs)
    canonical_test = canonical["test_raw"] / n_test
    log(f"  arm1 FINAL frozen-test deviance (canonical restart) = {canonical_test:.5f}  "
        f"(restart spread {spread:.6f}, min {min(test_devs):.5f} max {max(test_devs):.5f})")
    log(f"  vs shape D target {L.SHAPE_D_TARGET:.5f}  vs null {L.NULL_DEV:.5f}")

    return dict(status="ok", d_star=int(d_star), lam_L=float(lam_L_star), lam_M=float(lam_M_star),
                inner_val=float(best_val), grid_edge=edge, guard_pass=guard_pass,
                psi_check=psi_check, psi_inherited_total=float(inherited_tot),
                psi_reselected_total=float(reselect_tot),
                test_deviance=float(canonical_test), restart_spread=float(spread),
                restart_test_deviances=test_devs, canonical_seed=int(canonical["seed"]),
                shape_d_target=L.SHAPE_D_TARGET, null=L.NULL_DEV,
                diag_sweep=[{k: v for k, v in r.items() if k != "theta"} for r in diag],
                side_grid=[{k: v for k, v in r.items() if k != "theta"} for r in side],
                x0_final_theta=x0_final.tolist(),
                canonical_theta=canonical["theta"].tolist())


# ------------------------------------------------------------------ ARM 2 --

def run_arm2(d0, arm1_result):
    log("=" * 70)
    log("ARM 2: arm 1 + per-node loading multipliers (staged, greedy)")
    if arm1_result.get("status") != "ok":
        log("  arm1 did not pass its guard -- skipping arm2 (needs arm1's d*, lam_L*, lam_M*)")
        return dict(status="skipped_arm1_not_ok")

    D_fit, D_val, D_tr, D_te = d0["D_fit"], d0["D_val"], d0["D_tr"], d0["D_te"]
    psi0, n_bat, n_pit, ps = d0["psi0"], d0["n_bat"], d0["n_pit"], d0["ps"]
    n_val, n_test = d0["n_val"], d0["n_test"]
    d_star = arm1_result["d_star"]
    lam_L, lam_M = arm1_result["lam_L"], arm1_result["lam_M"]

    x0 = np.array(arm1_result["x0_final_theta"])  # unused directly; refit on F below
    alpha0, beta0, B, Q = L.node_free_fits(D_fit, psi0, d0["lam_bat0"], d0["lam_pit0"], n_bat, n_pit, ps)
    Lm0, f0, M0, g0 = L.svd_init(B, Q, d_star)
    warm = L.pack_shared(alpha0, beta0, f0, g0, Lm0, M0)
    # re-derive arm1's own F-fit at (d*, lam_L, lam_M) as the true starting point
    warm, _ = L.fit_shared(warm, D_fit, psi0, d_star, n_bat, n_pit, lam_L, lam_M,
                            np.ones(L.NNODE), np.ones(L.NNODE))
    base_val = L.score_shared(warm, D_val, psi0, d_star, n_bat, n_pit) / n_val
    log(f"  arm2 base (arm1's config, c_f=c_g=1): inner_val={base_val:.5f}")

    c_f = np.ones(L.NNODE); c_g = np.ones(L.NNODE)
    edge_hits = []
    lo, hi = L.C_GRID[0], L.C_GRID[-1]

    # pass 1: greedy per-node sweep of c_f, one node at a time, in node order
    for n, (name, _, _) in enumerate(L.NODES):
        cands = []
        for c in L.C_GRID:
            cf_try = c_f.copy(); cf_try[n] = c
            th, _ = L.fit_shared(warm, D_fit, psi0, d_star, n_bat, n_pit, lam_L, lam_M, cf_try, c_g)
            val = L.score_shared(th, D_val, psi0, d_star, n_bat, n_pit) / n_val
            cands.append((val, c, th))
        val_b, c_b, th_b = pick_plateau_1d(cands)
        c_f[n] = c_b; warm = th_b
        if c_b in (lo, hi):
            edge_hits.append(f"{name}:c_f={c_b:g}")
        log(f"  [arm2 pass1 c_f] {name:9} c_f*={c_b:g} val={val_b:.5f}")

    # pass 2: greedy per-node sweep of c_g, given pass-1's c_f
    for n, (name, _, _) in enumerate(L.NODES):
        cands = []
        for c in L.C_GRID:
            cg_try = c_g.copy(); cg_try[n] = c
            th, _ = L.fit_shared(warm, D_fit, psi0, d_star, n_bat, n_pit, lam_L, lam_M, c_f, cg_try)
            val = L.score_shared(th, D_val, psi0, d_star, n_bat, n_pit) / n_val
            cands.append((val, c, th))
        val_b, c_b, th_b = pick_plateau_1d(cands)
        c_g[n] = c_b; warm = th_b
        if c_b in (lo, hi):
            edge_hits.append(f"{name}:c_g={c_b:g}")
        log(f"  [arm2 pass2 c_g] {name:9} c_g*={c_b:g} val={val_b:.5f}")

    final_val = L.score_shared(warm, D_val, psi0, d_star, n_bat, n_pit) / n_val
    log(f"  arm2 SELECTED multipliers: c_f={c_f.tolist()} c_g={c_g.tolist()}")
    log(f"  arm2 inner_val after staged sweep = {final_val:.5f}  (base was {base_val:.5f}, "
        f"delta {final_val-base_val:+.6f})")
    if edge_hits:
        log(f"  arm2 GRID EDGE HITS: {edge_hits}")

    guard_pass = bool(final_val < L.NULL_DEV)
    log(f"  NULL GUARD: inner-val {final_val:.5f} vs null {L.NULL_DEV}  {'PASS' if guard_pass else 'FAIL'}")
    if not guard_pass:
        return dict(status="guard_failed", inner_val=final_val, d_star=d_star,
                    lam_L=lam_L, lam_M=lam_M, c_f=c_f.tolist(), c_g=c_g.tolist(), grid_edge=edge_hits)

    # ---- final fit + restarts on full TRAIN --------------------------------
    alphaT, betaT, BT, QT = L.node_free_fits(D_tr, psi0, d0["lam_bat0"], d0["lam_pit0"], n_bat, n_pit, ps)
    LmT, fT, MT, gT = L.svd_init(BT, QT, d_star)
    x0_final = L.pack_shared(alphaT, betaT, fT, gT, LmT, MT)

    def jitter(x0_base, seed):
        rng = np.random.RandomState(4000 + seed)
        if seed == 0:
            return x0_base.copy()
        alpha, beta, f, g, Lm, M = L.unpack_shared(x0_base, d_star, n_bat, n_pit, ps)
        f2 = f + rng.normal(scale=0.3 * (np.std(f) + 1e-6), size=f.shape)
        g2 = g + rng.normal(scale=0.3 * (np.std(g) + 1e-6), size=g.shape)
        L2 = Lm + rng.normal(scale=0.3 * (np.std(Lm) + 1e-6), size=Lm.shape)
        M2 = M + rng.normal(scale=0.3 * (np.std(M) + 1e-6), size=M.shape)
        return L.pack_shared(alpha, beta, f2, g2, L2, M2)

    canonical, restarts_log = restart_final(
        L.fit_shared, L.score_shared, x0_final, jitter, D_tr, D_te, psi0, d_star, n_bat, n_pit,
        extra_args=(lam_L, lam_M, c_f, c_g))
    test_devs = [r["test_raw"] / n_test for r in restarts_log]
    spread = max(test_devs) - min(test_devs)
    canonical_test = canonical["test_raw"] / n_test
    log(f"  arm2 FINAL frozen-test deviance (canonical restart) = {canonical_test:.5f}  "
        f"(restart spread {spread:.6f})")
    log(f"  vs shape D target {L.SHAPE_D_TARGET:.5f}  vs null {L.NULL_DEV:.5f}")

    return dict(status="ok", d_star=int(d_star), lam_L=float(lam_L), lam_M=float(lam_M),
                c_f=c_f.tolist(), c_g=c_g.tolist(), inner_val=float(final_val),
                base_inner_val=float(base_val), grid_edge=edge_hits, guard_pass=guard_pass,
                test_deviance=float(canonical_test), restart_spread=float(spread),
                restart_test_deviances=test_devs, canonical_seed=int(canonical["seed"]),
                shape_d_target=L.SHAPE_D_TARGET, null=L.NULL_DEV)


# ------------------------------------------------------------------ ARM 3 --

def run_arm3(d0):
    log("=" * 70)
    log("ARM 3: hybrid -- free per-node effects + low-rank channel on top")
    D_fit, D_val, D_tr, D_te = d0["D_fit"], d0["D_val"], d0["D_tr"], d0["D_te"]
    psi0, n_bat, n_pit, ps = d0["psi0"], d0["n_bat"], d0["n_pit"], d0["ps"]
    n_val, n_test = d0["n_val"], d0["n_test"]
    lam_bat0, lam_pit0 = d0["lam_bat0"], d0["lam_pit0"]

    alpha0, beta0, B, Q = L.node_free_fits(D_fit, psi0, lam_bat0, lam_pit0, n_bat, n_pit, ps)

    diag = []
    x0_by_d = {}
    for d in L.D_GRID:
        Lm, f, M, g = L.svd_init(B, Q, d)
        b0 = (B - Lm @ f.T).T
        q0 = (Q - M @ g.T).T
        x0 = L.pack_arm3(alpha0, beta0, f, g, Lm, M, b0, q0)
        x0_by_d[d] = x0
        warm = x0
        for lam in L.LAM_GRID:
            t0 = time.time()
            th, _ = L.fit_arm3(warm, D_fit, psi0, d, n_bat, n_pit, lam, lam, lam_bat0, lam_pit0)
            val = L.score_arm3(th, D_val, psi0, d, n_bat, n_pit) / n_val
            warm = th
            diag.append(dict(d=d, lam=lam, val=val, theta=th))
            log(f"  [arm3 diag] d={d} lam={lam:<7g} val={val:.5f} ({time.time()-t0:.1f}s)")
    best_diag = min(diag, key=lambda r: r["val"])
    d_star = best_diag["d"]
    log(f"  arm3 stage A: best diagonal d*={d_star} lam={best_diag['lam']:g} val={best_diag['val']:.5f}")

    side = [dict(lam_L=r["lam"], lam_M=r["lam"], val=r["val"], theta=r["theta"])
            for r in diag if r["d"] == d_star]
    for lb in L.LAM_GRID:
        warm = x0_by_d[d_star]
        for lp in L.LAM_GRID:
            if lb == lp:
                continue
            t0 = time.time()
            th, _ = L.fit_arm3(warm, D_fit, psi0, d_star, n_bat, n_pit, lb, lp, lam_bat0, lam_pit0)
            val = L.score_arm3(th, D_val, psi0, d_star, n_bat, n_pit) / n_val
            warm = th
            side.append(dict(lam_L=lb, lam_M=lp, val=val, theta=th))
            log(f"  [arm3 side] lam_L={lb:<7g} lam_M={lp:<7g} val={val:.5f} ({time.time()-t0:.1f}s)")

    cands = [(r["val"], r["lam_L"], r["lam_M"]) for r in side]
    best_val, lam_L_star, lam_M_star = L.select_plateau(cands)
    lo, hi = L.LAM_GRID[0], L.LAM_GRID[-1]
    edge = []
    if lam_L_star in (lo, hi):
        edge.append(f"lam_L={lam_L_star:g}")
    if lam_M_star in (lo, hi):
        edge.append(f"lam_M={lam_M_star:g}")
    log(f"  arm3 SELECTED: d*={d_star} lam_L*={lam_L_star:g} lam_M*={lam_M_star:g} "
        f"inner_val={best_val:.5f}" + (f"  <<< GRID EDGE {edge}" if edge else "  (interior)"))

    guard_pass = bool(best_val < L.NULL_DEV)
    log(f"  NULL GUARD: inner-val {best_val:.5f} vs null {L.NULL_DEV}  {'PASS' if guard_pass else 'FAIL'}")
    if not guard_pass:
        return dict(status="guard_failed", inner_val=best_val, d_star=d_star,
                    lam_L=lam_L_star, lam_M=lam_M_star, grid_edge=edge)

    # ---- final fit + restarts on full TRAIN --------------------------------
    alphaT, betaT, BT, QT = L.node_free_fits(D_tr, psi0, lam_bat0, lam_pit0, n_bat, n_pit, ps)
    LmT, fT, MT, gT = L.svd_init(BT, QT, d_star)
    b0T = (BT - LmT @ fT.T).T
    q0T = (QT - MT @ gT.T).T
    x0_final = L.pack_arm3(alphaT, betaT, fT, gT, LmT, MT, b0T, q0T)

    def jitter(x0_base, seed):
        rng = np.random.RandomState(5000 + seed)
        if seed == 0:
            return x0_base.copy()
        alpha, beta, f, g, Lm, M, b, q = L.unpack_arm3(x0_base, d_star, n_bat, n_pit, ps)
        f2 = f + rng.normal(scale=0.3 * (np.std(f) + 1e-6), size=f.shape)
        g2 = g + rng.normal(scale=0.3 * (np.std(g) + 1e-6), size=g.shape)
        L2 = Lm + rng.normal(scale=0.3 * (np.std(Lm) + 1e-6), size=Lm.shape)
        M2 = M + rng.normal(scale=0.3 * (np.std(M) + 1e-6), size=M.shape)
        return L.pack_arm3(alpha, beta, f2, g2, L2, M2, b, q)

    canonical, restarts_log = restart_final(
        L.fit_arm3, L.score_arm3, x0_final, jitter, D_tr, D_te, psi0, d_star, n_bat, n_pit,
        extra_args=(lam_L_star, lam_M_star, lam_bat0, lam_pit0))
    test_devs = [r["test_raw"] / n_test for r in restarts_log]
    spread = max(test_devs) - min(test_devs)
    canonical_test = canonical["test_raw"] / n_test

    # how much does the low-rank channel actually carry, at the canonical fit?
    alphaC, betaC, fC, gC, LC, MC, bC, qC = L.unpack_arm3(canonical["theta"], d_star, n_bat, n_pit, ps)
    latent_norm = float(np.linalg.norm(LC) * np.linalg.norm(fC) + np.linalg.norm(MC) * np.linalg.norm(gC))
    free_norm = float(np.linalg.norm(bC) + np.linalg.norm(qC))

    log(f"  arm3 FINAL frozen-test deviance (canonical restart) = {canonical_test:.5f}  "
        f"(restart spread {spread:.6f})")
    log(f"  vs shape D target {L.SHAPE_D_TARGET:.5f}  vs null {L.NULL_DEV:.5f}")
    log(f"  canonical fit: ||L||*||f||+||M||*||g|| = {latent_norm:.3f}  ||b||+||q|| = {free_norm:.3f}")

    return dict(status="ok", d_star=int(d_star), lam_L=float(lam_L_star), lam_M=float(lam_M_star),
                inner_val=float(best_val), grid_edge=edge, guard_pass=guard_pass,
                test_deviance=float(canonical_test), restart_spread=float(spread),
                restart_test_deviances=test_devs, canonical_seed=int(canonical["seed"]),
                latent_channel_norm=latent_norm, free_channel_norm=free_norm,
                shape_d_target=L.SHAPE_D_TARGET, null=L.NULL_DEV,
                diag_sweep=[{k: v for k, v in r.items() if k != "theta"} for r in diag],
                side_grid=[{k: v for k, v in r.items() if k != "theta"} for r in side])


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", help="grad | 1 | 2 | 3 | all")
    args = ap.parse_args()

    out = load_result()

    if args.arm in ("grad", "all"):
        err_shared = L.gradient_check_shared()
        err_arm3 = L.gradient_check_arm3()
        log(f"gradient check (arm1/arm2 objective, obj_shared): error = {err_shared:.3e}")
        log(f"gradient check (arm3 objective, obj_arm3):        error = {err_arm3:.3e}")
        out["grad_check"] = dict(obj_shared=float(err_shared), obj_arm3=float(err_arm3))
        if err_shared > 1e-4 or err_arm3 > 1e-4:
            log("GRADIENT CHECK FAILED -- refusing to fit anything until this is fixed.")
            out["status"] = "gradient_check_failed"
            save_result(out)
            return
        save_result(out)
        if args.arm == "grad":
            return

    d0 = None
    if args.arm in ("1", "2", "3", "all"):
        d0 = L.setup()
        out["null"] = L.NULL_DEV
        out["shape_d_target"] = L.SHAPE_D_TARGET
        out["structural_only_test_deviance"] = L.structural_only_dev(
            d0["D_te"], d0["psi0"], d0["n_bat"], d0["n_pit"], d0["ps"], d0["n_test"])
        log(f"structural-only (no player terms) frozen-test deviance = "
            f"{out['structural_only_test_deviance']:.5f}")
        save_result(out)

    if args.arm in ("1", "all"):
        out["arm1"] = run_arm1(d0)
        save_result(out)

    if args.arm in ("2", "all"):
        out["arm2"] = run_arm2(d0, out.get("arm1", {}))
        save_result(out)

    if args.arm in ("3", "all"):
        out["arm3"] = run_arm3(d0)
        save_result(out)

    log("")
    log("=" * 70)
    log(f"SUMMARY  null={L.NULL_DEV:.5f}  shape D target={L.SHAPE_D_TARGET:.5f}")
    for a in ("arm1", "arm2", "arm3"):
        if a in out and out[a].get("status") == "ok":
            log(f"  {a}: test={out[a]['test_deviance']:.5f}  "
                f"(delta vs shape D {out[a]['test_deviance']-L.SHAPE_D_TARGET:+.5f})  "
                f"spread={out[a]['restart_spread']:.6f}")
        elif a in out:
            log(f"  {a}: {out[a].get('status')}")


if __name__ == "__main__":
    main()
