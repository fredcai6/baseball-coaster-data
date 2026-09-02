"""Assemble spikes/latent_style/LATENT_STYLE.md from result.json. Pure
formatting -- no new computation -- so the prose file can be regenerated
without re-running any fit."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "result.json")))

CAT_ORDER = ["K", "BB", "HBP", "F", "G", "1B", "2B", "3B", "HR", "OTHER"]


def pct(x):
    return f"{100*x:.1f}" if x is not None else "-"


def player_row(row):
    r = row["rates"]
    cells = " | ".join(pct(r[c]) for c in CAT_ORDER)
    return f"| {row['player']} | {row['score']:+.3f} | {row['pa']} | {cells} |"


def axis_section(fh, side_label, axes):
    for ax in axes:
        fh.write(f"\n### {side_label} axis {ax['axis']}  (singular value {ax['singular_value']:.2f})\n\n")
        fh.write(f"Run-value correlation: {ax['run_value_corr']:+.3f} -> **{ax['interpretation']}**\n\n")
        fh.write("Node loadings, most negative to most positive:\n\n")
        fh.write("| node | loading |\n|---|---|\n")
        for node, val in ax["node_loadings"]:
            fh.write(f"| {node} | {val:+.3f} |\n")
        fh.write("\nTop 10 (most positive on this axis):\n\n")
        fh.write("| player | score | PA | " + " | ".join(CAT_ORDER) + " |\n")
        fh.write("|---|---|---|" + "---|" * len(CAT_ORDER) + "\n")
        for row in ax["top10"]:
            fh.write(player_row(row) + "\n")
        fh.write("\nBottom 10 (most negative on this axis):\n\n")
        fh.write("| player | score | PA | " + " | ".join(CAT_ORDER) + " |\n")
        fh.write("|---|---|---|" + "---|" * len(CAT_ORDER) + "\n")
        for row in ax["bottom10"]:
            fh.write(player_row(row) + "\n")


def test1_table(fh, side_key, side_label):
    s = d["test1"][side_key]
    fh.write(f"\n**{side_label}** -- {s['n_survive']} / {s['n_total']} players with "
              f">= {s['floor']} PA/BF in BOTH halves.\n\n")
    fh.write("Procrustes-aligned latent-coordinate split-half r (per axis):\n\n")
    fh.write("| axis | split-half r | Spearman-Brown r (full-sample) |\n|---|---|---|\n")
    for a in s["axis_r"]:
        fh.write(f"| {a['axis']} | {a['r_half']:.3f} | {a['r_spearman_brown']:.3f} |\n")
    fh.write("\nAlignment-free per-node predicted-effect (L @ f.T) split-half r:\n\n")
    fh.write("| node | split-half r | Spearman-Brown r (full-sample) |\n|---|---|---|\n")
    for nrow in s["node_r"]:
        fh.write(f"| {nrow['node']} | {nrow['r_half']:.3f} | {nrow['r_spearman_brown']:.3f} |\n")


def test2_table(fh, side_key, side_label):
    s = d["test2"][side_key]
    fh.write(f"\n**{side_label}**\n\n")
    fh.write("| variable | n qualified | latent3 R2 | free9 R2 | pca3 R2 |\n|---|---|---|---|---|\n")
    for var, res in s.items():
        r2 = res["r2"]
        fh.write(f"| {var} | {res['n']} | {r2['latent3']:+.3f} | {r2['free9']:+.3f} | {r2['pca3']:+.3f} |\n")


with open(os.path.join(HERE, "LATENT_STYLE.md"), "w") as fh:
    fh.write("# Latent style: is the 3-D player latent a meaningful summary?\n\n")
    fh.write(
        "Question: latent2's arm 1 (shared rank-3 batter/pitcher latent inside shape D's "
        "9-node binarised tree, Aranda-Ordaz link, d=3 lam_L=lam_M=40, psi inherited from "
        "step6 shape D) compresses 9 free per-node player effects into 3 numbers at "
        "essentially no cost in test deviance (3.95109 vs shape D's 3.94526 free-effects "
        "target). Is that 3-D space a MEANINGFUL summary, or an arbitrary rotation of a "
        "quality-plus-noise ellipsoid? Two tests below decide it; both use the FIXED "
        "hyperparameters already selected in spikes/latent2/result.json -- no new "
        "hyperparameter search was run here.\n"
    )

    ff = d["full_fit"]
    fh.write(f"\nCanonical full-train fit: d={ff['d']:g} lam_L=lam_M={ff['lam']:g}, "
              f"n_bat={ff['n_bat']} n_pit={ff['n_pit']} n_train_pa={ff['n_train_pa']}, "
              f"{ff['n_restarts']} restarts (canonical seed {ff['canonical_seed']}, "
              f"train-loss spread {ff['train_loss_spread']:.4f} -- restart spread at these "
              "hyperparameters was already shown negligible in latent2/result.json "
              "(8.35e-7), so 3 restarts rather than 5 were used here for all three fits "
              "(full/halfA/halfB) to keep each stage inside the foreground time budget).\n")

    hf = d["half_fits"]
    fh.write(f"\nHalves split: seed recorded in halves_split.json, stratified by season, "
              f"disjoint games. Half A: {hf['A']['n_games']} games / {hf['A']['n_pa']} PA. "
              f"Half B: {hf['B']['n_games']} games / {hf['B']['n_pa']} PA.\n")

    fh.write("\n## Test 1 -- split-half reliability (the decisive one)\n")
    fh.write(f"\nAlignment: {d['test1']['procrustes']}\n")
    test1_table(fh, "batters", "Batters")
    test1_table(fh, "pitchers", "Pitchers")

    fh.write(
        "\n**Reading the axes (batters):** axis 0 (r=0.616 / SB 0.762) and axis 1 "
        "(r=0.477 / SB 0.646) replicate reasonably; axis 2 (r=0.227 / SB 0.370) is "
        "borderline-to-weak. **Pitchers:** axis 2 (r=0.606 / SB 0.755) is the strongest, "
        "axes 0-1 (r~0.40-0.42, SB ~0.57-0.59) are moderate. The alignment-free per-node "
        "check tells the same story at the node level: most nodes land r=0.35-0.63 "
        "(SB 0.5-0.78), except hit_2B, which is NEGATIVE for both sides (batters -0.324, "
        "pitchers -0.457) -- consistent with step6_shapes.py's own finding that hit_2B's "
        "validation surface is flat/degenerate (pitcher identity carries ~no signal for "
        "2B once conditioned on reaching {2B,3B,HR}); the Spearman-Brown formula is not "
        "meaningful for negative r and is reported as-is for completeness, not as evidence "
        "of anti-correlation.\n"
    )

    fh.write("\n## Test 2 -- external validity\n")
    fh.write(f"\nMethod: {d['test2']['method']} Qualified: >= {d['test2']['qual_floor_pa']} "
              "training PA/BF.\n")
    fh.write(
        "\nCaveats stated, not hidden: `gb_rate` (derived from `bb_type`) is PARTIALLY "
        "in-target -- out_F/con_OUT directly model F vs G, so a high R2 there is expected "
        "and is not evidence of external validity. `pitches_per_pa`, `swing_miss_rate`, "
        "`called_strike_rate`, `first_pitch_strike_rate` are 2026-only (pitch_seq is null "
        "in 2024/2025), so their qualified-n is much smaller (~136-140) than the other "
        "variables (~255-355).\n"
    )
    test2_table(fh, "batters", "Batters")
    test2_table(fh, "pitchers", "Pitchers")

    fh.write("\n## Presentation: canonical axes\n")
    fh.write(
        "\nCanonical rotation: SVD of L_bat @ F_bat.T (batters) and M_pit @ G_pit.T "
        "(pitchers) separately, axes ordered by singular value. Sign is arbitrary (SVD "
        "sign ambiguity) -- read +/- as 'one side' / 'the other side', not as an absolute "
        "direction, per spikes/npmr/latent.md's convention. Category rates are computed "
        "on the SAME training games the latent was fit on.\n"
    )
    axis_section(fh, "Batter", d["presentation"]["batters"])
    axis_section(fh, "Pitcher", d["presentation"]["pitchers"])

print("wrote", os.path.join(HERE, "LATENT_STYLE.md"))
