"""Build per-player external descriptive variables from pa_table.csv and
player_profiles.csv, computed over the SAME training games as the latent
fit -- for Test 2 (external validity: does the latent predict things the
model never saw as a target).

None of these come from common.load_pa()'s row dicts (which don't carry
spray/bb_type/position) -- pa_table.csv is read directly, in file order,
which spikes/common.py's load_pa() also preserves (verified: row i of
load_pa() == row i of pa_table.csv), so we zip by index rather than re-join
on keys.
"""
import csv, os, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PA_TABLE = os.path.join(ROOT, "artifacts", "derived", "pa_table.csv")
PROFILES = os.path.join(ROOT, "profiles", "player_profiles.csv")

# pull_score: raw spray direction mapped to a handedness-relative pull axis.
# Raw values (checked against the data): '', 'left', 'left_center', 'center',
# 'right_center', 'right'. Absolute direction score right=+1 ... left=-1,
# then flipped for LHB so + always means "pulled" (RHB pulls to LF/'left',
# LHB pulls to RF/'right') and - means "opposite field".
_SPRAY_SCORE = {"left": -1.0, "left_center": -0.5, "center": 0.0,
                "right_center": 0.5, "right": 1.0}


def _pull_score(spray, bats):
    if not spray or spray not in _SPRAY_SCORE or bats not in ("L", "R"):
        return None
    s = _SPRAY_SCORE[spray]
    return -s if bats == "R" else s   # RHB: 'left' (s=-1) is pulled -> +1


def _first_pitch_strike(pitch_seq, n_pitches):
    if pitch_seq is None or n_pitches is None or n_pitches < 1:
        return None
    if pitch_seq == "":
        return 1.0   # ball put in play on pitch 1 -> a strike by definition
    c = pitch_seq[0]
    if c == "B":
        return 0.0
    if c == "H":
        return 0.0   # hit-by-pitch on pitch 1 -- not a strike
    return 1.0        # K (called) / S (swinging) / F (foul)


_POS_TOKEN = {
    "c": "C", "catcher": "C",
    "1b": "IF", "2b": "IF", "3b": "IF", "ss": "IF", "inf": "IF", "if": "IF",
    "utl": "IF", "util": "IF", "utility": "IF", "mif": "IF", "cif": "IF",
    "infield": "IF", "inf.": "IF", "shortstop": "IF", "third": "IF", "second": "IF", "first": "IF",
    "of": "OF", "cf": "OF", "rf": "OF", "lf": "OF", "outfield": "OF", "of$": "OF",
    "dh": "DH",
}
_PRIORITY = ("C", "IF", "OF", "DH")


def bucket_position(pos_str):
    """C > IF > OF > DH priority for multi-position strings (e.g. '1B/OF' ->
    IF wins over OF is arbitrary but consistent; catcher always wins since a
    part-time catcher is descriptively 'a catcher'). Pitcher-only / unparsed
    strings -> None (dropped, not bucketed as a fifth category -- documented
    in LATENT_STYLE.md as a scope cut, not silently coerced)."""
    if not pos_str:
        return None
    toks = re.split(r"[\/\s,]+", pos_str.strip().lower())
    found = {_POS_TOKEN[t] for t in toks if t in _POS_TOKEN}
    for p in _PRIORITY:
        if p in found:
            return p
    return None


def _safe_float(s):
    """A handful of profile rows have garbage/shifted values (e.g. weight_lb
    "6'4" -- a height string that landed in the wrong column). Skip those
    rather than crash; they're a data-entry defect in player_profiles.csv,
    not something this spike should silently coerce or repair."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_profile_index():
    """(season, person_id) -> dict(position, height_in, weight_lb)."""
    idx = {}
    with open(PROFILES) as fh:
        for r in csv.DictReader(fh):
            if not r["person_id"]:
                continue
            key = (int(r["season"]), r["person_id"])
            h = _safe_float(r["height_in"]) if r["height_in"] else None
            w = _safe_float(r["weight_lb"]) if r["weight_lb"] else None
            idx[key] = dict(position=r["position"] or None, height_in=h, weight_lb=w)
    return idx


def build(rows, train_g, BI, PI):
    """rows: common.load_pa() output (full, all rows, in pa_table.csv order).
    train_g: set of training game ids. Returns dict of per-player aggregate
    stat dicts for batters and pitchers, restricted to training-game PAs.
    """
    prof = load_profile_index()

    with open(PA_TABLE) as fh:
        pa_rows = list(csv.DictReader(fh))
    assert len(pa_rows) == len(rows), (len(pa_rows), len(rows))

    n_bat, n_pit = len(BI), len(PI)
    # accumulators, batters
    bat = dict(
        pull_sum=np.zeros(n_bat), pull_n=np.zeros(n_bat),
        gb_sum=np.zeros(n_bat), bb_type_n=np.zeros(n_bat),
        pitches_sum=np.zeros(n_bat), pitches_n=np.zeros(n_bat),
        swing_miss_sum=np.zeros(n_bat), called_strike_sum=np.zeros(n_bat), pitch_n=np.zeros(n_bat),
        fps_sum=np.zeros(n_bat), fps_n=np.zeros(n_bat),
        pos_votes=[dict() for _ in range(n_bat)],
        height_sum=np.zeros(n_bat), height_n=np.zeros(n_bat),
        weight_sum=np.zeros(n_bat), weight_n=np.zeros(n_bat),
        bats_votes=[dict() for _ in range(n_bat)],
    )
    pit = dict(
        gb_sum=np.zeros(n_pit), bb_type_n=np.zeros(n_pit),
        pitches_sum=np.zeros(n_pit), pitches_n=np.zeros(n_pit),
        swing_miss_sum=np.zeros(n_pit), called_strike_sum=np.zeros(n_pit), pitch_n=np.zeros(n_pit),
        fps_sum=np.zeros(n_pit), fps_n=np.zeros(n_pit),
        height_sum=np.zeros(n_pit), height_n=np.zeros(n_pit),
        weight_sum=np.zeros(n_pit), weight_n=np.zeros(n_pit),
        throws_votes=[dict() for _ in range(n_pit)],
    )

    for r, pr in zip(rows, pa_rows):
        if r["game_id"] not in train_g:
            continue
        bi = BI[r["batter"]]; pj = PI[r["pitcher"]]
        season = r["season"]

        spray = pr["spray"] or None
        ps = _pull_score(spray, r["bats"])
        if ps is not None:
            bat["pull_sum"][bi] += ps; bat["pull_n"][bi] += 1

        bb_type = pr["bb_type"] or None
        if bb_type in ("gb", "fb", "pu", "ld"):
            is_gb = 1.0 if bb_type == "gb" else 0.0
            bat["gb_sum"][bi] += is_gb; bat["bb_type_n"][bi] += 1
            pit["gb_sum"][pj] += is_gb; pit["bb_type_n"][pj] += 1

        seq, npi = r["pitch_seq"], r["n_pitches"]
        if seq is not None and npi is not None:
            bat["pitches_sum"][bi] += npi; bat["pitches_n"][bi] += 1
            pit["pitches_sum"][pj] += npi; pit["pitches_n"][pj] += 1
            nS = seq.count("S"); nK = seq.count("K")
            bat["swing_miss_sum"][bi] += nS; bat["called_strike_sum"][bi] += nK; bat["pitch_n"][bi] += npi
            pit["swing_miss_sum"][pj] += nS; pit["called_strike_sum"][pj] += nK; pit["pitch_n"][pj] += npi
            fps = _first_pitch_strike(seq, npi)
            if fps is not None:
                bat["fps_sum"][bi] += fps; bat["fps_n"][bi] += 1
                pit["fps_sum"][pj] += fps; pit["fps_n"][pj] += 1

        key = (season, r["batter_person"])
        pinfo = prof.get(key)
        if pinfo:
            b = bucket_position(pinfo["position"])
            if b:
                bat["pos_votes"][bi][b] = bat["pos_votes"][bi].get(b, 0) + 1
            if pinfo["height_in"]:
                bat["height_sum"][bi] += pinfo["height_in"]; bat["height_n"][bi] += 1
            if pinfo["weight_lb"]:
                bat["weight_sum"][bi] += pinfo["weight_lb"]; bat["weight_n"][bi] += 1
        if r["bats"] in ("L", "R", "S"):
            bat["bats_votes"][bi][r["bats"]] = bat["bats_votes"][bi].get(r["bats"], 0) + 1

        pkey = (season, r["pitcher_person"])
        ppinfo = prof.get(pkey)
        if ppinfo:
            if ppinfo["height_in"]:
                pit["height_sum"][pj] += ppinfo["height_in"]; pit["height_n"][pj] += 1
            if ppinfo["weight_lb"]:
                pit["weight_sum"][pj] += ppinfo["weight_lb"]; pit["weight_n"][pj] += 1
        if r["throws"] in ("L", "R", "S"):
            pit["throws_votes"][pj][r["throws"]] = pit["throws_votes"][pj].get(r["throws"], 0) + 1

    def safe_div(num, den):
        out = np.full(len(num), np.nan)
        m = den > 0
        out[m] = num[m] / den[m]
        return out

    def mode_of(votes_list):
        out = [None] * len(votes_list)
        for i, v in enumerate(votes_list):
            if v:
                out[i] = max(v.items(), key=lambda kv: kv[1])[0]
        return out

    batters_out = dict(
        pull_score=safe_div(bat["pull_sum"], bat["pull_n"]),
        gb_rate=safe_div(bat["gb_sum"], bat["bb_type_n"]),
        pitches_per_pa=safe_div(bat["pitches_sum"], bat["pitches_n"]),
        swing_miss_rate=safe_div(bat["swing_miss_sum"], bat["pitch_n"]),
        called_strike_rate=safe_div(bat["called_strike_sum"], bat["pitch_n"]),
        first_pitch_strike_rate=safe_div(bat["fps_sum"], bat["fps_n"]),
        height_in=safe_div(bat["height_sum"], bat["height_n"]),
        weight_lb=safe_div(bat["weight_sum"], bat["weight_n"]),
        position=np.array(mode_of(bat["pos_votes"]), dtype=object),
        bats=np.array(mode_of(bat["bats_votes"]), dtype=object),
    )
    pitchers_out = dict(
        gb_rate=safe_div(pit["gb_sum"], pit["bb_type_n"]),
        pitches_per_pa=safe_div(pit["pitches_sum"], pit["pitches_n"]),
        swing_miss_rate=safe_div(pit["swing_miss_sum"], pit["pitch_n"]),
        called_strike_rate=safe_div(pit["called_strike_sum"], pit["pitch_n"]),
        first_pitch_strike_rate=safe_div(pit["fps_sum"], pit["fps_n"]),
        height_in=safe_div(pit["height_sum"], pit["height_n"]),
        weight_lb=safe_div(pit["weight_sum"], pit["weight_n"]),
        throws=np.array(mode_of(pit["throws_votes"]), dtype=object),
    )
    return batters_out, pitchers_out
