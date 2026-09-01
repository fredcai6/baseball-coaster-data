#!/usr/bin/env python3
"""Join archived team-roster pages to the corpus, producing player profiles.

    python scripts/build_profiles.py            # -> profiles/player_profiles.csv
    python scripts/build_profiles.py --check    # summarize without writing

Recovers the field the corpus has nowhere else: **handedness**. `bats_side` is
null on all 41,713 player records and appears in no boxscore or play-by-play
page. It is on the league site's team-roster pages, which this reads from the
raw archive (never re-fetched here -- fetching and parsing are separate so a
parser bug costs no requests against a 12s pacing floor).

Unlike `artifacts/derived/pa_table.csv`, this output IS committed. The
distinction is reproducibility from the repo: the plate-appearance table
rebuilds from `games/**` in two seconds, but these profiles derive from raw
HTML that the caller contract keeps out of git and on one PC. Nothing in the
repo can regenerate them -- losing the archive means re-scraping a live site
that may have changed underneath us. That makes them source data, and they
carry a `provenance.json` recording every page they came from so the claim is
auditable rather than asserted.

Archived pages are located through the CHECKPOINT, never by globbing archive
filenames. `archive_filename` truncates the URL slug at 80 characters, so a
long team slug (`grandjunctionjackalopes`, `missoulapaddleheads`) loses the
`-view-roster` tail and a filename glob silently drops it -- 9 of 36 pages, a
quarter of the sweep, with no error. The README says this in as many words:
the checkpoint, not the directory listing, is the authority on what was
fetched.

The join is by URL slug suffix: a roster row links to
`/sports/bsb/<season>/players/<name><4-char-suffix>` where the suffix is the
first four characters of the corpus `player_id`. Four characters is not quite
unique -- the corpus holds exactly one within-season collision -- so a match is
confirmed against the player's display name before it is accepted.
"""

import argparse
import collections
import csv
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from bc_pipeline import pa_table  # noqa: E402
from bc_pipeline.roster import parse_roster  # noqa: E402

ARCHIVE = Path.home() / "bc-raw-archive"
CHECKPOINT = ARCHIVE / "checkpoint.json"
ROSTER_URL = re.compile(r"/sports/bsb/(\d{4})/teams/([a-z0-9\-]+)\?view=roster$")


def source_records():
    """Provenance for every roster page the profiles were built from."""
    checkpoint = json.loads(CHECKPOINT.read_text())
    out = []
    for url, entry in sorted(checkpoint.items()):
        if ROSTER_URL.search(url) and entry.get("status") == "done":
            out.append({"url": url,
                        "content_hash": entry["content_hash"],
                        "fetched_at": entry["fetched_at"]})
    return out


def archived_roster_pages():
    """(team_slug, path) for every roster page the checkpoint records as done."""
    if not CHECKPOINT.exists():
        return []
    checkpoint = json.loads(CHECKPOINT.read_text())
    out = []
    for url, entry in sorted(checkpoint.items()):
        m = ROSTER_URL.search(url)
        if not m or entry.get("status") != "done":
            continue
        path = Path(entry["archived_path"])
        if path.exists():
            out.append((m.group(2), path))
    return out

COLUMNS = [
    "season", "player_id", "person_id", "career_id", "name",
    "bats", "throws", "position", "height_in", "weight_lb", "dob",
    "hometown", "status", "jersey", "team_slug",
]


def normalize_height(raw):
    """Heights render as `5'11`, `6' 0` or `6-3` across seasons. Return inches."""
    if not raw:
        return None
    m = re.match(r"^\s*(\d)\s*(?:'|-)\s*(\d{1,2})\s*$", raw)
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))


def normalize_dob(raw):
    """DOB renders as `MM/DD/YYYY`, `YYYY-MM-DD` or `MM/DD/YY`. Return ISO."""
    if not raw:
        return None
    raw = raw.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$", raw)
    if not m:
        return None
    mo, da, yr = m.groups()
    yr = int(yr)
    if yr < 100:                      # two-digit year: these are all players
        yr += 1900 if yr > 30 else 2000
    return f"{yr:04d}-{int(mo):02d}-{int(da):02d}"


def load_corpus_index():
    """(season, player_id) -> {name, person_id, career_id} for every real id."""
    idx = {}
    for path in pa_table.iter_game_files(ROOT / "games"):
        game = json.loads(Path(path).read_text())
        season = game["season"]
        for pid, rec in game["players"].items():
            if pid.startswith("syn:"):
                continue
            idx[(season, pid)] = {
                "name": rec.get("name"),
                "person_id": rec.get("person_id"),
                "career_id": rec.get("career_id"),
            }
    return idx


def _norm_name(s):
    return "".join(c for c in (s or "").lower() if c.isalnum())


def build():
    corpus = load_corpus_index()
    by_prefix = collections.defaultdict(list)
    for (season, pid) in corpus:
        by_prefix[(season, pid[:4])].append(pid)

    rows, stats = [], collections.Counter()
    seen = set()
    pages = archived_roster_pages()
    for team_slug, path in pages:
        for r in parse_roster(path.read_text()):
            stats["roster_rows"] += 1
            cands = by_prefix.get((r["season"], r["id_prefix"]), [])
            if not cands:
                stats["no_corpus_match"] += 1
                continue
            if len(cands) > 1:
                # 4 chars is not unique corpus-wide; confirm by display name.
                cands = [c for c in cands
                         if r["slug"].startswith(_norm_name(corpus[(r["season"], c)]["name"])[:6])]
                stats["prefix_collision"] += 1
                if len(cands) != 1:
                    stats["collision_unresolved"] += 1
                    continue
            pid = cands[0]
            meta = corpus[(r["season"], pid)]
            if not r["slug"].startswith(_norm_name(meta["name"])[:6]):
                stats["name_disagreement"] += 1
                continue
            key = (r["season"], pid)
            if key in seen:
                stats["duplicate_row"] += 1
                continue
            seen.add(key)
            stats["joined"] += 1
            if r["bats"]:
                stats["with_bats"] += 1
            rows.append({
                "season": r["season"], "player_id": pid,
                "person_id": meta["person_id"], "career_id": meta["career_id"],
                "name": meta["name"],
                "bats": r["bats"], "throws": r["throws"],
                "position": r.get("position"),
                "height_in": normalize_height(r.get("height")),
                "weight_lb": r.get("weight"),
                "dob": normalize_dob(r.get("dob")),
                "hometown": r.get("hometown"), "status": r.get("status"),
                "jersey": r.get("jersey"), "team_slug": team_slug,
            })
    return rows, stats, corpus, len(pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "profiles" / "player_profiles.csv"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    rows, stats, corpus, n_pages = build()
    print("roster pages read:", n_pages)
    for k in ("roster_rows", "joined", "with_bats", "no_corpus_match",
              "prefix_collision", "collision_unresolved", "name_disagreement", "duplicate_row"):
        print(f"  {k:24s} {stats[k]}")

    have = {(r["season"], r["player_id"]) for r in rows if r["bats"]}
    print()
    print(f"{'season':>7} {'corpus real ids':>16} {'with bats':>10} {'pct':>7}")
    for season in sorted({s for s, _ in corpus}):
        tot = sum(1 for s, _ in corpus if s == season)
        got = sum(1 for s, _ in have if s == season)
        print(f"{season:>7} {tot:>16} {got:>10} {100*got/max(1,tot):>6.1f}%")

    bats = collections.Counter(r["bats"] for r in rows if r["bats"])
    throws = collections.Counter(r["throws"] for r in rows if r["throws"])
    n = sum(bats.values())
    print()
    print("bats  :", {k: f"{v} ({100*v/n:.1f}%)" for k, v in bats.most_common()})
    n2 = sum(throws.values())
    print("throws:", {k: f"{v} ({100*v/n2:.1f}%)" for k, v in throws.most_common()})

    if not args.check:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r[k] is None else r[k]) for k in COLUMNS})
        prov = {
            "generated_by": "scripts/build_profiles.py",
            "rows": len(rows),
            "with_bats": stats["with_bats"],
            "roster_pages": n_pages,
            "join": "roster slug 4-char suffix -> corpus player_id, confirmed by display name",
            "sources": source_records(),
        }
        prov_path = out.parent / "provenance.json"
        prov_path.write_text(json.dumps(prov, indent=1, sort_keys=True) + "\n")
        print()
        print(f"wrote {out} ({len(rows)} rows)")
        print(f"wrote {prov_path} ({len(prov['sources'])} source pages)")


if __name__ == "__main__":
    main()
