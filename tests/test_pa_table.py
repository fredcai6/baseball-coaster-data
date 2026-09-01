"""The plate-appearance table, checked against the boxscore.

`bc_pipeline.pa_table` flattens the play-by-play spine into one row per plate
appearance. It asserts nothing new -- every column is a fold over `events[]` --
which is exactly what makes it checkable: the boxscore is parsed from a
different region of the source page, so per-player AB/H/BB/SO totals folded
from the narrative and the same totals read from the box are two independent
readings of one truth.

That check earned its keep the first time it ran. The fold scored AB from the
outcome TYPE, and the corpus records only 36 of its 1,895 sacrifices as
`type: "sacrifice"` -- the other 1,859 are ordinary flyouts, groundouts,
lineouts, foul outs, popouts, fielders' choices and reached-on-errors carrying
a `SAC` or `sacrifice fly` MODIFIER. Eight types, one modifier pair. AB was
over-counted on 1,749 of 41,122 player-games and every other column was
perfect, so nothing but an independent oracle would have found it.

The numbers below are pinned so a re-parse that moves them has to say so.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from bc_pipeline import pa_table

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_DIR = REPO_ROOT / "games"
LEDGER_PATH = REPO_ROOT / "corrections" / "dispositions.json"

BOX_FIELDS = ("AB", "H", "BB", "SO")


def _disposed_game_ids():
    if not LEDGER_PATH.exists():
        return set()
    return {d["game_id"] for d in json.loads(LEDGER_PATH.read_text())["dispositions"]}


@pytest.fixture(scope="module")
def corpus_reconciliation():
    """Fold every game and compare against its box, bucketed by disposition."""
    disposed = _disposed_game_ids()
    agree = collections.Counter()
    disagree = collections.Counter()
    fields_wrong = collections.Counter()
    rows = 0

    for path in pa_table.iter_game_files(GAMES_DIR):
        game = json.loads(path.read_text())
        if game.get("record_shape") == "boxscore_only":
            continue
        folded = collections.defaultdict(collections.Counter)
        for r in pa_table.rows_for_game(game):
            rows += 1
            c = folded[r["batter_pid"]]
            c["AB"] += bool(r["is_ab"])
            c["H"] += bool(r["is_hit"])
            c["BB"] += bool(r["is_bb"])
            c["SO"] += bool(r["is_k"])
        bucket = "disposed" if game["game_id"] in disposed else "clean"
        for team_rows in (game.get("box") or {}).get("batting", {}).values():
            for brow in team_rows:
                pid = brow.get("player_id")
                if pid is None:
                    continue
                got = folded.get(pid, collections.Counter())
                bad = [f for f in BOX_FIELDS if got[f] != brow.get(f, 0)]
                if bad:
                    disagree[bucket] += 1
                    if bucket == "clean":
                        for f in bad:
                            fields_wrong[f] += 1
                else:
                    agree[bucket] += 1
    return {"agree": agree, "disagree": disagree, "fields": fields_wrong, "rows": rows}


def test_every_clean_player_game_reconciles_with_the_box(corpus_reconciliation):
    """AB/H/BB/SO folded from the narrative equal the box, on every clean game.

    This is the whole warrant for using the table as an analysis input. It is an
    equality and not a threshold on purpose: the fold re-derives nothing the box
    cannot check, so any disagreement outside a disposed game is a fold bug.
    """
    r = corpus_reconciliation
    assert r["disagree"]["clean"] == 0, (
        f"{r['disagree']['clean']} clean player-games disagree with the box "
        f"(by field: {dict(r['fields'])}). The fold, not the corpus, is wrong."
    )
    assert r["agree"]["clean"] > 40_000


def test_the_disposed_residual_stays_inside_the_ledger(corpus_reconciliation):
    """Disagreements survive only where the ledger already discloses a defect.

    A game whose source dropped plate appearances cannot reconcile, and should
    not be made to. What this pins is that the count does not grow: if a fold
    bug appears it will land in the clean bucket above, and if a NEW source
    defect appears this number moves and the ledger has to answer for it.
    """
    r = corpus_reconciliation
    assert r["disagree"]["disposed"] == 37, (
        f"disposed-bucket disagreements moved to {r['disagree']['disposed']} "
        "(was 37). Re-attribute them against corrections/dispositions.json "
        "before re-pinning."
    )


def test_the_table_covers_every_plate_appearance(corpus_reconciliation):
    """One row per plate appearance, no silent drops."""
    expected = 0
    for path in pa_table.iter_game_files(GAMES_DIR):
        game = json.loads(path.read_text())
        if game.get("record_shape") == "boxscore_only":
            continue
        for e in game.get("events") or []:
            if e.get("kind") != "plate_appearance":
                continue
            if (e.get("batter") or {}).get("player_id") and (e.get("pitcher") or {}).get("player_id"):
                expected += 1
    assert corpus_reconciliation["rows"] == expected


def test_sacrifice_is_read_from_the_modifier_not_the_type():
    """The regression pin for the bug the box caught.

    A sacrifice fly scored as `flyout` must not be an at-bat. Scoring it from
    the type alone passes every other column and fails only AB, which is why
    this is pinned by behaviour and not by a count.
    """
    sac_fly = pa_table.classify({"type": "flyout", "modifiers": ["sacrifice fly"]})
    assert sac_fly["is_sac"] is True
    assert sac_fly["is_ab"] is False

    sac_bunt = pa_table.classify({"type": "groundout", "modifiers": ["SAC", "bunt"]})
    assert sac_bunt["is_sac"] is True
    assert sac_bunt["is_ab"] is False

    ordinary = pa_table.classify({"type": "flyout", "modifiers": []})
    assert ordinary["is_sac"] is False
    assert ordinary["is_ab"] is True


def test_an_unknown_outcome_type_is_refused_not_absorbed():
    """A schema addition must be taught here, not silently scored as an out."""
    with pytest.raises(ValueError, match="unknown outcome type"):
        pa_table.classify({"type": "quantum_tunneling", "modifiers": []})


def test_join_keys_are_career_ids_and_they_are_populated():
    """`player_id` does not join across games; the table must carry what does.

    10% of player records hold a synthetic `syn:<side>:<n>` id assigned by box
    row order, so a naive cross-game join on `*_pid` mixes distinct people.
    """
    assert "batter_career" in pa_table.COLUMNS
    assert "pitcher_career" in pa_table.COLUMNS

    missing = 0
    total = 0
    for path in list(pa_table.iter_game_files(GAMES_DIR))[:200]:
        game = json.loads(path.read_text())
        for r in pa_table.rows_for_game(game):
            total += 1
            if not (r["batter_career"] and r["pitcher_career"]):
                missing += 1
    assert total > 0
    assert missing / total < 0.001, f"{missing}/{total} rows lack a career-id join key"
