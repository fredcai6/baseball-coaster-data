"""The dispositions ledger, and the reconciliation that keeps it honest.

Two kinds of test here, and the second kind is the point.

The unit tests fix the loader's contract: one entry per game, a pin that
matches the bytes, and an `audit` that reports in BOTH directions. The
corpus test then runs that audit against the real corpus, which is the only
thing that can catch the failure this file was built for -- a disposition
that was true when it was written and is not true now.

`20240508_04ck` and `20240528_6w90` were each recorded as unfixable in a
handoff and each became fixable the morning an unrelated oracle was
corrected. Nothing noticed. That is what
`test_the_ledger_and_the_corpus_agree_on_the_real_corpus` is for.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bc_pipeline import archive, dispositions, replay, reparse
from bc_pipeline.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schemas" / "disposition.schema.json"
LEDGER_PATH = REPO_ROOT / "corrections" / "dispositions.json"

#: sha256 of the empty string -- what an unreplayed game pins to.
EMPTY_DIGEST = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _write(tmp_path: Path, entries: list) -> Path:
    path = tmp_path / "dispositions.json"
    path.write_text(
        json.dumps({"schema_version": "1.0.0", "dispositions": entries}),
        encoding="utf-8",
    )
    return path


def _entry(game_id: str, warnings: list, **over) -> dict:
    entry = {
        "game_id": game_id,
        "state": "committed_no_replay",
        "class": "missing_plate_appearances",
        "checks_failed": sorted({w.split("]")[0].lstrip("[") for w in warnings}),
        "warnings_sha256": dispositions.warnings_sha256(warnings),
        "warnings_excerpt": sorted(warnings),
        "evidence": "x" * 80,
        "alternative_scored": "y" * 40,
    }
    entry.update(over)
    return entry


# --- the pin ----------------------------------------------------------------


def test_the_pin_ignores_order_and_repetition_but_not_content():
    """A check's emission order is a parser detail, and a caller that
    replays an already-replayed game sees every warning twice. Neither is a
    change in the FAILURE, and a pin that tripped on them would cry wolf
    until nobody read it. A different warning is a different failure."""
    a = dispositions.warnings_sha256(["[lob] two", "[pa_counts] one"])
    assert a == dispositions.warnings_sha256(["[pa_counts] one", "[lob] two"])
    assert a == dispositions.warnings_sha256(
        ["[lob] two", "[pa_counts] one", "[lob] two"]
    )
    assert a != dispositions.warnings_sha256(["[lob] two", "[pa_counts] ONE"])


def test_no_warnings_pins_to_the_empty_digest():
    """Not a special case -- a game that is never replayed has nothing to
    say, and hashing nothing is the honest way to record that."""
    assert dispositions.warnings_sha256([]) == EMPTY_DIGEST


# --- the loader -------------------------------------------------------------


def test_a_game_may_have_only_one_terminal_state(tmp_path: Path):
    path = _write(tmp_path, [_entry("20240101_aaaa", ["[lob] x"]),
                             _entry("20240101_aaaa", ["[lob] y"])])
    with pytest.raises(dispositions.DispositionError, match="duplicate"):
        dispositions.load(path)


def test_a_missing_ledger_is_empty_not_an_error(tmp_path: Path):
    assert dispositions.load(tmp_path / "nope.json") == {}


# --- the audit, in both directions ------------------------------------------


def test_a_failing_game_with_no_disposition_is_undisclosed(tmp_path: Path):
    path = _write(tmp_path, [])
    problems = dispositions.audit({"20240101_aaaa": ["[lob] x"]}, path=path)
    assert [p["problem"] for p in problems] == ["undisclosed"]


def test_a_disposition_whose_failure_changed_is_stale(tmp_path: Path):
    """The case that motivated the file. An oracle improves, the game now
    fails differently -- or not at all in the way the author argued about --
    and the argument on record no longer describes it."""
    path = _write(tmp_path, [_entry("20240101_aaaa", ["[lob] folded 2 != 1"])])
    problems = dispositions.audit(
        {"20240101_aaaa": ["[lob] folded 3 != 1"]}, path=path
    )
    assert [p["problem"] for p in problems] == ["stale"]


def test_a_disposition_for_a_game_that_now_passes_is_spent(tmp_path: Path):
    """04ck and 6w90, mechanised. Both were recorded as unfixable and both
    became fixable after an unrelated check was corrected; the record stayed
    put because nothing was checking it."""
    path = _write(tmp_path, [_entry("20240101_aaaa", ["[lob] x"])])
    problems = dispositions.audit({"20240101_aaaa": []}, path=path)
    assert [p["problem"] for p in problems] == ["spent"]


def test_a_clean_game_with_no_disposition_is_not_a_problem(tmp_path: Path):
    assert dispositions.audit({"20240101_aaaa": []}, path=_write(tmp_path, [])) == []


def test_a_disposition_naming_a_game_the_corpus_does_not_hold_is_absent(tmp_path: Path):
    path = _write(tmp_path, [_entry("20240101_zzzz", ["[lob] x"])])
    problems = dispositions.audit({}, committed_game_ids=[], path=path)
    assert [p["problem"] for p in problems] == ["absent"]


def test_a_not_committed_disposition_inverts_the_absence_check(tmp_path: Path):
    """`not_committed` asserts there is NO game file. A file appearing is
    then the discrepancy, and the same check has to catch it from the other
    side or the state means nothing."""
    entry = _entry("20240101_zzzz", [], state="not_committed",
                   **{"class": "no_baseball_on_the_page"})
    path = _write(tmp_path, [entry])
    assert dispositions.audit({}, committed_game_ids=[], path=path) == []
    problems = dispositions.audit({}, committed_game_ids=["20240101_zzzz"], path=path)
    assert [p["problem"] for p in problems] == ["absent"]


def test_an_unreplayed_committed_game_is_not_reported_spent(tmp_path: Path):
    """A boxscore-only record is committed and never handed to the replayer,
    so it contributes no warnings. That must not read as 'passes cleanly,
    delete the entry'."""
    entry = _entry("20240101_bbbb", [], state="committed_no_play_by_play",
                   **{"class": "no_play_by_play_published"})
    path = _write(tmp_path, [entry])
    assert dispositions.audit({}, committed_game_ids=["20240101_bbbb"], path=path) == []


# --- the committed ledger ---------------------------------------------------


def test_the_committed_ledger_validates_against_its_schema():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    doc = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(doc))
    assert errors == [], [(list(e.path), e.message) for e in errors]


def test_every_entry_names_a_scored_alternative_not_a_shrug():
    """`alternative_scored` is the field that stops this becoming a junk
    drawer. The bar is a NUMBER -- the score the alternative got, the size
    of the population it would regress, the count of the thing that
    forecloses it. A keyword check would pass on "we scored it and it did
    not work", which says nothing the next reader can act on."""
    for entry in json.loads(LEDGER_PATH.read_text(encoding="utf-8"))["dispositions"]:
        assert re.search(r"\d", entry["alternative_scored"]), entry["game_id"]


def test_our_own_oracles_misses_are_labelled_as_ours():
    """The one distinction this file cannot be allowed to blur. Every class
    but `oracle_residual` says the SOURCE failed to write something down;
    `oracle_residual` says we are wrong and the source is right."""
    ours = [
        e for e in json.loads(LEDGER_PATH.read_text(encoding="utf-8"))["dispositions"]
        if e["class"] == "oracle_residual"
    ]
    assert {e["game_id"] for e in ours} == {"20240818_iu7f", "20250715_pdfw"}
    for entry in ours:
        assert "OURS, NOT THE SOURCE'S" in entry["evidence"]


@lru_cache(maxsize=1)
def _replay_the_real_corpus():
    """(warnings_by_game, committed_game_ids), replaying every game once.

    Cached because both corpus tests below need it and a full replay of the
    corpus is the most expensive thing in this suite.
    """
    config = load_config(None)
    html_by_id = reparse._archived_html_by_game_id(config)
    ledger = dispositions.load(LEDGER_PATH)

    warnings_by_game = {}
    committed = []
    for path in sorted((REPO_ROOT / "games").rglob("*.json")):
        game = json.loads(path.read_text(encoding="utf-8"))
        game_id = game["game_id"]
        committed.append(game_id)
        entry = ledger.get(game_id)
        if entry is not None and entry["state"] in dispositions.NOT_REPLAYED_STATES:
            continue
        assert game_id in html_by_id, f"{game_id} has no archived HTML"
        html = Path(html_by_id[game_id]).read_text(encoding="utf-8", errors="replace")
        # Clear the committed warnings first: the pin must be what the
        # oracles say about this game NOW, not that plus a transcript of
        # what they said at parse time, which replay_game preserves.
        game.setdefault("meta", {}).setdefault("parse", {})["warnings"] = []
        replayed = replay.replay_game(game, html)
        warnings_by_game[game_id] = replayed["meta"]["parse"]["warnings"]
    return warnings_by_game, tuple(committed)


def test_the_ledger_and_the_corpus_agree_on_the_real_corpus():
    """The whole point. Replays every committed game and reconciles the
    result against the ledger in both directions -- undisclosed failures,
    stale pins, and spent entries all fail here.

    If this goes red after an oracle change, that is the test working: read
    the entries it names before deciding anything, because a game that was
    unfixable under the old oracle may not be under the new one."""
    warnings_by_game, committed = _replay_the_real_corpus()
    problems = dispositions.audit(
        warnings_by_game, committed_game_ids=committed, path=LEDGER_PATH
    )
    assert problems == [], "\n".join(
        f"{p['game_id']} [{p['problem']}] {p['detail']}" for p in problems
    )


def test_every_game_the_league_published_is_accounted_for():
    """The corpus's score against the SOURCE, not against itself.

    The denominator is deliberately not read from `completeness.json`, which
    this repository generates -- it is counted from the raw-archive
    checkpoint, which holds one entry per boxscore URL the league's own
    schedule pages yielded. 1,486: 498 in 2024, 490 in 2025, 498 in 2026.

    Accounted means replay-validating OR disposed. Neither term is a
    judgement call at this point: the first is measured, and the second is
    reconciled against the corpus by the test above, so a disposition can
    only count here if it still describes a game that genuinely fails."""
    config = load_config(None)
    checkpoint = archive.load_checkpoint(archive._resolve_checkpoint_path(config))
    discovered = {url for url in checkpoint if "/boxscores/" in url}
    assert len(discovered) == 1486, (
        "the archive no longer holds the 1,486 boxscore URLs the league "
        "published; re-run the schedule walk before trusting this score"
    )

    warnings_by_game, _committed = _replay_the_real_corpus()
    validating = {gid for gid, w in warnings_by_game.items() if not w}
    disposed = set(dispositions.load(LEDGER_PATH))
    assert not (validating & disposed), (
        "a game counted both as validating and as disposed -- the two terms "
        "must not overlap or the total is meaningless"
    )

    unaccounted = len(discovered) - len(validating) - len(disposed)
    assert unaccounted == 0, (
        f"{unaccounted} of {len(discovered)} published games are neither "
        f"replay-validating ({len(validating)}) nor disposed ({len(disposed)})"
    )
