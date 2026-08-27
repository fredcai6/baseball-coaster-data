"""Tests for bc_pipeline.refresh: the backfill -> frequency-regen -> guard
orchestration entrypoint (g2, issue #21).

Same technique as ``test_backfill.py``/``test_fetch.py``: an INJECTED fake
transport and a fake clock/sleep pair, so pacing logic genuinely runs, no
test sleeps for real, and no test invokes real git.
"""

from __future__ import annotations

import json
from pathlib import Path

from bc_pipeline import career_map, frequencies, person_map, refresh, schedule, team_map
from bc_pipeline.config import PipelineConfig
from bc_pipeline.fetcher import FetchResponse

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOP_SAMPLES_DIR = _REPO_ROOT / "tests" / "samples"

FINAL_HTML = (_TOP_SAMPLES_DIR / "boxscore_20260709_final.html").read_text(encoding="utf-8")


def _season_schedule_url(season: int) -> str:
    return schedule.build_schedule_urls([season])[0]


def _box_url(season: int, slug: str) -> str:
    return f"{schedule.DEFAULT_BASE_URL}/sports/bsb/{season}/boxscores/{slug}.xml"


def _schedule_html(box_slugs: list[str], season: int) -> str:
    rows = "\n".join(
        f'<div class="card event-row result" '
        f'data-boxscore="/sports/bsb/{season}/boxscores/{slug}.xml"></div>'
        for slug in box_slugs
    )
    return f"<html><body>{rows}</body></html>"


class FakeClock:
    """Same technique as test_backfill.py/test_fetch.py: clock advances only
    on sleep()."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self._t

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._t += seconds


def make_transport(response_map: dict[str, FetchResponse], call_log: list[str]):
    def transport(url: str) -> FetchResponse:
        call_log.append(url)
        if url not in response_map:
            raise AssertionError(f"unexpected fetch for url not in response_map: {url}")
        return response_map[url]

    return transport


def make_config(tmp_path: Path, seasons: list[int]) -> PipelineConfig:
    return PipelineConfig(
        seasons=seasons,
        min_interval_seconds=10,
        jitter_seconds=0,
        archive_root=str(tmp_path / "archive"),
        checkpoint_path=str(tmp_path / "archive" / "checkpoint.json"),
    )


def run_refresh_against(
    config,
    response_map,
    call_log,
    commits,
    repo_root,
    **kwargs,
):
    """Call refresh.run_refresh with the fake-transport/fake-clock idiom.
    ``call_log``/``commits`` are shared, appendable lists so a caller can
    inspect them (and their lengths before/after) across repeat calls."""
    transport = make_transport(response_map, call_log)

    def commit_fn(paths, message):
        commits.append((tuple(str(p) for p in paths), message))

    return refresh.run_refresh(
        config,
        transport,
        repo_root=repo_root,
        sleep_fn=FakeClock().sleep,
        clock_fn=FakeClock().now,
        jitter_fn=lambda lo, hi: 0,
        wall_clock_fn=lambda: 1_700_000_000.0,
        print_fn=lambda _msg: None,
        commit_fn=commit_fn,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 2a: pick-up proof -- one newly-final game -> committed + frequency regen +
# a distinct frequency-artifact commit.
# ---------------------------------------------------------------------------


def test_refresh_pickup_new_final_game_commits_game_and_regenerates_frequencies(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=200, body=FINAL_HTML),
    }

    call_log: list[str] = []
    commits: list = []
    result = run_refresh_against(
        config, response_map, call_log, commits, tmp_path, limit=None
    )

    assert not result.stopped_by_challenge
    assert result.backfill.seasons[2026].parsed == 1
    game_path = tmp_path / "games" / "2026" / "20260401_g1.json"
    assert game_path.exists()

    # The new game file was staged in SOME commit_fn call.
    committed_paths = [p for paths, _msg in commits for p in paths]
    assert str(game_path) in committed_paths

    # Frequency artifact regenerated with the new game reflected.
    freq_path = tmp_path / "artifacts" / "latest" / "frequencies.json"
    assert freq_path.exists()
    artifact = json.loads(freq_path.read_text(encoding="utf-8"))
    assert artifact["meta"]["games_included"]["total"] == 1
    assert result.frequency_status == "changed"
    assert result.frequency_commit_message == refresh.FREQUENCY_COMMIT_MESSAGE

    # A DISTINCT frequency-artifact commit call was made (its own call_fn
    # invocation, not folded into the game-file batch commit).
    freq_commits = [
        (paths, msg) for paths, msg in commits if msg == refresh.FREQUENCY_COMMIT_MESSAGE
    ]
    assert len(freq_commits) == 1
    assert freq_commits[0][0] == (str(freq_path),)


# ---------------------------------------------------------------------------
# 2b: no-op proof -- literal same-args re-run against a now-exhausted
# backlog makes 0 new fetches/commits and the frequency regen is a genuine
# no-op.
# ---------------------------------------------------------------------------


def test_refresh_noop_on_literal_same_args_rerun_against_exhausted_backlog(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, seasons=[2026])
    # A small, FIXED backlog (2 boxscore URLs), no --limit -- genuinely
    # exhausted by the first run (bounded-crawl-idempotency-criterion-cost).
    box_slugs = ["20260401_g1", "20260402_g2"]
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(box_slugs, 2026)
        ),
    }
    for slug in box_slugs:
        response_map[_box_url(2026, slug)] = FetchResponse(status_code=200, body=FINAL_HTML)

    call_log: list[str] = []
    commits: list = []
    same_kwargs = dict(limit=None)

    first = run_refresh_against(
        config, response_map, call_log, commits, tmp_path, **same_kwargs
    )
    assert first.backfill.seasons[2026].parsed == 2
    assert first.frequency_status == "changed"
    commits_after_first = len(commits)

    # The schedule page itself is legitimately re-fetched every run (that's
    # how a NEW final game would ever be picked up) -- only the per-boxscore
    # fetches must NOT repeat once their games are already committed. Clear
    # the shared call_log so the second run's fetches can be inspected in
    # isolation, same technique as test_backfill.py's own repeat-run test.
    call_log.clear()

    # Literal SAME args, second call.
    second = run_refresh_against(
        config, response_map, call_log, commits, tmp_path, **same_kwargs
    )

    assert not second.stopped_by_challenge
    assert second.backfill.seasons[2026].fetched == 0
    assert second.backfill.seasons[2026].parsed == 0
    assert second.backfill.seasons[2026].skipped_already_committed == 2
    # No new game-file commits and no new frequency commit.
    assert len(commits) == commits_after_first
    # No boxscore was even re-fetched this second run (fetch-side
    # idempotency, already proven in test_fetch.py/test_backfill.py,
    # re-confirmed here at the refresh level).
    for slug in box_slugs:
        assert _box_url(2026, slug) not in call_log
    # Frequency regeneration itself is a genuine no-op (identical games/**
    # snapshot -> identical artifact modulo generated_at).
    assert second.frequency_status == "no-op"
    assert second.frequency_commit_message is None


# ---------------------------------------------------------------------------
# 2c: challenge-stop path skips frequency regeneration entirely.
# ---------------------------------------------------------------------------


def test_refresh_challenge_stop_skips_frequency_regeneration(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=202, body="please wait"),
    }

    call_log: list[str] = []
    commits: list = []
    escalation_sleeps: list[float] = []
    result = run_refresh_against(
        config,
        response_map,
        call_log,
        commits,
        tmp_path,
        limit=None,
        escalation_sleep_fn=escalation_sleeps.append,
    )

    assert result.stopped_by_challenge
    assert result.frequency_status == "skipped-challenge"
    assert result.frequency_commit_message is None

    # No frequency artifact was ever written, and no frequency commit made.
    freq_path = tmp_path / "artifacts" / "latest" / "frequencies.json"
    assert not freq_path.exists()
    assert all(msg != refresh.FREQUENCY_COMMIT_MESSAGE for _paths, msg in commits)

    # The escalating-backoff policy still ran (inherited from
    # run_backfill_with_escalation, unchanged).
    assert escalation_sleeps == [60.0, 600.0, 3600.0]


# ---------------------------------------------------------------------------
# person map: regenerated alongside frequencies, and person_id drift measured
# (issue #41)
# ---------------------------------------------------------------------------


def test_refresh_regenerates_the_person_map_in_its_own_commit(tmp_path: Path) -> None:
    """The person map is a derived artifact on the same footing as
    frequencies: regenerated every refresh, committed separately so `git log`
    says which surface moved."""
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=200, body=FINAL_HTML),
    }
    commits: list = []
    result = run_refresh_against(config, response_map, [], commits, tmp_path, limit=None)

    map_path = tmp_path / "artifacts" / "latest" / "person_map.json"
    assert map_path.exists()
    artifact = json.loads(map_path.read_text(encoding="utf-8"))
    assert artifact["meta"]["games"] == 1
    assert result.person_map_status == "changed"
    assert result.person_map_commit_message == refresh.PERSON_MAP_COMMIT_MESSAGE

    map_commits = [
        (paths, msg) for paths, msg in commits if msg == refresh.PERSON_MAP_COMMIT_MESSAGE
    ]
    assert len(map_commits) == 1
    assert map_commits[0][0] == (str(map_path),)
    # Distinct from the frequency commit, not folded into it.
    assert refresh.PERSON_MAP_COMMIT_MESSAGE != refresh.FREQUENCY_COMMIT_MESSAGE


def test_refresh_challenge_stop_skips_the_person_map_too(tmp_path: Path) -> None:
    """Regenerating the identity layer over a PARTIAL corpus is worse than
    regenerating the frequency artifact over one: it would mint person ids
    from an incomplete roster picture."""
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=202, body="please wait"),
    }
    commits: list = []
    escalation_sleeps: list[float] = []
    result = run_refresh_against(
        config, response_map, [], commits, tmp_path, limit=None,
        escalation_sleep_fn=escalation_sleeps.append,
    )

    assert result.stopped_by_challenge
    assert result.person_map_status == "skipped-challenge"
    assert result.person_map_commit_message is None
    assert result.person_id_drift is None
    assert not (tmp_path / "artifacts" / "latest" / "person_map.json").exists()
    assert all(msg != refresh.PERSON_MAP_COMMIT_MESSAGE for _paths, msg in commits)


def test_person_id_drift_counts_records_a_reparse_would_fix() -> None:
    """A game backfilled since the last re-parse carries null person_id for
    its synthetic players while the regenerated map already links them. The
    refresh cannot fix that (games/** is write-once) -- it reports the count."""
    games = [
        {
            "game_id": "g1",
            "season": 2026,
            "players": {
                "realaaaaaaaaaaa1": {
                    "name": "Ann Real", "team_id": "t1", "person_id": "realaaaaaaaaaaa1",
                },
                # Backfilled after the last re-parse: not yet materialized.
                "syn:away:1": {"name": "Bob Ghost", "team_id": "t1", "person_id": None},
            },
        },
    ]
    fresh = person_map.build_person_map(games)
    assert refresh._person_id_drift(games, fresh) == 1

    # Once a re-parse materializes it, drift goes to zero.
    games[0]["players"]["syn:away:1"]["person_id"] = person_map.mint_person_id(
        2026, "t1", "Bob Ghost"
    )
    assert refresh._person_id_drift(games, fresh) == 0


def test_person_id_drift_counts_a_pre_1_7_0_file_as_drifted() -> None:
    """A file with no `person_id` key at all is exactly what a re-parse would
    fill in, so it must not read as "in sync"."""
    games = [{
        "game_id": "g1", "season": 2026,
        "players": {"realaaaaaaaaaaa1": {"name": "Ann Real", "team_id": "t1"}},
    }]
    assert refresh._person_id_drift(games, person_map.build_person_map(games)) == 1


def test_person_id_drift_is_zero_on_the_real_corpus() -> None:
    """The committed corpus and its committed map must agree; if this fails a
    re-parse is genuinely due."""
    repo_root = Path(__file__).resolve().parents[2]
    games = frequencies.load_games(repo_root / "games")
    fresh = person_map.build_person_map(games)
    assert refresh._person_id_drift(games, fresh) == 0


def test_refresh_regenerates_the_team_map_in_its_own_commit(tmp_path: Path) -> None:
    """The franchise registry is a third derived artifact on the same footing,
    with its own commit so `git log` names the surface that moved."""
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=200, body=FINAL_HTML),
    }
    commits: list = []
    result = run_refresh_against(config, response_map, [], commits, tmp_path, limit=None)

    map_path = tmp_path / "artifacts" / "latest" / "team_map.json"
    assert map_path.exists()
    assert result.team_map_status == "changed"
    assert result.team_map_commit_message == refresh.TEAM_MAP_COMMIT_MESSAGE

    map_commits = [
        (paths, msg) for paths, msg in commits if msg == refresh.TEAM_MAP_COMMIT_MESSAGE
    ]
    assert len(map_commits) == 1
    assert map_commits[0][0] == (str(map_path),)
    # Three distinct artifact commit messages, none folded into another.
    assert len({
        refresh.TEAM_MAP_COMMIT_MESSAGE,
        refresh.PERSON_MAP_COMMIT_MESSAGE,
        refresh.FREQUENCY_COMMIT_MESSAGE,
    }) == 3


def test_refresh_challenge_stop_skips_the_team_map_too(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=202, body="please wait"),
    }
    commits: list = []
    result = run_refresh_against(
        config, response_map, [], commits, tmp_path, limit=None,
        escalation_sleep_fn=[].append,
    )
    assert result.stopped_by_challenge
    assert result.team_map_status == "skipped-challenge"
    assert result.team_map_commit_message is None
    assert not (tmp_path / "artifacts" / "latest" / "team_map.json").exists()
    assert all(msg != refresh.TEAM_MAP_COMMIT_MESSAGE for _paths, msg in commits)


def test_corpus_franchise_id_needs_no_drift_counterpart() -> None:
    """franchise_id is a pure function of the team name in each file, so
    parse populates it directly and it cannot fall out of sync with the
    registry the way person_id can. Assert that on the real corpus."""
    repo_root = Path(__file__).resolve().parents[2]
    games = frequencies.load_games(repo_root / "games")
    art = team_map.build_team_map(games)
    for game in games:
        for side in ("home", "away"):
            team = game["teams"][side]
            expected = team_map.mint_franchise_id(team["name"])
            assert team.get("franchise_id") == expected, (game["game_id"], side)
            assert expected in art["franchises"]


def test_refresh_regenerates_the_career_map_in_its_own_commit(tmp_path: Path) -> None:
    config = make_config(tmp_path, seasons=[2026])
    response_map = {
        _season_schedule_url(2026): FetchResponse(
            status_code=200, body=_schedule_html(["20260401_g1"], 2026)
        ),
        _box_url(2026, "20260401_g1"): FetchResponse(status_code=200, body=FINAL_HTML),
    }
    commits: list = []
    result = run_refresh_against(config, response_map, [], commits, tmp_path, limit=None)

    map_path = tmp_path / "artifacts" / "latest" / "career_map.json"
    assert map_path.exists()
    assert result.career_map_status == "changed"
    assert result.career_map_commit_message == refresh.CAREER_MAP_COMMIT_MESSAGE
    career_commits = [
        (paths, msg) for paths, msg in commits if msg == refresh.CAREER_MAP_COMMIT_MESSAGE
    ]
    assert len(career_commits) == 1
    # Four artifact commit messages, all distinct.
    assert len({
        refresh.CAREER_MAP_COMMIT_MESSAGE, refresh.TEAM_MAP_COMMIT_MESSAGE,
        refresh.PERSON_MAP_COMMIT_MESSAGE, refresh.FREQUENCY_COMMIT_MESSAGE,
    }) == 4


def test_career_id_drift_counts_records_a_reparse_would_fix() -> None:
    """career_id drifts independently of person_id: adding a season can link
    new careers without changing a single person_id."""
    games = [{
        "game_id": "g1", "season": 2026, "date": "2026-05-01",
        "teams": {
            "home": {"team_id": "t1", "name": "H", "franchise_id": "franchise:aaaaaaaaaaaaaaaa"},
            "away": {"team_id": "t2", "name": "A", "franchise_id": "franchise:bbbbbbbbbbbbbbbb"},
        },
        "players": {
            "realaaaaaaaaaaa1": {
                "name": "Ann Real", "team_id": "t1", "positions": ["ss"],
                "person_id": "realaaaaaaaaaaa1", "career_id": None,
            },
        },
    }]
    fresh = career_map.build_career_map(games)
    assert refresh._career_id_drift(games, fresh) == 1
    games[0]["players"]["realaaaaaaaaaaa1"]["career_id"] = fresh["assignments"][
        "realaaaaaaaaaaa1"
    ]
    assert refresh._career_id_drift(games, fresh) == 0


def test_career_id_drift_is_zero_on_the_real_corpus() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    games = frequencies.load_games(repo_root / "games")
    assert refresh._career_id_drift(games, career_map.build_career_map(games)) == 0
