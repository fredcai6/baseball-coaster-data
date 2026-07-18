"""Tests for bc_pipeline.refresh: the backfill -> frequency-regen -> guard
orchestration entrypoint (g2, issue #21).

Same technique as ``test_backfill.py``/``test_fetch.py``: an INJECTED fake
transport and a fake clock/sleep pair, so pacing logic genuinely runs, no
test sleeps for real, and no test invokes real git.
"""

from __future__ import annotations

import json
from pathlib import Path

from bc_pipeline import refresh, schedule
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
