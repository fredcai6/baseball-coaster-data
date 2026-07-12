"""Schedule-page walker (g1: pure parse, no I/O).

Given a pioneerleague.com/PrestoSports season-schedule page as an HTML string,
extract the boxscore URLs of games marked FINAL. This module does zero network
I/O and zero filesystem I/O -- it is pure HTML-in, URL-list-out. It must stay a
dependency-free building block: no imports from a fetcher or archive module.

Final-game signal (verified against a real 2026 schedule-page snapshot, see
``.agent-work/archive/2026-07-11-explore-advanced-stats/samples/x1-archive-notes.md``):
a completed game's schedule row is an element carrying both the ``event-row``
class token and a ``data-boxscore="/sports/bsb/<yr>/boxscores/YYYYMMDD_xxxx.xml"``
attribute. There is no literal "Final" text in the DOM -- future/in-progress
games either omit ``data-boxscore`` entirely or expose the same boxscore path
only as a plain ``href`` link, so attribute presence (not text content) is the
signal this module keys off.

Implementation note: uses only ``html.parser.HTMLParser`` from the standard
library. No third-party HTML parsing dependency was added -- the extraction
rule (one attribute, on one element, identified by one class token) does not
need a DOM tree or CSS-selector engine, so stdlib is the minimal choice here.
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Iterable

DEFAULT_BASE_URL = "https://www.pioneerleague.com"
DEFAULT_SEASON_PATH = "sports/bsb/{year}/schedule"
DEFAULT_YEARS: tuple[int, ...] = (2026, 2025, 2024)


class _EventRowBoxscoreExtractor(HTMLParser):
    """Collects ``data-boxscore`` values from elements tagged ``event-row``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.boxscore_urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._handle_tag(attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._handle_tag(attrs)

    def _handle_tag(self, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        class_attr = attr_map.get("class") or ""
        class_tokens = class_attr.split()
        boxscore_url = attr_map.get("data-boxscore")
        if boxscore_url and "event-row" in class_tokens:
            self.boxscore_urls.append(boxscore_url)


def final_boxscore_urls(schedule_html: str) -> list[str]:
    """Return the boxscore URLs of FINAL games found in a schedule page.

    ``schedule_html`` is the full HTML text of a pioneerleague.com (or any
    PrestoSports-hosted team site, same markup) season-schedule page. A game
    is considered FINAL when its ``event-row`` element carries a
    ``data-boxscore`` attribute; games without that attribute (future or
    in-progress) are excluded. Order follows document order; duplicate rows
    (if any) are preserved as-is -- callers needing uniqueness can dedupe.
    """
    parser = _EventRowBoxscoreExtractor()
    parser.feed(schedule_html)
    parser.close()
    return parser.boxscore_urls


def build_schedule_urls(
    years: Iterable[int] = DEFAULT_YEARS,
    *,
    base_url: str = DEFAULT_BASE_URL,
) -> list[str]:
    """Build season schedule-page URLs for the given years, in the given order.

    Defaults to the three in-scope seasons (2026, 2025, 2024) in that order.
    Uses the verified ``/sports/bsb/<year>/schedule`` path (confirmed 200 and
    identical in shape to ``/schedule-all`` per recon notes) against
    ``https://www.pioneerleague.com``.
    """
    return [
        f"{base_url}/{DEFAULT_SEASON_PATH.format(year=year)}" for year in years
    ]
