"""Parse a PrestoSports team-roster page into player profile rows.

The corpus has no handedness: `players[].bats_side` is null on all 41,713
player records, and the boxscore/play-by-play pages the pipeline already reads
never mention it. The league site does carry it, but NOT on the individual
player page (`players?id=<player_id>`), which is stats-only -- it is on the
team roster page, `teams/<slug>?view=roster`, one bulk table per team-season.

That distinction is the whole reason this module exists at the team level: a
roster sweep is ~36 requests where a per-player sweep would be ~1,787.

The join back to the corpus is by URL SLUG, not by id. Roster rows link to
`/sports/bsb/<season>/players/<name><4-char-suffix>`, where the suffix is the
first four characters of the site's 16-char `player_id` -- the same value the
corpus stores as `players[].player_id`. Note the site's id is SEASON-SCOPED:
all 159 of our cross-season careers carry a different site id each year, so a
profile row is only ever bound to one (season, player_id), and `career_id`
remains the only cross-season key.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

# Header label -> the field name we emit. Columns are located by reading the
# header row rather than by fixed position: the page template emits some <th>
# cells conditionally, so position is not stable across seasons.
_HEADERS = {
    "#": "jersey",
    "name": "name",
    "position": "position",
    "year": "class_year",
    "status": "status",
    "height": "height",
    "weight": "weight",
    "bats": "bats",
    "throws": "throws",
    "dob": "dob",
    "hometown": "hometown",
}

_PLAYER_HREF = re.compile(r"/sports/bsb/(\d{4})/players/([a-z0-9\-]+)\s*$")

BATS_VALUES = {"L", "R", "S"}


class _RosterParser(HTMLParser):
    """Pull the single table whose caption is 'Team Roster'."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self.headers: list[str] = []
        self._in_table = False
        self._in_caption = False
        self._caption = ""
        self._in_thead = False
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._row_cells: list[str] = []
        self._row_href: str | None = None
        self._pending_table = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._pending_table = True
            self._caption = ""
        elif tag == "caption" and self._pending_table:
            self._in_caption = True
        elif tag == "thead" and self._in_table:
            self._in_thead = True
        elif tag == "tr" and self._in_table:
            self._row_cells = []
            self._row_href = None
        elif tag in ("td", "th") and self._in_table:
            self._in_cell = True
            self._cell_parts = []
        elif tag == "a" and self._in_cell and self._in_table:
            href = a.get("href", "")
            if _PLAYER_HREF.search(href):
                self._row_href = href

    def handle_endtag(self, tag):
        if tag == "caption" and self._in_caption:
            self._in_caption = False
            if "team roster" in self._caption.strip().lower():
                self._in_table = True
            self._pending_table = False
        elif tag == "thead":
            self._in_thead = False
        elif tag in ("td", "th") and self._in_cell:
            self._in_cell = False
            self._row_cells.append(" ".join("".join(self._cell_parts).split()))
        elif tag == "tr" and self._in_table:
            if self._in_thead:
                self.headers = [_HEADERS.get(c.strip().lower(), None) for c in self._row_cells]
            elif self._row_cells:
                row = {}
                for name, value in zip(self.headers, self._row_cells):
                    if name:
                        row[name] = value or None
                row["href"] = self._row_href
                self.rows.append(row)
        elif tag == "table" and self._in_table:
            self._in_table = False

    def handle_data(self, data):
        if self._in_caption:
            self._caption += data
        elif self._in_cell:
            self._cell_parts.append(data)


def parse_roster(html: str) -> list[dict]:
    """Return one profile dict per roster row.

    Each row carries `season` and `id_prefix` (the 4-char tail of the player
    slug, which is the first 4 characters of the corpus `player_id`). Rows with
    no player link are dropped -- they carry no way to join.
    """
    p = _RosterParser()
    p.feed(html)
    out = []
    for row in p.rows:
        href = row.pop("href", None)
        if not href:
            continue
        m = _PLAYER_HREF.search(href)
        if not m:
            continue
        season, slug = int(m.group(1)), m.group(2)
        row["season"] = season
        row["slug"] = slug
        row["id_prefix"] = slug[-4:]
        bats = (row.get("bats") or "").strip().upper()
        throws = (row.get("throws") or "").strip().upper()
        row["bats"] = bats if bats in BATS_VALUES else None
        row["throws"] = throws if throws in BATS_VALUES else None
        out.append(row)
    return out
