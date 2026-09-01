"""Extract game venue (ballpark) from the source page's HTML.

Venue had been sitting in the archive unread for the corpus's whole life: the
page carries an "Other Information" table with Location, Stadium, Attendance,
Duration, Weather and Umpires, and `parse.py` had never looked at it. Reading
it cost zero fetches.

It is not decoration. The designated home team is NOT always the team playing
at its own park -- the 2025 Colorado Springs Sky Sox played 26 of their 37
designated-home games in the opponent's ballpark, whole series at a time --
so `half == "bottom"` is a fact about batting order and not a proxy for
home-field advantage. Before this field nothing in the corpus could tell the
two apart, and every model on the board had been using one for the other.

Since schema 1.13.0 `parse.py` calls this during the corpus parse, so the
venue is IN `games/**` and every derived consumer rebuilds from the corpus
alone -- no raw archive required. That was deliberate: a venue table built
straight off the archive would have been the one `artifacts/derived/` cache
that could not be regenerated from a clone, quietly breaking the rule the rest
of that directory keeps.

Two raw fields appear on the source page's "Other Information" table,
inconsistently:

  * `Stadium` — the ballpark name, present on newer pages.
  * `Location` — sometimes a street address, sometimes a bare ballpark name,
    sometimes nothing more specific than "Boise, ID". Whichever the page
    carries is fed to `venue_raw`; `Stadium` wins when both are present.

Neither field is close to clean: 82 distinct raw strings across the 1,485-
game corpus reduce to 16 physical ballparks once typos ("Lindquist FIeld"),
case variants ("DEHLER PARK"), address forms, and city/state fallbacks are
folded together. `CANONICAL_VENUE` is that fold, built by hand from the
corpus and should be re-examined, not blindly extended, if a new raw string
shows up. `tests/test_venue.py` pins the counts (81 raw -> 16 parks) so an
edit that moves them has to say so.

The map is SEASON-BLIND, and one pair is only safe because of when it occurs:
'Colorado Springs' is the Vibes' 2024 home (UCHealth Park) while 'Colorado
Springs, CO' is their 2025 home (Blocktickets Park, after the move). Each
string appears in exactly one season, and a test pins that -- if a later fetch
puts either in the other's season this map returns the wrong park silently.
"""
from __future__ import annotations

from typing import Dict, Optional

from . import html_struct

#: Label -> raw value read off the source page's "Other Information" table
#: (a `<table class="... game-info ...">` whose rows are
#: `<strong>Label: </strong>` next to a `<span>value</span>`, except the
#: Umpires row, which has no `<span>` and is read from the row's last `<td>`).
OtherInfo = Dict[str, str]


def extract_other_info(html: str) -> OtherInfo:
    """Return the label->value pairs from the page's "Other Information"
    table: `Location`, `Stadium`, `Attendance`, `Duration`, `Weather`,
    `Umpires` — whichever of them the page actually carries. Empty dict if
    the page has no such table (observed on 2 of 1,485 archived games).
    """
    root = html_struct.parse_html(html)
    info: OtherInfo = {}
    for table in html_struct.find_all(root, "table"):
        local: OtherInfo = {}
        for row in html_struct.find_all(table, "tr"):
            strongs = html_struct.find_all(row, "strong")
            if not strongs:
                continue
            label = html_struct.text_of(strongs[0]).rstrip(":").strip()
            if not label:
                continue
            spans = html_struct.find_all(row, "span")
            if spans:
                value = html_struct.text_of(spans[0]).strip()
            else:
                # The Umpires row carries its value as plain text in the
                # second <td>, not a <span>.
                tds = [c for c in row.children if getattr(c, "tag", None) in ("td", "th")]
                value = html_struct.text_of(tds[-1]).strip() if len(tds) > 1 else ""
            if value:
                local[label] = value
        # Only accept a table that actually looks like the venue block —
        # guards against an unrelated table that happens to have <strong>
        # labels (e.g. the box score's 2B/3B/RBI header cells).
        if "Stadium" in local or "Location" in local:
            info.update(local)
    return info


def venue_raw(info: OtherInfo) -> Optional[str]:
    """`Stadium` when the page carries it, else `Location`, else None."""
    return info.get("Stadium") or info.get("Location") or None


#: raw string (exact, as extracted by `venue_raw`) -> canonical ballpark
#: name. Built by hand from every one of the 82 distinct raw strings this
#: corpus produces (2024-2026 seasons) — see the module docstring. A raw
#: string this map does not know maps to itself unchanged (see
#: `canonicalize`), so an unmapped new value degrades to "one more distinct
#: venue" rather than raising or silently disappearing.
CANONICAL_VENUE: Dict[str, str] = {
    # Oakland Ballers -- Raimondi Park, Oakland CA
    "Raimondi Park": "Raimondi Park",
    "Raimondi": "Raimondi Park",
    "Oakland, CA": "Raimondi Park",
    "449 50th Street, Oakland, CA, 94609": "Raimondi Park",
    # Ogden Raptors -- Lindquist Field, Ogden UT
    "Lindquist Field": "Lindquist Field",
    "Lindquist field": "Lindquist Field",
    "Lindquist FIeld": "Lindquist Field",
    "Lindquist, Field": "Lindquist Field",
    "lindquist Field": "Lindquist Field",
    "2330 Lincoln Avenue, Ogden, UT, 84401": "Lindquist Field",
    # Billings Mustangs -- Dehler Park, Billings MT
    "Dehler Park": "Dehler Park",
    "DEHLER PARK": "Dehler Park",
    "Deher Park": "Dehler Park",
    "Billings, MT": "Dehler Park",
    "Billngs, MT": "Dehler Park",
    "2611 9th Avenue North, Billings, MT, 59101": "Dehler Park",
    # Great Falls Voyagers -- Voyager Stadium (fka Centene Stadium), Great Falls MT
    "Voyager Stadium": "Voyager Stadium",
    "Voyagers Stadium": "Voyager Stadium",
    "Voyagers stadium": "Voyager Stadium",
    "Voyager": "Voyager Stadium",
    "Centene Stadium": "Voyager Stadium",
    "1015 25th Street North, Great Falls, MT, 59401": "Voyager Stadium",
    # Grand Junction Jackalopes -- Suplizio Field, Grand Junction CO
    "Suplizio Field": "Suplizio Field",
    "Grand Junction, CO": "Suplizio Field",
    # Missoula PaddleHeads -- Ogren Allegiance Park, Missoula MT
    "Ogren Allegiance": "Ogren Allegiance Park",
    "Allegiance Field": "Ogren Allegiance Park",
    "Ogren Park": "Ogren Allegiance Park",
    "Allegiance": "Ogren Allegiance Park",
    "Missoula, MT": "Ogren Allegiance Park",
    "allegiance field": "Ogren Allegiance Park",
    "ALLEGIANCE FIELD": "Ogren Allegiance Park",
    "MISSOULA": "Ogren Allegiance Park",
    "Allegiance Park": "Ogren Allegiance Park",
    "Ogren Alleigance": "Ogren Allegiance Park",
    "Missoula": "Ogren Allegiance Park",
    "Allegiant Field": "Ogren Allegiance Park",
    "Allegiance Stadium": "Ogren Allegiance Park",
    "700 Cregg Lane, Missoula, MT, 59801": "Ogren Allegiance Park",
    # Yuba-Sutter Freebirds -- Bryant Field (2025-26 name) / Dobbins Stadium
    # (2024 name), Marysville CA. Same physical park, stable within a season,
    # so left as two canonical labels rather than merged (merging would hide
    # a real rename from a season-scoped consumer).
    "Bryant Field": "Bryant Field",
    "Marysville, CA, 95901": "Bryant Field",
    "Dobbins Stadium": "Dobbins Stadium",
    "Dobbins Field": "Dobbins Stadium",
    # Glacier Range Riders -- Glacier Bank Park, Kalispell MT
    "Glacier Bank Park": "Glacier Bank Park",
    "Glacir Bank Park": "Glacier Bank Park",
    "glacier bank": "Glacier Bank Park",
    "Flathead Valley, MT": "Glacier Bank Park",
    "MT": "Glacier Bank Park",
    "25 McDermott Lane, Kalispell, MT, 59901": "Glacier Bank Park",
    # Boise Hawks -- Memorial Stadium, Garden City ID (Boise metro)
    "Memorial": "Memorial Stadium",
    "Memorial Stadium": "Memorial Stadium",
    "Boise, ID": "Memorial Stadium",
    "Boise, Id": "Memorial Stadium",
    "Boise, Idaho": "Memorial Stadium",
    "Boise ID": "Memorial Stadium",
    "Garden City, ID": "Memorial Stadium",
    "5600 N Glenwood Street, Garden City, ID, 83714": "Memorial Stadium",
    # Idaho Falls Chukars -- Melaleuca Field, Idaho Falls ID. Two distinct
    # addresses show up interleaved through 2026 with no date clustering, so
    # this is scorer/template variance, not a mid-season venue change.
    "Melaleuca Field": "Melaleuca Field",
    "Melaleuca": "Melaleuca Field",
    "IF": "Melaleuca Field",
    "Idaho Falls": "Melaleuca Field",
    "The Luc": "Melaleuca Field",
    "900 JIM GARCHOW WAY,": "Melaleuca Field",
    "900 JIM GARCHOW WAY, IDAHO FALLS, ID, 83402-4776": "Melaleuca Field",
    "568 West Elva, Idaho Falls, ID, 83406": "Melaleuca Field",
    # Long Beach Coast -- Blair Field, Long Beach CA
    "Blair Field": "Blair Field",
    "Long Beach, CA": "Blair Field",
    "4700 Deukmejian Drive, Long Beach, CA, 90804": "Blair Field",
    # Modesto Roadsters -- Modern Woodmen Field, Modesto CA
    "Modern Woodmen Field": "Modern Woodmen Field",
    "Modern Woodman Field": "Modern Woodmen Field",
    "601 Neece Drive, Modesto, CA, 95351": "Modern Woodmen Field",
    # Rocky Mountain Vibes -- UCHealth Park (2024, Colorado Springs) /
    # Blocktickets Park (2025, Fort Collins after relocation)
    "UCHealth Park": "UCHealth Park",
    "Colorado Springs": "UCHealth Park",
    "blocktickets PARK": "Blocktickets Park",
    "Blocktickets Park": "Blocktickets Park",
    "Colorado Springs, CO": "Blocktickets Park",
    "W Oak St, Fort Colli": "Blocktickets Park",
    # Colorado Springs Sky Sox 2024 -- 4Rivers Equipment Park, Windsor CO.
    # (2025 Sky Sox is the traveling-team anomaly -- see the module docstring
    # for `canonicalize`; the 2025 Sky Sox shared Blocktickets Park with the
    # Vibes for the 11 designated-home games they did not play on the road.)
    "4Rivers Equipment": "4Rivers Equipment Park",
    "4Rivers": "4Rivers Equipment Park",
    "Windsor, CO": "4Rivers Equipment Park",
    "Windsor, Colo.": "4Rivers Equipment Park",
    "WIndsor, CO": "4Rivers Equipment Park",
}


def canonicalize(raw: Optional[str]) -> Optional[str]:
    """Fold a raw `venue_raw` string onto its physical ballpark name.

    Unknown strings pass through unchanged rather than being dropped or
    raising, so a new raw form widens the corpus's venue count by one
    (visibly, in the distinct-venue count `tests/test_venue.py` pins)
    instead of silently vanishing.
    """
    if raw is None:
        return None
    return CANONICAL_VENUE.get(raw, raw)
