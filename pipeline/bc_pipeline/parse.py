"""Parse raw HTML into a canonical game file under ``games/<season>/<game_id>.json``.

Future responsibility: turn the raw boxscore + play-by-play HTML into the asserted
primitives the schema defines (players, lineups, linescore, box, events, unparsed).
No parsing logic lives here yet.

# Implemented in issue #18/#19
"""
