"""Fetch raw boxscore/PBP HTML from the source site to local disk.

Future responsibility: retrieve the raw HTML for a game and store it on the PC
outside git (raw HTML is never committed).

# Implemented in issue #18/#19

g2 (this gate) implements the paced-fetcher-and-challenge-detection half in
:mod:`bc_pipeline.fetcher`; this module re-exports its public surface so
callers can do ``from bc_pipeline.fetch import PacedFetcher`` as well as
``from bc_pipeline.fetcher import PacedFetcher``. The raw-archive/checkpoint
writer (g3, storage) is NOT implemented here yet.
"""

from __future__ import annotations

from bc_pipeline.fetcher import (
    CHALLENGE_BACKOFF_SECONDS,
    ChallengeDetected,
    FetchResponse,
    FetchResult,
    PacedFetcher,
    Transport,
)

__all__ = [
    "CHALLENGE_BACKOFF_SECONDS",
    "ChallengeDetected",
    "FetchResponse",
    "FetchResult",
    "PacedFetcher",
    "Transport",
]
