"""Real HTTP transport (g4): the ONE place actual network code is allowed to
exist anywhere in this issue.

Everything else in the pipeline (schedule walking, pacing, challenge
detection, archiving, orchestration) is developed and tested against an
injected fake transport. This module supplies the production implementation
of that same seam -- a callable matching
:class:`bc_pipeline.fetcher.Transport` (``Callable[[str], FetchResponse]``)
that does a real ``requests.get`` and returns a
:class:`~bc_pipeline.fetcher.FetchResponse`.

This module is intentionally NOT exercised by any test in this gate (per the
handoff's test mode: "the real-transport function itself may be test-after
or inspection-only ... do not write a test that hits real network in this
gate"). ``pipeline/tests/test_transport.py`` only inspects its shape
(importable, callable, right arity) -- it never calls it.
"""

from __future__ import annotations

import requests

from bc_pipeline.fetcher import FetchResponse

#: How long to wait for a response before giving up. Generous but bounded --
#: a hung connection should not stall the paced fetch loop indefinitely.
DEFAULT_TIMEOUT_SECONDS: float = 30.0

#: A plain, honest user agent. Not disguised as a browser -- this pipeline
#: identifies itself.
DEFAULT_USER_AGENT: str = "baseball-coaster-pipeline/0.1 (+https://github.com/)"


def real_transport(url: str) -> FetchResponse:
    """Fetch ``url`` over real HTTP and return a :class:`FetchResponse`.

    Matches the :class:`bc_pipeline.fetcher.Transport` protocol so it can be
    handed directly to :class:`~bc_pipeline.fetcher.PacedFetcher` (or to
    :func:`bc_pipeline.fetch.run_pipeline`) in place of a fake transport --
    same orchestration code path, real network on this one seam.

    Does not raise on non-2xx status codes; it hands the status code and
    body straight to the caller, exactly like a fake test transport would,
    so challenge detection (``PacedFetcher``) sees real 202s/WAF pages the
    same way it sees fake ones in tests.
    """
    response = requests.get(
        url,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )
    return FetchResponse(status_code=response.status_code, body=response.text)
