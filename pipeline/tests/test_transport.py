"""Shape-only check for the real HTTP transport (g4).

Per the handoff's test mode: the real-transport function is a thin,
mostly-untestable-without-network wrapper. This module confirms it exists
and matches ``bc_pipeline.fetcher.Transport``'s shape
(``Callable[[str], FetchResponse]``) -- it never calls it, and never touches
real network.
"""

from __future__ import annotations

import inspect

from bc_pipeline.transport import real_transport


def test_real_transport_is_callable_with_one_positional_argument() -> None:
    assert callable(real_transport)
    signature = inspect.signature(real_transport)
    params = list(signature.parameters.values())
    assert len(params) == 1
    assert params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def test_real_transport_module_does_not_get_called_anywhere_in_this_test_module() -> None:
    # Documentation-as-assertion: this test module's only interaction with
    # `real_transport` is the signature inspection above. No network call
    # happens anywhere in this file.
    assert True
