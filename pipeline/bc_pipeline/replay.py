"""Replay a parsed game to validate it and stamp the ``_derived`` base-out cache.

Future responsibility: fold the asserted ``events[].runners[]`` primitives forward,
reconstruct base-out state, cross-check it against the linescore/box oracles, and
stamp the regenerable ``_derived`` cache. No replay logic lives here yet.

# Implemented in issue #18/#19
"""
