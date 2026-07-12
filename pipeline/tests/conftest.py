"""Test-collection bootstrap: make ``bc_pipeline`` importable without an install.

The pipeline package (``pipeline/bc_pipeline``) is not pip-installed (editable
or otherwise) in this worktree yet, so pytest's default rootdir insertion
(which adds ``pipeline/tests`` to ``sys.path``, not ``pipeline/``) leaves
``import bc_pipeline`` unresolvable. Insert the ``pipeline/`` directory itself
so tests can import the package directly, matching the documented
verification command (``python -m pytest pipeline/tests -k schedule -q`` run
from the repo root).
"""

import sys
from pathlib import Path

_PIPELINE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PIPELINE_ROOT not in sys.path:
    sys.path.insert(0, _PIPELINE_ROOT)
