"""Static AST import-guard test (spec D2 independence, issue #19 gate g6).

Protected intent: ``replay.py`` must be an INDEPENDENT oracle/validator. If it
ever imports ``parse`` (directly or via ``from . import parse`` /
``from bc_pipeline import parse`` / ``from bc_pipeline.parse import ...``), a
single shared table-reading bug could fool both the parser and its own
replay check -- defeating the entire gate. This test parses ``replay.py``'s
own source with ``ast`` (never actually importing/exec'ing anything parser-
related) and fails loudly if any import statement names the ``parse``
module, at any depth.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPLAY_PATH = (
    Path(__file__).resolve().parent.parent / "pipeline" / "bc_pipeline" / "replay.py"
)


def _imported_module_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # node.module is None for a bare `from . import X` -- capture the
            # imported names themselves in that case too.
            if node.module:
                names.append(node.module)
            for alias in node.names:
                names.append(alias.name)
    return names


def test_replay_source_does_not_import_parse():
    source = REPLAY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(REPLAY_PATH))
    imported = _imported_module_names(tree)
    offending = [
        name
        for name in imported
        if name == "parse" or name.endswith(".parse") or name == "bc_pipeline.parse"
    ]
    assert offending == [], (
        f"replay.py must not import parse.py (spec D2 independence); "
        f"found import(s): {offending}"
    )


def test_replay_module_does_not_have_parse_in_sys_modules_dependency():
    # Belt-and-suspenders runtime check: importing bc_pipeline.replay alone
    # (fresh interpreter state within this test process is not guaranteed,
    # but we can at least assert replay's own module dict never binds a
    # name literally called `parse` pointing at the parse module).
    import bc_pipeline.replay as replay_module

    assert "parse" not in replay_module.__dict__, (
        "replay.py's namespace must never bind a `parse` module reference"
    )
