"""html_struct — generic stdlib-only DOM helpers over ``html.parser``.

Deliberately GENERIC. This module builds a lightweight DOM tree and offers
navigation helpers plus a play-by-play (PBP) *pane* locator. It must NOT
interpret the linescore or box tables (which column is R/H/E, which row is
which team) — that semantic interpretation is written independently, twice,
by the parser (later gate) and the replayer's oracle (later gate), so a
table-reading bug cannot be inherited by both (spec D2 independence). This
module only ever looks at structural facts: tag names, attributes, and
verbatim text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Iterator, List, Optional, Tuple, Union

# HTML5 void elements: never receive a matching end tag.
_VOID_ELEMENTS = frozenset(
    {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }
)

_PANE_ID_RE = re.compile(r"^pbp-inning-(\d+)$")


@dataclass
class Node:
    """A single element in the lightweight DOM tree.

    ``tag`` is ``None`` only for the synthetic document root. ``children`` is
    a mixed list of ``Node`` (elements) and ``str`` (raw text runs).
    """

    tag: Optional[str]
    attrs: dict = field(default_factory=dict)
    children: List[Union["Node", str]] = field(default_factory=list)


@dataclass
class Cell:
    """One ordered PBP play cell within an inning pane."""

    text: str
    is_strong: bool


class _DOMBuilder(HTMLParser):
    """Builds a `Node` tree from HTML text, tolerant of malformed markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node(tag=None)
        self._stack: List[Node] = [self.root]

    def handle_starttag(self, tag: str, attrs) -> None:
        node = Node(tag=tag, attrs=dict(attrs))
        self._stack[-1].children.append(node)
        if tag not in _VOID_ELEMENTS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs) -> None:
        # Explicit self-closing form, e.g. <br/>. Never pushed onto the stack.
        node = Node(tag=tag, attrs=dict(attrs))
        self._stack[-1].children.append(node)

    def handle_endtag(self, tag: str) -> None:
        # Close the nearest matching open element; ignore unmatched/stray
        # end tags rather than raising, since real-world HTML is often
        # imperfectly balanced.
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                return

    def handle_data(self, data: str) -> None:
        if data:
            self._stack[-1].children.append(data)


def parse_html(text: str) -> Node:
    """Parse ``text`` into a lightweight DOM tree rooted at a synthetic Node.

    Uses stdlib `html.parser.HTMLParser` only. Void/self-closing tags (br,
    img, input, hr, meta, link, ...) never expect a matching end tag.
    """
    builder = _DOMBuilder()
    builder.feed(text)
    builder.close()
    return builder.root


def _walk_nodes(node: Node) -> Iterator[Node]:
    """Yield ``node`` and every descendant `Node` in document (pre-)order."""
    yield node
    for child in node.children:
        if isinstance(child, Node):
            yield from _walk_nodes(child)


def find_by_id(root: Node, id_: str) -> Optional[Node]:
    """Return the first descendant (or root) whose ``id`` attribute matches."""
    for node in _walk_nodes(root):
        if node.attrs.get("id") == id_:
            return node
    return None


def find_all_by_class(root: Node, cls: str) -> List[Node]:
    """Return every descendant (or root) whose ``class`` list contains ``cls``."""
    out = []
    for node in _walk_nodes(root):
        classes = (node.attrs.get("class") or "").split()
        if cls in classes:
            out.append(node)
    return out


def find_all(root: Node, tag: str) -> List[Node]:
    """Return every descendant (or root) with the given tag name."""
    return [node for node in _walk_nodes(root) if node.tag == tag]


def iter_rows(table_node: Node) -> Iterator[Node]:
    """Yield every `<tr>` descendant of ``table_node``, in document order."""
    yield from find_all(table_node, "tr")


def cell_texts(row: Node) -> List[str]:
    """Return the whitespace-collapsed text of each direct `<td>`/`<th>` child."""
    return [
        text_of(child)
        for child in row.children
        if isinstance(child, Node) and child.tag in ("td", "th")
    ]


def text_of(node: Node) -> str:
    """Concatenate all descendant text under ``node``, trimming ONLY the
    surrounding whitespace of the whole string.

    Internal whitespace (mid-content newlines, tabs, multi-space runs) is
    preserved verbatim: later gates need the exact narrative string (for
    ``unparsed[]`` fidelity and so the parser can decide its own narrative
    normalization). Collapsing internal whitespace here would bake a
    normalization decision into the wrong, generic seam.
    """
    parts: List[str] = []

    def walk(n: Union[Node, str]) -> None:
        if isinstance(n, str):
            parts.append(n)
        else:
            for c in n.children:
                walk(c)

    walk(node)
    return "".join(parts).strip()


def is_strong(node: Node) -> bool:
    """True iff every significant (non-whitespace-only) direct child of
    ``node`` is a `<strong>` element — i.e. the node's rendered content is
    entirely wrapped in `<strong>`. Used later to detect scoring plays.
    """
    significant = [
        c for c in node.children if not (isinstance(c, str) and c.strip() == "")
    ]
    if not significant:
        return False
    return all(isinstance(c, Node) and c.tag == "strong" for c in significant)


def iter_pbp_panes(root: Node) -> List[Tuple[int, List[Cell]]]:
    """For each `id="pbp-inning-N"` pane, return the ordered play cells.

    Each pane yields ``(inning, cells)`` where ``cells`` are the ordered
    `<td class="text">` descendants of that pane (verbatim, whitespace-
    trimmed text plus an `is_strong` flag). Top/bottom halves are NOT split
    here — that is parser semantics for a later gate; this just returns the
    ordered cells for the whole inning pane.
    """
    panes: List[Tuple[int, List[Cell]]] = []
    for node in _walk_nodes(root):
        id_ = node.attrs.get("id") or ""
        m = _PANE_ID_RE.match(id_)
        if not m:
            continue
        inning = int(m.group(1))
        cells = [
            Cell(text=text_of(td), is_strong=is_strong(td))
            for td in find_all_by_class(node, "text")
            if td.tag == "td"
        ]
        panes.append((inning, cells))
    return panes


def has_pbp_panes(root: Node) -> bool:
    """True iff at least one `id="pbp-inning-N"` pane exists.

    This is the non-final / negative-path detector: pages without PBP panes
    (e.g. a "today"/pre-game page) have none.
    """
    for node in _walk_nodes(root):
        id_ = node.attrs.get("id") or ""
        if _PANE_ID_RE.match(id_):
            return True
    return False
