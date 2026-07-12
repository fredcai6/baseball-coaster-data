"""Raw archive writer + checkpoint-backed idempotency index (g3).

Given a :class:`~bc_pipeline.fetcher.FetchResult` (url, status_code, body,
fetched_at), this module:

    1. Writes ``body`` to an immutable file under a configured archive root
       (:data:`PipelineConfig.archive_root`), named by the naming contract
       from the issue text: ``source-url + fetch-timestamp + content-hash``
       (see :func:`archive_filename` for the exact format). It NEVER
       overwrites an existing file -- a name collision raises
       ``FileExistsError`` rather than silently clobbering data.

    2. Maintains a JSON checkpoint file at
       :data:`PipelineConfig.checkpoint_path` mapping
       ``source-url -> {archived_path, fetched_at, content_hash, status}``.

The checkpoint -- NOT the archive directory's filenames -- is the sole,
authoritative answer to "have I already fetched this URL". This is a
deliberate, already-decided design choice (decision:raw-archive-layout): the
naming contract embeds a fetch-timestamp that by definition differs on every
fetch, so a filename-glob/re-derivation existence check is undecidable
*before* a fetch happens (you don't know the timestamp of a fetch that
hasn't occurred yet). :func:`should_fetch` and :func:`should_fetch_url`
therefore only ever consult the checkpoint dict/file -- never the archive
directory's contents.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from bc_pipeline.config import PipelineConfig
from bc_pipeline.fetcher import FetchResult

#: Number of hex characters kept from the full sha256 digest in filenames.
#: Truncated is fine per the handoff -- this is a collision-avoidance aid,
#: not a security hash; the checkpoint (not the filename) is the
#: authoritative idempotency index regardless.
CONTENT_HASH_HEX_LENGTH: int = 16

#: Longest allowed length of the URL-derived slug portion of a filename,
#: to keep archived filenames from becoming unwieldy for long URLs.
MAX_URL_SLUG_LENGTH: int = 80

_SLUG_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def content_hash(body: str) -> str:
    """Return a truncated sha256 hex digest of ``body``."""
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return digest[:CONTENT_HASH_HEX_LENGTH]


def _slugify_url(url: str) -> str:
    """Turn ``url`` into a filesystem-safe slug (no scheme separators, etc.)."""
    slug = _SLUG_UNSAFE_CHARS.sub("-", url).strip("-")
    return slug[:MAX_URL_SLUG_LENGTH] or "url"


def archive_filename(url: str, fetched_at: float, body: str) -> str:
    """Compute the archive filename for one fetch.

    Format (documented naming contract): ``<url-slug>__<fetched-at-us>__<hash>.html``

    - ``url-slug``: the source URL with any character outside
      ``[A-Za-z0-9._-]`` collapsed to ``-`` (scheme, host, path, query all
      folded into one slug), truncated to :data:`MAX_URL_SLUG_LENGTH` chars.
    - ``fetched-at-us``: the fetch timestamp (``FetchResult.fetched_at``,
      seconds since the epoch) as an integer *microsecond* epoch count, so
      two fetches of the same URL are distinguishable to sub-second
      precision.
    - ``hash``: :func:`content_hash` of the body (truncated sha256 hex).

    Worked example::

        archive_filename(
            "https://example.test/box/42?season=2025",
            1700000000.123456,
            "<html>...</html>",
        )
        # -> "https---example.test-box-42-season-2025__1700000000123456__<16-hex-chars>.html"
    """
    slug = _slugify_url(url)
    ts_us = int(round(fetched_at * 1_000_000))
    digest = content_hash(body)
    return f"{slug}__{ts_us}__{digest}.html"


def resolve_archive_root(config: PipelineConfig) -> Path:
    """Return the absolute, resolved archive root directory for ``config``."""
    return Path(config.archive_root).resolve()


def _resolve_checkpoint_path(config: PipelineConfig) -> Path:
    return Path(config.checkpoint_path).resolve()


# --- Checkpoint I/O -------------------------------------------------------


def load_checkpoint(checkpoint_path: Path | str) -> dict[str, dict[str, Any]]:
    """Load the checkpoint JSON at ``checkpoint_path``.

    Returns an empty dict if the file does not exist yet (first run).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(checkpoint_path: Path | str, data: dict[str, dict[str, Any]]) -> None:
    """Write ``data`` to ``checkpoint_path`` as JSON.

    Written via write-to-temp-then-rename in the same directory: a crash or
    interruption mid-write leaves the previous checkpoint file intact rather
    than a half-written/corrupt JSON file (cheap safety measure explicitly
    flagged as worthwhile in the handoff, given g5's kill+resume demo).
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp_name, path)  # atomic on both POSIX and Windows (NTFS)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


# --- Idempotency queries: checkpoint-keyed, never filename-derived -------


def should_fetch(checkpoint: dict[str, dict[str, Any]], url: str) -> bool:
    """Pure checkpoint-dict lookup: should ``url`` be fetched?

    Returns ``False`` only if ``url`` is present in ``checkpoint`` with
    ``status == "done"``; ``True`` otherwise (unseen URL, or present but not
    done). Never touches the filesystem -- no archive-directory scan/glob,
    by design (decision:raw-archive-layout).
    """
    entry = checkpoint.get(url)
    if entry is None:
        return True
    return entry.get("status") != "done"


def should_fetch_url(config: PipelineConfig, url: str) -> bool:
    """Load the checkpoint from ``config.checkpoint_path`` and check ``url``.

    Convenience wrapper around :func:`should_fetch` for callers that only
    have the config, not an already-loaded checkpoint dict.
    """
    checkpoint = load_checkpoint(_resolve_checkpoint_path(config))
    return should_fetch(checkpoint, url)


# --- Archive writer --------------------------------------------------------


def archive_result(result: FetchResult, config: PipelineConfig) -> dict[str, Any]:
    """Archive ``result.body`` to disk and record it in the checkpoint.

    Steps:
        1. Compute the archive path (:func:`archive_filename` under
           :func:`resolve_archive_root`).
        2. Raise :class:`FileExistsError` if that exact path already exists
           (never overwrite -- should be near-impossible given the
           timestamp+hash naming contract, but guarded regardless).
        3. Write the body to that path.
        4. Load the existing checkpoint, add/replace this URL's entry, and
           save it back (write-temp-then-rename).

    Returns the checkpoint entry written for ``result.url``.
    """
    archive_root = resolve_archive_root(config)
    archive_root.mkdir(parents=True, exist_ok=True)

    filename = archive_filename(result.url, result.fetched_at, result.body)
    archived_path = archive_root / filename

    if archived_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing archive file: {archived_path}"
        )

    archived_path.write_text(result.body, encoding="utf-8")

    entry = {
        "archived_path": str(archived_path),
        "fetched_at": result.fetched_at,
        "content_hash": content_hash(result.body),
        "status": "done",
    }

    checkpoint_path = _resolve_checkpoint_path(config)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint[result.url] = entry
    save_checkpoint(checkpoint_path, checkpoint)

    return entry
