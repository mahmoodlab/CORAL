"""Atomic file writes via ``<path>.tmp`` + :func:`os.replace`.

Guarantees that no partial-write file is visible at the target path.
``os.replace`` is atomic on POSIX, and is implemented via
``MoveFileEx`` on Windows (atomic in normal cases; edge cases exist
for files held open by other processes). Used by CORAL for
``state.json``, run manifests, ``summary.md`` — anywhere a partial
write could corrupt downstream readers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:  # noqa: ANN401
    """Write ``data`` to ``path`` atomically as pretty-printed JSON.

    Writes first to ``<path>.tmp``, then renames over ``path``.

    Args:
        path: Target file path. Parent directory must already exist.
        data: JSON-serialisable Python object.

    Raises:
        TypeError: If ``data`` contains values ``json.dumps`` cannot
            serialise (e.g. raw ``Path`` or ``datetime``).
        FileNotFoundError: If the parent directory does not exist.

    Example:
        >>> import json, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "demo.json"
        ...     atomic_write_json(fp, {"a": 1, "b": [2, 3]})
        ...     json.loads(fp.read_text()) == {"a": 1, "b": [2, 3]}
        True
    """
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_write_text(path: Path, content: str) -> None:
    r"""Write ``content`` to ``path`` atomically.

    A trailing newline is appended if missing.

    Args:
        path: Target file path.
        content: Text to write.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     fp = Path(d) / "demo.txt"
        ...     atomic_write_text(fp, "hello")
        ...     fp.read_text()
        'hello\n'
    """
    if not content.endswith("\n"):
        content = content + "\n"
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)
