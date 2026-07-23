"""File-based locks for cohort processing (ported from TRIDENT ``IO.py``).

Sentinel ``<path>.lock`` files let N processors split a cohort without
colliding: a processor skips locked-or-done slides, locks the one it is
working, and removes the lock when finished. The lock carries ``{pid,
hostname, created_at}`` so a crashed run's **stale** lock can be reported and
cleared. Pure stdlib — no torch, importable everywhere.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any


def _lock_path(path: str | Path, suffix: str | None) -> Path:
    """``<path>.lock`` (or ``<path>_<suffix>.lock``) next to ``path``."""
    p = Path(path)
    name = p.name if suffix is None else f"{p.name}_{suffix}"
    return p.with_name(f"{name}.lock")


def create_lock(path: str | Path, *, suffix: str | None = None) -> None:
    """Write a ``<path>.lock`` sentinel with ``{pid, hostname, created_at}``.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "slide.zarr"
        ...     create_lock(p)
        ...     is_locked(p)
        True
    """
    lock = _lock_path(path, suffix)
    payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": time.time(),
    }
    lock.write_text(json.dumps(payload))


def is_locked(path: str | Path, *, suffix: str | None = None) -> bool:
    """Return whether ``path`` is locked (its ``.lock`` file exists).

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     is_locked(Path(d) / "slide.zarr")  # nothing locked yet
        False
    """
    return _lock_path(path, suffix).exists()


def remove_lock(path: str | Path, *, suffix: str | None = None) -> None:
    """Remove ``path``'s lock — best-effort, tolerant if already gone.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "slide.zarr"
        ...     create_lock(p)
        ...     remove_lock(p)
        ...     remove_lock(p)  # idempotent — no error if absent
        ...     is_locked(p)
        False
    """
    # best-effort marker; tolerate races / permission quirks
    with contextlib.suppress(OSError):
        _lock_path(path, suffix).unlink(missing_ok=True)


def read_lock(
    path: str | Path, *, suffix: str | None = None
) -> dict[str, Any] | None:
    """Read the lock payload (``{pid, hostname, created_at}``) or ``None``.

    Used to report a **stale** lock (a crashed run's) — its pid/hostname/age.

    Example:
        >>> import os, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "slide.zarr"
        ...     create_lock(p)
        ...     read_lock(p)["pid"] == os.getpid()
        True
    """
    lock = _lock_path(path, suffix)
    if not lock.exists():
        return None
    try:
        return json.loads(lock.read_text())
    except (OSError, json.JSONDecodeError):
        return None


_DEFAULT_MAX_AGE = 24 * 3600


def _pid_is_running(pid: int) -> bool:
    """Return whether process ``pid`` is alive on this host.

    Example:
        >>> import os
        >>> _pid_is_running(os.getpid())
        True
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    return True


def _is_stale_lock_file(lock_path: Path, *, max_age_seconds: float) -> bool:
    """Whether the lock at ``lock_path`` is stale (a crashed run's).

    Same-host locks are judged by their PID alone (dead = stale, alive =
    live — age is never consulted, so a long-running job is not
    reclaimed). Cross-host / no-pid / unreadable locks fall back to age.
    """
    if not lock_path.exists():
        return False
    try:
        payload: dict[str, Any] = json.loads(lock_path.read_text())
    except (OSError, json.JSONDecodeError):
        payload = {}

    pid = payload.get("pid")
    if pid is not None and payload.get("hostname") == socket.gethostname():
        try:
            return not _pid_is_running(int(pid))
        except (TypeError, ValueError):
            pass  # unparseable pid → fall back to age

    age_ref = payload.get("created_at")
    if age_ref is None:
        try:
            age_ref = lock_path.stat().st_mtime
        except OSError:
            return False
    try:
        return (time.time() - float(age_ref)) >= max_age_seconds
    except (TypeError, ValueError):
        return False


def is_stale_lock(
    path: str | Path,
    *,
    suffix: str | None = None,
    max_age_seconds: float = _DEFAULT_MAX_AGE,
) -> bool:
    """Whether ``path``'s lock is stale (a crashed run's).

    Stale when its holder is a dead same-host process, or (cross-host /
    unreadable / legacy) when older than ``max_age_seconds``. A live
    same-host holder is never stale.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "slide.zarr"
        ...     create_lock(p)  # held by this live process
        ...     is_stale_lock(p)
        False
    """
    return _is_stale_lock_file(
        _lock_path(path, suffix), max_age_seconds=max_age_seconds
    )


def clear_locks(
    directory: str | Path,
    *,
    stale_only: bool,
    max_age_seconds: float = _DEFAULT_MAX_AGE,
) -> dict[str, int]:
    """Remove ``.lock`` files under ``directory`` (recursively).

    With ``stale_only`` only stale locks are removed (see
    :func:`is_stale_lock`); otherwise **every** lock is removed (the
    ``--clear-locks`` force path — use only when no run is active).

    Returns ``{"scanned", "removed", "kept"}``.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     clear_locks(Path(d), stale_only=True)
        {'scanned': 0, 'removed': 0, 'kept': 0}
    """
    scanned = removed = kept = 0
    for lock_path in Path(directory).rglob("*.lock"):
        scanned += 1
        drop = not stale_only or _is_stale_lock_file(
            lock_path, max_age_seconds=max_age_seconds
        )
        if not drop:
            kept += 1
            continue
        try:
            lock_path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            kept += 1
    return {"scanned": scanned, "removed": removed, "kept": kept}
