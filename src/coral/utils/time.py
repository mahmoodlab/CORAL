"""ISO-8601 timestamps and epoch helpers."""

from __future__ import annotations

import time
from datetime import datetime


def now_iso() -> str:
    """Return the current local time as an ISO-8601 string with TZ.

    Example output: ``"2026-05-21T15:30:45-04:00"``. Sort-friendly
    lexicographically when machines share a timezone. Matches the
    format used by TRIDENT and other Mahmood Lab tooling.

    Note:
        Returns LOCAL time with offset. CORAL v0.1 runs on a single
        machine; if multi-GPU / multi-host (FFL) adds distributed
        workers, revisit this to use UTC for cross-machine
        consistency.

    Returns:
        Timestamp string.

    Example:
        >>> stamp = now_iso()
        >>> "T" in stamp
        True
        >>> len(stamp) >= 19
        True
    """
    return datetime.now().astimezone().isoformat(timespec="seconds")


def epoch_now() -> float:
    """Return the current epoch time as seconds since the Unix epoch.

    Returns:
        Current time as a float.

    Example:
        >>> import time
        >>> abs(epoch_now() - time.time()) < 1.0
        True
    """
    return time.time()


def fmt_duration(seconds: float) -> str:
    """Format a duration for CLI / per-slide result lines.

    Sub-second times use three decimals so fast steps (e.g. patching)
    do not collapse to a misleading ``0.0s``.

    Example:
        >>> fmt_duration(0.042)
        '0.042s'
        >>> fmt_duration(8.4)
        '8.4s'
        >>> fmt_duration(92)
        '1m 32s'
    """
    if seconds < 1.0:
        return f"{seconds:.3f}s"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"
