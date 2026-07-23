"""Progress-bar helpers shared by the CLI and slide methods.

Keeps one active tqdm bar in a contextvar so stage result text can update
live as a postfix. Completed bars use ``leave=True`` so they remain in the
terminal log. When the bar is disabled (CI / pipes), the status is written
as a plain ``desc: status`` line instead.

On success the bar is filled to 100% before close. On cancel (Ctrl+C) or a
``FAILED`` / ``cancelled`` status, progress is left where it stopped — it
does **not** jump to 100%.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from tqdm.std import tqdm

__all__ = [
    "activate_bar",
    "bar_status",
    "bar_write",
    "get_active_bar",
    "per_image_bar",
]

_active_bar: ContextVar[Any | None] = ContextVar(
    "coral_active_bar", default=None
)


def bar_write(msg: str) -> None:
    """Write a line without breaking a live tqdm bar."""
    tqdm.write(msg)


def get_active_bar() -> Any | None:  # noqa: ANN401
    """Return the current progress bar, if any."""
    return _active_bar.get()


def bar_status(msg: str) -> None:
    """Record stage result on the active bar (and for disabled-bar logs).

    Strips a leading 6-space indent so the same string works as a
    standalone line or as an on-bar status. When no bar is active, writes
    the message immediately.
    """
    text = msg[6:] if msg.startswith("      ") else msg
    bar = _active_bar.get()
    if bar is None:
        bar_write(msg if msg.startswith("      ") else f"      {text}")
        return
    bar._coral_status = text  # noqa: SLF001 — stash for close / CI fallback
    if not bar.disable:
        bar.set_postfix_str(text, refresh=True)


def _is_error_status(status: str | None) -> bool:
    """True when the bar should keep its last ``n`` (not fill to 100%)."""
    if status is None:
        return False
    head = status.split("—", 1)[0].strip().lower()
    return head.startswith(("failed", "cancelled", "interrupted"))


def _finish_bar(
    pbar: Any,  # noqa: ANN401
    *,
    desc: str,
    fill: bool,
) -> None:
    """Close the bar; fill to 100% only on a clean finish."""
    status = getattr(pbar, "_coral_status", None)
    if status is not None and not pbar.disable:
        pbar.set_postfix_str(status, refresh=True)
    total = pbar.total
    if (
        fill
        and not _is_error_status(status)
        and total is not None
        and pbar.n < total
    ):
        pbar.update(total - pbar.n)
    pbar.close()
    if status is not None and pbar.disable:
        bar_write(f"{desc}: {status}")


@contextmanager
def per_image_bar(
    *,
    desc: str,
    total: int,
    unit: str = "image",
    disable: bool | None = None,
) -> Iterator[Any]:
    """One image's bar; stays in the log with its final postfix.

    ``leave=True`` so the completed bar (with stage result in the
    postfix) remains visible. Inner work may call :meth:`tqdm.reset` to
    replace ``total`` with a data-derived count (e.g. CARTA tile batches).
    When disabled, the status is written as ``{desc}: {status}`` on close.
    Ctrl+C / ``FAILED`` leave the bar at its last tick (no jump to 100%).

    Example:
        >>> with per_image_bar(desc="demo", total=1, disable=True) as bar:
        ...     bar_status("42% tissue")
        ...     bar.update(1)
    """
    pbar: Any = tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=True,
        disable=disable,
    )
    pbar._coral_status = None  # noqa: SLF001
    token = _active_bar.set(pbar)
    try:
        yield pbar
    except BaseException:
        if getattr(pbar, "_coral_status", None) is None:
            # Ctrl+C and other aborts — keep partial progress visible.
            bar_status("cancelled")
        _active_bar.reset(token)
        _finish_bar(pbar, desc=desc, fill=False)
        raise
    else:
        _active_bar.reset(token)
        _finish_bar(pbar, desc=desc, fill=True)


@contextmanager
def activate_bar(pbar: Any) -> Iterator[Any]:  # noqa: ANN401
    """Register an existing tqdm bar as the active status target."""
    pbar._coral_status = getattr(pbar, "_coral_status", None)  # noqa: SLF001
    token = _active_bar.set(pbar)
    try:
        yield pbar
    except BaseException:
        if getattr(pbar, "_coral_status", None) is None:
            bar_status("cancelled")
        _active_bar.reset(token)
        desc = getattr(pbar, "desc", None) or "done"
        _finish_bar(pbar, desc=str(desc), fill=False)
        raise
    else:
        _active_bar.reset(token)
        desc = getattr(pbar, "desc", None) or "done"
        _finish_bar(pbar, desc=str(desc), fill=True)
