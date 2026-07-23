"""The active run's id, ambient for state writers.

``run_ledger`` sets the current run id on entry; ``save_state`` reads it
to stamp each slide's ``last_run_id`` with the run that last touched it.
A plain module global would not nest or stay thread-safe, so this uses a
:class:`~contextvars.ContextVar`. Outside any run, ``current_run_id()``
is ``None`` and state writers leave ``last_run_id`` untouched.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_run_id: ContextVar[str | None] = ContextVar("coral_run_id", default=None)


def set_run_id(run_id: str | None) -> Token[str | None]:
    """Set the active run id; return a token for :func:`reset_run_id`.

    Example:
        >>> tok = set_run_id("abc123")
        >>> current_run_id()
        'abc123'
        >>> reset_run_id(tok)
        >>> current_run_id() is None
        True
    """
    return _run_id.set(run_id)


def reset_run_id(token: Token[str | None]) -> None:
    """Restore the run id to its value before this token's set."""
    _run_id.reset(token)


def current_run_id() -> str | None:
    """The active run's id, or ``None`` outside a run ledger."""
    return _run_id.get()
