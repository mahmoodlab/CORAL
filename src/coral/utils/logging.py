"""Rich logging setup for CORAL command-line entry points.

CORAL's library code only *emits* log records (module-level loggers); it
never configures logging itself. A consumer — typically
the CLI — opts in by calling :func:`setup_logging`, which routes CORAL's
records through a coloured Rich handler so a user sees clean, meaningful
per-step output without the library imposing handlers on embedders.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from rich.highlighter import NullHighlighter
from rich.logging import RichHandler

__all__ = ["add_run_logfile", "remove_run_logfile", "setup_logging"]

logger = logging.getLogger(__name__)


class _BlankAwareRichHandler(RichHandler):
    """RichHandler with two CORAL-CLI tweaks.

    - An empty message renders as a true blank line (section spacing),
      not an ``INFO``-prefixed line.
    - A record carrying ``no_highlight=True`` (via the logger's
      ``extra=``) skips Rich's auto-highlighter, so an explicitly coloured
      line (the green/red marker map) is not recoloured — while every
      other line keeps the default number/path highlighting.
    """

    _null = NullHighlighter()

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() == "":
            self.console.print()
            return
        if getattr(record, "no_highlight", False):
            saved, self.highlighter = self.highlighter, self._null
            try:
                super().emit(record)
            finally:
                self.highlighter = saved
            return
        super().emit(record)


def setup_logging(level: int = logging.INFO) -> None:
    """Install a Rich handler on the ``coral`` logger (idempotent).

    Shows INFO-level narration + WARNING/ERROR from any ``coral.*`` logger,
    coloured by level, without timestamps or source paths (clean for a
    CLI). Safe to call repeatedly — each call reinstalls a fresh handler
    so its console binds to the current stdio.

    Args:
        level: Minimum level shown (default :data:`logging.INFO`).

    Example:
        >>> import logging
        >>> setup_logging(logging.INFO)
        >>> logging.getLogger("coral").level == logging.INFO
        True
    """
    coral_logger = logging.getLogger("coral")
    coral_logger.setLevel(level)
    for existing in list(coral_logger.handlers):
        if isinstance(existing, RichHandler):
            coral_logger.removeHandler(existing)
    handler = _BlankAwareRichHandler(
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    coral_logger.addHandler(handler)
    coral_logger.propagate = False


def add_run_logfile(
    job_dir: str | Path, run_id: str
) -> logging.FileHandler | None:
    """Tee the ``coral`` logger to ``<job_dir>/logs/<run_id>.log``.

    Attaches a plain-text ``FileHandler`` (INFO+) to the ``coral`` logger so
    a run's full narration is persisted next to its run manifest. Child
    loggers (``coral.cli.*``, ``coral.io.*``, …) propagate up, so one handler
    captures the whole run. Returns the handler (pass it to
    :func:`remove_run_logfile`), or ``None`` if the file could not be
    opened — logging must never fail a run.

    Args:
        job_dir: The run's job directory (``logs/`` is created in it).
        run_id: The run id; the log is ``logs/<run_id>.log``.

    Returns:
        The attached handler, or ``None`` on failure.

    Example:
        >>> import logging, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as d:
        ...     h = add_run_logfile(d, "abc123")
        ...     logging.getLogger("coral.demo").info("hi")
        ...     remove_run_logfile(h)
        ...     (Path(d) / "logs" / "abc123.log").exists()
        True
    """
    try:
        logs_dir = Path(job_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(
            logs_dir / f"{run_id}.log", encoding="utf-8"
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        coral_logger = logging.getLogger("coral")
        if coral_logger.getEffectiveLevel() > logging.INFO:
            coral_logger.setLevel(logging.INFO)
        coral_logger.addHandler(handler)
        return handler
    except Exception as exc:  # noqa: BLE001 — logging must not fail a run
        logger.debug("could not open run logfile: %s", exc)
        return None


def remove_run_logfile(handler: logging.FileHandler | None) -> None:
    """Detach + close a handler from :func:`add_run_logfile` (tolerant).

    Args:
        handler: The handler returned by :func:`add_run_logfile`, or
            ``None`` (a no-op).

    Example:
        >>> remove_run_logfile(None)  # no-op, no error
    """
    if handler is None:
        return
    with contextlib.suppress(Exception):
        logging.getLogger("coral").removeHandler(handler)
        handler.flush()
        handler.close()
