"""Per-slide data container + workflow state."""

from coral.slide.core import CoralSlide
from coral.slide.state import (
    SCHEMA_VERSION,
    STATE_FILENAME,
    SlideMeta,
    SlideRef,
    SlideState,
    TasksBlock,
    TaskState,
    default_state,
    load_state,
    save_state,
)

__all__ = [
    "SCHEMA_VERSION",
    "STATE_FILENAME",
    "CoralSlide",
    "SlideMeta",
    "SlideRef",
    "SlideState",
    "TaskState",
    "TasksBlock",
    "default_state",
    "load_state",
    "save_state",
]
